#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <inttypes.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/mempolicy.h>
#include <poll.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
#include <infiniband/verbs.h>
#endif

namespace {

static constexpr uint32_t PIPELINE_BRICK_MAGIC       = 0x5042524b;
static constexpr uint32_t PIPELINE_BRICK_VERSION     = 4;
static constexpr uint32_t PIPELINE_FLAG_WANT_LOGITS  = 1u << 0;
static constexpr uint32_t PIPELINE_FLAG_STOP         = 1u << 1;
static constexpr uint32_t PIPELINE_FLAG_TOKEN        = 1u << 2;
static constexpr uint32_t PIPELINE_FLAG_HIDDEN_F16   = 1u << 3;
static constexpr uint32_t PIPELINE_FLAG_DECODE       = 1u << 4;
static constexpr uint32_t PIPELINE_FLAG_TRAFFIC_ONLY = 1u << 5;
static constexpr uint32_t PIPELINE_SLOT_EMPTY        = 0;
static constexpr uint32_t PIPELINE_SLOT_FULL         = 1;
static constexpr uint32_t PIPELINE_N_SLOTS           = 64;
// Default model geometry (Qwen3-4B). Override with --n-layer / --n-embd for other models.
static constexpr int32_t  DEFAULT_N_LAYER            = 36;
static constexpr int32_t  DEFAULT_N_EMBD             = 2560;

static const char * PIPELINE_PARALLEL_SYSTEM_PROMPT =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:
Recommend a nice restaurant in the area.
Assistant:
I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User:
Who is Richard Feynman?
Assistant:
Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
)";

enum class doorbell_mode {
    write,
    poll,
    ioctl,
};

enum class transport_kind {
    ntb_mw,
    cxl,
    ib_rdma,
    zni_rdma,
};

enum class hidden_dtype {
    f32,
    f16,
};

enum class domain_mode {
    none,
    single,
    dual,
};

struct pipeline_args {
    std::string model;
    std::string prompt;
    std::string prompt_file;
    std::string tx_mw;
    std::string rx_mw;
    std::string tx_doorbell;
    std::string rx_doorbell;
    std::string numa_cpus;
    std::string head_numa;
    std::string tail_numa;
    std::string tail_kv_numa;
    std::string stage_numa;
    std::string up_tx_mw;
    std::string up_rx_mw;
    std::string down_tx_mw;
    std::string down_rx_mw;
    std::string up_rdmadev;
    std::string down_rdmadev;
    std::string up_rdma_local_info;
    std::string up_rdma_peer_info;
    std::string down_rdma_local_info;
    std::string down_rdma_peer_info;

    llama_pipeline_brick_role role = LLAMA_PIPELINE_BRICK_ROLE_NONE;
    transport_kind transport = transport_kind::ntb_mw;
    transport_kind up_transport = transport_kind::ntb_mw;
    transport_kind down_transport = transport_kind::ntb_mw;
    doorbell_mode db_mode = doorbell_mode::write;
    domain_mode domain = domain_mode::none;

    int32_t brick_id      = -1;
    int32_t peer_brick_id = -1;
    int32_t layer_start   = -1;
    int32_t layer_end     = -1;
    int32_t ctx_size      = 2048;
    int32_t threads       = 8;
    int32_t n_predict     = 16;
    int32_t bricks        = 2;
    int32_t stage_id      = -1;
    int32_t stage_count   = 0;
    int32_t parallel      = 1;
    int32_t numa_tp       = 1;
    int32_t tp_rank       = 0;
    int32_t tp_size       = 1;
    int32_t prefill_chunk = 32;
    int32_t pipeline_microbatch = 0;
    int32_t n_layer       = DEFAULT_N_LAYER;  // override via --n-layer for non-default models
    int32_t n_embd_arg    = DEFAULT_N_EMBD;   // override via --n-embd for non-default models
    int32_t stream_kv_sink   = 16;
    int32_t stream_kv_recent = 128;
    int32_t rdma_port = 1;
    int32_t rdma_gid_index = 0;
    int32_t rdma_mtu = 1024;

    hidden_dtype hidden_type = hidden_dtype::f16;

    bool hardware = false;
    bool single_system = false;
    bool self_test_ntb = false;
    bool verbose = false;
    bool quiet = false;
    bool stream_kv = false;
    bool async_pipeline = false;
    int32_t naive_transfer_mult = 1;  // 1=normal (hidden state only), >1=simulate naive pipeline that transfers N-1 extra copies of activation-size data per send
    bool naive_kv_cross = false;      // simulate naive scenario: tail's KV cache is on peer CPU, every attention reads KV cross-CXL
    int32_t naive_kv_cross_node = -1; // node where the simulated remote KV cache lives (peer CPU's node)
    std::vector<std::string> async_prompts;
};

struct brick_packet_header {
    uint32_t magic;
    uint32_t version;
    uint32_t slot;
    uint32_t flags;
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  pos;
    int32_t  seq_id;
    uint64_t payload_bytes;
};

struct token_payload {
    int32_t seq_id;
    int32_t token;
};

struct hidden_token_meta {
    int32_t seq_id;
    int32_t pos;
    uint32_t flags;
    uint32_t reserved;
};

struct recv_packet {
    brick_packet_header header;
    std::vector<uint8_t> payload;
};

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

static size_t transport_window_size(size_t max_payload_bytes) {
    const size_t slot_size = align_up(sizeof(uint32_t) + sizeof(brick_packet_header) + max_payload_bytes, 64);
    return slot_size * PIPELINE_N_SLOTS;
}

static void sleep_us(long usec) {
    timespec ts;
    ts.tv_sec = usec / 1000000;
    ts.tv_nsec = (usec % 1000000) * 1000;
    nanosleep(&ts, nullptr);
}

static std::string role_name(llama_pipeline_brick_role role) {
    switch (role) {
        case LLAMA_PIPELINE_BRICK_ROLE_HEAD: return "head";
        case LLAMA_PIPELINE_BRICK_ROLE_TAIL: return "tail";
        case LLAMA_PIPELINE_BRICK_ROLE_STAGE: return "stage";
        default: return "none";
    }
}

static std::string tp_stats_role_label(const pipeline_args & args) {
    if (args.stage_id >= 0) {
        return "stage" + std::to_string(args.stage_id);
    }
    return role_name(args.role);
}

static void exit_tp_child(const pipeline_args & args, int rc) {
    const std::string label = tp_stats_role_label(args);
    ggml_tp_print_stats("pipeline-brick TP all-reduce:", label.c_str());
    fflush(stderr);
    _exit(rc);
}

static std::string transport_name(transport_kind kind) {
    switch (kind) {
        case transport_kind::ntb_mw:   return "ntb-mw";
        case transport_kind::cxl:      return "cxl";
        case transport_kind::ib_rdma:  return "ib-rdma";
        case transport_kind::zni_rdma: return "zni-rdma";
    }
    return "unknown";
}

static transport_kind parse_transport_kind(const std::string & value) {
    if (value == "ntb-mw") {
        return transport_kind::ntb_mw;
    }
    if (value == "cxl") {
        return transport_kind::cxl;
    }
    if (value == "ib-rdma") {
        return transport_kind::ib_rdma;
    }
    if (value == "zni-rdma") {
        return transport_kind::zni_rdma;
    }
    throw std::runtime_error("unknown transport: " + value + " (supported: ntb-mw, cxl, ib-rdma, zni-rdma)");
}

static const char * hidden_dtype_name(hidden_dtype type) {
    switch (type) {
        case hidden_dtype::f32:  return "f32";
        case hidden_dtype::f16: return "f16";
    }
    return "unknown";
}

static size_t hidden_dtype_size(hidden_dtype type) {
    switch (type) {
        case hidden_dtype::f32:  return sizeof(float);
        case hidden_dtype::f16: return sizeof(ggml_fp16_t);
    }
    return sizeof(float);
}

static hidden_dtype parse_hidden_dtype(const std::string & value) {
    if (value == "f32") {
        return hidden_dtype::f32;
    }
    if (value == "f16") {
        return hidden_dtype::f16;
    }
    throw std::runtime_error("unknown --hidden-dtype: " + value);
}

static int32_t max_micro_batch_tokens(const pipeline_args & args) {
    return std::max(args.parallel, args.parallel * args.prefill_chunk);
}

// Effective model geometry. Values <=0 fall back to the Qwen3-4B default.
static int32_t effective_n_layer(const pipeline_args & args) {
    return args.n_layer > 0 ? args.n_layer : DEFAULT_N_LAYER;
}
static int32_t effective_n_embd(const pipeline_args & args) {
    return args.n_embd_arg > 0 ? args.n_embd_arg : DEFAULT_N_EMBD;
}
static int32_t effective_split_layer(const pipeline_args & args) {
    return effective_n_layer(args) / 2;
}

static size_t max_hidden_payload_bytes(const pipeline_args & args) {
    const size_t n_tokens = (size_t) max_micro_batch_tokens(args);
    return n_tokens * sizeof(hidden_token_meta) +
        n_tokens * (size_t) effective_n_embd(args) * hidden_dtype_size(args.hidden_type);
}

class fd_handle {
public:
    fd_handle() = default;
    explicit fd_handle(int fd) : fd(fd) {}
    fd_handle(const fd_handle &) = delete;
    fd_handle & operator=(const fd_handle &) = delete;

    fd_handle(fd_handle && other) noexcept {
        fd = other.fd;
        other.fd = -1;
    }

    fd_handle & operator=(fd_handle && other) noexcept {
        if (this != &other) {
            reset();
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~fd_handle() {
        reset();
    }

    int get() const {
        return fd;
    }

    bool valid() const {
        return fd >= 0;
    }

    void reset(int new_fd = -1) {
        if (fd >= 0) {
            close(fd);
        }
        fd = new_fd;
    }

private:
    int fd = -1;
};

class ntb_mw_transport {
public:
    ntb_mw_transport() = default;
    ntb_mw_transport(const ntb_mw_transport &) = delete;
    ntb_mw_transport & operator=(const ntb_mw_transport &) = delete;

    ntb_mw_transport(ntb_mw_transport && other) noexcept {
        *this = std::move(other);
    }

    ntb_mw_transport & operator=(ntb_mw_transport && other) noexcept {
        if (this != &other) {
            if (tx_base != MAP_FAILED) {
                munmap(tx_base, window_size);
            }
            if (rx_base != MAP_FAILED) {
                munmap(rx_base, window_size);
            }
            tx_mw_fd = std::move(other.tx_mw_fd);
            rx_mw_fd = std::move(other.rx_mw_fd);
            tx_db_fd = std::move(other.tx_db_fd);
            rx_db_fd = std::move(other.rx_db_fd);
            tx_base = other.tx_base;
            rx_base = other.rx_base;
            slot_size = other.slot_size;
            window_size = other.window_size;
            tx_slot = other.tx_slot;
            rx_slot = other.rx_slot;
            db_mode = other.db_mode;
            other.tx_base = MAP_FAILED;
            other.rx_base = MAP_FAILED;
            other.slot_size = 0;
            other.window_size = 0;
            other.tx_slot = 0;
            other.rx_slot = 0;
        }
        return *this;
    }

    ~ntb_mw_transport() {
        if (tx_base != MAP_FAILED) {
            munmap(tx_base, window_size);
        }
        if (rx_base != MAP_FAILED) {
            munmap(rx_base, window_size);
        }
    }

    static ntb_mw_transport open_transport(const pipeline_args & args, size_t max_payload_bytes) {
        ntb_mw_transport t;
        t.db_mode = args.db_mode;
        t.slot_size = align_up(sizeof(uint32_t) + sizeof(brick_packet_header) + max_payload_bytes, 64);
        t.window_size = t.slot_size * PIPELINE_N_SLOTS;

        t.tx_mw_fd.reset(open_required(args.tx_mw, O_RDWR | O_SYNC, "--tx-mw"));
        t.rx_mw_fd.reset(open_required(args.rx_mw, O_RDWR | O_SYNC, "--rx-mw"));

        if (args.db_mode == doorbell_mode::write) {
            t.tx_db_fd.reset(open_required(args.tx_doorbell, O_RDWR | O_SYNC, "--tx-doorbell"));
            t.rx_db_fd.reset(open_required(args.rx_doorbell, O_RDWR | O_SYNC, "--rx-doorbell"));
        } else if (args.db_mode == doorbell_mode::ioctl) {
            throw std::runtime_error("--doorbell-mode ioctl needs a vendor ioctl adapter; use write or poll for this build");
        }

        t.tx_base = mmap(nullptr, t.window_size, PROT_READ | PROT_WRITE, MAP_SHARED, t.tx_mw_fd.get(), 0);
        if (t.tx_base == MAP_FAILED) {
            throw std::runtime_error("mmap --tx-mw failed: " + std::string(strerror(errno)));
        }

        t.rx_base = mmap(nullptr, t.window_size, PROT_READ | PROT_WRITE, MAP_SHARED, t.rx_mw_fd.get(), 0);
        if (t.rx_base == MAP_FAILED) {
            throw std::runtime_error("mmap --rx-mw failed: " + std::string(strerror(errno)));
        }

        t.clear_tx_slots();
        fprintf(stderr,
                "pipeline-brick transport: ntb-mw slots=%u slot_size=%zu window_size=%zu doorbell=%s\n",
                PIPELINE_N_SLOTS, t.slot_size, t.window_size, doorbell_mode_name(args.db_mode).c_str());
        return t;
    }

    void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) {
        const uint64_t payload_bytes = (uint64_t) n_embd * sizeof(float);
        send_raw(seq_id, pos, 1, n_embd, flags, payload, payload_bytes);
    }

    void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        send_raw(-1, pos, n_tokens, n_embd, flags, payload, payload_bytes);
    }

    void send_token(int32_t seq_id, llama_token token) {
        const token_payload payload = { seq_id, (int32_t) token };
        send_raw(seq_id, -1, 0, 0, PIPELINE_FLAG_TOKEN, &payload, sizeof(payload));
    }

    void send_stop(int32_t seq_id, int32_t pos) {
        send_raw(seq_id, pos, 0, 0, PIPELINE_FLAG_STOP, nullptr, 0);
    }

    recv_packet recv() {
        while (true) {
            uint8_t * ptr = rx_slot_ptr(rx_slot);
            uint32_t * state = reinterpret_cast<uint32_t *>(ptr);
            const uint32_t state_value = __atomic_load_n(state, __ATOMIC_ACQUIRE);
            if (state_value == PIPELINE_SLOT_FULL) {
                const auto * header = reinterpret_cast<const brick_packet_header *>(ptr + sizeof(uint32_t));
                if (header->magic != PIPELINE_BRICK_MAGIC ||
                        header->version != PIPELINE_BRICK_VERSION ||
                        header->slot != rx_slot) {
                    throw std::runtime_error("invalid pipeline-brick packet header");
                }

                recv_packet out;
                out.header = *header;
                if (out.header.payload_bytes > 0) {
                    out.payload.resize(out.header.payload_bytes);
                    memcpy(out.payload.data(), ptr + sizeof(uint32_t) + sizeof(brick_packet_header), out.payload.size());
                }

                __atomic_store_n(state, PIPELINE_SLOT_EMPTY, __ATOMIC_RELEASE);
                rx_slot = (rx_slot + 1) % PIPELINE_N_SLOTS;
                return out;
            }

            wait_for_rx();
        }
    }

private:
    static int open_required(const std::string & path, int flags, const char * name) {
        if (path.empty()) {
            throw std::runtime_error(std::string(name) + " is required");
        }
        int fd = open(path.c_str(), flags);
        if (fd < 0) {
            throw std::runtime_error(std::string("open ") + name + " failed for " + path + ": " + strerror(errno));
        }
        return fd;
    }

    static std::string doorbell_mode_name(doorbell_mode mode) {
        switch (mode) {
            case doorbell_mode::write: return "write";
            case doorbell_mode::poll:  return "poll";
            case doorbell_mode::ioctl: return "ioctl";
        }
        return "unknown";
    }

    uint8_t * tx_slot_ptr(uint32_t slot) const {
        return static_cast<uint8_t *>(tx_base) + slot_size * slot;
    }

    uint8_t * rx_slot_ptr(uint32_t slot) const {
        return static_cast<uint8_t *>(rx_base) + slot_size * slot;
    }

    void clear_tx_slots() {
        for (uint32_t slot = 0; slot < PIPELINE_N_SLOTS; ++slot) {
            uint8_t * ptr = tx_slot_ptr(slot);
            auto * state = reinterpret_cast<uint32_t *>(ptr);
            __atomic_store_n(state, PIPELINE_SLOT_EMPTY, __ATOMIC_RELEASE);
        }
    }

    void send_raw(int32_t seq_id, int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        if (sizeof(uint32_t) + sizeof(brick_packet_header) + payload_bytes > slot_size) {
            throw std::runtime_error("pipeline-brick payload exceeds slot size");
        }

        uint8_t * ptr = tx_slot_ptr(tx_slot);
        auto * state = reinterpret_cast<uint32_t *>(ptr);
        while (__atomic_load_n(state, __ATOMIC_ACQUIRE) != PIPELINE_SLOT_EMPTY) {
            sleep_us(50);
        }

        auto * header = reinterpret_cast<brick_packet_header *>(ptr + sizeof(uint32_t));
        header->magic = PIPELINE_BRICK_MAGIC;
        header->version = PIPELINE_BRICK_VERSION;
        header->slot = tx_slot;
        header->flags = flags;
        header->n_tokens = n_tokens;
        header->n_embd = n_embd;
        header->pos = pos;
        header->seq_id = seq_id;
        header->payload_bytes = payload_bytes;

        if (payload_bytes > 0) {
            memcpy(ptr + sizeof(uint32_t) + sizeof(brick_packet_header), payload, payload_bytes);
        }

        __atomic_store_n(state, PIPELINE_SLOT_FULL, __ATOMIC_RELEASE);
        notify_peer();
        tx_slot = (tx_slot + 1) % PIPELINE_N_SLOTS;
    }

    void notify_peer() {
        if (db_mode != doorbell_mode::write) {
            return;
        }

        const uint64_t one = 1;
        if (write(tx_db_fd.get(), &one, sizeof(one)) < 0) {
            throw std::runtime_error("doorbell write failed: " + std::string(strerror(errno)));
        }
    }

    void wait_for_rx() {
        if (db_mode == doorbell_mode::poll) {
            sleep_us(50);
            return;
        }

        pollfd pfd;
        pfd.fd = rx_db_fd.get();
        pfd.events = POLLIN;
        pfd.revents = 0;

        int rc = poll(&pfd, 1, 100);
        if (rc < 0) {
            if (errno == EINTR) {
                return;
            }
            throw std::runtime_error("doorbell poll failed: " + std::string(strerror(errno)));
        }
        if (rc == 0) {
            return;
        }

        uint64_t value = 0;
        if (read(rx_db_fd.get(), &value, sizeof(value)) < 0 && errno != EAGAIN && errno != EINTR) {
            throw std::runtime_error("doorbell read failed: " + std::string(strerror(errno)));
        }
    }

    fd_handle tx_mw_fd;
    fd_handle rx_mw_fd;
    fd_handle tx_db_fd;
    fd_handle rx_db_fd;
    void * tx_base = MAP_FAILED;
    void * rx_base = MAP_FAILED;
    size_t slot_size = 0;
    size_t window_size = 0;
    uint32_t tx_slot = 0;
    uint32_t rx_slot = 0;
    doorbell_mode db_mode = doorbell_mode::write;
};

static void bind_shared_window_to_numa(const std::string & path, size_t size, const std::string & node_spec, const char * label);

class cxl_transport {
public:
    cxl_transport() = default;
    cxl_transport(const cxl_transport &) = delete;
    cxl_transport & operator=(const cxl_transport &) = delete;

    cxl_transport(cxl_transport && other) noexcept {
        *this = std::move(other);
    }

    cxl_transport & operator=(cxl_transport && other) noexcept {
        if (this != &other) {
            if (tx_base != MAP_FAILED) {
                munmap(tx_base, window_size);
            }
            if (rx_base != MAP_FAILED) {
                munmap(rx_base, window_size);
            }
            tx_base = other.tx_base;
            rx_base = other.rx_base;
            slot_size = other.slot_size;
            window_size = other.window_size;
            tx_slot = other.tx_slot;
            rx_slot = other.rx_slot;
            other.tx_base = MAP_FAILED;
            other.rx_base = MAP_FAILED;
            other.slot_size = 0;
            other.window_size = 0;
            other.tx_slot = 0;
            other.rx_slot = 0;
        }
        return *this;
    }

    ~cxl_transport() {
        if (tx_base != MAP_FAILED) {
            munmap(tx_base, window_size);
        }
        if (rx_base != MAP_FAILED) {
            munmap(rx_base, window_size);
        }
    }

    static cxl_transport open_transport(const pipeline_args & args, size_t max_payload_bytes) {
        cxl_transport t;
        t.slot_size = align_up(sizeof(uint32_t) + sizeof(brick_packet_header) + max_payload_bytes, 64);
        t.window_size = t.slot_size * PIPELINE_N_SLOTS;

        open_or_create_file(args.tx_mw, t.window_size, "--tx-mw");
        open_or_create_file(args.rx_mw, t.window_size, "--rx-mw");

        if (!args.tail_numa.empty()) {
            bind_shared_window_to_numa(args.tx_mw, t.window_size, args.tail_numa, "cxl-h2t");
        }
        if (!args.head_numa.empty()) {
            bind_shared_window_to_numa(args.rx_mw, t.window_size, args.head_numa, "cxl-t2h");
        }

        fd_handle tx_fd(open(args.tx_mw.c_str(), O_RDWR | O_SYNC));
        if (!tx_fd.valid()) {
            throw std::runtime_error("cxl open --tx-mw failed: " + std::string(strerror(errno)));
        }
        fd_handle rx_fd(open(args.rx_mw.c_str(), O_RDWR | O_SYNC));
        if (!rx_fd.valid()) {
            throw std::runtime_error("cxl open --rx-mw failed: " + std::string(strerror(errno)));
        }

        t.tx_base = mmap(nullptr, t.window_size, PROT_READ | PROT_WRITE, MAP_SHARED, tx_fd.get(), 0);
        if (t.tx_base == MAP_FAILED) {
            throw std::runtime_error("cxl mmap --tx-mw failed: " + std::string(strerror(errno)));
        }
        t.rx_base = mmap(nullptr, t.window_size, PROT_READ | PROT_WRITE, MAP_SHARED, rx_fd.get(), 0);
        if (t.rx_base == MAP_FAILED) {
            throw std::runtime_error("cxl mmap --rx-mw failed: " + std::string(strerror(errno)));
        }

        t.clear_tx_slots();

        fprintf(stderr,
                "pipeline-brick transport: cxl slots=%u slot_size=%zu window_size=%zu\n",
                PIPELINE_N_SLOTS, t.slot_size, t.window_size);
        return t;
    }

    void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) {
        const uint64_t payload_bytes = (uint64_t) n_embd * sizeof(float);
        send_raw(seq_id, pos, 1, n_embd, flags, payload, payload_bytes);
    }

    void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        send_raw(-1, pos, n_tokens, n_embd, flags, payload, payload_bytes);
    }

    void send_token(int32_t seq_id, llama_token token) {
        const token_payload payload = { seq_id, (int32_t) token };
        send_raw(seq_id, -1, 0, 0, PIPELINE_FLAG_TOKEN, &payload, sizeof(payload));
    }

    void send_stop(int32_t seq_id, int32_t pos) {
        send_raw(seq_id, pos, 0, 0, PIPELINE_FLAG_STOP, nullptr, 0);
    }

    recv_packet recv() {
        while (true) {
            uint8_t * ptr = rx_slot_ptr(rx_slot);
            uint32_t * state = reinterpret_cast<uint32_t *>(ptr);
            const uint32_t state_value = __atomic_load_n(state, __ATOMIC_ACQUIRE);
            if (state_value == PIPELINE_SLOT_FULL) {
                const auto * header = reinterpret_cast<const brick_packet_header *>(ptr + sizeof(uint32_t));
                if (header->magic != PIPELINE_BRICK_MAGIC ||
                        header->version != PIPELINE_BRICK_VERSION ||
                        header->slot != rx_slot) {
                    throw std::runtime_error("invalid pipeline-brick packet header");
                }

                recv_packet out;
                out.header = *header;
                if (out.header.payload_bytes > 0) {
                    out.payload.resize(out.header.payload_bytes);
                    memcpy(out.payload.data(), ptr + sizeof(uint32_t) + sizeof(brick_packet_header), out.payload.size());
                }

                __atomic_store_n(state, PIPELINE_SLOT_EMPTY, __ATOMIC_RELEASE);
                rx_slot = (rx_slot + 1) % PIPELINE_N_SLOTS;
                return out;
            }

            wait_for_rx();
        }
    }

private:
    static void open_or_create_file(const std::string & path, size_t size, const char * name) {
        int fd = open(path.c_str(), O_RDWR | O_SYNC);
        if (fd < 0 && errno == ENOENT) {
            fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd < 0) {
                if (errno == EEXIST) {
                    fd = open(path.c_str(), O_RDWR | O_SYNC);
                } else {
                    throw std::runtime_error(std::string("cxl create ") + name + " failed for " + path + ": " + strerror(errno));
                }
            } else {
                if (ftruncate(fd, (off_t) size) != 0) {
                    close(fd);
                    throw std::runtime_error(std::string("cxl ftruncate ") + name + " failed for " + path + ": " + strerror(errno));
                }
            }
        }
        if (fd < 0) {
            throw std::runtime_error(std::string("cxl open ") + name + " failed for " + path + ": " + strerror(errno));
        }
        close(fd);
    }

    uint8_t * tx_slot_ptr(uint32_t slot) const {
        return static_cast<uint8_t *>(tx_base) + slot_size * slot;
    }

    uint8_t * rx_slot_ptr(uint32_t slot) const {
        return static_cast<uint8_t *>(rx_base) + slot_size * slot;
    }

    void clear_tx_slots() {
        for (uint32_t slot = 0; slot < PIPELINE_N_SLOTS; ++slot) {
            uint8_t * ptr = tx_slot_ptr(slot);
            auto * state = reinterpret_cast<uint32_t *>(ptr);
            __atomic_store_n(state, PIPELINE_SLOT_EMPTY, __ATOMIC_RELEASE);
        }
    }

    void send_raw(int32_t seq_id, int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        if (sizeof(uint32_t) + sizeof(brick_packet_header) + payload_bytes > slot_size) {
            throw std::runtime_error("pipeline-brick payload exceeds slot size");
        }

        uint8_t * ptr = tx_slot_ptr(tx_slot);
        auto * state = reinterpret_cast<uint32_t *>(ptr);
        while (__atomic_load_n(state, __ATOMIC_ACQUIRE) != PIPELINE_SLOT_EMPTY) {
#if defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#else
            sleep_us(10);
#endif
        }

        auto * header = reinterpret_cast<brick_packet_header *>(ptr + sizeof(uint32_t));
        header->magic = PIPELINE_BRICK_MAGIC;
        header->version = PIPELINE_BRICK_VERSION;
        header->slot = tx_slot;
        header->flags = flags;
        header->n_tokens = n_tokens;
        header->n_embd = n_embd;
        header->pos = pos;
        header->seq_id = seq_id;
        header->payload_bytes = payload_bytes;

        if (payload_bytes > 0) {
            memcpy(ptr + sizeof(uint32_t) + sizeof(brick_packet_header), payload, payload_bytes);
        }

        __atomic_store_n(state, PIPELINE_SLOT_FULL, __ATOMIC_RELEASE);
        tx_slot = (tx_slot + 1) % PIPELINE_N_SLOTS;
    }

    void wait_for_rx() {
#if defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
#else
        sleep_us(10);
#endif
    }

    void * tx_base = MAP_FAILED;
    void * rx_base = MAP_FAILED;
    size_t slot_size = 0;
    size_t window_size = 0;
    uint32_t tx_slot = 0;
    uint32_t rx_slot = 0;
};

struct rdma_peer_info {
    uint32_t qpn = 0;
    uint32_t psn = 0;
    int gid_index = 0;
    int port = 1;
    std::array<uint8_t, 16> gid = {};
};

static std::string bytes_to_hex(const uint8_t * data, size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        out << std::setw(2) << (unsigned) data[i];
    }
    return out.str();
}

static std::array<uint8_t, 16> hex_to_gid(const std::string & value) {
    if (value.size() != 32) {
        throw std::runtime_error("RDMA gid must be 32 hex characters");
    }
    std::array<uint8_t, 16> out = {};
    for (size_t i = 0; i < out.size(); ++i) {
        const std::string byte = value.substr(i * 2, 2);
        out[i] = (uint8_t) std::stoul(byte, nullptr, 16);
    }
    return out;
}

static void write_rdma_info_file(const std::string & path, const rdma_peer_info & info) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write RDMA local info file: " + path);
    }
    out << "qpn=" << info.qpn << "\n";
    out << "psn=" << info.psn << "\n";
    out << "port=" << info.port << "\n";
    out << "gid_index=" << info.gid_index << "\n";
    out << "gid=" << bytes_to_hex(info.gid.data(), info.gid.size()) << "\n";
}

static rdma_peer_info read_rdma_info_file(const std::string & path) {
    constexpr int max_attempts = 60000;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        rdma_peer_info info;
        std::ifstream in(path);
        if (in) {
            try {
                std::string line;
                while (std::getline(in, line)) {
                    const size_t eq = line.find('=');
                    if (eq == std::string::npos) {
                        continue;
                    }
                    const std::string key = line.substr(0, eq);
                    const std::string value = line.substr(eq + 1);
                    if (key == "qpn") {
                        info.qpn = (uint32_t) std::stoul(value);
                    } else if (key == "psn") {
                        info.psn = (uint32_t) std::stoul(value);
                    } else if (key == "port") {
                        info.port = std::stoi(value);
                    } else if (key == "gid_index") {
                        info.gid_index = std::stoi(value);
                    } else if (key == "gid") {
                        info.gid = hex_to_gid(value);
                    }
                }
                if (info.qpn != 0) {
                    return info;
                }
            } catch (const std::exception &) {
                // Peer info may be visible while the user or shell is still writing it.
            }
        }
        if (attempt == 0) {
            fprintf(stderr, "pipeline-brick rdma: waiting for peer info file %s\n", path.c_str());
        }
        sleep_us(10000);
    }
    throw std::runtime_error("timed out waiting for valid RDMA peer info file: " + path);
}

class ib_rdma_transport {
public:
    ib_rdma_transport() = default;
    ib_rdma_transport(const ib_rdma_transport &) = delete;
    ib_rdma_transport & operator=(const ib_rdma_transport &) = delete;

    ib_rdma_transport(ib_rdma_transport && other) noexcept {
        *this = std::move(other);
    }

    ib_rdma_transport & operator=(ib_rdma_transport && other) noexcept {
        if (this != &other) {
            cleanup();
            move_from(other);
        }
        return *this;
    }

    ~ib_rdma_transport() {
        cleanup();
    }

    static ib_rdma_transport open_transport(const pipeline_args & args, size_t max_payload_bytes, bool downstream) {
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
        ib_rdma_transport t;
        t.max_message = sizeof(brick_packet_header) + max_payload_bytes;
        t.rx_stride = align_up(t.max_message, 64);
        t.rx_depth = PIPELINE_N_SLOTS;
        t.tx_buf.resize(t.max_message);
        t.rx_buf.resize(t.rx_stride * t.rx_depth);
        t.dev_name = downstream ? args.down_rdmadev : args.up_rdmadev;
        t.local_info_path = downstream ? args.down_rdma_local_info : args.up_rdma_local_info;
        t.peer_info_path = downstream ? args.down_rdma_peer_info : args.up_rdma_peer_info;
        t.ib_port = args.rdma_port;
        t.gid_index = args.rdma_gid_index;
        t.path_mtu = mtu_from_bytes(args.rdma_mtu);

        t.open_device();
        t.create_resources();
        t.query_local_info();
        write_rdma_info_file(t.local_info_path, t.local_info);
        if (t.peer_info_path.empty()) {
            throw std::runtime_error("RDMA requires --rdma-peer-info or --up/--down-rdma-peer-info");
        }
        rdma_peer_info peer = read_rdma_info_file(t.peer_info_path);
        t.activate(peer);
        t.post_all_recvs();

        fprintf(stderr,
                "pipeline-brick transport: ib-rdma dev=%s port=%d gid_index=%d qpn=%u peer_qpn=%u max_message=%zu\n",
                t.dev_name.empty() ? "*" : t.dev_name.c_str(), t.ib_port, t.gid_index,
                t.local_info.qpn, peer.qpn, t.max_message);
        return t;
#else
        (void) args;
        (void) max_payload_bytes;
        (void) downstream;
        throw std::runtime_error("ib-rdma transport was not built because libibverbs was not found");
#endif
    }

    void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) {
        const uint64_t payload_bytes = (uint64_t) n_embd * sizeof(float);
        send_raw(seq_id, pos, 1, n_embd, flags, payload, payload_bytes);
    }

    void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        send_raw(-1, pos, n_tokens, n_embd, flags, payload, payload_bytes);
    }

    void send_token(int32_t seq_id, llama_token token) {
        const token_payload payload = { seq_id, (int32_t) token };
        send_raw(seq_id, -1, 0, 0, PIPELINE_FLAG_TOKEN, &payload, sizeof(payload));
    }

    void send_stop(int32_t seq_id, int32_t pos) {
        send_raw(seq_id, pos, 0, 0, PIPELINE_FLAG_STOP, nullptr, 0);
    }

    recv_packet recv() {
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
        ibv_wc wc = {};
        poll_cq(rcq, wc);
        if (wc.opcode != IBV_WC_RECV || wc.byte_len < sizeof(brick_packet_header)) {
            throw std::runtime_error("invalid RDMA receive completion");
        }
        const uint32_t slot = (uint32_t) wc.wr_id;
        if (slot >= rx_depth) {
            throw std::runtime_error("invalid RDMA receive slot");
        }
        const uint8_t * ptr = rx_buf.data() + rx_stride * slot;
        const auto * header = reinterpret_cast<const brick_packet_header *>(ptr);
        if (header->magic != PIPELINE_BRICK_MAGIC || header->version != PIPELINE_BRICK_VERSION) {
            throw std::runtime_error("invalid pipeline-brick RDMA packet header");
        }
        if (sizeof(brick_packet_header) + header->payload_bytes != wc.byte_len) {
            throw std::runtime_error("invalid pipeline-brick RDMA packet length");
        }

        recv_packet out;
        out.header = *header;
        if (out.header.payload_bytes > 0) {
            out.payload.resize(out.header.payload_bytes);
            memcpy(out.payload.data(), ptr + sizeof(brick_packet_header), out.payload.size());
        }
        post_recv(slot);
        return out;
#else
        throw std::runtime_error("ib-rdma transport was not built because libibverbs was not found");
#endif
    }

private:
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
    static ibv_mtu mtu_from_bytes(int bytes) {
        if (bytes <= 256)  return IBV_MTU_256;
        if (bytes <= 512)  return IBV_MTU_512;
        if (bytes <= 1024) return IBV_MTU_1024;
        if (bytes <= 2048) return IBV_MTU_2048;
        return IBV_MTU_4096;
    }

    void open_device() {
        int ndev = 0;
        ibv_device ** devs = ibv_get_device_list(&ndev);
        if (!devs || ndev == 0) {
            throw std::runtime_error("ibv_get_device_list found no RDMA devices");
        }
        for (int i = 0; i < ndev; ++i) {
            const char * name = ibv_get_device_name(devs[i]);
            if (!dev_name.empty() && dev_name != name) {
                continue;
            }
            ctx = ibv_open_device(devs[i]);
            if (ctx) {
                break;
            }
        }
        ibv_free_device_list(devs);
        if (!ctx) {
            throw std::runtime_error("failed to open requested RDMA device");
        }
    }

    void create_resources() {
        pd = ibv_alloc_pd(ctx);
        if (!pd) {
            throw std::runtime_error("ibv_alloc_pd failed");
        }
        scq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
        rcq = ibv_create_cq(ctx, (int) rx_depth + 4, nullptr, nullptr, 0);
        if (!scq || !rcq) {
            throw std::runtime_error("ibv_create_cq failed");
        }

        ibv_qp_init_attr qia = {};
        qia.send_cq = scq;
        qia.recv_cq = rcq;
        qia.qp_type = IBV_QPT_RC;
        qia.cap.max_send_wr = 16;
        qia.cap.max_recv_wr = rx_depth + 4;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        qia.cap.max_inline_data = 256;
        qp = ibv_create_qp(pd, &qia);
        if (!qp) {
            throw std::runtime_error("ibv_create_qp failed");
        }
        max_inline = qia.cap.max_inline_data;

        tx_mr = ibv_reg_mr(pd, tx_buf.data(), tx_buf.size(), IBV_ACCESS_LOCAL_WRITE);
        rx_mr = ibv_reg_mr(pd, rx_buf.data(), rx_buf.size(), IBV_ACCESS_LOCAL_WRITE);
        if (!tx_mr || !rx_mr) {
            throw std::runtime_error("ibv_reg_mr failed");
        }
    }

    void query_local_info() {
        ibv_port_attr port_attr = {};
        if (ibv_query_port(ctx, ib_port, &port_attr) != 0 || port_attr.state != IBV_PORT_ACTIVE) {
            throw std::runtime_error("RDMA port is not active");
        }
        ibv_gid gid = {};
        if (ibv_query_gid(ctx, ib_port, gid_index, &gid) != 0) {
            throw std::runtime_error("ibv_query_gid failed");
        }
        local_info.qpn = qp->qp_num;
        local_info.psn = qp->qp_num & 0xffffff;
        local_info.port = ib_port;
        local_info.gid_index = gid_index;
        memcpy(local_info.gid.data(), &gid, local_info.gid.size());
    }

    void activate(const rdma_peer_info & peer) {
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_INIT;
            attr.port_num = ib_port;
            attr.pkey_index = 0;
            attr.qp_access_flags = 0;
            if (ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
                throw std::runtime_error("ibv_modify_qp INIT failed");
            }
        }
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RTR;
            attr.path_mtu = path_mtu;
            attr.dest_qp_num = peer.qpn;
            attr.rq_psn = peer.psn;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer = 12;
            attr.ah_attr.is_global = 1;
            memcpy(&attr.ah_attr.grh.dgid, peer.gid.data(), peer.gid.size());
            attr.ah_attr.grh.sgid_index = gid_index;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.dlid = 0;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = ib_port;
            if (ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
                throw std::runtime_error("ibv_modify_qp RTR failed");
            }
        }
        {
            ibv_qp_attr attr = {};
            attr.qp_state = IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.sq_psn = local_info.psn;
            attr.max_rd_atomic = 1;
            if (ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
                throw std::runtime_error("ibv_modify_qp RTS failed");
            }
        }
    }

    void post_all_recvs() {
        for (uint32_t i = 0; i < rx_depth; ++i) {
            post_recv(i);
        }
    }

    void post_recv(uint32_t slot) {
        ibv_sge sge = {};
        sge.addr = (uintptr_t) (rx_buf.data() + rx_stride * slot);
        sge.length = (uint32_t) rx_stride;
        sge.lkey = rx_mr->lkey;

        ibv_recv_wr wr = {};
        ibv_recv_wr * bad = nullptr;
        wr.wr_id = slot;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (ibv_post_recv(qp, &wr, &bad) != 0) {
            throw std::runtime_error("ibv_post_recv failed");
        }
    }

    void poll_cq(ibv_cq * cq, ibv_wc & wc) {
        while (true) {
            const int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) {
                throw std::runtime_error("ibv_poll_cq failed");
            }
            if (n == 0) {
                sleep_us(50);
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                throw std::runtime_error(std::string("RDMA completion failed: ") + ibv_wc_status_str(wc.status));
            }
            return;
        }
    }
#endif

    void send_raw(int32_t seq_id, int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
        const size_t bytes = sizeof(brick_packet_header) + (size_t) payload_bytes;
        if (bytes > tx_buf.size()) {
            throw std::runtime_error("pipeline-brick RDMA payload exceeds message buffer");
        }
        auto * header = reinterpret_cast<brick_packet_header *>(tx_buf.data());
        header->magic = PIPELINE_BRICK_MAGIC;
        header->version = PIPELINE_BRICK_VERSION;
        header->slot = 0;
        header->flags = flags;
        header->n_tokens = n_tokens;
        header->n_embd = n_embd;
        header->pos = pos;
        header->seq_id = seq_id;
        header->payload_bytes = payload_bytes;
        if (payload_bytes > 0) {
            memcpy(tx_buf.data() + sizeof(brick_packet_header), payload, payload_bytes);
        }

        ibv_sge sge = {};
        sge.addr = (uintptr_t) tx_buf.data();
        sge.length = (uint32_t) bytes;
        sge.lkey = tx_mr->lkey;

        ibv_send_wr wr = {};
        ibv_send_wr * bad = nullptr;
        wr.wr_id = 0;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        if (bytes <= max_inline) {
            wr.send_flags |= IBV_SEND_INLINE;
        }
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (ibv_post_send(qp, &wr, &bad) != 0) {
            throw std::runtime_error("ibv_post_send failed");
        }
        ibv_wc wc = {};
        poll_cq(scq, wc);
#else
        (void) seq_id;
        (void) pos;
        (void) n_tokens;
        (void) n_embd;
        (void) flags;
        (void) payload;
        (void) payload_bytes;
        throw std::runtime_error("ib-rdma transport was not built because libibverbs was not found");
#endif
    }

    void cleanup() {
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
        if (tx_mr) ibv_dereg_mr(tx_mr);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        if (qp) ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd) ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
        tx_mr = nullptr;
        rx_mr = nullptr;
        qp = nullptr;
        scq = nullptr;
        rcq = nullptr;
        pd = nullptr;
        ctx = nullptr;
#endif
    }

    void move_from(ib_rdma_transport & other) {
        max_message = other.max_message;
        rx_stride = other.rx_stride;
        rx_depth = other.rx_depth;
        ib_port = other.ib_port;
        gid_index = other.gid_index;
        dev_name = std::move(other.dev_name);
        local_info_path = std::move(other.local_info_path);
        peer_info_path = std::move(other.peer_info_path);
        tx_buf = std::move(other.tx_buf);
        rx_buf = std::move(other.rx_buf);
        local_info = other.local_info;
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
        ctx = other.ctx;
        pd = other.pd;
        scq = other.scq;
        rcq = other.rcq;
        qp = other.qp;
        tx_mr = other.tx_mr;
        rx_mr = other.rx_mr;
        path_mtu = other.path_mtu;
        max_inline = other.max_inline;
        other.ctx = nullptr;
        other.pd = nullptr;
        other.scq = nullptr;
        other.rcq = nullptr;
        other.qp = nullptr;
        other.tx_mr = nullptr;
        other.rx_mr = nullptr;
#endif
    }

    size_t max_message = 0;
    size_t rx_stride = 0;
    uint32_t rx_depth = 0;
    int ib_port = 1;
    int gid_index = 0;
    std::string dev_name;
    std::string local_info_path;
    std::string peer_info_path;
    std::vector<uint8_t> tx_buf;
    std::vector<uint8_t> rx_buf;
    rdma_peer_info local_info;
#if defined(LLAMA_PIPELINE_BRICK_IBVERBS)
    ibv_context * ctx = nullptr;
    ibv_pd * pd = nullptr;
    ibv_cq * scq = nullptr;
    ibv_cq * rcq = nullptr;
    ibv_qp * qp = nullptr;
    ibv_mr * tx_mr = nullptr;
    ibv_mr * rx_mr = nullptr;
    ibv_mtu path_mtu = IBV_MTU_1024;
    uint32_t max_inline = 0;
#endif
};

class zni_transport {
public:
    static zni_transport open_transport(const pipeline_args &, size_t, bool) {
        throw std::runtime_error("zni-rdma transport requires the ZNI SDK adapter; this build only provides the common transport interface");
    }

    void send_hidden(int32_t, int32_t, int32_t, uint32_t, const float *) {
        throw std::runtime_error("zni-rdma transport is not available in this build");
    }

    void send_hidden_payload(int32_t, int32_t, int32_t, uint32_t, const void *, uint64_t) {
        throw std::runtime_error("zni-rdma transport is not available in this build");
    }

    void send_token(int32_t, llama_token) {
        throw std::runtime_error("zni-rdma transport is not available in this build");
    }

    void send_stop(int32_t, int32_t) {
        throw std::runtime_error("zni-rdma transport is not available in this build");
    }

    recv_packet recv() {
        throw std::runtime_error("zni-rdma transport is not available in this build");
    }
};

class transport_box {
public:
    template<typename T>
    explicit transport_box(T && impl) : self(new model<T>(std::move(impl))) {}

    void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) {
        self->send_hidden(seq_id, pos, n_embd, flags, payload);
    }

    void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        self->send_hidden_payload(pos, n_tokens, n_embd, flags, payload, payload_bytes);
    }

    void send_token(int32_t seq_id, llama_token token) {
        self->send_token(seq_id, token);
    }

    void send_stop(int32_t seq_id, int32_t pos) {
        self->send_stop(seq_id, pos);
    }

    recv_packet recv() {
        return self->recv();
    }

private:
    struct iface {
        virtual ~iface() = default;
        virtual void send_hidden(int32_t, int32_t, int32_t, uint32_t, const float *) = 0;
        virtual void send_hidden_payload(int32_t, int32_t, int32_t, uint32_t, const void *, uint64_t) = 0;
        virtual void send_token(int32_t, llama_token) = 0;
        virtual void send_stop(int32_t, int32_t) = 0;
        virtual recv_packet recv() = 0;
    };

    template<typename T>
    struct model : iface {
        explicit model(T && impl) : impl(std::move(impl)) {}
        void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) override {
            impl.send_hidden(seq_id, pos, n_embd, flags, payload);
        }
        void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) override {
            impl.send_hidden_payload(pos, n_tokens, n_embd, flags, payload, payload_bytes);
        }
        void send_token(int32_t seq_id, llama_token token) override {
            impl.send_token(seq_id, token);
        }
        void send_stop(int32_t seq_id, int32_t pos) override {
            impl.send_stop(seq_id, pos);
        }
        recv_packet recv() override {
            return impl.recv();
        }
        T impl;
    };

    std::unique_ptr<iface> self;
};

static bool contains(const char * text, const char * needle) {
    return strstr(text, needle) != nullptr;
}

static void pipeline_brick_log_callback(ggml_log_level level, const char * text, void * user_data) {
    const bool verbose = user_data && *static_cast<const bool *>(user_data);

    if (!verbose) {
        if (level == GGML_LOG_LEVEL_CONT) {
            return;
        }

        const bool keep =
            level == GGML_LOG_LEVEL_ERROR ||
            contains(text, "llama_kv_cache: size =") ||
            (contains(text, "llama_kv_cache:") && contains(text, "KV buffer size")) ||
            contains(text, "pipeline-brick KV NUMA") ||
            contains(text, "verify_numa_buffer_pages:");

        if (!keep) {
            return;
        }
    }

    fputs(text, stderr);
}

static void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s --domain-mode single --model PATH --ctx-size N --threads N --n-predict N\n"
            "          --parallel N --head-numa 0-3 --tail-numa 4-7 [--tp-size 1|2|4]\n"
            "          (--prompt TEXT | --async-pipeline --prompt-file PATH --pipeline-microbatch N)\n"
            "          [--tail-kv-numa 0-3]\n"
            "       %s --domain-mode dual --stage-id N --stage-count 4 --model PATH --ctx-size N --threads N\n"
            "          --parallel N --numa-cpus CPU-RANGES --up-transport ntb-mw|cxl|ib-rdma|zni-rdma\n"
            "          --down-transport ntb-mw|cxl|ib-rdma|zni-rdma\n"
            "       %s --single-system --model PATH --ctx-size N --threads N --n-predict N\n"
            "          --parallel N --head-numa 0-3 --tail-numa 4-7 [--prefill-chunk N]\n"
            "          (--prompt TEXT | --async-pipeline --prompt-file PATH --pipeline-microbatch N)\n"
            "          [--hidden-dtype f16|f32] [--stream-kv] [--n-layer N] [--n-embd N]\n"
            "          [--stream-kv-sink N] [--stream-kv-recent N]\n"
            "          [--tp-size N] [--tail-kv-numa NODES] [--naive-transfer-mult N]\n"
            "          [--naive-kv-cross [--naive-kv-cross-node N]] [--quiet] [--verbose]\n"
            "       %s --single-system --stage-count 4 --stage-numa '0-1;2-3;4-5;6-7'\n"
            "          --transport ntb-mw|cxl --model PATH --prompt TEXT --ctx-size N --threads N\n"
            "          --n-predict N --parallel N [--stream-kv] [--quiet] [--verbose]\n"
            "       %s --hardware --stage-id N --stage-count 4 --model PATH --ctx-size N --threads N\n"
            "          --parallel N --numa-cpus CPU-RANGES --up-transport ntb-mw|cxl|ib-rdma|zni-rdma\n"
            "          --down-transport ntb-mw|cxl|ib-rdma|zni-rdma [--up-tx-mw PATH --up-rx-mw PATH]\n"
            "          [--down-tx-mw PATH --down-rx-mw PATH] [--up-rdma-local-info PATH --up-rdma-peer-info PATH]\n"
            "          [--down-rdma-local-info PATH --down-rdma-peer-info PATH] [--rdma-dev DEV]\n"
            "       %s --hardware --role head|tail --model PATH --ctx-size N --threads N --n-predict N --bricks 2\n"
            "          --brick-id N --peer-brick-id N --layer-start N --layer-end N --parallel N\n"
            "          --numa-tp N --tp-size N --numa-cpus CPU-RANGES --transport ntb-mw|cxl\n"
            "          --tx-mw PATH --rx-mw PATH [--tx-doorbell PATH --rx-doorbell PATH]\n"
            "          [--doorbell-mode write|poll|ioctl] [--head-numa NODES --tail-numa NODES]\n"
            "          [--prompt TEXT | --async-pipeline --prompt-file PATH --pipeline-microbatch N]\n"
            "          [--prefill-chunk N] [--hidden-dtype f16|f32] [--stream-kv]\n"
            "          [--stream-kv-sink N] [--stream-kv-recent N] [--quiet] [--verbose]\n"
            "       %s --self-test-ntb --role head|tail --brick-id N --peer-brick-id N\n"
            "          --transport ntb-mw --tx-mw PATH --rx-mw PATH\n",
            prog, prog, prog, prog, prog, prog, prog);
}

static doorbell_mode parse_doorbell_mode(const std::string & value) {
    if (value == "write") {
        return doorbell_mode::write;
    }
    if (value == "poll") {
        return doorbell_mode::poll;
    }
    if (value == "ioctl") {
        return doorbell_mode::ioctl;
    }
    throw std::runtime_error("unknown --doorbell-mode: " + value);
}

static llama_pipeline_brick_role parse_role(const std::string & value) {
    if (value == "head") {
        return LLAMA_PIPELINE_BRICK_ROLE_HEAD;
    }
    if (value == "tail") {
        return LLAMA_PIPELINE_BRICK_ROLE_TAIL;
    }
    if (value == "stage") {
        return LLAMA_PIPELINE_BRICK_ROLE_STAGE;
    }
    throw std::runtime_error("unknown --role: " + value);
}

static domain_mode parse_domain_mode(const std::string & value) {
    if (value == "single") {
        return domain_mode::single;
    }
    if (value == "dual") {
        return domain_mode::dual;
    }
    throw std::runtime_error("unknown --domain-mode: " + value + " (supported: single, dual)");
}

static pipeline_args parse_args(int argc, char ** argv) {
    pipeline_args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--hardware") {
            args.hardware = true;
        } else if (key == "--single-system") {
            args.single_system = true;
        } else if (key == "--self-test-ntb") {
            args.self_test_ntb = true;
        } else if (key == "--domain-mode") {
            args.domain = parse_domain_mode(need_value(key.c_str()));
        } else if (key == "--verbose") {
            args.verbose = true;
        } else if (key == "--quiet") {
            args.quiet = true;
        } else if (key == "--model" || key == "-m") {
            args.model = need_value(key.c_str());
        } else if (key == "--prompt" || key == "-p") {
            args.prompt = need_value(key.c_str());
        } else if (key == "--prompt-file") {
            args.prompt_file = need_value(key.c_str());
        } else if (key == "--role") {
            args.role = parse_role(need_value(key.c_str()));
        } else if (key == "--brick-id") {
            args.brick_id = std::stoi(need_value(key.c_str()));
        } else if (key == "--peer-brick-id") {
            args.peer_brick_id = std::stoi(need_value(key.c_str()));
        } else if (key == "--layer-start") {
            args.layer_start = std::stoi(need_value(key.c_str()));
        } else if (key == "--layer-end") {
            args.layer_end = std::stoi(need_value(key.c_str()));
        } else if (key == "--ctx-size" || key == "-c") {
            args.ctx_size = std::stoi(need_value(key.c_str()));
        } else if (key == "--threads" || key == "-t") {
            args.threads = std::stoi(need_value(key.c_str()));
        } else if (key == "--n-predict" || key == "-n") {
            args.n_predict = std::stoi(need_value(key.c_str()));
        } else if (key == "--bricks") {
            args.bricks = std::stoi(need_value(key.c_str()));
        } else if (key == "--stage-id") {
            args.stage_id = std::stoi(need_value(key.c_str()));
        } else if (key == "--stage-count") {
            args.stage_count = std::stoi(need_value(key.c_str()));
        } else if (key == "--parallel") {
            args.parallel = std::stoi(need_value(key.c_str()));
        } else if (key == "--prefill-chunk") {
            args.prefill_chunk = std::stoi(need_value(key.c_str()));
        } else if (key == "--pipeline-microbatch") {
            args.pipeline_microbatch = std::stoi(need_value(key.c_str()));
        } else if (key == "--hidden-dtype") {
            args.hidden_type = parse_hidden_dtype(need_value(key.c_str()));
        } else if (key == "--async-pipeline") {
            args.async_pipeline = true;
        } else if (key == "--stream-kv") {
            args.stream_kv = true;
        } else if (key == "--naive-transfer-mult") {
            args.naive_transfer_mult = std::stoi(need_value(key.c_str()));
        } else if (key == "--naive-kv-cross") {
            args.naive_kv_cross = true;
        } else if (key == "--naive-kv-cross-node") {
            args.naive_kv_cross_node = std::stoi(need_value(key.c_str()));
        } else if (key == "--stream-kv-sink") {
            args.stream_kv_sink = std::stoi(need_value(key.c_str()));
        } else if (key == "--stream-kv-recent") {
            args.stream_kv_recent = std::stoi(need_value(key.c_str()));
        } else if (key == "--ema-kv") {
            throw std::runtime_error("--ema-kv has been removed from pipeline-brick; use --stream-kv");
        } else if (key == "--ema-kv-keep") {
            (void) need_value(key.c_str());
            throw std::runtime_error("--ema-kv-keep has been removed from pipeline-brick; use --stream-kv-sink/--stream-kv-recent");
        } else if (key == "--ema-kv-recent") {
            (void) need_value(key.c_str());
            throw std::runtime_error("--ema-kv-recent has been removed from pipeline-brick; use --stream-kv-recent");
        } else if (key == "--ema-kv-sink") {
            (void) need_value(key.c_str());
            throw std::runtime_error("--ema-kv-sink has been removed from pipeline-brick; use --stream-kv-sink");
        } else if (key == "--ema-alpha") {
            (void) need_value(key.c_str());
            throw std::runtime_error("--ema-alpha has been removed from pipeline-brick; use --stream-kv");
        } else if (key == "--ema-sync") {
            throw std::runtime_error("--ema-sync has been removed from pipeline-brick; use --stream-kv");
        } else if (key == "--numa-tp") {
            args.numa_tp = std::stoi(need_value(key.c_str()));
        } else if (key == "--tp-size") {
            args.tp_size = std::stoi(need_value(key.c_str()));
        } else if (key == "--n-layer") {
            args.n_layer = std::stoi(need_value(key.c_str()));
        } else if (key == "--n-embd") {
            args.n_embd_arg = std::stoi(need_value(key.c_str()));
        } else if (key == "--numa-cpus") {
            args.numa_cpus = need_value(key.c_str());
        } else if (key == "--head-numa") {
            args.head_numa = need_value(key.c_str());
        } else if (key == "--tail-numa") {
            args.tail_numa = need_value(key.c_str());
        } else if (key == "--tail-kv-numa") {
            args.tail_kv_numa = need_value(key.c_str());
        } else if (key == "--stage-numa") {
            args.stage_numa = need_value(key.c_str());
        } else if (key == "--transport") {
            args.transport = parse_transport_kind(need_value(key.c_str()));
            args.up_transport = args.transport;
            args.down_transport = args.transport;
        } else if (key == "--up-transport") {
            args.up_transport = parse_transport_kind(need_value(key.c_str()));
        } else if (key == "--down-transport") {
            args.down_transport = parse_transport_kind(need_value(key.c_str()));
        } else if (key == "--tx-mw") {
            args.tx_mw = need_value(key.c_str());
        } else if (key == "--rx-mw") {
            args.rx_mw = need_value(key.c_str());
        } else if (key == "--up-tx-mw") {
            args.up_tx_mw = need_value(key.c_str());
        } else if (key == "--up-rx-mw") {
            args.up_rx_mw = need_value(key.c_str());
        } else if (key == "--down-tx-mw") {
            args.down_tx_mw = need_value(key.c_str());
        } else if (key == "--down-rx-mw") {
            args.down_rx_mw = need_value(key.c_str());
        } else if (key == "--rdma-dev") {
            args.up_rdmadev = need_value(key.c_str());
            args.down_rdmadev = args.up_rdmadev;
        } else if (key == "--up-rdma-dev") {
            args.up_rdmadev = need_value(key.c_str());
        } else if (key == "--down-rdma-dev") {
            args.down_rdmadev = need_value(key.c_str());
        } else if (key == "--rdma-port") {
            args.rdma_port = std::stoi(need_value(key.c_str()));
        } else if (key == "--rdma-gid-index") {
            args.rdma_gid_index = std::stoi(need_value(key.c_str()));
        } else if (key == "--rdma-mtu") {
            args.rdma_mtu = std::stoi(need_value(key.c_str()));
        } else if (key == "--rdma-local-info") {
            args.up_rdma_local_info = need_value(key.c_str());
            args.down_rdma_local_info = args.up_rdma_local_info;
        } else if (key == "--rdma-peer-info") {
            args.up_rdma_peer_info = need_value(key.c_str());
            args.down_rdma_peer_info = args.up_rdma_peer_info;
        } else if (key == "--up-rdma-local-info") {
            args.up_rdma_local_info = need_value(key.c_str());
        } else if (key == "--up-rdma-peer-info") {
            args.up_rdma_peer_info = need_value(key.c_str());
        } else if (key == "--down-rdma-local-info") {
            args.down_rdma_local_info = need_value(key.c_str());
        } else if (key == "--down-rdma-peer-info") {
            args.down_rdma_peer_info = need_value(key.c_str());
        } else if (key == "--tx-doorbell") {
            args.tx_doorbell = need_value(key.c_str());
        } else if (key == "--rx-doorbell") {
            args.rx_doorbell = need_value(key.c_str());
        } else if (key == "--doorbell-mode") {
            args.db_mode = parse_doorbell_mode(need_value(key.c_str()));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

static std::vector<int> parse_cpu_list(const std::string & spec) {
    std::vector<int> cpus;
    size_t start = 0;
    while (start < spec.size()) {
        size_t end = spec.find(',', start);
        if (end == std::string::npos) {
            end = spec.size();
        }
        std::string part = spec.substr(start, end - start);
        if (!part.empty()) {
            size_t dash = part.find('-');
            if (dash == std::string::npos) {
                cpus.push_back(std::stoi(part));
            } else {
                int first = std::stoi(part.substr(0, dash));
                int last = std::stoi(part.substr(dash + 1));
                if (last < first) {
                    throw std::runtime_error("invalid --numa-cpus range: " + part);
                }
                for (int cpu = first; cpu <= last; ++cpu) {
                    cpus.push_back(cpu);
                }
            }
        }
        start = end + 1;
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

static std::vector<int> parse_cpu_list_ordered(const std::string & spec) {
    std::vector<int> cpus;
    size_t start = 0;
    while (start < spec.size()) {
        size_t end = spec.find(',', start);
        if (end == std::string::npos) {
            end = spec.size();
        }
        std::string part = spec.substr(start, end - start);
        if (!part.empty()) {
            size_t dash = part.find('-');
            if (dash == std::string::npos) {
                cpus.push_back(std::stoi(part));
            } else {
                int first = std::stoi(part.substr(0, dash));
                int last = std::stoi(part.substr(dash + 1));
                if (last < first) {
                    throw std::runtime_error("invalid CPU range: " + part);
                }
                for (int cpu = first; cpu <= last; ++cpu) {
                    cpus.push_back(cpu);
                }
            }
        }
        start = end + 1;
    }
    return cpus;
}

static std::vector<std::string> split_stage_specs(const std::string & spec) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= spec.size()) {
        size_t end = spec.find(';', start);
        if (end == std::string::npos) {
            end = spec.size();
        }
        std::string part = spec.substr(start, end - start);
        if (!part.empty()) {
            out.push_back(part);
        }
        if (end == spec.size()) {
            break;
        }
        start = end + 1;
    }
    return out;
}

static std::string join_cpu_list(const std::vector<int> & cpus) {
    std::ostringstream out;
    for (size_t i = 0; i < cpus.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << cpus[i];
    }
    return out.str();
}

static std::string read_text_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to read " + path);
    }
    std::string value;
    std::getline(in, value);
    return value;
}

static std::string cpus_for_numa_nodes(const std::string & node_spec) {
    std::vector<int> nodes = parse_cpu_list(node_spec);
    std::vector<int> cpus;

    for (int node : nodes) {
        const std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
        std::vector<int> node_cpus = parse_cpu_list(read_text_file(path));
        cpus.insert(cpus.end(), node_cpus.begin(), node_cpus.end());
    }

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    if (cpus.empty()) {
        throw std::runtime_error("NUMA node spec did not resolve to any CPUs: " + node_spec);
    }
    return join_cpu_list(cpus);
}

static std::vector<int> numa_nodes_for_cpu_spec(const std::string & cpu_spec) {
    std::vector<int> cpus = parse_cpu_list(cpu_spec);
    std::vector<int> nodes;

    if (cpus.empty()) {
        return nodes;
    }

    for (int node = 0; node < 1024; ++node) {
        const std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
        std::ifstream in(path);
        if (!in) {
            break;
        }

        std::string line;
        std::getline(in, line);
        std::vector<int> node_cpus = parse_cpu_list(line);
        for (int cpu : cpus) {
            if (std::find(node_cpus.begin(), node_cpus.end(), cpu) != node_cpus.end()) {
                nodes.push_back(node);
                break;
            }
        }
    }

    return nodes;
}

static std::vector<unsigned long> make_nodemask(const std::vector<int> & nodes) {
    const int bits_per_word = (int) (sizeof(unsigned long) * CHAR_BIT);
    const int max_node = *std::max_element(nodes.begin(), nodes.end());
    std::vector<unsigned long> mask((max_node / bits_per_word) + 1, 0);

    for (int node : nodes) {
        if (node < 0) {
            throw std::runtime_error("negative NUMA node id is invalid");
        }
        mask[node / bits_per_word] |= 1UL << (node % bits_per_word);
    }

    return mask;
}

static void bind_shared_window_to_numa(const std::string & path, size_t size, const std::string & node_spec, const char * label) {
    std::vector<int> nodes = parse_cpu_list(node_spec);
    if (nodes.empty()) {
        throw std::runtime_error(std::string(label) + " NUMA node spec is empty");
    }

    fd_handle fd(open(path.c_str(), O_RDWR | O_SYNC));
    if (!fd.valid()) {
        throw std::runtime_error("failed to open shared window " + path + ": " + strerror(errno));
    }

    void * base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
    if (base == MAP_FAILED) {
        throw std::runtime_error("failed to mmap shared window " + path + ": " + strerror(errno));
    }

    std::vector<unsigned long> mask = make_nodemask(nodes);
    const int bits_per_word = (int) (sizeof(unsigned long) * CHAR_BIT);
    const int max_node_id = *std::max_element(nodes.begin(), nodes.end());
    const unsigned long maxnode = (unsigned long) ((max_node_id / bits_per_word + 1) * bits_per_word);
    const long rc = syscall(SYS_mbind, base, size, MPOL_BIND, mask.data(), maxnode, 0);
    if (rc != 0) {
        const int err = errno;
        munmap(base, size);
        throw std::runtime_error(std::string("mbind failed for ") + label + " shared window on NUMA " + node_spec + ": " + strerror(err));
    }

    memset(base, 0, size);
    if (msync(base, size, MS_SYNC) != 0) {
        const int err = errno;
        munmap(base, size);
        throw std::runtime_error(std::string("msync failed for ") + label + " shared window: " + strerror(err));
    }

    munmap(base, size);
    fprintf(stderr, "pipeline-brick numa: bound %s shared window to NUMA nodes %s\n", label, node_spec.c_str());
}

// Allocate an anonymous mmap region pinned to a single NUMA node via mbind.
// node<0 means no binding (let the OS place it).
static void * alloc_on_node(size_t bytes, int node) {
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "alloc_on_node: mmap failed: %s\n", strerror(errno));
        return nullptr;
    }
    if (node >= 0) {
        const int bits_per_word = (int)(sizeof(unsigned long) * 8);
        std::vector<unsigned long> mask((node / bits_per_word) + 1, 0);
        mask[node / bits_per_word] |= 1UL << (node % bits_per_word);
        const unsigned long maxnode = (unsigned long)mask.size() * bits_per_word;
        const long rc = syscall(SYS_mbind, p, bytes, MPOL_BIND, mask.data(), maxnode, 0);
        if (rc != 0) {
            fprintf(stderr, "alloc_on_node: mbind(node=%d) failed: %s\n", node, strerror(errno));
            munmap(p, bytes);
            return nullptr;
        }
    }
    memset(p, 0, bytes); // first-touch on the bound node
    return p;
}

static void bind_to_cpus(const std::string & cpu_spec) {
    if (cpu_spec.empty()) {
        return;
    }

    std::vector<int> cpus = parse_cpu_list(cpu_spec);
    if (cpus.empty()) {
        throw std::runtime_error("--numa-cpus did not contain any CPU id");
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        CPU_SET(cpu, &set);
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        throw std::runtime_error("sched_setaffinity failed: " + std::string(strerror(errno)));
    }

    fprintf(stderr, "pipeline-brick numa: bound process to %zu CPUs from '%s'\n", cpus.size(), cpu_spec.c_str());
}

static void init_ggml_numa_for_binding(const pipeline_args & args) {
    std::vector<int> nodes = numa_nodes_for_cpu_spec(args.numa_cpus);
    if (nodes.size() == 1) {
        fprintf(stderr, "pipeline-brick numa: skip ggml NUMA init for single-node CPU binding node=%d\n", nodes[0]);
        return;
    }

    llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
}

static bool parse_numa_maps_node_token(const std::string & token, int & node, uint64_t & pages) {
    if (token.size() < 4 || token[0] != 'N') {
        return false;
    }

    size_t pos = 1;
    while (pos < token.size() && token[pos] >= '0' && token[pos] <= '9') {
        ++pos;
    }
    if (pos == 1 || pos >= token.size() || token[pos] != '=') {
        return false;
    }

    node = std::stoi(token.substr(1, pos - 1));
    pages = std::stoull(token.substr(pos + 1));
    return true;
}

static std::vector<uint64_t> read_numa_maps_pages() {
    std::ifstream in("/proc/self/numa_maps");
    std::vector<uint64_t> pages_by_node;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            int node = -1;
            uint64_t pages = 0;
            if (!parse_numa_maps_node_token(token, node, pages)) {
                continue;
            }
            if (node >= (int) pages_by_node.size()) {
                pages_by_node.resize((size_t) node + 1, 0);
            }
            pages_by_node[(size_t) node] += pages;
        }
    }
    return pages_by_node;
}

// Pipeline profiler: five-category breakdown for performance attribution.
// compute is the residual after subtracting communication, waiting, TP, and
// simulated KV traffic; it includes model compute and other local work.
struct pipeline_profiler {
    int64_t cxl_send_us = 0;   // sender-visible enqueue/copy time, including slot backpressure
    int64_t wait_recv_us = 0;  // blocking recv() wall time, including polling and payload copy
    int64_t kv_cxl_read_us = 0; // naive-KV-cross: simulated cross-CXL KV cache read time
    int64_t cxl_send_calls = 0;
    uint64_t cxl_send_bytes = 0;
    int64_t wait_recv_calls = 0;
    int64_t kv_cxl_read_calls = 0;
    uint64_t kv_cxl_read_bytes = 0;
};

static pipeline_profiler g_pipe_prof;

// naive-kv-cross simulation: a buffer pinned to a peer-CPU node, read each decode
// step to mimic a naive scenario where the brick's KV cache lives on the peer CPU
// and every attention reads K/V cross-CXL. Size grows with current sequence length.
static void * g_naive_kv_buf = nullptr;
static size_t g_naive_kv_buf_size = 0;

static void init_naive_kv_buf(int node, size_t max_bytes) {
    if (max_bytes == 0) return;
    g_naive_kv_buf = alloc_on_node(max_bytes, node);
    if (g_naive_kv_buf) {
        g_naive_kv_buf_size = max_bytes;
    }
}

// Read `bytes` from the peer-pinned buffer, accumulate time into kv_cxl_read_us.
// bytes = summed local K/V width across layers * cur_pos * n_sequences * elem_size.
static void simulate_kv_cxl_read(
        uint64_t n_kv_width, int32_t cur_pos, int32_t n_sequences) {
    if (!g_naive_kv_buf || g_naive_kv_buf_size == 0) return;
    const size_t elem_size = 2; // F16
    size_t bytes = (size_t) n_kv_width * (size_t) cur_pos
            * (size_t) n_sequences * elem_size;
    if (bytes > g_naive_kv_buf_size) bytes = g_naive_kv_buf_size;
    const int64_t t0 = ggml_time_us();
    // read the bytes to pull them across CXL; volatile sink prevents elision
    volatile uint8_t sink = 0;
    const uint8_t * p = (const uint8_t *) g_naive_kv_buf;
    for (size_t i = 0; i < bytes; i += 64) sink ^= p[i];
    asm volatile("" :: "r"(sink) : "memory");
    g_pipe_prof.kv_cxl_read_us += ggml_time_us() - t0;
    g_pipe_prof.kv_cxl_read_calls++;
    g_pipe_prof.kv_cxl_read_bytes += bytes;
}

struct pipe_timer {
    int64_t & acc;
    int64_t t0;
    explicit pipe_timer(int64_t & a) : acc(a), t0(ggml_time_us()) {}
    ~pipe_timer() { acc += ggml_time_us() - t0; }
};

template<typename Transport>
static void send_hidden_payload_profiled(
        const pipeline_args & args,
        Transport & transport,
        int32_t pos,
        int32_t n_tokens,
        int32_t n_embd,
        uint32_t flags,
        const void * payload,
        uint64_t payload_bytes) {
    if (args.tp_size <= 1 || args.tp_rank == 0) {
        pipe_timer timer(g_pipe_prof.cxl_send_us);
        g_pipe_prof.cxl_send_calls++;
        g_pipe_prof.cxl_send_bytes += payload_bytes;
        transport.send_hidden_payload(
                pos, n_tokens, n_embd, flags, payload, payload_bytes);
        return;
    }
    transport.send_hidden_payload(
            pos, n_tokens, n_embd, flags, payload, payload_bytes);
}

// Five-category breakdown. Percentages use the per-rank inference interval.
// Each TP rank prints its own (g_pipe_prof and tp stats are per-process after fork).
static void print_pipeline_profile(const pipeline_args & args, double infer_s) {
    const int64_t total_us = (int64_t)(infer_s * 1e6);
    if (total_us <= 0) {
        return;
    }
    const int64_t cxl_us    = g_pipe_prof.cxl_send_us;
    const int64_t wait_us   = g_pipe_prof.wait_recv_us;
    const int64_t reduce_us = ggml_tp_total_us();
    const int64_t kvcxl_us  = g_pipe_prof.kv_cxl_read_us;
    const int64_t compute_us = total_us - cxl_us - wait_us - reduce_us - kvcxl_us;
    const auto pct = [&](int64_t v) { return (v < 0 ? 0 : v) * 100.0 / total_us; };
    const char * rank_tag = args.tp_size > 1 ? "rank=" : "";

    fprintf(stderr,
            "pipeline-brick profile: role=%s %s%d total=%.2f ms | compute=%.1f%% cxl_send=%.1f%% wait_recv=%.1f%% tp_reduce=%.1f%% kv_cxl_read=%.1f%%\n",
            role_name(args.role).c_str(), rank_tag, args.tp_rank,
            infer_s * 1000.0,
            pct(compute_us), pct(cxl_us), pct(wait_us), pct(reduce_us), pct(kvcxl_us));
    fprintf(stderr,
            "pipeline-brick profile: cxl_send calls=%lld bytes=%llu us=%lld | wait_recv calls=%lld us=%lld | tp_reduce us=%lld | kv_cxl_read calls=%lld bytes=%llu us=%lld\n",
            (long long) g_pipe_prof.cxl_send_calls,
            (unsigned long long) g_pipe_prof.cxl_send_bytes, (long long) cxl_us,
            (long long) g_pipe_prof.wait_recv_calls, (long long) wait_us,
            (long long) reduce_us, (long long) g_pipe_prof.kv_cxl_read_calls,
            (unsigned long long) g_pipe_prof.kv_cxl_read_bytes, (long long) kvcxl_us);
}

struct pipeline_profile_snapshot {
    int64_t wall_us;
    int64_t cxl_send_us;
    int64_t wait_recv_us;
    int64_t tp_reduce_us;
    int64_t kv_cxl_read_us;
};

static pipeline_profile_snapshot capture_pipeline_profile() {
    return {
        ggml_time_us(),
        g_pipe_prof.cxl_send_us,
        g_pipe_prof.wait_recv_us,
        ggml_tp_total_us(),
        g_pipe_prof.kv_cxl_read_us,
    };
}

static void print_pipeline_phase_profile(
        const pipeline_args & args,
        const char * phase,
        const pipeline_profile_snapshot & begin,
        const pipeline_profile_snapshot & end) {
    const int64_t total_us  = end.wall_us - begin.wall_us;
    if (total_us <= 0) {
        return;
    }
    const int64_t cxl_us    = end.cxl_send_us - begin.cxl_send_us;
    const int64_t wait_us   = end.wait_recv_us - begin.wait_recv_us;
    const int64_t reduce_us = end.tp_reduce_us - begin.tp_reduce_us;
    const int64_t kvcxl_us  = end.kv_cxl_read_us - begin.kv_cxl_read_us;
    const int64_t compute_us = total_us - cxl_us - wait_us - reduce_us - kvcxl_us;
    const auto pct = [&](int64_t v) { return (v < 0 ? 0 : v) * 100.0 / total_us; };
    const char * rank_tag = args.tp_size > 1 ? "rank=" : "";

    fprintf(stderr,
            "pipeline-brick phase-profile: phase=%s role=%s %s%d total=%.2f ms | compute=%.1f%% cxl_send=%.1f%% wait_recv=%.1f%% tp_reduce=%.1f%% kv_cxl_read=%.1f%%\n",
            phase, role_name(args.role).c_str(), rank_tag, args.tp_rank,
            total_us / 1000.0,
            pct(compute_us), pct(cxl_us), pct(wait_us), pct(reduce_us), pct(kvcxl_us));
    fprintf(stderr,
            "pipeline-brick phase-profile: phase=%s cxl_send us=%lld | wait_recv us=%lld | tp_reduce us=%lld | kv_cxl_read us=%lld\n",
            phase, (long long) cxl_us, (long long) wait_us,
            (long long) reduce_us, (long long) kvcxl_us);
}

static void print_numa_pages_line(
        const pipeline_args & args,
        const char * kind,
        const std::vector<uint64_t> & pages_by_node,
        const std::vector<uint64_t> * before = nullptr) {
    const long page_size = sysconf(_SC_PAGESIZE);
    uint64_t total_pages = 0;
    fprintf(stderr,
            "pipeline-brick numa-maps: role=%s rank=%d/%d pid=%ld tp=tp%d kind=%s model=%s pages",
            role_name(args.role).c_str(), args.tp_rank, args.tp_size, (long) getpid(),
            args.tp_rank, kind, args.model.c_str());

    const size_t n_nodes = before != nullptr ? std::max(pages_by_node.size(), before->size()) : pages_by_node.size();
    for (size_t i = 0; i < n_nodes; ++i) {
        int64_t value = i < pages_by_node.size() ? (int64_t) pages_by_node[i] : 0;
        if (before != nullptr) {
            const uint64_t prev = i < before->size() ? (*before)[i] : 0;
            value -= (int64_t) prev;
        }
        if (value == 0) {
            continue;
        }
        total_pages += value > 0 ? (uint64_t) value : 0;
        fprintf(stderr, " N%zu=%" PRId64, i, value);
    }

    const double mib = page_size > 0 ? (double) total_pages * (double) page_size / 1048576.0 : 0.0;
    fprintf(stderr, " total=%" PRIu64 " total_mib=%.1f\n", total_pages, mib);
}

static int tail_kv_numa_node(const pipeline_args & args) {
    if (args.role != LLAMA_PIPELINE_BRICK_ROLE_TAIL || args.tail_kv_numa.empty()) {
        return -1;
    }

    const std::vector<int> nodes = parse_cpu_list_ordered(args.tail_kv_numa);
    if (args.tp_rank < 0 || args.tp_rank >= (int32_t) nodes.size()) {
        throw std::runtime_error("Tail TP rank has no matching --tail-kv-numa node");
    }
    return nodes[args.tp_rank];
}

static bool stage_mode_enabled(const pipeline_args & args);

static void apply_domain_mode_defaults(pipeline_args & args) {
    switch (args.domain) {
        case domain_mode::none:
            break;
        case domain_mode::single:
            args.single_system = true;
            break;
        case domain_mode::dual:
            args.hardware = true;
            if (args.stage_count == 0) {
                args.stage_count = 4;
            }
            break;
    }
}

static void prepare_async_prompts(pipeline_args & args) {
    if (!args.async_pipeline || args.prompt_file.empty()) {
        return;
    }

    std::ifstream input(args.prompt_file);
    if (!input) {
        throw std::runtime_error("failed to open --prompt-file: " + args.prompt_file);
    }

    std::string line;
    int32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            throw std::runtime_error("--prompt-file contains an empty prompt at line " + std::to_string(line_number));
        }
        args.async_prompts.push_back(std::move(line));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading --prompt-file: " + args.prompt_file);
    }
}

static void validate_args(const pipeline_args & args) {
    if (!args.hardware && !args.single_system && !args.self_test_ntb) {
        throw std::runtime_error("real-machine build requires --hardware, --single-system, or --self-test-ntb");
    }
    const bool stage_mode = stage_mode_enabled(args);
    if (args.domain == domain_mode::single) {
        if (args.hardware || args.self_test_ntb) {
            throw std::runtime_error("--domain-mode single cannot be combined with --hardware or --self-test-ntb");
        }
        if (args.stage_id >= 0 || args.stage_count != 0 || args.role == LLAMA_PIPELINE_BRICK_ROLE_STAGE ||
                !args.stage_numa.empty()) {
            throw std::runtime_error("--domain-mode single uses the two-stage head/tail path and cannot be combined with stage options");
        }
    }
    if (args.domain == domain_mode::dual) {
        if (args.single_system || args.self_test_ntb) {
            throw std::runtime_error("--domain-mode dual cannot be combined with --single-system or --self-test-ntb");
        }
        if (args.stage_count != 4) {
            throw std::runtime_error("--domain-mode dual requires --stage-count 4");
        }
        if (args.stage_id < 0) {
            throw std::runtime_error("--domain-mode dual requires --stage-id N");
        }
    }
    if (!stage_mode && args.bricks != 2) {
        throw std::runtime_error("pipeline-brick requires --bricks 2");
    }
    if (args.tp_size <= 0) {
        throw std::runtime_error("--tp-size must be positive");
    }
    if (args.tp_size != 1 && args.tp_size != 2 && args.tp_size != 4) {
        throw std::runtime_error("pipeline-brick TP prototype supports only --tp-size 1, 2, or 4");
    }
    if (args.ctx_size <= 0) {
        throw std::runtime_error("--ctx-size must be positive");
    }
    if (args.threads <= 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (args.naive_transfer_mult <= 0) {
        throw std::runtime_error("--naive-transfer-mult must be positive");
    }
    if (!args.tail_kv_numa.empty()) {
        if (!args.single_system || stage_mode) {
            throw std::runtime_error("--tail-kv-numa is supported only in single-system Head/Tail mode");
        }
        if (args.naive_kv_cross) {
            throw std::runtime_error("--tail-kv-numa cannot be combined with --naive-kv-cross");
        }
        std::vector<int> nodes;
        try {
            nodes = parse_cpu_list_ordered(args.tail_kv_numa);
        } catch (const std::exception &) {
            throw std::runtime_error("--tail-kv-numa contains an invalid or negative NUMA node");
        }
        if ((int32_t) nodes.size() != args.tp_size) {
            throw std::runtime_error("--tail-kv-numa must contain exactly --tp-size NUMA nodes");
        }
        for (int node : nodes) {
            if (node < 0) {
                throw std::runtime_error("--tail-kv-numa node ids must be non-negative");
            }
        }
        std::vector<int> unique_nodes = nodes;
        std::sort(unique_nodes.begin(), unique_nodes.end());
        if (std::adjacent_find(unique_nodes.begin(), unique_nodes.end()) != unique_nodes.end()) {
            throw std::runtime_error("--tail-kv-numa must not contain duplicate NUMA nodes");
        }
    }
    if (args.async_pipeline) {
        if (args.prompt_file.empty()) {
            throw std::runtime_error("--async-pipeline requires --prompt-file");
        }
        if (!args.prompt.empty()) {
            throw std::runtime_error("--async-pipeline cannot be combined with --prompt");
        }
        if (args.pipeline_microbatch <= 0 || args.pipeline_microbatch > args.parallel) {
            throw std::runtime_error("--pipeline-microbatch must be between 1 and --parallel");
        }
        if ((int32_t) args.async_prompts.size() != args.parallel) {
            throw std::runtime_error("--prompt-file must contain exactly --parallel non-empty lines");
        }
        if (args.parallel > (int32_t) PIPELINE_N_SLOTS) {
            throw std::runtime_error("--async-pipeline requires --parallel <= 64");
        }
        if (args.stream_kv) {
            throw std::runtime_error("--async-pipeline cannot be combined with --stream-kv");
        }
        if (args.naive_kv_cross) {
            throw std::runtime_error("--async-pipeline cannot be combined with --naive-kv-cross");
        }
    } else {
        if (!args.prompt_file.empty()) {
            throw std::runtime_error("--prompt-file requires --async-pipeline");
        }
        if (args.pipeline_microbatch != 0) {
            throw std::runtime_error("--pipeline-microbatch requires --async-pipeline");
        }
    }
    if (args.stream_kv) {
        if (args.stream_kv_sink < 0 || args.stream_kv_recent < 0) {
            throw std::runtime_error("--stream-kv-sink and --stream-kv-recent must be non-negative");
        }
        if (args.stream_kv_sink + args.stream_kv_recent <= 0) {
            throw std::runtime_error("--stream-kv requires at least one retained token");
        }
    }
    if (args.single_system) {
        if (args.model.empty()) {
            throw std::runtime_error("--model is required");
        }
        if (!args.async_pipeline && args.prompt.empty()) {
            throw std::runtime_error("--prompt is required");
        }
        if (stage_mode && args.stage_numa.empty()) {
            throw std::runtime_error("--stage-numa is required for single-system stage mode");
        }
        if (!stage_mode && (args.head_numa.empty() || args.tail_numa.empty())) {
            throw std::runtime_error("--head-numa and --tail-numa are required for --single-system");
        }
        if (args.parallel <= 0) {
            throw std::runtime_error("--parallel must be positive");
        }
        if (args.n_predict <= 0) {
            throw std::runtime_error("--n-predict must be positive");
        }
        if (args.prefill_chunk <= 0) {
            throw std::runtime_error("--prefill-chunk must be positive");
        }
        return;
    }
    if (!stage_mode && args.role == LLAMA_PIPELINE_BRICK_ROLE_NONE) {
        throw std::runtime_error("--role head|tail is required");
    }
    if (!stage_mode && (args.brick_id < 0 || args.peer_brick_id < 0)) {
        throw std::runtime_error("--brick-id and --peer-brick-id are required");
    }
    if (args.parallel <= 0) {
        throw std::runtime_error("--parallel must be positive");
    }
    if (args.numa_tp <= 0) {
        throw std::runtime_error("--numa-tp must be positive");
    }
    if (args.prefill_chunk <= 0) {
        throw std::runtime_error("--prefill-chunk must be positive");
    }
    if (stage_mode && args.tp_size > 1 && args.stage_numa.empty()) {
        throw std::runtime_error("--stage-numa is required for stage mode with --tp-size > 1");
    }
    if (!stage_mode && args.transport == transport_kind::ntb_mw &&
            args.db_mode == doorbell_mode::write && (args.tx_doorbell.empty() || args.rx_doorbell.empty())) {
        throw std::runtime_error("--tx-doorbell and --rx-doorbell are required for --doorbell-mode write");
    }
    if (stage_mode && args.db_mode == doorbell_mode::write &&
            (args.up_transport == transport_kind::ntb_mw || args.down_transport == transport_kind::ntb_mw)) {
        throw std::runtime_error("stage mode with ntb-mw requires --doorbell-mode poll");
    }
    if (!args.self_test_ntb) {
        if (args.model.empty()) {
            throw std::runtime_error("--model is required");
        }
        if (!args.async_pipeline &&
                ((!stage_mode && args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD && args.prompt.empty()) ||
                 (stage_mode && args.stage_id == 0 && args.prompt.empty()))) {
            throw std::runtime_error("--prompt is required on head");
        }
        if (args.n_predict <= 0) {
            throw std::runtime_error("--n-predict must be positive");
        }
        const bool layer_range_defaulted = stage_mode && args.layer_start < 0 && args.layer_end < 0;
        if (!layer_range_defaulted &&
                (args.layer_start < 0 || args.layer_end <= args.layer_start || args.layer_end > effective_n_layer(args))) {
            throw std::runtime_error("invalid --layer-start/--layer-end");
        }
        if (!stage_mode) {
            const int32_t n_layer = effective_n_layer(args);
            const int32_t split = effective_split_layer(args);
            if (n_layer % 2 != 0) {
                fprintf(stderr,
                        "pipeline-brick: n_layer=%d is odd, Head/Tail split is uneven (%d vs %d layers)\n",
                        n_layer, split, n_layer - split);
            }
            if (args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD && (args.layer_start != 0 || args.layer_end != split)) {
                throw std::runtime_error("head must use --layer-start 0 --layer-end " + std::to_string(split));
            }
            if (args.role == LLAMA_PIPELINE_BRICK_ROLE_TAIL && (args.layer_start != split || args.layer_end != n_layer)) {
                throw std::runtime_error("tail must use --layer-start " + std::to_string(split) +
                                         " --layer-end " + std::to_string(n_layer));
            }
        }
    }
}

static llama_model * load_brick_model(const pipeline_args & args) {
    std::vector<uint64_t> numa_pages_before;
    if (args.tp_size > 1) {
        numa_pages_before = read_numa_maps_pages();
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.pipeline_brick_enabled = true;
    mparams.pipeline_brick_role = args.role;
    mparams.pipeline_brick_layer_start = args.layer_start;
    mparams.pipeline_brick_layer_end = args.layer_end;
    mparams.pipeline_brick_tp_rank = args.tp_rank;
    mparams.pipeline_brick_tp_size = args.tp_size;
    mparams.pipeline_brick_kv_numa_node = tail_kv_numa_node(args);
    if (mparams.pipeline_brick_kv_numa_node >= 0) {
        fprintf(stderr,
                "pipeline-brick KV NUMA: role=%s rank=%d/%d target_node=%d\n",
                role_name(args.role).c_str(), args.tp_rank, args.tp_size,
                mparams.pipeline_brick_kv_numa_node);
    }
    if (args.tp_size > 1) {
        mparams.use_mmap = false; // load TP shards into rank-local memory after CPU binding
    }
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (args.tp_size > 1) {
        const std::vector<uint64_t> numa_pages_after = read_numa_maps_pages();
        print_numa_pages_line(args, "after_model_load", numa_pages_after);
        print_numa_pages_line(args, "model_load_delta", numa_pages_after, &numa_pages_before);
    }
    return model;
}

static llama_context * make_context(llama_model * model, const pipeline_args & args, bool embeddings) {
    llama_context_params cparams = llama_context_default_params();
    const int32_t max_batch = max_micro_batch_tokens(args);
    cparams.n_ctx = args.ctx_size * args.parallel;
    cparams.n_batch = max_batch;
    cparams.n_ubatch = max_batch;
    cparams.n_seq_max = args.parallel;
    cparams.n_outputs_max = max_batch;
    cparams.n_threads = args.threads;
    cparams.n_threads_batch = args.threads;
    cparams.embeddings = embeddings;
    cparams.no_perf = true;
    if (args.stream_kv) {
        cparams.stream_kv_enabled = true;
        cparams.stream_kv_active = false;
        cparams.stream_kv_sink = args.stream_kv_sink;
        cparams.stream_kv_recent = args.stream_kv_recent;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        throw std::runtime_error("failed to create llama context");
    }
    llama_set_embeddings(ctx, embeddings);
    return ctx;
}

static void set_batch_entry(llama_batch & batch, int32_t i, int32_t seq_id, int32_t pos, bool want_output) {
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i] = want_output ? 1 : 0;
}

static llama_token greedy_sample(llama_context * ctx, const llama_vocab * vocab, int32_t batch_index) {
    float * logits = llama_get_logits_ith(ctx, batch_index);
    if (!logits) {
        throw std::runtime_error("failed to get logits");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token best = 0;
    float best_score = -INFINITY;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i]) && logits[i] > best_score) {
            best_score = logits[i];
            best = i;
        }
    }
    return best;
}

static size_t hidden_data_offset(int32_t n_tokens) {
    return (size_t) n_tokens * sizeof(hidden_token_meta);
}

static size_t hidden_payload_bytes(int32_t n_tokens, int32_t n_embd, hidden_dtype type) {
    return hidden_data_offset(n_tokens) + (size_t) n_tokens * (size_t) n_embd * hidden_dtype_size(type);
}

static std::vector<uint8_t> make_hidden_payload(
        llama_context * ctx,
        const std::vector<hidden_token_meta> & metas,
        int32_t n_embd,
        hidden_dtype type) {
    const int32_t n_tokens = (int32_t) metas.size();
    std::vector<uint8_t> payload(hidden_payload_bytes(n_tokens, n_embd, type));
    memcpy(payload.data(), metas.data(), metas.size() * sizeof(hidden_token_meta));

    uint8_t * data = payload.data() + hidden_data_offset(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * hidden = llama_get_embeddings_ith(ctx, i);
        if (!hidden) {
            throw std::runtime_error("failed to get hidden states");
        }

        if (type == hidden_dtype::f32) {
            memcpy(data + (size_t) i * n_embd * sizeof(float), hidden, (size_t) n_embd * sizeof(float));
        } else {
            ggml_fp32_to_fp16_row(hidden, reinterpret_cast<ggml_fp16_t *>(data) + (size_t) i * n_embd, n_embd);
        }
    }

    return payload;
}

static bool decode_hidden_payload(
        const recv_packet & packet,
        hidden_dtype type,
        int32_t n_embd,
        std::vector<hidden_token_meta> & metas,
        std::vector<float> & hidden_f32) {
    const int32_t n_tokens = packet.header.n_tokens;
    if (n_tokens <= 0 || packet.header.n_embd != n_embd) {
        return false;
    }

    const size_t expected = hidden_payload_bytes(n_tokens, n_embd, type);
    if (packet.payload.size() != expected) {
        return false;
    }

    metas.resize(n_tokens);
    memcpy(metas.data(), packet.payload.data(), (size_t) n_tokens * sizeof(hidden_token_meta));

    hidden_f32.resize((size_t) n_tokens * n_embd);
    const uint8_t * data = packet.payload.data() + hidden_data_offset(n_tokens);
    if (type == hidden_dtype::f32) {
        memcpy(hidden_f32.data(), data, hidden_f32.size() * sizeof(float));
    } else {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(data), hidden_f32.data(), (int64_t) hidden_f32.size());
    }

    return true;
}

template<typename Transport>
static llama_token read_token_for_seq(Transport & transport, int32_t seq_id, std::vector<token_payload> & pending) {
    for (auto it = pending.begin(); it != pending.end(); ++it) {
        if (it->seq_id == seq_id) {
            const llama_token token = (llama_token) it->token;
            pending.erase(it);
            return token;
        }
    }

    while (true) {
        int64_t rw0 = ggml_time_us();
        recv_packet packet = transport.recv();
        g_pipe_prof.wait_recv_us += ggml_time_us() - rw0;
        g_pipe_prof.wait_recv_calls++;
        if (!(packet.header.flags & PIPELINE_FLAG_TOKEN) || packet.payload.size() != sizeof(token_payload)) {
            throw std::runtime_error("expected token packet on reverse channel");
        }
        token_payload payload;
        memcpy(&payload, packet.payload.data(), sizeof(payload));
        if (payload.seq_id == seq_id) {
            return (llama_token) payload.token;
        }
        pending.push_back(payload);
    }
}

template<typename Transport>
static int run_head_sync(const pipeline_args & args, Transport & transport) {
    llama_backend_init();
    bind_to_cpus(args.numa_cpus);
    init_ggml_numa_for_binding(args);

    llama_model * model = load_brick_model(args);
    if (!model) {
        fprintf(stderr, "pipeline-brick head: failed to load model\n");
        return 2;
    }

    llama_context * ctx = make_context(model, args, true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t t_infer_start = ggml_time_us();
    const std::string prompt =
        std::string(PIPELINE_PARALLEL_SYSTEM_PROMPT) +
        "User:\n" + args.prompt + "\nAssistant:\n";
    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, false);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "pipeline-brick head: empty prompt after tokenization\n");
        return 2;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd != effective_n_embd(args)) {
        fprintf(stderr, "pipeline-brick head: expected n_embd=%d, got %d (use --n-embd to override)\n", effective_n_embd(args), n_embd);
        return 2;
    }

    fprintf(stderr,
            "pipeline-brick head: brick=%d peer=%d layers [%d,%d), parallel=%d, n_embd=%d, numa_tp=%d\n",
            args.brick_id, args.peer_brick_id, args.layer_start, args.layer_end, args.parallel, n_embd, args.numa_tp);
    fprintf(stderr,
            "pipeline-brick head: llama-parallel prompt template enabled, prompt tokens=%zu\n",
            prompt_tokens.size());
    fprintf(stderr,
            "pipeline-brick head: micro-batch max=%d prefill_chunk=%d hidden_dtype=%s bytes_per_token=%zu\n",
            max_micro_batch_tokens(args), args.prefill_chunk, hidden_dtype_name(args.hidden_type),
            (size_t) n_embd * hidden_dtype_size(args.hidden_type));
    if (args.stream_kv) {
        fprintf(stderr,
                "pipeline-brick head: stream-kv enabled sink=%d recent=%d\n",
                args.stream_kv_sink, args.stream_kv_recent);
    }

    llama_batch batch = llama_batch_init(max_micro_batch_tokens(args), 0, 1);
    const int64_t n_total_prompt = (int64_t) prompt_tokens.size() * args.parallel;
    const int64_t n_total_gen = (int64_t) args.n_predict * args.parallel;
    int64_t n_gen_read = 0;

    if (args.stream_kv) {
        llama_set_stream_kv_active(ctx, false);
    }

    const pipeline_profile_snapshot prefill_start = capture_pipeline_profile();
    for (size_t offset = 0; offset < prompt_tokens.size(); offset += args.prefill_chunk) {
        const size_t chunk = std::min((size_t) args.prefill_chunk, prompt_tokens.size() - offset);
        batch.n_tokens = (int32_t) chunk * args.parallel;
        std::vector<hidden_token_meta> metas;
        metas.reserve(batch.n_tokens);

        int32_t bi = 0;
        for (size_t j = 0; j < chunk; ++j) {
            const int32_t pos = (int32_t) (offset + j);
            const bool want_logits = (offset + j + 1 == prompt_tokens.size());
            for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
                batch.token[bi] = prompt_tokens[offset + j];
                set_batch_entry(batch, bi, seq_id, pos, true);
                metas.push_back({ seq_id, pos, want_logits ? PIPELINE_FLAG_WANT_LOGITS : 0u, 0u });
                ++bi;
            }
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "pipeline-brick head: llama_decode failed at prompt offset=%zu n_tokens=%d\n", offset, batch.n_tokens);
            return 3;
        }

        std::vector<uint8_t> payload;
        try {
            payload = make_hidden_payload(ctx, metas, n_embd, args.hidden_type);
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return 3;
        }

        uint32_t flags = args.hidden_type == hidden_dtype::f16 ? PIPELINE_FLAG_HIDDEN_F16 : 0u;
        send_hidden_payload_profiled(
                args, transport, (int32_t) offset, batch.n_tokens, n_embd,
                flags, payload.data(), payload.size());
        // naive-transfer-mult: simulate naive pipeline that additionally transfers (mult-1) copies
        // of activation-size data each send (mimics sending per-layer activations instead of just
        // the boundary hidden state). Re-sends the same payload bytes to match data volume.
        for (int32_t m = 1; m < args.naive_transfer_mult; ++m) {
            send_hidden_payload_profiled(
                    args, transport, (int32_t) offset, batch.n_tokens, n_embd,
                    flags | PIPELINE_FLAG_TRAFFIC_ONLY, payload.data(), payload.size());
        }
        if (!args.quiet) {
            fprintf(stderr, "pipeline-brick head: sent prefill pos=%zu n_tokens=%d bytes=%zu dtype=%s naive_mult=%d\n",
                    offset, batch.n_tokens, payload.size(), hidden_dtype_name(args.hidden_type), args.naive_transfer_mult);
        }
    }

    std::vector<token_payload> pending_tokens;
    pipeline_profile_snapshot decode_start = {};
    bool decode_phase_started = false;
    for (int32_t step = 1; step < args.n_predict; ++step) {
        const int32_t decode_pos = (int32_t) prompt_tokens.size() + step - 1;
        batch.n_tokens = args.parallel;
        std::vector<hidden_token_meta> metas;
        metas.reserve(batch.n_tokens);

        for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
            llama_token token = 0;
            try {
                token = read_token_for_seq(transport, seq_id, pending_tokens);
                ++n_gen_read;
            } catch (const std::exception & e) {
                fprintf(stderr, "pipeline-brick head: %s\n", e.what());
                return 3;
            }

            batch.token[seq_id] = token;
            set_batch_entry(batch, seq_id, seq_id, decode_pos, true);
            metas.push_back({ seq_id, decode_pos, PIPELINE_FLAG_WANT_LOGITS, 0u });
        }

        if (!decode_phase_started) {
            decode_start = capture_pipeline_profile();
            decode_phase_started = true;
        }

        if (args.stream_kv) {
            const bool sparse_active = decode_pos + 1 > args.stream_kv_sink + args.stream_kv_recent;
            llama_set_stream_kv_active(ctx, sparse_active);
            if (args.verbose) {
                fprintf(stderr, "pipeline-brick head: stream-kv decode pos=%d sparse=%d\n", decode_pos, sparse_active ? 1 : 0);
            }
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "pipeline-brick head: llama_decode failed at generated step=%d n_tokens=%d\n", step, batch.n_tokens);
            return 3;
        }

        std::vector<uint8_t> payload;
        try {
            payload = make_hidden_payload(ctx, metas, n_embd, args.hidden_type);
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return 3;
        }

        uint32_t flags = args.hidden_type == hidden_dtype::f16 ? PIPELINE_FLAG_HIDDEN_F16 : 0u;
        send_hidden_payload_profiled(
                args, transport, decode_pos, batch.n_tokens, n_embd,
                flags | PIPELINE_FLAG_WANT_LOGITS | PIPELINE_FLAG_DECODE,
                payload.data(), payload.size());
        for (int32_t m = 1; m < args.naive_transfer_mult; ++m) {
            send_hidden_payload_profiled(
                    args, transport, decode_pos, batch.n_tokens, n_embd,
                    flags | PIPELINE_FLAG_DECODE | PIPELINE_FLAG_TRAFFIC_ONLY,
                    payload.data(), payload.size());
        }
        if (!args.quiet) {
            fprintf(stderr, "pipeline-brick head: sent decode pos=%d n_tokens=%d bytes=%zu dtype=%s naive_mult=%d\n",
                    decode_pos, batch.n_tokens, payload.size(), hidden_dtype_name(args.hidden_type), args.naive_transfer_mult);
        }
    }

    for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
        try {
            (void) read_token_for_seq(transport, seq_id, pending_tokens);
            ++n_gen_read;
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return 3;
        }
    }

    if (!decode_phase_started) {
        decode_start = capture_pipeline_profile();
        decode_phase_started = true;
    }
    const pipeline_profile_snapshot phase_end = capture_pipeline_profile();
    const int64_t t_infer_end = phase_end.wall_us;
    transport.send_stop(0, (int32_t) prompt_tokens.size() + args.n_predict - 1);
    const double infer_s = (t_infer_end - t_infer_start) / 1e6;
    const double prefill_s = (decode_start.wall_us - prefill_start.wall_us) / 1e6;
    const double decode_s = (phase_end.wall_us - decode_start.wall_us) / 1e6;
    const int64_t n_decode_steps = args.n_predict - 1;
    const int64_t n_decode_tokens = n_decode_steps * (int64_t) args.parallel;
    if (args.tp_size <= 1 || args.tp_rank == 0) {
        fprintf(stderr, "pipeline-brick perf: inference time %.2f s\n", infer_s);
        fprintf(stderr, "pipeline-brick perf: total prompt tokens %lld, speed %.2f t/s\n",
                (long long) n_total_prompt, n_total_prompt / infer_s);
        fprintf(stderr, "pipeline-brick perf: total gen tokens %lld, speed %.2f t/s\n",
                (long long) n_total_gen, n_total_gen / infer_s);
        fprintf(stderr, "pipeline-brick perf: measured gen tokens %lld / target %lld\n",
                (long long) n_gen_read, (long long) n_total_gen);
        fprintf(stderr, "pipeline-brick perf: total tokens %lld, speed %.2f t/s\n",
                (long long) (n_total_prompt + n_total_gen), (n_total_prompt + n_total_gen) / infer_s);
        fprintf(stderr,
                "pipeline-brick phase: prefill time %.3f s, prompt tokens %lld, speed %.2f t/s, ttft %.2f ms\n",
                prefill_s, (long long) n_total_prompt,
                prefill_s > 0.0 ? n_total_prompt / prefill_s : 0.0,
                prefill_s * 1000.0);
        fprintf(stderr,
                "pipeline-brick phase: decode time %.3f s, decode tokens %lld, speed %.2f t/s, steps %lld, tpot %.3f ms/step\n",
                decode_s, (long long) n_decode_tokens,
                decode_s > 0.0 ? n_decode_tokens / decode_s : 0.0,
                (long long) n_decode_steps,
                n_decode_steps > 0 ? decode_s * 1000.0 / n_decode_steps : 0.0);
    }
    print_pipeline_phase_profile(args, "prefill", prefill_start, decode_start);
    print_pipeline_phase_profile(args, "decode", decode_start, phase_end);
    print_pipeline_profile(args, infer_s);
    if (args.stream_kv) {
        llama_set_stream_kv_active(ctx, false);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

template<typename Transport>
static int run_head_async(const pipeline_args & args, Transport & transport) {
    llama_backend_init();
    bind_to_cpus(args.numa_cpus);
    init_ggml_numa_for_binding(args);

    llama_model * model = load_brick_model(args);
    if (!model) {
        fprintf(stderr, "pipeline-brick head: failed to load model\n");
        return 2;
    }

    llama_context * ctx = make_context(model, args, true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int64_t t_infer_start = ggml_time_us();
    std::vector<std::vector<llama_token>> prompt_tokens(args.parallel);
    size_t max_prompt_tokens = 0;
    int64_t n_total_prompt = 0;
    for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
        const std::string prompt =
            std::string(PIPELINE_PARALLEL_SYSTEM_PROMPT) +
            "User:\n" + args.async_prompts[seq_id] + "\nAssistant:\n";
        prompt_tokens[seq_id] = common_tokenize(vocab, prompt, false);
        if (prompt_tokens[seq_id].empty()) {
            fprintf(stderr, "pipeline-brick head: empty prompt after tokenization for seq=%d\n", seq_id);
            return 2;
        }
        if ((int64_t) prompt_tokens[seq_id].size() + args.n_predict > args.ctx_size) {
            fprintf(stderr,
                    "pipeline-brick head: seq=%d requires %zu prompt tokens + %d generated tokens, ctx_size=%d\n",
                    seq_id, prompt_tokens[seq_id].size(), args.n_predict, args.ctx_size);
            return 2;
        }
        max_prompt_tokens = std::max(max_prompt_tokens, prompt_tokens[seq_id].size());
        n_total_prompt += (int64_t) prompt_tokens[seq_id].size();
    }

    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd != effective_n_embd(args)) {
        fprintf(stderr, "pipeline-brick head: expected n_embd=%d, got %d (use --n-embd to override)\n", effective_n_embd(args), n_embd);
        return 2;
    }

    const int32_t n_groups =
        (args.parallel + args.pipeline_microbatch - 1) / args.pipeline_microbatch;
    fprintf(stderr,
            "pipeline-brick head: brick=%d peer=%d layers [%d,%d), parallel=%d, n_embd=%d, numa_tp=%d\n",
            args.brick_id, args.peer_brick_id, args.layer_start, args.layer_end, args.parallel, n_embd, args.numa_tp);
    fprintf(stderr,
            "pipeline-brick head: async-pipeline decode-only prompts=%d microbatch=%d groups=%d prompt_file=%s\n",
            args.parallel, args.pipeline_microbatch, n_groups, args.prompt_file.c_str());
    fprintf(stderr,
            "pipeline-brick head: micro-batch max=%d prefill_chunk=%d hidden_dtype=%s bytes_per_token=%zu\n",
            max_micro_batch_tokens(args), args.prefill_chunk, hidden_dtype_name(args.hidden_type),
            (size_t) n_embd * hidden_dtype_size(args.hidden_type));
    for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
        fprintf(stderr, "pipeline-brick head: async prompt seq=%d tokens=%zu\n",
                seq_id, prompt_tokens[seq_id].size());
    }

    llama_batch batch = llama_batch_init(max_micro_batch_tokens(args), 0, 1);
    const int64_t n_total_gen = (int64_t) args.n_predict * args.parallel;
    int64_t n_gen_read = 0;
    int64_t decode_dispatches = 0;

    const pipeline_profile_snapshot prefill_start = capture_pipeline_profile();
    for (size_t offset = 0; offset < max_prompt_tokens; offset += args.prefill_chunk) {
        const size_t end = std::min(offset + (size_t) args.prefill_chunk, max_prompt_tokens);
        batch.n_tokens = 0;
        std::vector<hidden_token_meta> metas;
        metas.reserve((end - offset) * (size_t) args.parallel);

        for (size_t pos = offset; pos < end; ++pos) {
            for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
                if (pos >= prompt_tokens[seq_id].size()) {
                    continue;
                }
                const int32_t bi = batch.n_tokens++;
                const bool want_logits = pos + 1 == prompt_tokens[seq_id].size();
                batch.token[bi] = prompt_tokens[seq_id][pos];
                set_batch_entry(batch, bi, seq_id, (int32_t) pos, true);
                metas.push_back({ seq_id, (int32_t) pos, want_logits ? PIPELINE_FLAG_WANT_LOGITS : 0u, 0u });
            }
        }

        if (batch.n_tokens == 0) {
            continue;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "pipeline-brick head: llama_decode failed at async prompt offset=%zu n_tokens=%d\n",
                    offset, batch.n_tokens);
            return 3;
        }

        std::vector<uint8_t> payload;
        try {
            payload = make_hidden_payload(ctx, metas, n_embd, args.hidden_type);
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return 3;
        }

        uint32_t flags = args.hidden_type == hidden_dtype::f16 ? PIPELINE_FLAG_HIDDEN_F16 : 0u;
        send_hidden_payload_profiled(
                args, transport, (int32_t) offset, batch.n_tokens, n_embd,
                flags, payload.data(), payload.size());
        for (int32_t m = 1; m < args.naive_transfer_mult; ++m) {
            send_hidden_payload_profiled(
                    args, transport, (int32_t) offset, batch.n_tokens, n_embd,
                    flags | PIPELINE_FLAG_TRAFFIC_ONLY, payload.data(), payload.size());
        }
        if (!args.quiet) {
            fprintf(stderr,
                    "pipeline-brick head: sent async prefill pos=%zu n_tokens=%d bytes=%zu dtype=%s naive_mult=%d\n",
                    offset, batch.n_tokens, payload.size(), hidden_dtype_name(args.hidden_type),
                    args.naive_transfer_mult);
        }
    }

    std::vector<token_payload> pending_tokens;
    std::vector<llama_token> current_tokens(args.parallel);
    for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
        try {
            current_tokens[seq_id] = read_token_for_seq(transport, seq_id, pending_tokens);
            ++n_gen_read;
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return 3;
        }
    }
    const pipeline_profile_snapshot decode_start = capture_pipeline_profile();

    auto dispatch_group = [&](int32_t group_id, int32_t step) -> bool {
        const int32_t seq_begin = group_id * args.pipeline_microbatch;
        const int32_t seq_end = std::min(seq_begin + args.pipeline_microbatch, args.parallel);
        batch.n_tokens = seq_end - seq_begin;
        std::vector<hidden_token_meta> metas;
        metas.reserve(batch.n_tokens);

        for (int32_t seq_id = seq_begin; seq_id < seq_end; ++seq_id) {
            const int32_t bi = seq_id - seq_begin;
            const int32_t decode_pos = (int32_t) prompt_tokens[seq_id].size() + step - 1;
            batch.token[bi] = current_tokens[seq_id];
            set_batch_entry(batch, bi, seq_id, decode_pos, true);
            metas.push_back({ seq_id, decode_pos, PIPELINE_FLAG_WANT_LOGITS, 0u });
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr,
                    "pipeline-brick head: llama_decode failed at async generated step=%d group=%d n_tokens=%d\n",
                    step, group_id, batch.n_tokens);
            return false;
        }

        std::vector<uint8_t> payload;
        try {
            payload = make_hidden_payload(ctx, metas, n_embd, args.hidden_type);
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick head: %s\n", e.what());
            return false;
        }

        const int32_t header_pos = metas.front().pos;
        uint32_t flags = args.hidden_type == hidden_dtype::f16 ? PIPELINE_FLAG_HIDDEN_F16 : 0u;
        send_hidden_payload_profiled(
                args, transport, header_pos, batch.n_tokens, n_embd,
                flags | PIPELINE_FLAG_WANT_LOGITS | PIPELINE_FLAG_DECODE,
                payload.data(), payload.size());
        for (int32_t m = 1; m < args.naive_transfer_mult; ++m) {
            send_hidden_payload_profiled(
                    args, transport, header_pos, batch.n_tokens, n_embd,
                    flags | PIPELINE_FLAG_DECODE | PIPELINE_FLAG_TRAFFIC_ONLY,
                    payload.data(), payload.size());
        }
        ++decode_dispatches;
        if (!args.quiet) {
            fprintf(stderr,
                    "pipeline-brick head: sent async decode step=%d group=%d seq=[%d,%d) pos=%d n_tokens=%d bytes=%zu\n",
                    step, group_id, seq_begin, seq_end, header_pos, batch.n_tokens, payload.size());
        }
        return true;
    };

    auto receive_group = [&](int32_t group_id) -> bool {
        const int32_t seq_begin = group_id * args.pipeline_microbatch;
        const int32_t seq_end = std::min(seq_begin + args.pipeline_microbatch, args.parallel);
        for (int32_t seq_id = seq_begin; seq_id < seq_end; ++seq_id) {
            try {
                current_tokens[seq_id] = read_token_for_seq(transport, seq_id, pending_tokens);
                ++n_gen_read;
            } catch (const std::exception & e) {
                fprintf(stderr, "pipeline-brick head: %s\n", e.what());
                return false;
            }
        }
        return true;
    };

    if (args.n_predict > 1) {
        for (int32_t group_id = 0; group_id < n_groups; ++group_id) {
            if (!dispatch_group(group_id, 1)) {
                return 3;
            }
        }

        for (int32_t step = 2; step < args.n_predict; ++step) {
            for (int32_t group_id = 0; group_id < n_groups; ++group_id) {
                if (!receive_group(group_id) || !dispatch_group(group_id, step)) {
                    return 3;
                }
            }
        }

        for (int32_t group_id = 0; group_id < n_groups; ++group_id) {
            if (!receive_group(group_id)) {
                return 3;
            }
        }
    }

    const pipeline_profile_snapshot phase_end = capture_pipeline_profile();
    const int64_t t_infer_end = phase_end.wall_us;
    transport.send_stop(0, (int32_t) max_prompt_tokens + args.n_predict - 1);
    const double infer_s = (t_infer_end - t_infer_start) / 1e6;
    const double prefill_s = (decode_start.wall_us - prefill_start.wall_us) / 1e6;
    const double decode_s = (phase_end.wall_us - decode_start.wall_us) / 1e6;
    const int64_t n_decode_steps = args.n_predict - 1;
    const int64_t n_decode_tokens = n_decode_steps * (int64_t) args.parallel;
    if (args.tp_size <= 1 || args.tp_rank == 0) {
        fprintf(stderr,
                "pipeline-brick async: decode-only prompts=%d microbatch=%d groups=%d decode_dispatches=%lld\n",
                args.parallel, args.pipeline_microbatch, n_groups, (long long) decode_dispatches);
        fprintf(stderr, "pipeline-brick perf: inference time %.2f s\n", infer_s);
        fprintf(stderr, "pipeline-brick perf: total prompt tokens %lld, speed %.2f t/s\n",
                (long long) n_total_prompt, n_total_prompt / infer_s);
        fprintf(stderr, "pipeline-brick perf: total gen tokens %lld, speed %.2f t/s\n",
                (long long) n_total_gen, n_total_gen / infer_s);
        fprintf(stderr, "pipeline-brick perf: measured gen tokens %lld / target %lld\n",
                (long long) n_gen_read, (long long) n_total_gen);
        fprintf(stderr, "pipeline-brick perf: total tokens %lld, speed %.2f t/s\n",
                (long long) (n_total_prompt + n_total_gen), (n_total_prompt + n_total_gen) / infer_s);
        fprintf(stderr,
                "pipeline-brick phase: prefill time %.3f s, prompt tokens %lld, speed %.2f t/s, ttft %.2f ms\n",
                prefill_s, (long long) n_total_prompt,
                prefill_s > 0.0 ? n_total_prompt / prefill_s : 0.0,
                prefill_s * 1000.0);
        fprintf(stderr,
                "pipeline-brick phase: decode time %.3f s, decode tokens %lld, speed %.2f t/s, steps %lld, tpot %.3f ms/step\n",
                decode_s, (long long) n_decode_tokens,
                decode_s > 0.0 ? n_decode_tokens / decode_s : 0.0,
                (long long) n_decode_steps,
                n_decode_steps > 0 ? decode_s * 1000.0 / n_decode_steps : 0.0);
    }
    print_pipeline_phase_profile(args, "prefill", prefill_start, decode_start);
    print_pipeline_phase_profile(args, "decode", decode_start, phase_end);
    print_pipeline_profile(args, infer_s);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

template<typename Transport>
static int run_head(const pipeline_args & args, Transport & transport) {
    return args.async_pipeline ? run_head_async(args, transport) : run_head_sync(args, transport);
}

template<typename Transport>
static int run_tail(const pipeline_args & args, Transport & transport) {
    llama_backend_init();
    bind_to_cpus(args.numa_cpus);
    init_ggml_numa_for_binding(args);

    llama_model * model = load_brick_model(args);
    if (!model) {
        fprintf(stderr, "pipeline-brick tail: failed to load model\n");
        return 2;
    }

    llama_context * ctx = make_context(model, args, false);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd != effective_n_embd(args)) {
        fprintf(stderr, "pipeline-brick tail: expected n_embd=%d, got %d (use --n-embd to override)\n", effective_n_embd(args), n_embd);
        return 2;
    }

    fprintf(stderr,
            "pipeline-brick tail: brick=%d peer=%d layers [%d,%d), parallel=%d, n_embd=%d, numa_tp=%d\n",
            args.brick_id, args.peer_brick_id, args.layer_start, args.layer_end, args.parallel, n_embd, args.numa_tp);
    fprintf(stderr,
            "pipeline-brick tail: micro-batch max=%d prefill_chunk=%d hidden_dtype=%s bytes_per_token=%zu\n",
            max_micro_batch_tokens(args), args.prefill_chunk, hidden_dtype_name(args.hidden_type),
            (size_t) n_embd * hidden_dtype_size(args.hidden_type));
    if (args.stream_kv) {
        fprintf(stderr,
                "pipeline-brick tail: stream-kv enabled sink=%d recent=%d\n",
                args.stream_kv_sink, args.stream_kv_recent);
    }

    llama_batch batch = llama_batch_init(max_micro_batch_tokens(args), n_embd, 1);
    std::vector<std::string> outputs(args.parallel);
    const int64_t n_target_gen = (int64_t) args.n_predict * args.parallel;
    int64_t n_gen_sent = 0;

    if (args.stream_kv) {
        llama_set_stream_kv_active(ctx, false);
    }

    // naive-kv-cross: pre-allocate a peer-pinned buffer sized for the worst-case
    // KV cache of this brick (all layers, full ctx). Read each decode step to
    // simulate a naive scenario where this brick's KV cache is on the peer CPU.
    uint64_t naive_kv_width = 0;
    if (args.naive_kv_cross) {
        const int64_t n_kv_layers = (int64_t) args.layer_end - args.layer_start;
        for (int32_t il = args.layer_start; il < args.layer_end; ++il) {
            const int32_t n_k = llama_model_n_embd_k_gqa(model, il);
            const int32_t n_v = llama_model_n_embd_v_gqa(model, il);
            if (n_k <= 0 || n_v <= 0 || n_k % args.tp_size != 0 || n_v % args.tp_size != 0) {
                fprintf(stderr,
                        "pipeline-brick tail: invalid TP-local KV geometry at layer %d: k=%d v=%d tp_size=%d\n",
                        il, n_k, n_v, args.tp_size);
                return 2;
            }
            naive_kv_width += (uint64_t) (n_k + n_v) / (uint64_t) args.tp_size;
        }
        const int32_t n_kv_embd = llama_model_n_embd_k_gqa(model, args.layer_start) / args.tp_size;
        const size_t elem_size = 2; // F16
        const size_t max_kv_bytes = (size_t) naive_kv_width
                * (size_t) args.ctx_size * (size_t) args.parallel * elem_size;
        const int kv_node = args.naive_kv_cross_node >= 0 ? args.naive_kv_cross_node : 0;
        fprintf(stderr,
                "pipeline-brick tail: naive-kv-cross enabled kv_layers=%lld kv_k_embd/rank=%d ctx=%d parallel=%d node=%d max_kv=%.1f MiB\n",
                (long long) n_kv_layers, n_kv_embd, args.ctx_size, args.parallel,
                kv_node, (double) max_kv_bytes / (1024.0 * 1024.0));
        init_naive_kv_buf(kv_node, max_kv_bytes);
        if (!g_naive_kv_buf) {
            fprintf(stderr, "pipeline-brick tail: naive-kv-cross buffer allocation failed\n");
            return 2;
        }
    }

    const int64_t t_infer_start = ggml_time_us();
    int64_t first_recv_us = 0;
    int64_t final_stop_recv_us = 0;
    bool first_packet = true;
    while (true) {
        int64_t rw0 = ggml_time_us();
        recv_packet packet = transport.recv();
        const int64_t waited = ggml_time_us() - rw0;
        const brick_packet_header & header = packet.header;
        // Exclude the first packet's startup wait and the final STOP wait from
        // pipeline-bubble accounting.
        if (first_packet) {
            first_recv_us = waited;
        } else if (header.flags & PIPELINE_FLAG_STOP) {
            final_stop_recv_us = waited;
        } else {
            g_pipe_prof.wait_recv_us += waited;
            g_pipe_prof.wait_recv_calls++;
        }
        first_packet = false;
        if (header.flags & PIPELINE_FLAG_STOP) {
            break;
        }
        if (header.flags & PIPELINE_FLAG_TRAFFIC_ONLY) {
            continue;
        }
        const bool packet_f16 = (header.flags & PIPELINE_FLAG_HIDDEN_F16) != 0;
        if (packet_f16 != (args.hidden_type == hidden_dtype::f16)) {
            fprintf(stderr, "pipeline-brick tail: hidden dtype mismatch\n");
            return 3;
        }

        std::vector<hidden_token_meta> metas;
        std::vector<float> hidden_f32;
        if (!decode_hidden_payload(packet, args.hidden_type, n_embd, metas, hidden_f32)) {
            fprintf(stderr, "pipeline-brick tail: invalid hidden payload\n");
            return 3;
        }

        if (!args.quiet) {
            fprintf(stderr, "pipeline-brick tail: recv pos=%d n_tokens=%d bytes=%llu dtype=%s\n",
                    header.pos, header.n_tokens, (unsigned long long) header.payload_bytes,
                    hidden_dtype_name(args.hidden_type));
        }

        batch.n_tokens = header.n_tokens;
        memcpy(batch.embd, hidden_f32.data(), hidden_f32.size() * sizeof(float));
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const hidden_token_meta & meta = metas[i];
            if (meta.seq_id < 0 || meta.seq_id >= args.parallel) {
                fprintf(stderr, "pipeline-brick tail: invalid seq_id %d\n", meta.seq_id);
                return 3;
            }
            set_batch_entry(batch, i, meta.seq_id, meta.pos, (meta.flags & PIPELINE_FLAG_WANT_LOGITS) != 0);
        }

        if (args.stream_kv) {
            const bool is_decode_packet = (header.flags & PIPELINE_FLAG_DECODE) != 0;
            if (is_decode_packet) {
                const bool sparse_active = header.pos + 1 > args.stream_kv_sink + args.stream_kv_recent;
                llama_set_stream_kv_active(ctx, sparse_active);
                if (args.verbose) {
                    fprintf(stderr, "pipeline-brick tail: stream-kv decode pos=%d sparse=%d\n", header.pos, sparse_active ? 1 : 0);
                }
            } else {
                llama_set_stream_kv_active(ctx, false);
            }
        }

        // naive-kv-cross: simulate reading this brick's KV cache from the peer CPU
        // before attention. KV size grows with sequence length (header.pos+1).
        if (args.naive_kv_cross && g_naive_kv_buf && (header.flags & PIPELINE_FLAG_DECODE)) {
            simulate_kv_cxl_read(naive_kv_width, header.pos + 1, header.n_tokens);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "pipeline-brick tail: llama_decode failed at pos=%d n_tokens=%d\n", header.pos, batch.n_tokens);
            return 3;
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const hidden_token_meta & meta = metas[i];
            if (meta.flags & PIPELINE_FLAG_WANT_LOGITS) {
                llama_token token = greedy_sample(ctx, vocab, i);
                transport.send_token(meta.seq_id, token);
                ++n_gen_sent;
                const std::string piece = common_token_to_piece(vocab, token, false);
                outputs[meta.seq_id] += piece;
            }
        }
    }

    const int64_t t_infer_end = ggml_time_us();
    const double infer_s =
        (t_infer_end - t_infer_start - first_recv_us - final_stop_recv_us) / 1e6;
    if (args.tp_size <= 1 || args.tp_rank == 0) {
        for (int32_t seq_id = 0; seq_id < args.parallel; ++seq_id) {
            printf("[seq %d] %s\n", seq_id, outputs[seq_id].c_str());
        }
        fflush(stdout);
        fprintf(stderr, "pipeline-brick tail: measured sent gen tokens %lld / target %lld\n",
                (long long) n_gen_sent, (long long) n_target_gen);
    }
    print_pipeline_profile(args, infer_s);
    if (args.stream_kv) {
        llama_set_stream_kv_active(ctx, false);
    }
    if (g_naive_kv_buf) {
        munmap(g_naive_kv_buf, g_naive_kv_buf_size);
        g_naive_kv_buf = nullptr;
        g_naive_kv_buf_size = 0;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

template<typename Upstream, typename Downstream>
static int run_middle_stage(const pipeline_args & args, Upstream & upstream, Downstream & downstream) {
    llama_backend_init();
    bind_to_cpus(args.numa_cpus);
    init_ggml_numa_for_binding(args);

    llama_model * model = load_brick_model(args);
    if (!model) {
        fprintf(stderr, "pipeline-brick stage%d: failed to load model\n", args.stage_id);
        return 2;
    }

    llama_context * ctx = make_context(model, args, true);
    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd != effective_n_embd(args)) {
        fprintf(stderr, "pipeline-brick stage%d: expected n_embd=%d, got %d (use --n-embd to override)\n", args.stage_id, effective_n_embd(args), n_embd);
        return 2;
    }

    fprintf(stderr,
            "pipeline-brick stage%d: brick=%d layers [%d,%d), parallel=%d, n_embd=%d, up/down transport active\n",
            args.stage_id, args.brick_id, args.layer_start, args.layer_end, args.parallel, n_embd);
    fprintf(stderr,
            "pipeline-brick stage%d: micro-batch max=%d hidden_dtype=%s bytes_per_token=%zu\n",
            args.stage_id, max_micro_batch_tokens(args), hidden_dtype_name(args.hidden_type),
            (size_t) n_embd * hidden_dtype_size(args.hidden_type));
    if (args.stream_kv) {
        fprintf(stderr,
                "pipeline-brick stage%d: stream-kv enabled sink=%d recent=%d\n",
                args.stage_id, args.stream_kv_sink, args.stream_kv_recent);
        llama_set_stream_kv_active(ctx, false);
    }

    llama_batch batch = llama_batch_init(max_micro_batch_tokens(args), n_embd, 1);

    while (true) {
        recv_packet packet = upstream.recv();
        const brick_packet_header & header = packet.header;
        if (header.flags & PIPELINE_FLAG_STOP) {
            downstream.send_stop(header.seq_id, header.pos);
            break;
        }
        if (header.flags & PIPELINE_FLAG_TRAFFIC_ONLY) {
            downstream.send_hidden_payload(
                    header.pos, header.n_tokens, header.n_embd, header.flags,
                    packet.payload.data(), packet.payload.size());
            continue;
        }

        const bool packet_f16 = (header.flags & PIPELINE_FLAG_HIDDEN_F16) != 0;
        if (packet_f16 != (args.hidden_type == hidden_dtype::f16)) {
            fprintf(stderr, "pipeline-brick stage%d: hidden dtype mismatch\n", args.stage_id);
            return 3;
        }

        std::vector<hidden_token_meta> metas;
        std::vector<float> hidden_f32;
        if (!decode_hidden_payload(packet, args.hidden_type, n_embd, metas, hidden_f32)) {
            fprintf(stderr, "pipeline-brick stage%d: invalid hidden payload\n", args.stage_id);
            return 3;
        }

        batch.n_tokens = header.n_tokens;
        memcpy(batch.embd, hidden_f32.data(), hidden_f32.size() * sizeof(float));
        int32_t want_tokens = 0;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const hidden_token_meta & meta = metas[i];
            if (meta.seq_id < 0 || meta.seq_id >= args.parallel) {
                fprintf(stderr, "pipeline-brick stage%d: invalid seq_id %d\n", args.stage_id, meta.seq_id);
                return 3;
            }
            const bool want_output = (meta.flags & PIPELINE_FLAG_WANT_LOGITS) != 0;
            set_batch_entry(batch, i, meta.seq_id, meta.pos, want_output);
            if (want_output) {
                ++want_tokens;
            }
        }

        if (args.stream_kv) {
            const bool is_decode_packet = (header.flags & PIPELINE_FLAG_DECODE) != 0;
            if (is_decode_packet) {
                const bool sparse_active = header.pos + 1 > args.stream_kv_sink + args.stream_kv_recent;
                llama_set_stream_kv_active(ctx, sparse_active);
                if (args.verbose) {
                    fprintf(stderr, "pipeline-brick stage%d: stream-kv decode pos=%d sparse=%d\n",
                            args.stage_id, header.pos, sparse_active ? 1 : 0);
                }
            } else {
                llama_set_stream_kv_active(ctx, false);
            }
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "pipeline-brick stage%d: llama_decode failed at pos=%d n_tokens=%d\n",
                    args.stage_id, header.pos, batch.n_tokens);
            return 3;
        }

        std::vector<uint8_t> payload;
        try {
            payload = make_hidden_payload(ctx, metas, n_embd, args.hidden_type);
        } catch (const std::exception & e) {
            fprintf(stderr, "pipeline-brick stage%d: %s\n", args.stage_id, e.what());
            return 3;
        }

        downstream.send_hidden_payload(header.pos, batch.n_tokens, n_embd, header.flags, payload.data(), payload.size());
        for (int32_t i = 0; i < want_tokens; ++i) {
            recv_packet token = downstream.recv();
            if (!(token.header.flags & PIPELINE_FLAG_TOKEN) || token.payload.size() != sizeof(token_payload)) {
                fprintf(stderr, "pipeline-brick stage%d: expected downstream token packet\n", args.stage_id);
                return 3;
            }
            token_payload payload_token;
            memcpy(&payload_token, token.payload.data(), sizeof(payload_token));
            upstream.send_token(payload_token.seq_id, (llama_token) payload_token.token);
        }
    }

    if (args.stream_kv) {
        llama_set_stream_kv_active(ctx, false);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

static int run_self_test_ntb(const pipeline_args & args) {
    ntb_mw_transport transport = ntb_mw_transport::open_transport(args, 4 * sizeof(float));

    if (args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD) {
        const float payload[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        transport.send_hidden(0, 0, 4, PIPELINE_FLAG_WANT_LOGITS, payload);
        recv_packet token = transport.recv();
        if (!(token.header.flags & PIPELINE_FLAG_TOKEN) || token.payload.size() != sizeof(token_payload)) {
            fprintf(stderr, "pipeline-brick self-test: head did not receive token response\n");
            return 1;
        }
        fprintf(stderr, "pipeline-brick self-test: head sent hidden payload and received token response\n");
    } else {
        recv_packet packet = transport.recv();
        if (packet.header.n_embd != 4 || packet.payload.size() != 4 * sizeof(float)) {
            fprintf(stderr, "pipeline-brick self-test: tail received invalid hidden packet\n");
            return 1;
        }
        transport.send_token(packet.header.seq_id, 42);
        fprintf(stderr, "pipeline-brick self-test: tail received hidden payload and sent token response\n");
    }
    return 0;
}

static pipeline_args make_link_args(const pipeline_args & args, bool downstream) {
    pipeline_args link = args;
    if (downstream) {
        if (!args.down_tx_mw.empty()) {
            link.tx_mw = args.down_tx_mw;
        }
        if (!args.down_rx_mw.empty()) {
            link.rx_mw = args.down_rx_mw;
        }
        link.transport = args.down_transport;
    } else {
        if (!args.up_tx_mw.empty()) {
            link.tx_mw = args.up_tx_mw;
        }
        if (!args.up_rx_mw.empty()) {
            link.rx_mw = args.up_rx_mw;
        }
        link.transport = args.up_transport;
    }
    return link;
}

static std::unique_ptr<transport_box> open_transport_box(const pipeline_args & args, transport_kind kind, size_t max_payload_bytes, bool downstream) {
    pipeline_args link = make_link_args(args, downstream);
    link.transport = kind;
    switch (kind) {
        case transport_kind::ntb_mw:
            return std::make_unique<transport_box>(ntb_mw_transport::open_transport(link, max_payload_bytes));
        case transport_kind::cxl:
            return std::make_unique<transport_box>(cxl_transport::open_transport(link, max_payload_bytes));
        case transport_kind::ib_rdma:
            return std::make_unique<transport_box>(ib_rdma_transport::open_transport(args, max_payload_bytes, downstream));
        case transport_kind::zni_rdma:
            return std::make_unique<transport_box>(zni_transport::open_transport(args, max_payload_bytes, downstream));
    }
    throw std::runtime_error("invalid transport kind");
}

struct tp_broadcast_shm {
    volatile int sequence;
    volatile int ack_count;
    uint32_t payload_size;
    uint32_t reserved;
    brick_packet_header header;
};

template<typename Transport>
struct tp_transport_proxy {
    Transport * real;
    int tp_rank;
    int tp_size;
    tp_broadcast_shm * shm;
    size_t shm_size;
    int seen_sequence = 0;

    void send_hidden(int32_t seq_id, int32_t pos, int32_t n_embd, uint32_t flags, const float * payload) {
        if (tp_rank == 0) {
            real->send_hidden(seq_id, pos, n_embd, flags, payload);
        }
    }

    void send_hidden_payload(int32_t pos, int32_t n_tokens, int32_t n_embd, uint32_t flags, const void * payload, uint64_t payload_bytes) {
        if (tp_rank == 0) {
            real->send_hidden_payload(pos, n_tokens, n_embd, flags, payload, payload_bytes);
        }
    }

    void send_token(int32_t seq_id, llama_token token) {
        if (tp_rank == 0) {
            real->send_token(seq_id, token);
        }
    }

    void send_stop(int32_t seq_id, int32_t pos) {
        if (tp_rank == 0) {
            real->send_stop(seq_id, pos);
        }
    }

    recv_packet recv() {
        uint8_t * payload_base = (uint8_t *) shm + align_up(sizeof(tp_broadcast_shm), 64);
        const size_t payload_cap = shm_size - align_up(sizeof(tp_broadcast_shm), 64);

        if (tp_rank == 0) {
            if (seen_sequence > 0) {
                while (__atomic_load_n(&shm->ack_count, __ATOMIC_ACQUIRE) < tp_size - 1) {
#if defined(__aarch64__)
                    __asm__ volatile("yield" ::: "memory");
#endif
                }
            }

            recv_packet pkt = real->recv();
            if (pkt.payload.size() > payload_cap) {
                throw std::runtime_error("TP broadcast payload exceeds shared buffer");
            }

            shm->header = pkt.header;
            shm->payload_size = (uint32_t) pkt.payload.size();
            if (!pkt.payload.empty()) {
                memcpy(payload_base, pkt.payload.data(), pkt.payload.size());
            }

            __atomic_store_n(&shm->ack_count, 0, __ATOMIC_RELEASE);
            ++seen_sequence;
            __atomic_store_n(&shm->sequence, seen_sequence, __ATOMIC_RELEASE);
            return pkt;
        }

        while (__atomic_load_n(&shm->sequence, __ATOMIC_ACQUIRE) == seen_sequence) {
#if defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#endif
        }

        seen_sequence = __atomic_load_n(&shm->sequence, __ATOMIC_ACQUIRE);

        recv_packet pkt;
        pkt.header = shm->header;
        const uint32_t payload_size = shm->payload_size;
        if (payload_size > payload_cap) {
            throw std::runtime_error("TP broadcast payload exceeds shared buffer");
        }
        if (payload_size > 0) {
            pkt.payload.resize(payload_size);
            memcpy(pkt.payload.data(), payload_base, payload_size);
        }

        __sync_fetch_and_add(&shm->ack_count, 1);
        return pkt;
    }
};

static std::string make_tp_model_path(const std::string & base_path, int tp_rank) {
    const size_t dot = base_path.rfind('.');
    if (dot == std::string::npos) {
        return base_path + ".tp" + std::to_string(tp_rank);
    }
    return base_path.substr(0, dot) + ".tp" + std::to_string(tp_rank) + base_path.substr(dot);
}

static std::string tp_rank_numa_node(const pipeline_args & args, int tp_rank) {
    std::string spec;
    if (stage_mode_enabled(args)) {
        std::vector<std::string> stage_specs = split_stage_specs(args.stage_numa);
        if (args.stage_id < 0 || args.stage_id >= (int) stage_specs.size()) {
            throw std::runtime_error("--stage-numa must contain a NUMA spec for this --stage-id");
        }
        spec = stage_specs[args.stage_id];
    } else {
        spec = args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD ? args.head_numa : args.tail_numa;
    }
    if (spec.empty()) {
        throw std::runtime_error("NUMA node spec is required when --tp-size > 1");
    }

    std::vector<int> nodes = parse_cpu_list_ordered(spec);
    if ((int) nodes.size() < args.tp_size) {
        throw std::runtime_error("NUMA node list does not contain enough nodes for --tp-size");
    }

    if (args.tp_size == 2 && nodes.size() == 4) {
        const size_t begin = (size_t) tp_rank * 2;
        return join_cpu_list({ nodes[begin], nodes[begin + 1] });
    }

    return std::to_string(nodes[tp_rank]);
}

static int tp_rank_primary_numa_node(const pipeline_args & args, int tp_rank) {
    const std::vector<int> nodes = parse_cpu_list_ordered(tp_rank_numa_node(args, tp_rank));
    if (nodes.empty()) {
        throw std::runtime_error("TP rank NUMA node list is empty");
    }
    return nodes.front();
}

static size_t tp_broadcast_size(const pipeline_args & args) {
    return align_up(sizeof(tp_broadcast_shm), 64) + max_hidden_payload_bytes(args) + 4096;
}

static size_t tp_all_reduce_size(const pipeline_args & args) {
    const size_t max_elements = (size_t) effective_n_embd(args) * (size_t) max_micro_batch_tokens(args);
    return align_up(sizeof(uint64_t) * 8, 64) + max_elements * sizeof(float) * (size_t) args.tp_size + 65536;
}

static int run_hardware_tp(const pipeline_args & args) {
    const int tp_size = args.tp_size;
    const size_t bc_size = tp_broadcast_size(args);
    const size_t ar_size = tp_all_reduce_size(args);
    const size_t shm_size = align_up(bc_size, 64) + ar_size;
    const size_t max_payload_bytes = max_hidden_payload_bytes(args);

    const int shm_node = tp_rank_primary_numa_node(args, 0);
    void * shm = alloc_on_node(shm_size, shm_node);
    if (!shm) {
        throw std::runtime_error("TP shared allocation failed on NUMA node " + std::to_string(shm_node));
    }

    auto * bc_shm = (tp_broadcast_shm *) shm;
    void * ar_shm = (uint8_t *) shm + align_up(bc_size, 64);

    fprintf(stderr, "pipeline-brick TP: role=%s tp_size=%d shm_node=%d broadcast=%zu all_reduce=%zu\n",
            role_name(args.role).c_str(), tp_size, shm_node, bc_size, ar_size);

    std::vector<pid_t> children;
    for (int r = 0; r < tp_size; ++r) {
        const pid_t pid = fork();
        if (pid < 0) {
            munmap(shm, shm_size);
            throw std::runtime_error("TP fork failed: " + std::string(strerror(errno)));
        }

        if (pid == 0) {
            pipeline_args child_args = args;
            child_args.tp_rank = r;
            child_args.tp_size = tp_size;
            child_args.numa_tp = 1;
            child_args.threads = std::max(1, args.threads / tp_size);
            child_args.model = make_tp_model_path(args.model, r);
            child_args.numa_cpus = cpus_for_numa_nodes(tp_rank_numa_node(args, r));

            ggml_tp_shm_init(r, tp_size, ar_shm, ar_size);

            fprintf(stderr,
                    "pipeline-brick TP: role=%s rank=%d/%d threads=%d cpus=%s model=%s\n",
                    role_name(args.role).c_str(), r, tp_size, child_args.threads,
                    child_args.numa_cpus.c_str(), child_args.model.c_str());

            try {
                if (args.transport == transport_kind::cxl) {
                    std::unique_ptr<cxl_transport> real;
                    if (r == 0) {
                        real = std::make_unique<cxl_transport>(cxl_transport::open_transport(args, max_payload_bytes));
                    }
                    tp_transport_proxy<cxl_transport> transport{ real.get(), r, tp_size, bc_shm, bc_size };
                    const int rc = args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD ?
                        run_head(child_args, transport) : run_tail(child_args, transport);
                    exit_tp_child(child_args, rc);
                }

                std::unique_ptr<ntb_mw_transport> real;
                if (r == 0) {
                    real = std::make_unique<ntb_mw_transport>(ntb_mw_transport::open_transport(args, max_payload_bytes));
                }
                tp_transport_proxy<ntb_mw_transport> transport{ real.get(), r, tp_size, bc_shm, bc_size };
                const int rc = args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD ?
                    run_head(child_args, transport) : run_tail(child_args, transport);
                exit_tp_child(child_args, rc);
            } catch (const std::exception & e) {
                fprintf(stderr, "pipeline-brick TP rank %d: %s\n", r, e.what());
                _exit(1);
            }
        }

        children.push_back(pid);
    }

    int exit_code = 0;
    for (pid_t pid : children) {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            exit_code = 1;
            continue;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            exit_code = 1;
        }
    }

    munmap(shm, shm_size);
    return exit_code;
}

static bool stage_mode_enabled(const pipeline_args & args) {
    return args.stage_count > 0 || args.stage_id >= 0 || args.role == LLAMA_PIPELINE_BRICK_ROLE_STAGE;
}

static pipeline_args normalize_stage_args(const pipeline_args & args) {
    pipeline_args out = args;
    if (out.stage_count == 0) {
        out.stage_count = 4;
    }
    if (out.stage_count != 4) {
        throw std::runtime_error("pipeline-brick stage mode currently supports --stage-count 4");
    }
    if (!out.stage_numa.empty() && (int) split_stage_specs(out.stage_numa).size() != out.stage_count) {
        throw std::runtime_error("--stage-numa must contain one semicolon-separated NUMA spec per stage");
    }
    if (out.stage_id < 0 || out.stage_id >= out.stage_count) {
        throw std::runtime_error("--stage-id must be in [0, stage-count)");
    }
    if (out.layer_start < 0 && out.layer_end < 0) {
        const int n_layer = effective_n_layer(args);
        const int layers_per_stage = n_layer / out.stage_count;
        out.layer_start = out.stage_id * layers_per_stage;
        out.layer_end = out.stage_id == out.stage_count - 1 ? n_layer : out.layer_start + layers_per_stage;
    }
    if (out.role == LLAMA_PIPELINE_BRICK_ROLE_NONE) {
        if (out.stage_id == 0) {
            out.role = LLAMA_PIPELINE_BRICK_ROLE_HEAD;
        } else if (out.stage_id == out.stage_count - 1) {
            out.role = LLAMA_PIPELINE_BRICK_ROLE_TAIL;
        } else {
            out.role = LLAMA_PIPELINE_BRICK_ROLE_STAGE;
        }
    }
    if (out.brick_id < 0) {
        out.brick_id = out.stage_id;
    }
    if (out.peer_brick_id < 0) {
        out.peer_brick_id = out.stage_id < out.stage_count - 1 ? out.stage_id + 1 : out.stage_id - 1;
    }
    return out;
}

static int run_hardware_stage_tp(const pipeline_args & raw_args) {
    pipeline_args args = normalize_stage_args(raw_args);
    const int tp_size = args.tp_size;
    const size_t bc_size = tp_broadcast_size(args);
    const size_t ar_size = tp_all_reduce_size(args);
    const size_t up_bc_offset = 0;
    const size_t down_bc_offset = align_up(bc_size, 64);
    const size_t ar_offset = down_bc_offset + align_up(bc_size, 64);
    const size_t shm_size = ar_offset + ar_size;
    const size_t max_payload_bytes = max_hidden_payload_bytes(args);

    const int shm_node = tp_rank_primary_numa_node(args, 0);
    void * shm = alloc_on_node(shm_size, shm_node);
    if (!shm) {
        throw std::runtime_error(
                "stage TP shared allocation failed on NUMA node " + std::to_string(shm_node));
    }

    auto * up_bc_shm = (tp_broadcast_shm *) ((uint8_t *) shm + up_bc_offset);
    auto * down_bc_shm = (tp_broadcast_shm *) ((uint8_t *) shm + down_bc_offset);
    void * ar_shm = (uint8_t *) shm + ar_offset;

    fprintf(stderr,
            "pipeline-brick stage%d TP: layers [%d,%d) role=%s tp_size=%d shm_node=%d broadcast=%zu all_reduce=%zu\n",
            args.stage_id, args.layer_start, args.layer_end, role_name(args.role).c_str(),
            tp_size, shm_node, bc_size, ar_size);

    std::vector<pid_t> children;
    for (int r = 0; r < tp_size; ++r) {
        const pid_t pid = fork();
        if (pid < 0) {
            munmap(shm, shm_size);
            throw std::runtime_error("stage TP fork failed: " + std::string(strerror(errno)));
        }

        if (pid == 0) {
            pipeline_args child_args = args;
            child_args.tp_rank = r;
            child_args.tp_size = tp_size;
            child_args.numa_tp = 1;
            child_args.threads = std::max(1, args.threads / tp_size);
            child_args.model = make_tp_model_path(args.model, r);
            child_args.numa_cpus = cpus_for_numa_nodes(tp_rank_numa_node(args, r));

            ggml_tp_shm_init(r, tp_size, ar_shm, ar_size);

            fprintf(stderr,
                    "pipeline-brick stage%d TP: rank=%d/%d threads=%d cpus=%s model=%s\n",
                    args.stage_id, r, tp_size, child_args.threads,
                    child_args.numa_cpus.c_str(), child_args.model.c_str());

            try {
                if (args.stage_id == 0) {
                    std::unique_ptr<transport_box> down_real;
                    if (r == 0) {
                        down_real = open_transport_box(child_args, child_args.down_transport, max_payload_bytes, true);
                    }
                    tp_transport_proxy<transport_box> down{ down_real.get(), r, tp_size, down_bc_shm, bc_size };
                    const int rc = run_head(child_args, down);
                    exit_tp_child(child_args, rc);
                }

                if (args.stage_id == args.stage_count - 1) {
                    std::unique_ptr<transport_box> up_real;
                    if (r == 0) {
                        up_real = open_transport_box(child_args, child_args.up_transport, max_payload_bytes, false);
                    }
                    tp_transport_proxy<transport_box> up{ up_real.get(), r, tp_size, up_bc_shm, bc_size };
                    const int rc = run_tail(child_args, up);
                    exit_tp_child(child_args, rc);
                }

                std::unique_ptr<transport_box> up_real;
                std::unique_ptr<transport_box> down_real;
                if (r == 0) {
                    up_real = open_transport_box(child_args, child_args.up_transport, max_payload_bytes, false);
                    down_real = open_transport_box(child_args, child_args.down_transport, max_payload_bytes, true);
                }
                tp_transport_proxy<transport_box> up{ up_real.get(), r, tp_size, up_bc_shm, bc_size };
                tp_transport_proxy<transport_box> down{ down_real.get(), r, tp_size, down_bc_shm, bc_size };
                const int rc = run_middle_stage(child_args, up, down);
                exit_tp_child(child_args, rc);
            } catch (const std::exception & e) {
                fprintf(stderr, "pipeline-brick stage%d TP rank %d: %s\n", args.stage_id, r, e.what());
                _exit(1);
            }
        }

        children.push_back(pid);
    }

    int exit_code = 0;
    for (pid_t pid : children) {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "pipeline-brick stage%d TP: child %d exited abnormally\n",
                    args.stage_id, (int) pid);
            exit_code = 1;
        }
    }

    munmap(shm, shm_size);
    return exit_code;
}

static int run_hardware_stage(const pipeline_args & raw_args) {
    pipeline_args args = normalize_stage_args(raw_args);
    if (args.tp_size > 1) {
        return run_hardware_stage_tp(args);
    }
    const size_t max_payload_bytes = max_hidden_payload_bytes(args);

    if (args.stage_id == 0) {
        auto down = open_transport_box(args, args.down_transport, max_payload_bytes, true);
        return run_head(args, *down);
    }
    if (args.stage_id == args.stage_count - 1) {
        auto up = open_transport_box(args, args.up_transport, max_payload_bytes, false);
        return run_tail(args, *up);
    }

    auto up = open_transport_box(args, args.up_transport, max_payload_bytes, false);
    auto down = open_transport_box(args, args.down_transport, max_payload_bytes, true);
    return run_middle_stage(args, *up, *down);
}

static int run_hardware(const pipeline_args & args) {
    const size_t max_payload_bytes = max_hidden_payload_bytes(args);

    if (stage_mode_enabled(args)) {
        return run_hardware_stage(args);
    }

    if (args.tp_size > 1) {
        return run_hardware_tp(args);
    }

    // ponytail: numa_tp is a NUMA-span info label only (logged in head/tail);
    // NUMA-aware TP is provided by --tp-size. No single-process tensor-level split.

    if (args.transport == transport_kind::cxl) {
        cxl_transport transport = cxl_transport::open_transport(args, max_payload_bytes);
        if (args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD) {
            return run_head(args, transport);
        }
        if (args.role == LLAMA_PIPELINE_BRICK_ROLE_TAIL) {
            return run_tail(args, transport);
        }
        throw std::runtime_error("invalid role");
    }

    ntb_mw_transport transport = ntb_mw_transport::open_transport(args, max_payload_bytes);

    if (args.role == LLAMA_PIPELINE_BRICK_ROLE_HEAD) {
        return run_head(args, transport);
    }
    if (args.role == LLAMA_PIPELINE_BRICK_ROLE_TAIL) {
        return run_tail(args, transport);
    }
    throw std::runtime_error("invalid role");
}

static void create_shared_window_file(const std::string & path, size_t size) {
    fd_handle fd(open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600));
    if (!fd.valid()) {
        throw std::runtime_error("failed to create shared window " + path + ": " + strerror(errno));
    }
    if (ftruncate(fd.get(), (off_t) size) != 0) {
        throw std::runtime_error("failed to size shared window " + path + ": " + strerror(errno));
    }
}

static pipeline_args make_single_system_child_args(
        const pipeline_args & args,
        llama_pipeline_brick_role role,
        const std::string & h2t_path,
        const std::string & t2h_path) {
    pipeline_args child = args;
    child.single_system = false;
    child.hardware = true;
    child.db_mode = doorbell_mode::poll;
    child.role = role;
    child.numa_tp = 4;
    child.tx_doorbell.clear();
    child.rx_doorbell.clear();

    if (role == LLAMA_PIPELINE_BRICK_ROLE_HEAD) {
        child.brick_id = 0;
        child.peer_brick_id = 1;
        child.layer_start = 0;
        child.layer_end = effective_split_layer(args);
        child.numa_cpus = cpus_for_numa_nodes(args.head_numa);
        child.tx_mw = h2t_path;
        child.rx_mw = t2h_path;
    } else {
        child.brick_id = 1;
        child.peer_brick_id = 0;
        child.layer_start = effective_split_layer(args);
        child.layer_end = effective_n_layer(args);
        child.numa_cpus = cpus_for_numa_nodes(args.tail_numa);
        child.tx_mw = t2h_path;
        child.rx_mw = h2t_path;
    }

    return child;
}

static int run_single_system_stages(const pipeline_args & args) {
    if (args.stage_count != 4) {
        throw std::runtime_error("single-system stage mode requires --stage-count 4");
    }
    std::vector<std::string> numa_specs = split_stage_specs(args.stage_numa);
    if ((int) numa_specs.size() != args.stage_count) {
        throw std::runtime_error("--stage-numa must contain four semicolon-separated NUMA specs");
    }

    const size_t max_payload_bytes = max_hidden_payload_bytes(args);
    const size_t window_size = transport_window_size(max_payload_bytes);
    const std::string base = "/dev/shm/llama-pipeline-brick-stage-" + std::to_string(getpid());

    std::vector<std::string> fwd(args.stage_count - 1);
    std::vector<std::string> rev(args.stage_count - 1);
    for (int i = 0; i < args.stage_count - 1; ++i) {
        fwd[i] = base + "-s" + std::to_string(i) + "-to-s" + std::to_string(i + 1);
        rev[i] = base + "-s" + std::to_string(i + 1) + "-to-s" + std::to_string(i);
        create_shared_window_file(fwd[i], window_size);
        create_shared_window_file(rev[i], window_size);
        if (args.transport == transport_kind::cxl) {
            bind_shared_window_to_numa(fwd[i], window_size, numa_specs[i + 1], "stage-forward");
            bind_shared_window_to_numa(rev[i], window_size, numa_specs[i], "stage-reverse");
        }
    }

    fprintf(stderr, "pipeline-brick single-system stages: stage_count=%d window_size=%zu\n", args.stage_count, window_size);

    std::vector<pid_t> pids;
    for (int sid = args.stage_count - 1; sid >= 0; --sid) {
        pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("fork stage failed: " + std::string(strerror(errno)));
        }
        if (pid == 0) {
            pipeline_args child = args;
            child.single_system = false;
            child.hardware = true;
            child.stage_id = sid;
            child.role = sid == 0 ? LLAMA_PIPELINE_BRICK_ROLE_HEAD :
                (sid == args.stage_count - 1 ? LLAMA_PIPELINE_BRICK_ROLE_TAIL : LLAMA_PIPELINE_BRICK_ROLE_STAGE);
            child.brick_id = sid;
            child.peer_brick_id = sid == args.stage_count - 1 ? sid - 1 : sid + 1;
            child.layer_start = -1;
            child.layer_end = -1;
            child.numa_cpus = cpus_for_numa_nodes(numa_specs[sid]);
            child.db_mode = doorbell_mode::poll;
            child.up_transport = args.transport;
            child.down_transport = args.transport;
            child.tx_doorbell.clear();
            child.rx_doorbell.clear();
            if (sid > 0) {
                child.up_rx_mw = fwd[sid - 1];
                child.up_tx_mw = rev[sid - 1];
            }
            if (sid < args.stage_count - 1) {
                child.down_tx_mw = fwd[sid];
                child.down_rx_mw = rev[sid];
            }
            try {
                const int rc = run_hardware(child);
                _exit(rc);
            } catch (const std::exception & e) {
                fprintf(stderr, "pipeline-brick stage%d: %s\n", sid, e.what());
                _exit(1);
            }
        }
        pids.push_back(pid);
        sleep_us(200000);
    }

    int exit_code = 0;
    for (pid_t pid : pids) {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            exit_code = 1;
        }
    }
    for (int i = 0; i < args.stage_count - 1; ++i) {
        unlink(fwd[i].c_str());
        unlink(rev[i].c_str());
    }
    return exit_code;
}

static int run_single_system(const pipeline_args & args) {
    if (stage_mode_enabled(args)) {
        return run_single_system_stages(args);
    }

    const size_t max_payload_bytes = max_hidden_payload_bytes(args);
    const size_t window_size = transport_window_size(max_payload_bytes);
    const std::string base = "/dev/shm/llama-pipeline-brick-" + std::to_string(getpid());
    const std::string h2t_path = base + "-h2t";
    const std::string t2h_path = base + "-t2h";

    create_shared_window_file(h2t_path, window_size);
    create_shared_window_file(t2h_path, window_size);
    if (args.transport == transport_kind::cxl) {
        std::string head_window_numa = args.head_numa;
        std::string tail_window_numa = args.tail_numa;
        if (args.tp_size > 1) {
            pipeline_args head_bind_args = args;
            pipeline_args tail_bind_args = args;
            head_bind_args.role = LLAMA_PIPELINE_BRICK_ROLE_HEAD;
            tail_bind_args.role = LLAMA_PIPELINE_BRICK_ROLE_TAIL;
            head_window_numa = tp_rank_numa_node(head_bind_args, 0);
            tail_window_numa = tp_rank_numa_node(tail_bind_args, 0);
        }
        bind_shared_window_to_numa(h2t_path, window_size, tail_window_numa, "head-to-tail");
        bind_shared_window_to_numa(t2h_path, window_size, head_window_numa, "tail-to-head");
    }

    fprintf(stderr,
            "pipeline-brick single-system: head NUMA=%s tail NUMA=%s window_size=%zu\n",
            args.head_numa.c_str(), args.tail_numa.c_str(), window_size);

    pipeline_args tail_args = make_single_system_child_args(args, LLAMA_PIPELINE_BRICK_ROLE_TAIL, h2t_path, t2h_path);
    pipeline_args head_args = make_single_system_child_args(args, LLAMA_PIPELINE_BRICK_ROLE_HEAD, h2t_path, t2h_path);

    pid_t tail_pid = fork();
    if (tail_pid < 0) {
        throw std::runtime_error("fork tail failed: " + std::string(strerror(errno)));
    }
    if (tail_pid == 0) {
        const int rc = run_hardware(tail_args);
        _exit(rc);
    }

    sleep_us(200000);
                                                            
    pid_t head_pid = fork();
    if (head_pid < 0) {
        throw std::runtime_error("fork head failed: " + std::string(strerror(errno)));
    }
    if (head_pid == 0) {
        const int rc = run_hardware(head_args);
        _exit(rc);
    }

    int status_head = 0;
    int status_tail = 0;
    waitpid(head_pid, &status_head, 0);
    waitpid(tail_pid, &status_tail, 0);

    unlink(h2t_path.c_str());
    unlink(t2h_path.c_str());

    if (!WIFEXITED(status_head) || WEXITSTATUS(status_head) != 0) {
        fprintf(stderr, "pipeline-brick single-system: head exited abnormally\n");
        return 1;
    }
    if (!WIFEXITED(status_tail) || WEXITSTATUS(status_tail) != 0) {
        fprintf(stderr, "pipeline-brick single-system: tail exited abnormally\n");
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        pipeline_args args = parse_args(argc, argv);
        apply_domain_mode_defaults(args);
        prepare_async_prompts(args);
        validate_args(args);
        llama_log_set(pipeline_brick_log_callback, &args.verbose);

        fprintf(stderr, "pipeline-brick: mode=%s role=%s brick=%d peer=%d\n",
                args.self_test_ntb ? "self-test-ntb" : "hardware",
                role_name(args.role).c_str(), args.brick_id, args.peer_brick_id);

        if (args.self_test_ntb) {
            return run_self_test_ntb(args);
        }
        if (args.single_system) {
            return run_single_system(args);
        }
        return run_hardware(args);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }
}
