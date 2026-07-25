// numa-latency-bench: microarchitecture-level memory latency measurement.
//
// Measures single-access latency for three memory paths on a NUMA-CXL platform:
//   1. local  NUMA: data on the same NUMA node as the computing core
//   2. remote NUMA: data on a different NUMA node of the same CPU (intra-CPU)
//   3. cross  CXL : data on a NUMA node belonging to the peer CPU (via CXL link)
//
// Method: pointer-chasing over a shuffled linked list. Each load depends on the
// previous one, so the CPU cannot overlap loads -> measured time per step is the
// raw memory access latency. We use mmap + mbind(MPOL_BIND) to pin pages to a
// target NUMA node, and sched_setaffinity to pin the computing thread to a core.
//
// No libnuma dependency: uses linux/mempolicy.h (SYS_mbind) and sched.h, both
// kernel-native, so it builds anywhere glibc is present.
//
// Usage:
//   llama-numa-latency-bench --local-node 0 --remote-node 1 --cxl-node 4
//       --local-cpu 0 --remote-cpu 0 --cxl-cpu 0 [--size-mb 64] [--iters 20000000]
//
// On a single-NUMA machine, only --local-node makes sense; remote/cxl are skipped.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <sched.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

struct args_t {
    int local_node  = -1;
    int remote_node = -1;
    int cxl_node    = -1;
    int local_cpu   = -1;
    int remote_cpu  = -1;
    int cxl_cpu     = -1;
    size_t size_mb  = 64;
    size_t iters    = 20000000;
    std::string mode = "both";   // latency | bandwidth | both | tp-allreduce
    int bw_threads  = 8;         // threads per path for bandwidth test
    int bw_sweeps   = 20;        // full-region sweeps for bandwidth
    std::string bw_mode = "read"; // read | write | rw (bandwidth sub-mode)
    // Rank input buffers live on tp_data_node; computing threads run on
    // tp_compute_cpu..tp_compute_cpu+tp_threads-1. The output buffer is local
    // to the compute CPU unless tp_output_node is explicitly supplied.
    int tp_ranks       = 2;      // number of TP ranks to sum
    int tp_elem_mb     = 8;      // data per rank in MiB (activation-scale)
    int tp_data_node   = 0;      // node where the shared reduce region lives
    int tp_output_node = -1;     // node for rank-local output; inferred from compute CPU
    int tp_compute_cpu = 0;      // first core for computing threads
    int tp_threads     = 8;      // compute threads
    int tp_sweeps      = 100;    // all-reduce iterations
};

static void usage(const char * prog) {
    fprintf(stderr,
        "numa-latency-bench: measure local / remote-NUMA / cross-CXL memory latency.\n"
        "usage: %s --local-node N --remote-node N --cxl-node N \\\n"
        "         --local-cpu C --remote-cpu C --cxl-cpu C [--size-mb MB] [--iters N]\n"
        "\n"
        "  --local-node   NUMA node id for the local  memory path (same node as local-cpu)\n"
        "  --remote-node  NUMA node id for the remote memory path (same CPU, other node)\n"
        "  --cxl-node     NUMA node id for the cross-CXL path (peer CPU's node)\n"
        "  --local-cpu    cpu id to run on for the local  measurement\n"
        "  --remote-cpu   compute cpu for the remote measurement; keep equal to local-cpu for path comparison\n"
        "  --cxl-cpu      compute cpu for the cross-CXL measurement; keep equal to local-cpu for path comparison\n"
        "  --size-mb      working-set size in MiB (default 64)\n"
        "  --iters        pointer-chase iterations per measurement (default 20000000)\n"
        "  --mode         latency | bandwidth | both (default both)\n"
        "  --bw-threads   threads for bandwidth test, uses cores cpu,cpu+1,... (default 8)\n"
        "  --bw-sweeps    full-region sweeps for bandwidth (default 20)\n"
        "  --bw-mode      read | write | rw (bandwidth sub-mode, default read)\n"
        "\n"
        "tp-allreduce mode (isolates the TP reduction kernel):\n"
        "  --tp-ranks       number of ranks to sum (default 2)\n"
        "  --tp-elem-mb     data per rank in MiB, activation-scale (default 8)\n"
        "  --tp-data-node   node where the shared reduce region lives (default 0)\n"
        "  --tp-output-node node for rank-local output (default: infer from compute CPU)\n"
        "  --tp-compute-cpu first core for computing threads (default 0)\n"
        "  --tp-threads     compute threads (default 8)\n"
        "  --tp-sweeps      all-reduce iterations (default 100)\n"
        "\n"
        "Each path needs both a --*-node and a --*-cpu. Paths without both are skipped.\n"
        "On a single-NUMA machine, pass only --local-node/--local-cpu.\n",
        prog);
}

static bool parse_args(int argc, char ** argv, args_t & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const std::string & k) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", k.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--local-node")       { const char *v=need(key); if(!v)return false; a.local_node=atoi(v); }
        else if (key == "--remote-node") { const char *v=need(key); if(!v)return false; a.remote_node=atoi(v); }
        else if (key == "--cxl-node")    { const char *v=need(key); if(!v)return false; a.cxl_node=atoi(v); }
        else if (key == "--local-cpu")   { const char *v=need(key); if(!v)return false; a.local_cpu=atoi(v); }
        else if (key == "--remote-cpu")  { const char *v=need(key); if(!v)return false; a.remote_cpu=atoi(v); }
        else if (key == "--cxl-cpu")     { const char *v=need(key); if(!v)return false; a.cxl_cpu=atoi(v); }
        else if (key == "--size-mb")     { const char *v=need(key); if(!v)return false; a.size_mb=(size_t)atoll(v); }
        else if (key == "--iters")       { const char *v=need(key); if(!v)return false; a.iters=(size_t)atoll(v); }
        else if (key == "--mode")        { const char *v=need(key); if(!v)return false; a.mode=v; }
        else if (key == "--bw-threads")  { const char *v=need(key); if(!v)return false; a.bw_threads=atoi(v); }
        else if (key == "--bw-sweeps")   { const char *v=need(key); if(!v)return false; a.bw_sweeps=atoi(v); }
        else if (key == "--bw-mode")     { const char *v=need(key); if(!v)return false; a.bw_mode=v; }
        else if (key == "--tp-ranks")       { const char *v=need(key); if(!v)return false; a.tp_ranks=atoi(v); }
        else if (key == "--tp-elem-mb")     { const char *v=need(key); if(!v)return false; a.tp_elem_mb=atoi(v); }
        else if (key == "--tp-data-node")   { const char *v=need(key); if(!v)return false; a.tp_data_node=atoi(v); }
        else if (key == "--tp-output-node") { const char *v=need(key); if(!v)return false; a.tp_output_node=atoi(v); }
        else if (key == "--tp-compute-cpu") { const char *v=need(key); if(!v)return false; a.tp_compute_cpu=atoi(v); }
        else if (key == "--tp-threads")     { const char *v=need(key); if(!v)return false; a.tp_threads=atoi(v); }
        else if (key == "--tp-sweeps")      { const char *v=need(key); if(!v)return false; a.tp_sweeps=atoi(v); }
        else if (key == "--help" || key == "-h") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown argument: %s\n", key.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

static bool bind_cpu(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        fprintf(stderr, "invalid CPU id: %d\n", cpu);
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity(cpu=%d) failed: %s\n", cpu, strerror(errno));
        return false;
    }
    return true;
}

static int numa_node_for_cpu(int cpu) {
    for (int node = 0; node < 1024; ++node) {
        char path[160];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/node%d", cpu, node);
        if (access(path, F_OK) == 0) {
            return node;
        }
    }
    return -1;
}

// Pin a shared anon mmap region to a single NUMA node via mbind(MPOL_BIND).
static void * alloc_on_node(size_t bytes, int node) {
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return nullptr;
    }
    if (node >= 0) {
        const int bits_per_word = (int)(sizeof(unsigned long) * 8);
        std::vector<unsigned long> mask((node / bits_per_word) + 1, 0);
        mask[node / bits_per_word] |= 1UL << (node % bits_per_word);
        const unsigned long maxnode = (unsigned long)mask.size() * bits_per_word;
        // Set the policy before first touch; the anonymous mapping has no
        // physical pages to migrate yet.
        const long rc = syscall(SYS_mbind, p, bytes, MPOL_BIND, mask.data(), maxnode, 0);
        if (rc != 0) {
            fprintf(stderr, "mbind(node=%d) failed: %s\n", node, strerror(errno));
            munmap(p, bytes);
            return nullptr;
        }
    }
    // first-touch: force physical allocation on the bound node
    memset(p, 0, bytes);
    return p;
}

// Each linked-list node occupies one full 64-byte cache line. The next index
// sits at the start; the rest is padding. This defeats the hardware prefetcher:
// accesses jump between cache lines in a shuffled order, so the CPU cannot
// predict the next address and must stall on each load -> raw latency exposed.
struct cache_line_node {
    uint32_t next;
    uint8_t  pad[60];
} __attribute__((aligned(64)));

static constexpr size_t CACHE_LINE = 64;

// Build a shuffled circular linked list of cache-line nodes.
// mem points to a size_bytes region; n_nodes = size_bytes / 64.
static void build_shuffled_chain(void * mem, size_t n_nodes, uint64_t seed) {
    auto * nodes = (cache_line_node *)mem;
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> perm(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) perm[i] = (uint32_t)i;
    for (size_t i = n_nodes; i > 1; --i) {
        size_t j = rng() % i;
        std::swap(perm[i - 1], perm[j]);
    }
    for (size_t i = 0; i < n_nodes; ++i) {
        nodes[perm[i]].next = perm[(i + 1) % n_nodes];
    }
}

// Returns nanoseconds per pointer-chase step (one cache-line load per step).
static double measure_chain(void * mem, size_t iters) {
    auto * nodes = (cache_line_node *)mem;
    uint32_t idx = 0;
    // warmup: bring the chain access pattern into a stable state
    for (size_t i = 0; i < 1024; ++i) idx = nodes[idx].next;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < iters; ++i) {
        idx = nodes[idx].next;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    // prevent compiler from optimizing the loop away
    asm volatile("" :: "r"(idx) : "memory");

    double ns = (double)((t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec));
    return ns / (double)iters;
}

// Measure bandwidth. mode: "read" = pure read, "write" = pure write, "rw" = read+write.
// Returns GB/s. bytes_moved per sweep: read->1x, write->1x, rw->2x (read+write).
//
// Key correctness points:
//  - 64-bit (uint64_t) accesses, not byte: each load/store moves 8 bytes, and we
//    step by 8 so every byte in the region is touched once per sweep. Byte access
//    underutilizes the load units and under-reports bandwidth.
//  - All threads are created and bound first, then released by a shared atomic
//    flag so they start the timed sweep together. Thread creation and teardown
//    are outside the clock interval.
//  - sink accumulation + asm memory barrier prevents the compiler from eliding loads.
static double measure_bandwidth(void * mem, size_t size_bytes, int core,
                                size_t sweeps, int n_threads, const std::string & mode) {
    if ((size_t) n_threads > size_bytes / sizeof(uint64_t)) {
        fprintf(stderr, "bandwidth thread count exceeds the number of uint64 elements\n");
        return -1.0;
    }
    const size_t per_thread = size_bytes / n_threads;
    std::atomic<int> ready{0};      // threads signal they are bound and waiting
    std::atomic<int> go{0};         // main signals start
    std::atomic<int> done{0};
    std::atomic<bool> bind_failed{false};
    std::atomic<int64_t> end_ns{0};

    auto mark_done = [&]() {
        if (done.fetch_add(1, std::memory_order_seq_cst) + 1 == n_threads) {
            struct timespec t;
            clock_gettime(CLOCK_MONOTONIC, &t);
            end_ns.store(
                    (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec,
                    std::memory_order_release);
        }
    };

    auto worker = [&](int tid) {
        const bool bound = bind_cpu(core + tid);
        if (!bound) {
            bind_failed.store(true, std::memory_order_seq_cst);
        }
        uint64_t * base = (uint64_t *)((uint8_t *)mem + (size_t)tid * per_thread);
        const size_t n_u64 = ((tid == n_threads - 1) ? (size_bytes - per_thread * (n_threads - 1)) : per_thread) / sizeof(uint64_t);
        uint64_t sink = 0;
        ready.fetch_add(1, std::memory_order_seq_cst);
        while (go.load(std::memory_order_seq_cst) == 0) { /* spin */ }

        if (!bound) {
            mark_done();
            return;
        } else if (mode == "read") {
            for (size_t s = 0; s < sweeps; ++s) {
                for (size_t i = 0; i < n_u64; ++i) sink ^= base[i];
                asm volatile("" ::: "memory");
            }
        } else if (mode == "write") {
            for (size_t s = 0; s < sweeps; ++s) {
                for (size_t i = 0; i < n_u64; ++i) base[i] = sink + i + s;
                asm volatile("" ::: "memory");
            }
        } else { // rw
            for (size_t s = 0; s < sweeps; ++s) {
                for (size_t i = 0; i < n_u64; ++i) {
                    sink ^= base[i];
                    base[i] = sink + i + s;
                }
                asm volatile("" ::: "memory");
            }
        }
        asm volatile("" :: "r"(sink) : "memory");
        mark_done();
    };

    // create + bind all threads first (not timed)
    std::vector<std::thread> ths;
    for (int t = 0; t < n_threads; ++t) ths.emplace_back(worker, t);
    while (ready.load(std::memory_order_seq_cst) < n_threads) { /* wait for all bound */ }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const int64_t start_ns = (int64_t) t0.tv_sec * 1000000000LL + t0.tv_nsec;
    go.store(1, std::memory_order_seq_cst);  // release all threads at once
    for (auto & th : ths) th.join();

    if (bind_failed.load(std::memory_order_seq_cst)) {
        return -1.0;
    }

    const double ns = (double) (end_ns.load(std::memory_order_acquire) - start_ns);
    const double bytes_per_sweep = (mode == "rw") ? 2.0 : 1.0;
    return (double)sweeps * bytes_per_sweep * (double)size_bytes / 1e9 / (ns / 1e9);
}

struct path_result {
    const char * name;
    int node;
    int cpu;
    double ns_per_access = -1;  // latency, ns/access (-1 = not measured)
    double gbps          = -1;  // bandwidth, GB/s   (-1 = not measured)
};

// Measure the TP reduction kernel, mirroring ggml_tp_reduce_sum_f32:
//   out[i] = sum over r of shm[r*nelem + i]
// The shared region `shm` (holding all ranks' data back-to-back) is pinned to
// data_node; the computing threads run on compute_cpu, compute_cpu+1, ...
// If data_node is on the peer CPU, every load crosses CXL. This isolates the
// reduction phase; rank barriers and rank-local copies are measured separately
// by pipeline-brick's TP profiler. Returns effective GB/s.
static double measure_tp_allreduce(void * shm, size_t total_bytes, int n_ranks,
                                   int compute_cpu, int n_threads,
                                   size_t sweeps, int output_node) {
    const size_t per_rank = total_bytes / n_ranks;       // bytes per rank
    const size_t nelem    = per_rank / sizeof(float);    // floats per rank
    if ((size_t) n_threads > nelem) {
        fprintf(stderr, "TP thread count exceeds the number of F32 elements per rank\n");
        return -1.0;
    }
    const size_t per_thread = nelem / n_threads;
    // The reduced output is rank-local in pipeline-brick. Keeping it on the
    // compute CPU's node prevents a cross-CXL input test from also charging
    // unrelated remote stores.
    void * out_mem = alloc_on_node(per_rank, output_node);
    if (!out_mem) return -1.0;
    float * out = (float *)out_mem;

    std::atomic<int> ready{0}, go{0}, done{0};
    std::atomic<bool> bind_failed{false};
    std::atomic<int64_t> end_ns{0};

    auto mark_done = [&]() {
        if (done.fetch_add(1, std::memory_order_seq_cst) + 1 == n_threads) {
            struct timespec t;
            clock_gettime(CLOCK_MONOTONIC, &t);
            end_ns.store(
                    (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec,
                    std::memory_order_release);
        }
    };

    auto worker = [&](int tid) {
        const bool bound = bind_cpu(compute_cpu + tid);
        if (!bound) {
            bind_failed.store(true, std::memory_order_seq_cst);
        }
        const size_t begin = tid * per_thread;
        const size_t end   = (tid == n_threads - 1) ? nelem : (tid + 1) * per_thread;
        float * base = (float *)shm;
        ready.fetch_add(1, std::memory_order_seq_cst);
        while (go.load(std::memory_order_seq_cst) == 0) { /* spin */ }

        if (!bound) {
            mark_done();
            return;
        }
        for (size_t s = 0; s < sweeps; ++s) {
            size_t i = begin;
#if defined(__aarch64__)
            // NEON 4-way sum, matching ggml_tp_reduce_sum_f32 style (TP=4 fast path).
            // General n_ranks handled in scalar tail.
            if (n_ranks >= 4) {
                const float * r0 = base;
                const float * r1 = base + nelem;
                const float * r2 = base + 2*nelem;
                const float * r3 = base + 3*nelem;
                for (; i + 4 <= end; i += 4) {
                    const float32x4_t v0 = vld1q_f32(r0 + i);
                    const float32x4_t v1 = vld1q_f32(r1 + i);
                    const float32x4_t v2 = vld1q_f32(r2 + i);
                    const float32x4_t v3 = vld1q_f32(r3 + i);
                    const float32x4_t sum01 = vaddq_f32(v0, v1);
                    const float32x4_t sum23 = vaddq_f32(v2, v3);
                    vst1q_f32(out + i, vaddq_f32(sum01, sum23));
                }
            }
#endif
            for (; i < end; ++i) {
                float sum = 0.0f;
                for (int r = 0; r < n_ranks; ++r) {
                    sum += base[(size_t)r * nelem + i];
                }
                out[i] = sum;
            }
            asm volatile("" ::: "memory");
        }
        mark_done();
    };

    std::vector<std::thread> ths;
    for (int t = 0; t < n_threads; ++t) ths.emplace_back(worker, t);
    while (ready.load(std::memory_order_seq_cst) < n_threads) { /* wait */ }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const int64_t start_ns = (int64_t) t0.tv_sec * 1000000000LL + t0.tv_nsec;
    go.store(1, std::memory_order_seq_cst);
    for (auto & th : ths) th.join();

    const double ns = (double) (end_ns.load(std::memory_order_acquire) - start_ns);
    munmap(out_mem, per_rank);
    if (bind_failed.load(std::memory_order_seq_cst)) {
        return -1.0;
    }
    // bytes moved: read n_ranks * per_rank + write per_rank (out)
    const double bytes_per_sweep = (double)(n_ranks + 1) * (double)per_rank;
    return (double)sweeps * bytes_per_sweep / 1e9 / (ns / 1e9);
}

static bool run_path(const char * name, int node, int cpu, size_t size_bytes, size_t iters,
                     const args_t & a, path_result & out) {
    if (node < 0 || cpu < 0) {
        return false;
    }
    const bool do_lat = (a.mode == "latency" || a.mode == "both");
    const bool do_bw  = (a.mode == "bandwidth" || a.mode == "both");
    fprintf(stderr, "numa-latency-bench: measuring %-12s node=%d cpu=%d size=%.1f MiB mode=%s\n",
            name, node, cpu, (double)size_bytes / (1024.0 * 1024.0), a.mode.c_str());
    if (!bind_cpu(cpu)) {
        return false;
    }
    void * mem = alloc_on_node(size_bytes, node);
    if (!mem) {
        fprintf(stderr, "  alloc_on_node failed, skipping %s\n", name);
        return false;
    }

    if (do_lat) {
        const size_t n_nodes = size_bytes / CACHE_LINE;
        build_shuffled_chain(mem, n_nodes, (uint64_t)node * 7 + (uint64_t)cpu);
        double best = 1e30;
        for (int t = 0; t < 3; ++t) {
            double ns = measure_chain(mem, iters);
            if (ns < best) best = ns;
        }
        out.ns_per_access = best;
        fprintf(stderr, "  %s latency: %.1f ns/access\n", name, best);
    }

    if (do_bw) {
        // bandwidth test uses sequential read+write sweeps; threads start at
        // cpu, cpu+1, ... so caller must pass a cpu range that all belong to
        // the same node (we only validate the first cpu's node via alloc).
        double best = 0;
        for (int t = 0; t < 3; ++t) {
            double g = measure_bandwidth(mem, size_bytes, cpu, a.bw_sweeps, a.bw_threads, a.bw_mode);
            if (g <= 0) {
                munmap(mem, size_bytes);
                return false;
            }
            if (g > best) best = g;
        }
        out.gbps = best;
        fprintf(stderr, "  %s bandwidth(%s): %.1f GB/s (%d threads)\n", name, a.bw_mode.c_str(), best, a.bw_threads);
    }

    munmap(mem, size_bytes);
    out.name = name;
    out.node = node;
    out.cpu = cpu;
    return true;
}

int main(int argc, char ** argv) {
    args_t a;
    if (!parse_args(argc, argv, a)) {
        return 1;
    }
    if (a.mode != "latency" && a.mode != "bandwidth" &&
            a.mode != "both" && a.mode != "tp-allreduce") {
        fprintf(stderr, "invalid --mode: %s\n", a.mode.c_str());
        return 1;
    }
    if (a.bw_mode != "read" && a.bw_mode != "write" && a.bw_mode != "rw") {
        fprintf(stderr, "invalid --bw-mode: %s\n", a.bw_mode.c_str());
        return 1;
    }
    if (a.size_mb == 0 || a.iters == 0 || a.bw_threads <= 0 || a.bw_sweeps <= 0) {
        fprintf(stderr, "size, iterations, bandwidth threads, and sweeps must be positive\n");
        return 1;
    }
    if (a.mode == "tp-allreduce" &&
            ((a.tp_ranks != 2 && a.tp_ranks != 4) || a.tp_elem_mb <= 0 ||
             a.tp_threads <= 0 || a.tp_sweeps <= 0 || a.tp_data_node < 0 ||
             a.tp_compute_cpu < 0 || a.tp_output_node < -1)) {
        fprintf(stderr, "tp-allreduce requires ranks=2|4 and positive size/threads/sweeps with valid node/cpu\n");
        return 1;
    }
    const size_t size_bytes = a.size_mb * 1024 * 1024;
    if (size_bytes < 4096) {
        fprintf(stderr, "--size-mb too small (need >= 1)\n");
        return 1;
    }

    std::vector<path_result> results;
    path_result r{};

    // tp-allreduce mode isolates the reduction kernel. Rank input buffers live
    // on tp_data_node, while the output is local to the compute CPU by default.
    // Rank-local copies and inter-rank barriers are intentionally excluded.
    if (a.mode == "tp-allreduce") {
        if (a.tp_output_node < 0) {
            a.tp_output_node = numa_node_for_cpu(a.tp_compute_cpu);
        }
        if (a.tp_output_node < 0) {
            fprintf(stderr, "cannot infer NUMA node for compute CPU %d; pass --tp-output-node\n",
                    a.tp_compute_cpu);
            return 1;
        }
        const size_t per_rank = (size_t)a.tp_elem_mb * 1024 * 1024;
        const size_t total    = per_rank * (size_t)a.tp_ranks;
        fprintf(stderr, "numa-latency-bench: tp-allreduce ranks=%d per_rank=%d MiB data_node=%d output_node=%d compute_cpu=%d threads=%d sweeps=%d\n",
                a.tp_ranks, a.tp_elem_mb, a.tp_data_node, a.tp_output_node,
                a.tp_compute_cpu, a.tp_threads, a.tp_sweeps);
        void * shm = alloc_on_node(total, a.tp_data_node);
        if (!shm) {
            fprintf(stderr, "alloc_on_node(data_node=%d) failed\n", a.tp_data_node);
            return 1;
        }
        // init each rank's data (non-zero so reads are not optimized away)
        float * base = (float *)shm;
        const size_t nelem = per_rank / sizeof(float);
        for (int rk = 0; rk < a.tp_ranks; ++rk) {
            for (size_t i = 0; i < nelem; ++i) base[(size_t)rk*nelem + i] = (float)((rk + i) & 0xff);
        }
        double best = 0;
        for (int t = 0; t < 3; ++t) {
            double g = measure_tp_allreduce(shm, total, a.tp_ranks, a.tp_compute_cpu,
                                             a.tp_threads, a.tp_sweeps, a.tp_output_node);
            if (g <= 0) {
                munmap(shm, total);
                return 1;
            }
            if (g > best) best = g;
        }
        munmap(shm, total);
        printf("\n===== TP all-reduce bandwidth =====\n");
        printf("ranks=%d  per_rank=%d MiB  data_node=%d  output_node=%d  compute_cpu=%d  threads=%d\n",
               a.tp_ranks, a.tp_elem_mb, a.tp_data_node, a.tp_output_node,
               a.tp_compute_cpu, a.tp_threads);
        const double bytes_per_sweep = (double)(a.tp_ranks + 1) * (double)per_rank;
        printf("reduce-kernel bandwidth: %.2f GB/s  (bytes/sweep = %.1f MiB)\n",
               best, bytes_per_sweep / (1024.0 * 1024.0));
        // per-reduce latency = time for one sweep = bytes_per_sweep / bandwidth
        const double per_reduce_us = best > 0 ? bytes_per_sweep / (best * 1e9) * 1e6 : 0;
        printf("per-reduce latency:   %.2f us  (best of 3, sweeps=%d, per_rank=%d MiB)\n",
               per_reduce_us, a.tp_sweeps, a.tp_elem_mb);
        return 0;
    }

    // Order matters: measure local first, then remote, then cross-CXL, so the
    // computing core moves outward and each measurement is independent.
    if (run_path("local",        a.local_node,  a.local_cpu,  size_bytes, a.iters, a, r)) results.push_back(r);
    if (run_path("remote-NUMA",  a.remote_node, a.remote_cpu, size_bytes, a.iters, a, r)) results.push_back(r);
    if (run_path("cross-CXL",    a.cxl_node,    a.cxl_cpu,    size_bytes, a.iters, a, r)) results.push_back(r);

    if (results.empty()) {
        fprintf(stderr, "no valid path measured. pass --local-node/--local-cpu at minimum.\n");
        return 1;
    }

    const bool has_lat = std::any_of(results.begin(), results.end(),
                                     [](const path_result & x){ return x.ns_per_access > 0; });
    const bool has_bw  = std::any_of(results.begin(), results.end(),
                                     [](const path_result & x){ return x.gbps > 0; });

    if (has_lat) {
        printf("\n===== memory latency summary =====\n");
        printf("%-14s | %-10s | %-8s | %s\n", "path", "numa-node", "cpu", "ns/access");
        printf("-------------------------------------------------\n");
        double local_ns = -1;
        for (const auto & x : results) {
            if (x.ns_per_access < 0) continue;
            printf("%-14s | %-10d | %-8d | %8.1f\n", x.name, x.node, x.cpu, x.ns_per_access);
            if (std::string(x.name) == "local") local_ns = x.ns_per_access;
        }
        if (local_ns > 0) {
            printf("\n----- latency ratio vs local -----\n");
            for (const auto & x : results) {
                if (x.ns_per_access < 0) continue;
                printf("%-14s : %.2fx\n", x.name, x.ns_per_access / local_ns);
            }
        }
    }

    if (has_bw) {
        printf("\n===== memory bandwidth summary =====\n");
        printf("%-14s | %-10s | %-8s | %s\n", "path", "numa-node", "cpu", "GB/s");
        printf("-------------------------------------------------\n");
        double local_bw = -1;
        for (const auto & x : results) {
            if (x.gbps < 0) continue;
            printf("%-14s | %-10d | %-8d | %8.1f\n", x.name, x.node, x.cpu, x.gbps);
            if (std::string(x.name) == "local") local_bw = x.gbps;
        }
        if (local_bw > 0) {
            printf("\n----- bandwidth ratio vs local (local/x = how many times slower) -----\n");
            for (const auto & x : results) {
                if (x.gbps < 0) continue;
                printf("%-14s : %.2fx slower (%.1f / %.1f GB/s)\n",
                       x.name, local_bw / x.gbps, x.gbps, local_bw);
            }
        }
    }
    return 0;
}
