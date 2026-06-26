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
//       --local-cpu 0 --remote-cpu 16 --cxl-cpu 64 [--size-mb 64] [--iters 20000000]
//
// On a single-NUMA machine, only --local-node makes sense; remote/cxl are skipped.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
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

struct args_t {
    int local_node  = -1;
    int remote_node = -1;
    int cxl_node    = -1;
    int local_cpu   = -1;
    int remote_cpu  = -1;
    int cxl_cpu     = -1;
    size_t size_mb  = 64;
    size_t iters    = 20000000;
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
        "  --remote-cpu   cpu id to run on for the remote measurement\n"
        "  --cxl-cpu      cpu id to run on for the cross-CXL measurement\n"
        "  --size-mb      working-set size in MiB (default 64)\n"
        "  --iters        pointer-chase iterations per measurement (default 20000000)\n"
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
        else if (key == "--help" || key == "-h") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown argument: %s\n", key.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

static void bind_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity(cpu=%d) failed: %s\n", cpu, strerror(errno));
    }
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
        // MPOL_MF_MOVE | MPOL_MF_STRICT: migrate + enforce. Pages are freshly
        // allocated (anon), so this just sets policy before first touch.
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

// Build a shuffled circular linked list of indices over an array of size n.
// Each slot holds the index of the next slot. Pointer-chasing this list gives
// data-dependent loads that cannot be pipelined, exposing raw latency.
static void build_shuffled_chain(uint32_t * next, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = (uint32_t)i;
    for (size_t i = n; i > 1; --i) {
        size_t j = rng() % i;
        std::swap(perm[i - 1], perm[j]);
    }
    for (size_t i = 0; i < n; ++i) next[perm[i]] = perm[(i + 1) % n];
}

// Returns nanoseconds per pointer-chase step.
static double measure_chain(uint32_t * next, size_t iters) {
    uint32_t idx = 0;
    // warmup: bring the chain access pattern into a stable state
    for (size_t i = 0; i < 1024; ++i) idx = next[idx];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < iters; ++i) {
        idx = next[idx];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    // prevent compiler from optimizing the loop away
    asm volatile("" :: "r"(idx) : "memory");

    double ns = (double)((t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec));
    return ns / (double)iters;
}

struct path_result {
    const char * name;
    int node;
    int cpu;
    double ns_per_access;
};

static bool run_path(const char * name, int node, int cpu, size_t size_bytes, size_t iters,
                     path_result & out) {
    if (node < 0 || cpu < 0) {
        return false;
    }
    fprintf(stderr, "numa-latency-bench: measuring %-12s node=%d cpu=%d size=%.1f MiB iters=%zu\n",
            name, node, cpu, (double)size_bytes / (1024.0 * 1024.0), iters);
    bind_cpu(cpu);
    void * mem = alloc_on_node(size_bytes, node);
    if (!mem) {
        fprintf(stderr, "  alloc_on_node failed, skipping %s\n", name);
        return false;
    }
    const size_t n_slots = size_bytes / sizeof(uint32_t);
    uint32_t * next = (uint32_t *)mem;
    build_shuffled_chain(next, n_slots, (uint64_t)node * 7 + (uint64_t)cpu);

    // run 3 trials, take the min (least noise)
    double best = 1e30;
    for (int t = 0; t < 3; ++t) {
        double ns = measure_chain(next, iters);
        if (ns < best) best = ns;
    }
    munmap(mem, size_bytes);
    out.name = name;
    out.node = node;
    out.cpu = cpu;
    out.ns_per_access = best;
    fprintf(stderr, "  %s: %.1f ns/access\n", name, best);
    return true;
}

int main(int argc, char ** argv) {
    args_t a;
    if (!parse_args(argc, argv, a)) {
        return 1;
    }
    const size_t size_bytes = a.size_mb * 1024 * 1024;
    if (size_bytes < 4096) {
        fprintf(stderr, "--size-mb too small (need >= 1)\n");
        return 1;
    }

    std::vector<path_result> results;
    path_result r{};

    // Order matters: measure local first, then remote, then cross-CXL, so the
    // computing core moves outward and each measurement is independent.
    if (run_path("local",        a.local_node,  a.local_cpu,  size_bytes, a.iters, r)) results.push_back(r);
    if (run_path("remote-NUMA",  a.remote_node, a.remote_cpu, size_bytes, a.iters, r)) results.push_back(r);
    if (run_path("cross-CXL",    a.cxl_node,    a.cxl_cpu,    size_bytes, a.iters, r)) results.push_back(r);

    if (results.empty()) {
        fprintf(stderr, "no valid path measured. pass --local-node/--local-cpu at minimum.\n");
        return 1;
    }

    printf("\n===== memory latency summary =====\n");
    printf("%-14s | %-10s | %-8s | %s\n", "path", "numa-node", "cpu", "ns/access");
    printf("-------------------------------------------------\n");
    for (const auto & x : results) {
        printf("%-14s | %-10d | %-8d | %8.1f\n", x.name, x.node, x.cpu, x.ns_per_access);
    }

    // print ratios relative to local, if local was measured
    double local_ns = -1;
    for (const auto & x : results) {
        if (std::string(x.name) == "local") local_ns = x.ns_per_access;
    }
    if (local_ns > 0) {
        printf("\n===== ratio vs local =====\n");
        for (const auto & x : results) {
            printf("%-14s : %.2fx\n", x.name, x.ns_per_access / local_ns);
        }
    }
    return 0;
}
