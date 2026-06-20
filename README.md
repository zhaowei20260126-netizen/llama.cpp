# Feibric

Feibric is a CPU-oriented large language model inference system built on top of llama.cpp. It targets dual-card Phytium CPU platforms and focuses on improving Qwen3-4B style model inference through pipeline partitioning, local resource binding, lightweight hidden-state transport, and sparse KV-cache friendly decoding.

The original upstream llama.cpp README has been kept as [readme_orgin](readme_orgin).

## Project Overview

Feibric splits a Transformer model into pipeline bricks:

- **Head Brick**: token embedding and the first half of Transformer layers.
- **Tail Brick**: the second half of Transformer layers, final normalization, and LM head.
- **Transport layer**: transfers hidden states from Head to Tail and returns generated token/control packets.
- **Locality control**: binds each brick to the CPU card / NUMA group where its weights and KV cache are expected to stay local.

Compared with a single full-model CPU process, this design reduces per-process memory pressure, keeps each brick's KV cache local to its assigned CPU resources, and allows prefill/decode work to overlap across the two pipeline stages.

## Main Features

- **Two-stage pipeline inference** for Qwen3-4B compatible GGUF models.
- **Dual-card Phytium CPU execution** through `llama-pipeline-brick`.
- **BF16 hidden-state transport** to reduce inter-brick payload size compared with F32.
- **Micro-batch prefill** through `--prefill-chunk`.
- **Streaming KV sparse attention path** through `--stream-kv`, `--stream-kv-sink`, and `--stream-kv-recent`.
- **NUMA-aware binding** through `--head-numa`, `--tail-numa`, `--stage-numa`, and `--numa-cpus`.
- **TP prototype controls** through `--tp-size 1|2|4`.
- **Multiple transport modes** in the prototype: shared memory / CXL-like memory-window mode, NTB memory-window mode, and optional IB RDMA when libibverbs is available on Linux.

## Experimental Effect

On the dual-card Phytium CPU test platform, Feibric improves CPU inference throughput by combining:

- layer-level pipeline partitioning,
- BF16 hidden-state transfer,
- prefill micro-batching,
- local KV cache ownership per brick,
- StreamingLLM-style sparse KV attention.

In the measured Qwen3-4B Q4_K_M runs:

| Scenario | Baseline | Feibric best | Speedup |
| --- | ---: | ---: | ---: |
| batch=4, about 1K prompt, 128 output tokens | 18.64 tokens/s | 53.83 tokens/s | 2.89x |
| batch=8, 256 output tokens | 20.06 tokens/s | 55.79 tokens/s | 2.78x |

The main gain comes from better CPU resource utilization and lower cross-stage communication cost. TP=2 is the preferred setting in the current prototype; TP=4 is supported for comparison but may introduce higher synchronization overhead.

## Build

Recommended platform: Linux on the target Phytium CPU machine.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --target llama-pipeline-brick -j$(nproc)
```

The executable is generated as:

```bash
./build/bin/llama-pipeline-brick
```

If IB RDMA support is needed, install the system libibverbs development package before configuring CMake. The build script automatically enables `LLAMA_PIPELINE_BRICK_IBVERBS` when both the library and headers are found.

## Model Preparation

Prepare a local GGUF model path before running:

```bash
export MODEL=/path/to/Qwen3-4B-Instruct-Q4_K_M.gguf
```

For the current prototype, Qwen3-4B compatible GGUF models are the primary target. The default layer split is 18/18.

## Single-System Dual-Card Run

Use this mode when the two Phytium CPU cards are visible in one Linux system. Replace the resource ranges with the actual values reported by `numactl -H` and `lscpu` on the target machine.

```bash
export HEAD_NUMA="HEAD_NUMA_RANGE"
export TAIL_NUMA="TAIL_NUMA_RANGE"

./build/bin/llama-pipeline-brick \
  --single-system \
  --model "$MODEL" \
  --prompt "Briefly introduce the Feibric inference architecture." \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 128 \
  --parallel 4 \
  --head-numa "$HEAD_NUMA" \
  --tail-numa "$TAIL_NUMA" \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --stream-kv \
  --stream-kv-sink 16 \
  --stream-kv-recent 128 \
  --tp-size 2
```

Parameter notes:

- `--head-numa`: NUMA nodes assigned to the Head Brick.
- `--tail-numa`: NUMA nodes assigned to the Tail Brick.
- `--parallel`: number of parallel sequences.
- `--prefill-chunk`: prompt micro-batch size.
- `--hidden-dtype bf16`: sends BF16 hidden states between bricks.
- `--stream-kv`: enables sparse KV-cache friendly decoding.
- `--tp-size 2`: enables the TP prototype with two ranks per brick.

## Two-Process Hardware Run

Use this mode when Head and Tail are launched as explicit processes. Start Tail first, then Head.

Tail process:

```bash
export HEAD_NUMA="HEAD_NUMA_RANGE"
export TAIL_NUMA="TAIL_NUMA_RANGE"
export HEAD_CPUS="HEAD_CPU_RANGE"
export TAIL_CPUS="TAIL_CPU_RANGE"

./build/bin/llama-pipeline-brick \
  --hardware \
  --role tail \
  --model "$MODEL" \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 128 \
  --bricks 2 \
  --brick-id 1 \
  --peer-brick-id 0 \
  --layer-start 18 \
  --layer-end 36 \
  --parallel 4 \
  --numa-tp 1 \
  --tp-size 2 \
  --numa-cpus "$TAIL_CPUS" \
  --transport cxl \
  --tx-mw /dev/shm/feibric-t2h \
  --rx-mw /dev/shm/feibric-h2t \
  --head-numa "$HEAD_NUMA" \
  --tail-numa "$TAIL_NUMA" \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --stream-kv
```

Head process:

```bash
export HEAD_NUMA="HEAD_NUMA_RANGE"
export TAIL_NUMA="TAIL_NUMA_RANGE"
export HEAD_CPUS="HEAD_CPU_RANGE"
export TAIL_CPUS="TAIL_CPU_RANGE"

./build/bin/llama-pipeline-brick \
  --hardware \
  --role head \
  --model "$MODEL" \
  --prompt "Briefly introduce the Feibric inference architecture." \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 128 \
  --bricks 2 \
  --brick-id 0 \
  --peer-brick-id 1 \
  --layer-start 0 \
  --layer-end 18 \
  --parallel 4 \
  --numa-tp 1 \
  --tp-size 2 \
  --numa-cpus "$HEAD_CPUS" \
  --transport cxl \
  --tx-mw /dev/shm/feibric-h2t \
  --rx-mw /dev/shm/feibric-t2h \
  --head-numa "$HEAD_NUMA" \
  --tail-numa "$TAIL_NUMA" \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --stream-kv
```

If the platform exposes NTB/CXL memory-window device files, replace the `/dev/shm/feibric-*` paths with the corresponding device or mapped window paths and set `--transport ntb-mw` or `--transport cxl` as appropriate.

## Four-Stage / Multi-Domain Prototype

The tool also contains a four-stage prototype for extended platforms:

```bash
export STAGE_NUMA="STAGE0_NUMA;STAGE1_NUMA;STAGE2_NUMA;STAGE3_NUMA"

./build/bin/llama-pipeline-brick \
  --single-system \
  --stage-count 4 \
  --stage-numa "$STAGE_NUMA" \
  --transport cxl \
  --model "$MODEL" \
  --prompt "Briefly introduce the Feibric inference architecture." \
  --ctx-size 2048 \
  --threads 32 \
  --n-predict 128 \
  --parallel 4 \
  --hidden-dtype bf16 \
  --stream-kv \
  --tp-size 2
```

For dual-domain hardware, use `--domain-mode dual`, `--stage-id`, `--stage-count 4`, and the `--up-transport` / `--down-transport` options. IB RDMA requires `--rdma-dev`, local info files, and peer info files for each RDMA link.

## Repository Layout

- `tools/pipeline-brick/pipeline-brick.cpp`: main Feibric pipeline inference tool.
- `tools/pipeline-brick/CMakeLists.txt`: build target definition.
- `tp/`: TP prototype code and related backend changes.
- `src/`, `common/`, `ggml/`: upstream llama.cpp and ggml runtime code used by Feibric.

## Notes

- This repository keeps only source code and necessary build files. Experiment reports, intermediate documents, private logs, and submission-only materials are intentionally excluded from the public branch.
- The default GitHub branch only removes those files from the latest version; it does not rewrite old Git history.
- The current `llama-pipeline-brick` target is Linux-oriented because the real-machine path uses Linux NUMA and memory policy APIs.
