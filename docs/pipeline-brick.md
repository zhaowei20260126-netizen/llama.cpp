# Pipeline Brick hardware prototype

This is a private-fork experiment for the Phytium two-Brick Pipeline Brick plan.

It is not llama.cpp RPC, TCP, or CUDA. On the confirmed Phytium machine, both cards belong to one Linux system:

- card0: NUMA nodes `0-3`
- card1: NUMA nodes `4-7`

The recommended mode is `--single-system`, which launches one head process and one tail process inside the same Linux system:

- brick0/head: Qwen3-4B layers `[0,18)`, token embedding, no final norm, no lm_head.
- brick1/tail: Qwen3-4B layers `[18,36)`, final norm, lm_head.
- forward link: shared-memory ring buffer carrying `float32[1][2560]` hidden states.
- reverse link: shared-memory ring buffer carrying small token packets.
- payload per generated or prompt token: `2560 * sizeof(float) = 10240` bytes.
- scope: Qwen3-4B GGUF only, CPU-only.

Important limitation: `--numa-tp` is currently a runtime configuration and affinity check in this tool. llama.cpp's CPU backend does not yet expose true CPU-NUMA tensor-parallel devices, so Q4_K_M matrix sharding across NUMA nodes still needs backend work before it is a real tensor split.

## Build

```sh
cmake -B build -DLLAMA_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build -j 32 --target llama-pipeline-brick
```

## Single-system NUMA binding

Use this mode first on the real Phytium machine:

```sh
./build/bin/llama-pipeline-brick \
  --single-system \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "请用三句话介绍一下 llama.cpp 是什么。" \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --head-numa 0-3 \
  --tail-numa 4-7
```

`--head-numa 0-3` is translated by reading `/sys/devices/system/node/node*/cpulist`, then the head process is bound with `sched_setaffinity`. `--tail-numa 4-7` does the same for the tail process.

The launcher creates two shared-memory ring buffers in `/dev/shm`:

- head -> tail: hidden states from layer 17 output to layer 18 input, bound with `mbind` to tail NUMA nodes `4-7`.
- tail -> head: generated token control packets, bound with `mbind` to head NUMA nodes `0-3`.

This is a process-level Memory Window style transport inside one Linux system: the sender writes into the receiver-side NUMA memory window, then the receiver reads locally. Whether the physical traffic crosses MCIO depends on the platform routing for remote NUMA memory access.

## Explicit hardware transport model

Each process opens two Memory Windows:

- `--tx-mw`: local mapping used by this Brick to write packets to the peer.
- `--rx-mw`: local mapping where this Brick receives packets from the peer.

With `--doorbell-mode write`, each process also opens:

- `--tx-doorbell`: write here after publishing a packet.
- `--rx-doorbell`: poll/read here while waiting for peer packets.

If the driver does not expose a readable Doorbell device yet, use `--doorbell-mode poll`. That mode polls Memory Window slot state directly and does not require Doorbell paths.

The code does not hard-code Phytium device names because those come from the actual NTB/CXL driver. Replace the `/dev/REPLACE_*` paths below with the real device nodes on the two machines.

## Link self-test

Start tail first:

```sh
sudo ./build/bin/llama-pipeline-brick \
  --self-test-ntb \
  --role tail \
  --brick-id 1 \
  --peer-brick-id 0 \
  --transport ntb-mw \
  --doorbell-mode write \
  --tx-mw /dev/REPLACE_TAIL_TO_HEAD_MW \
  --rx-mw /dev/REPLACE_HEAD_TO_TAIL_MW \
  --tx-doorbell /dev/REPLACE_TAIL_DOORBELL \
  --rx-doorbell /dev/REPLACE_HEAD_DOORBELL
```

Then start head:

```sh
sudo ./build/bin/llama-pipeline-brick \
  --self-test-ntb \
  --role head \
  --brick-id 0 \
  --peer-brick-id 1 \
  --transport ntb-mw \
  --doorbell-mode write \
  --tx-mw /dev/REPLACE_HEAD_TO_TAIL_MW \
  --rx-mw /dev/REPLACE_TAIL_TO_HEAD_MW \
  --tx-doorbell /dev/REPLACE_HEAD_DOORBELL \
  --rx-doorbell /dev/REPLACE_TAIL_DOORBELL
```

Expected result:

- head sends one hidden-state packet.
- tail receives the packet and sends one token packet back.
- head receives the token packet and exits.

## Explicit two-process inference

Start tail first:

```sh
sudo ./build/bin/llama-pipeline-brick \
  --hardware \
  --role tail \
  --brick-id 1 \
  --peer-brick-id 0 \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --bricks 2 \
  --layer-start 18 \
  --layer-end 36 \
  --numa-tp 4 \
  --numa-cpus "0-15,16-31,32-47,48-63" \
  --transport ntb-mw \
  --doorbell-mode write \
  --tx-mw /dev/REPLACE_TAIL_TO_HEAD_MW \
  --rx-mw /dev/REPLACE_HEAD_TO_TAIL_MW \
  --tx-doorbell /dev/REPLACE_TAIL_DOORBELL \
  --rx-doorbell /dev/REPLACE_HEAD_DOORBELL
```

Then start head:

```sh
sudo ./build/bin/llama-pipeline-brick \
  --hardware \
  --role head \
  --brick-id 0 \
  --peer-brick-id 1 \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "请用三句话介绍一下 llama.cpp 是什么。" \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --bricks 2 \
  --layer-start 0 \
  --layer-end 18 \
  --numa-tp 4 \
  --numa-cpus "0-15,16-31,32-47,48-63" \
  --transport ntb-mw \
  --doorbell-mode write \
  --tx-mw /dev/REPLACE_HEAD_TO_TAIL_MW \
  --rx-mw /dev/REPLACE_TAIL_TO_HEAD_MW \
  --tx-doorbell /dev/REPLACE_HEAD_DOORBELL \
  --rx-doorbell /dev/REPLACE_TAIL_DOORBELL
```

For the confirmed single Linux system that sees all 8 NUMA nodes, prefer `--single-system`. If you manually launch two processes, bind them separately:

- head process: `--numa-cpus "0-15,16-31,32-47,48-63"`
- tail process: `--numa-cpus "64-79,80-95,96-111,112-127"`

## Expected logs

The logs should include:

```text
pipeline-brick head: brick=0 peer=1 layers [0,18), parallel=4, n_embd=2560
pipeline-brick tail: brick=1 peer=0 layers [18,36), parallel=4, n_embd=2560
pipeline-brick transport: ntb-mw slots=64 slot_size=...
pipeline-brick head: hidden payload per token = 10240 bytes
pipeline-brick tail: hidden payload per token = 10240 bytes
pipeline-brick head: sent seq=0 pos=...
pipeline-brick tail: recv seq=0 pos=...
```

With `--parallel 4`, the `seq` values should interleave. That confirms micro-batch scheduling is filling the pipeline instead of running one sequence to completion before starting the next.

## Greedy consistency check

Run a normal single-process baseline with greedy top-1:

```sh
./build/bin/llama-cli \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 64 \
  -c 2048 \
  -n 16 \
  --no-warmup \
  --no-conversation \
  --temp 0 \
  -p "请用三句话介绍一下 llama.cpp 是什么。"
```

Then run the hardware pipeline with `--parallel 1 --n-predict 16`. The target is that the first 16 generated tokens match. If text diverges, compare top-1 logits first, because one small floating-point difference can change later autoregressive tokens.
