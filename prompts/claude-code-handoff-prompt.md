# Claude Code 接手提示词：Pipeline Brick 双飞腾 CPU 推理原型

你现在接手的是一个 llama.cpp 私有 fork，用于研电赛飞腾赛道赛题 3：高效模型推理。请先按下面信息建立上下文，再阅读代码和文档。回答和改代码时默认用中文，工程术语第一次出现请解释清楚，不要假设我已经完全理解。

## 1. 项目目标

我们要做的是一个面向双飞腾 CPU 的 Pipeline Brick 推理原型。核心想法不是 GPU 推理，也不是 llama.cpp 自带 RPC/TCP，而是：

- 把 Qwen3-4B 模型按层切成两段。
- 第一张 CPU 卡运行 head Brick，负责 token embedding 和前半层。
- 第二张 CPU 卡运行 tail Brick，负责后半层、final norm 和 lm_head。
- 两个 Brick 中间只传 hidden states，不传完整 KV cache。
- 多请求场景下用 micro-batch 做流水线，让 head 和 tail 尽量重叠执行。
- 当前用 mmap shared memory 模拟 CXL/NTB Memory Window，用 poll 语义模拟 Doorbell。

这不是 upstream-ready 的通用功能，是比赛用私有 fork 实验代码。

## 2. 实机硬件环境

实机是一套双飞腾 CPU、单 Linux 系统：

- 一共 128 核，8 个 NUMA node。
- 第一张卡：NUMA `0-3`，CPU `0-63`。
- 第二张卡：NUMA `4-7`，CPU `64-127`。
- 每张卡 4 个 NUMA，每个 NUMA 16 核。
- 模型文件路径通常是：

```sh
models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

模型是 Qwen3-4B-Instruct-2507 GGUF Q4_K_M，当前代码固定按 Qwen3-4B 的结构处理：

- 层数：36。
- hidden size：2560。
- head Brick：layers `[0,18)`。
- tail Brick：layers `[18,36)`。

## 3. 当前已经实现的内容

主要代码在：

```text
tools/pipeline-brick/pipeline-brick.cpp
```

新增工具构建目标：

```text
llama-pipeline-brick
```

当前实现包括：

1. 双进程 head/tail 原型

`--single-system` 模式会在同一个 Linux 系统里 fork 出 head 和 tail 两个进程。

head 绑定第一张卡：

```sh
--head-numa 0-3
```

tail 绑定第二张卡：

```sh
--tail-numa 4-7
```

2. 模型分层执行

head 执行：

```text
token embedding + layers [0,18)
```

tail 执行：

```text
layers [18,36) + final norm + lm_head
```

tail 不重新做 token embedding，而是通过 `batch.embd` 接收 head 传来的 hidden states。

3. hidden states 传输

head 到 tail 传输的是每个 token 的 hidden states。Qwen3-4B hidden size 是 2560：

```text
float32: 2560 x 4 = 10240 bytes/token
bf16:    2560 x 2 = 5120 bytes/token
```

当前默认使用：

```sh
--hidden-dtype bf16
```

也就是把 hidden states 转成 BF16 传输，tail 端再转回 float32 填入 `batch.embd`。

4. micro-batch 调度

micro-batch 是流水线的一次传输和执行单位，不等于整个 batch。

例如：

```text
parallel = 4
prefill-chunk = 32
每条 seq prompt 长度 = 100 token
```

那么整个 batch 是：

```text
4 x 100 = 400 token
```

但 prefill 阶段每次 micro-batch 是：

```text
parallel x prefill-chunk = 4 x 32 = 128 token
```

执行顺序大致是：

```text
t0: head 处理 seq0-3 的 token 0-31
t1: tail 处理 token 0-31，同时 head 处理 token 32-63
t2: tail 处理 token 32-63，同时 head 处理 token 64-95
t3: tail 处理 token 64-95，同时 head 处理 token 96-99
```

decode 阶段每轮每个 sequence 只有一个当前 token，所以：

```text
decode micro-batch = parallel
```

如果 `parallel=4`，decode 阶段每次 micro-batch 是 4 token。

5. 传输结构

当前使用 ring buffer。包里包含：

- header：magic/version/slot/flags/n_tokens/n_embd/pos/seq_id/payload_bytes。
- metadata：每个 token 的 `seq_id`、`pos`、`want_logits` 等信息。
- hidden payload：`[n_tokens][n_embd]`，格式为 bf16 或 f32。

6. prompt template 对齐

为了和 `llama-parallel` 公平比较，`llama-pipeline-brick` 内部已经加了和 `llama-parallel` 对齐的 prompt template 处理。日志里应看到类似：

```text
pipeline-brick head: llama-parallel prompt template enabled, prompt tokens=287
```

或者长 prompt 场景：

```text
pipeline-brick head: llama-parallel prompt template enabled, prompt tokens=643
```

## 4. 当前没有真正实现的内容

不要把下面这些说成已经完成：

1. 真实 CXL/NTB 驱动

当前是 mmap shared memory 模拟，不是真实 NTB BAR、CXL Memory Window 或厂商字符设备。

2. 真实 Doorbell 中断

当前是 poll 语义模拟，不是硬件 Doorbell interrupt。

3. DMA

当前 head 仍然要把 hidden states 写入 shared buffer，tail 再读出来。还不是 DMA engine 搬运。

4. 严格零拷贝

目前不是严格零拷贝。head 计算 buffer 到 shared buffer 有拷贝，bf16 转换也有开销；tail 还要把 bf16 转回 float32 放进 `batch.embd`。

5. 真正的 NUMA tensor parallel

当前只是 CPU affinity / NUMA 绑核，不是把矩阵乘法真实切到 4 个 NUMA 上做 tensor parallel。

6. predictor / sparse attention

暂时不做预测器，不做 Quest，不做 sparse attention。

## 5. 构建命令

在飞腾实机上推荐先关闭不必要目标，避免 mtmd 等组件缺文件导致 CMake 报错：

```sh
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF

cmake --build build -j 32 --target llama-pipeline-brick
```

如果需要 baseline：

```sh
cmake --build build -j 32 --target llama-parallel
```

注意：这个项目版本里不一定有 `llama-cli` target，曾经 `cmake --build build --target help | grep llama` 看到的是 `llama-parallel`、`llama-bench`、`llama-pipeline-brick` 等目标。

## 6. 已验证的运行命令

### 6.1 batch=2 短 prompt baseline

```sh
mkdir -p bench-logs

/usr/bin/time -p -o bench-logs/baseline_parallel2.time \
./build/bin/llama-parallel \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 128 \
  -c 4096 \
  -n 64 \
  -np 2 \
  -ns 2 \
  -p "请用三句话介绍一下 llama.cpp 是什么。" \
  --top-k 1 \
  > bench-logs/baseline_parallel2.out \
  2> bench-logs/baseline_parallel2.log
```

### 6.2 batch=2 短 prompt Pipeline Brick

```sh
mkdir -p bench-logs

/usr/bin/time -p -o bench-logs/pipeline_p2_microbf16.time \
./build/bin/llama-pipeline-brick \
  --single-system \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "请用三句话介绍一下 llama.cpp 是什么。" \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 64 \
  --parallel 2 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --quiet \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  > bench-logs/pipeline_p2_microbf16.out \
  2> bench-logs/pipeline_p2_microbf16.log
```

### 6.3 batch=4 长 prompt baseline

先创建 prompt 文件：

```sh
mkdir -p bench-logs prompts

cat > prompts/long_1k_prompt.txt <<'EOF'
请围绕 llama.cpp、CPU 大模型推理、NUMA 亲和性、流水线并行、KV cache 管理、prefill 和 decode 阶段的性能瓶颈，写一段技术分析。要求说明纯 CPU 推理为什么通常受内存带宽和矩阵乘法效率限制，说明长上下文下 KV cache 访问为什么会影响吞吐，说明多 sequence 并发时 batching 如何提高硬件利用率。然后进一步分析在双飞腾 CPU 系统中，将 Qwen3-4B 按层切分为 head Brick 和 tail Brick 的意义：head Brick 负责 embedding 和前 18 层，tail Brick 负责后 18 层、final norm 和 lm_head；两端之间只传 hidden states，不传完整 KV cache，也不通过 TCP 或 RPC。请解释这种方式相比单进程完整模型推理的潜在优势，包括两颗 CPU 同时工作、内存压力分散、每个 Brick 只维护本地层的 KV cache，以及通过 micro-batch 让 head 和 tail 在不同 token 批次上重叠执行。还要说明这种方案的局限，例如 hidden states 传输仍有开销，BF16 可以减少传输量但会带来格式转换，真实 CXL 或 NTB Memory Window 和 Doorbell 才能更接近硬件方案，当前 mmap shared memory 只能作为单系统验证。最后请用比较客观的语气总结：在 batch 较小、prompt 较短时，流水线并行可能被调度和传输开销抵消；在 prompt 较长、batch 较大、micro-batch 设置合理时，流水线并行才更可能体现优势。
EOF
```

运行 baseline：

```sh
/usr/bin/time -p -o bench-logs/baseline_parallel4_long1k.time \
./build/bin/llama-parallel \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 128 \
  -c 8192 \
  -n 64 \
  -np 4 \
  -ns 4 \
  -f prompts/long_1k_prompt.txt \
  --top-k 1 \
  > bench-logs/baseline_parallel4_long1k.out \
  2> bench-logs/baseline_parallel4_long1k.log
```

### 6.4 batch=4 长 prompt Pipeline Brick

注意：当前 `llama-pipeline-brick` 不支持 `--prompt-file`，要用 shell 变量把文件内容读入 `--prompt`。

```sh
mkdir -p bench-logs

LONG_PROMPT="$(cat prompts/long_1k_prompt.txt)"

/usr/bin/time -p -o bench-logs/pipeline_p4_long1k_microbf16.time \
./build/bin/llama-pipeline-brick \
  --single-system \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "$LONG_PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --quiet \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  > bench-logs/pipeline_p4_long1k_microbf16.out \
  2> bench-logs/pipeline_p4_long1k_microbf16.log
```

## 7. 已测结果

### 7.1 batch=2 短 prompt

baseline `llama-parallel`：

```text
real 47.08
user 3636.43
sys 926.12
```

日志关键值：

```text
n_parallel = 2, n_sequences = 2
prompt = 287
Total prompt tokens: 574
Total gen tokens: 128
```

Pipeline Brick：

```text
real 37.25
user 2420.29
sys 52.97
```

结论：

```text
baseline throughput = 128 / 47.08 = 2.72 tok/s
pipeline throughput = 128 / 37.25 = 3.44 tok/s
speedup = 47.08 / 37.25 = 1.26x
```

### 7.2 batch=4 长 prompt

baseline `llama-parallel`：

```text
real 100.49
user 8046.93
sys 1390.54
```

日志关键值：

```text
n_parallel = 4, n_sequences = 4
prompt = 643
Total prompt tokens: 2572
Total gen tokens: 256
Total gen speed: 2.60 t/s
```

Pipeline Brick：

```text
real 91.85
user 6939.97
sys 86.07
```

结论：

```text
baseline throughput = 256 / 100.49 = 2.55 tok/s
pipeline throughput = 256 / 91.85 = 2.79 tok/s
speedup = 100.49 / 91.85 = 1.09x
```

这个结果要客观看：Pipeline Brick 在两个场景都比 baseline 快，但 batch=4 长 prompt 的提升只有 1.09x，不要夸大。原因可能是长 prompt + 多 batch 时 `llama-parallel` 的 continuous batching 已经比较有效，而 Pipeline Brick 仍有 hidden states 传输、bf16 转换和双进程同步开销。

## 8. 判断当前实现是否真有流水线

当前实现确实有软件原型级流水线，不是单纯分层。

证据：

- prefill 阶段不是单 token，而是 micro-batch。
- batch=2、`prefill-chunk=32` 时，日志可看到 `n_tokens=64`。
- batch=4、`prefill-chunk=32` 时，日志可看到 micro-batch max=128。
- head 和 tail 是两个进程，分别绑定两张 CPU 卡。
- head 发送 hidden states 后可继续处理后续 micro-batch，tail 负责消费前面的 hidden micro-batch。

但它不是最终硬件版流水线。更准确的表述是：

```text
已经实现双 Brick 分层 + micro-batch + shared memory 传输的软件流水线原型；
尚未实现真实 CXL/NTB、DMA、硬件 Doorbell 和 NUMA tensor parallel。
```

## 9. 接手时优先阅读的文件

请按这个顺序读：

1. `docs/pipeline-brick-experiment-report.md`
2. `docs/pipeline-brick-run-single-system.md`
3. `docs/pipeline-brick.md`
4. `tools/pipeline-brick/pipeline-brick.cpp`
5. 如需理解 llama.cpp graph 分段，再查模型 graph 相关改动。

## 10. 接下来可能要做的事

如果继续优化性能，优先考虑：

1. 给 `llama-pipeline-brick` 增加 `--prompt-file`，避免长 prompt 只能用 shell 变量传。
2. 增加更详细但低开销的性能统计，比如 prefill time、decode time、head wait time、tail wait time、transport time。
3. 测 `parallel=1/2/4/8` 和 `prefill-chunk=16/32/64/128`，找实机最佳点。
4. 减少 bf16 转换和 memcpy 开销。
5. 如果拿到真实 CXL/NTB 设备路径，再替换当前 mmap/poll transport。
6. 谨慎评估 NUMA tensor parallel。当前只是绑核，真正 tensor parallel 工程量很大，不一定是第一优先级。

## 11. 回答用户时的风格要求

用户不希望被附和。请客观回答，尤其不要把“跑通”说成“完全实现方案”。如果有问题，直接说。

术语第一次出现要解释，例如：

- micro-batch（微批，即流水线里一次处理和传输的一小组 token）。
- Memory Window（内存窗口，即两端都能访问或映射的一段共享内存区域）。
- Doorbell（门铃通知，即写完数据后通知对端读取）。
- DMA（直接内存访问，即设备搬数据时尽量不让 CPU 参与逐字节拷贝）。
- NUMA tensor parallel（NUMA 张量并行，即把同一层矩阵计算切到多个 NUMA 节点上并行算）。

不要写得像宣传稿。当前最稳妥的总体结论是：

```text
当前 Pipeline Brick 原型已经实现了双卡分层推理、多 sequence micro-batch 流水线、bf16 hidden states 传输和 NUMA 绑核，并在实机上相对 llama-parallel 取得 1.09x 到 1.26x 的端到端提升。它符合比赛方案的主干思路，但仍是软件原型，真实 CXL/NTB、DMA、硬件 Doorbell 和 NUMA tensor parallel 尚未落地。
```
