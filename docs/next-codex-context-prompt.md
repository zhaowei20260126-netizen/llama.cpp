# 给下一个 Codex 会话的上下文 Prompt

## 1. 对话与项目背景

我正在参加研电赛飞腾赛道赛题 3：高效模型推理。当前项目基于 `llama.cpp` 私有 fork 开发，目标不是提交 upstream，而是在飞腾双 CPU 实机上验证一个 Pipeline Brick 推理原型。

已有方案文档包括：

- `研电赛清单.pdf`
- `基于板间桥接的流水线并行推理架构研究方案.docx`
- `Pipeline_Brick相比GPU的核心优势分析.docx`
- `llama.cpp_CPU稀疏注意力内核调研.docx`
- `Tang 等 - 2024 - Quest Query-Aware Sparsity for Efficient Long-Context LLM Inference.pdf`

最初考虑过 Quest / Sparse Attention，但后续计划变更，当前不做稀疏注意力，也不做预测器。重点改成：

```text
片间流水线并行 + 多 batch/micro-batch 调度 + hidden states 低开销传输
```

真实实机环境：

```text
双飞腾 CPU
一个 Linux 系统能看到 8 个 NUMA node
第一张卡：NUMA 0-3，CPU 0-63
第二张卡：NUMA 4-7，CPU 64-127
总内存约 256GB
```

当前模型：

```text
models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
Qwen3-4B
36 层
hidden size = 2560
GGUF Q4_K_M 量化
模型文件约 2.4GB
```

## 2. 目前代码实现

主要改动在：

```text
tools/pipeline-brick/pipeline-brick.cpp
```

新增工具目标：

```text
llama-pipeline-brick
```

构建命令：

```sh
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF

cmake --build build -j 32 --target llama-pipeline-brick llama-parallel
```

当前实现的是双 Brick 原型：

```text
head Brick:
  token embedding + layers [0,18)

tail Brick:
  layers [18,36) + final norm + lm_head
```

head 到 tail 只传 hidden states，不传完整 KV cache。tail 采样得到 token 后，通过反向通道把 token/control 小消息传回 head。

当前支持：

- `--single-system`：同一个 Linux 系统下 fork 出 head/tail 两个进程。
- `--head-numa 0-3`：head 绑定第一张卡 CPU 0-63。
- `--tail-numa 4-7`：tail 绑定第二张卡 CPU 64-127。
- head-to-tail shared buffer 绑定到 tail 侧 NUMA 4-7。
- tail-to-head shared buffer 绑定到 head 侧 NUMA 0-3。
- shared memory / mmap ring buffer 模拟 CXL/NTB Memory Window。
- poll 方式模拟 Doorbell。
- `--parallel N` 支持多条 sequence。
- `--prefill-chunk N` 支持 prefill 阶段 micro-batch。
- `--hidden-dtype bf16|f32`，默认 bf16。
- `--quiet` 关闭逐 micro-batch 日志，避免影响性能测试。

hidden states 传输格式：

```text
Qwen3 hidden size = 2560
f32:  2560 x 4 = 10240 bytes/token
bf16: 2560 x 2 = 5120 bytes/token
```

bf16 只作为传输格式。tail 收到后会转回 f32，再填入 `batch.embd`。

## 3. Micro-batch 的实际含义

Micro-batch（微批，即流水线里一次传递的一小批 token）不等于整个 batch。

假设：

```text
parallel = 4
prefill-chunk = 32
每条 sequence 长度 = 100 token
```

整个 batch 是：

```text
4 条 sequence x 100 token = 400 token
```

prefill 阶段一个 micro-batch 是：

```text
4 条 sequence x 每条 32 token = 128 token
```

执行方式：

```text
第 1 个 micro-batch:
  seq0 token 0-31
  seq1 token 0-31
  seq2 token 0-31
  seq3 token 0-31

第 2 个 micro-batch:
  seq0 token 32-63
  seq1 token 32-63
  seq2 token 32-63
  seq3 token 32-63
```

decode 阶段每轮每条 sequence 只有一个当前 token，因此：

```text
decode micro-batch = parallel
```

例如 `parallel=4` 时，每轮 decode micro-batch 是 4 个 token。

## 4. 流水线并行是否真的实现

这里必须客观表述。

当前代码不是完整意义上的 decode 流水线。准确说：

```text
Prefill 阶段：有 micro-batch 流水线重叠。
Decode 阶段：没有真正的 head/tail 流水线重叠，是锁步交替的分层推理。
```

prefill 阶段：

```text
head 算 chunk0 前半层 -> 发送 hidden
head 立即算 chunk1 前半层

tail 收 chunk0 hidden -> 算后半层
tail 再收 chunk1 hidden -> 算后半层
```

这部分有 head/tail overlap。

decode 阶段：

```text
head step N:
  用当前 token 算 layers [0,18)
  发送 hidden 给 tail
  阻塞等待 tail 返回 token_N+1

tail step N:
  收 hidden
  算 layers [18,36)
  greedy sample 得到 token_N+1
  发回 head
```

head 的 step N+1 必须等 tail 的 step N token 输出。因此对于同一条 sequence 的标准自回归 decode，当前实现是：

```text
head -> tail -> head -> tail
```

不是：

```text
head 和 tail 在不同 decode step 上持续重叠
```

当前最诚实的表述：

```text
当前 Pipeline Brick 原型实现了按层切分的双 Brick 推理。
在 prefill 阶段，head/tail 通过 micro-batch 形成流水线重叠；
在 decode 阶段，由于自回归 token 依赖，当前实现为 head/tail 锁步交替执行，不存在真正的流水线重叠。
```

## 5. 与方案文档的符合程度

方案文档主线是：

```text
按层切分模型
多个 Brick 之间做流水线并行
通过 PCIe NTB / CXL Memory Window 传 hidden states
Doorbell 低延迟通知
多请求 micro-batch 填满流水线
```

当前已经实现：

- Qwen3-4B 按层切分。
- head/tail 两进程分段运行。
- head/tail 分别绑定不同 NUMA/CPU 卡。
- head-to-tail 只传 hidden states。
- tail-to-head 只传 token/control。
- prefill micro-batch 流水线。
- bf16 hidden states 传输。
- 与 `llama-parallel` 对齐的 prompt template。

部分实现或模拟：

- Memory Window：当前是 mmap shared memory 模拟。
- Doorbell：当前是 poll 模拟。
- CXL/NTB：未接真实硬件驱动。
- 零拷贝：不是严格零拷贝，head 仍需写 shared buffer，tail 仍需 bf16->f32 转换。

未实现：

- 真实 CXL/NTB 驱动。
- DMA。
- 硬件 Doorbell 中断。
- 真正 NUMA 内 tensor parallel。
- predictor。
- sparse attention。
- decode 阶段真正异步流水线。
- dynamic continuous batching。

## 6. NUMA 内张量并行的判断

方案文档里原始主线并没有明确要求 NUMA 内 tensor parallel。后面老师提到过：

```text
NUMA 级张量并行 + 节点级流水线并行 + CXL 传输优化 + 预测器
```

但结合当前 Qwen3-4B Q4_K_M 场景，我的判断是：现在不应该优先做 NUMA 内张量并行。

原因：

- Qwen3-4B Q4 权重只有约 2.4GB，单张 CPU 卡内存完全放得下。
- 每个 Brick 已经绑定 64 个 CPU core，OpenMP 已经在做多线程计算。
- 真正 NUMA tensor parallel 需要切 Q/K/V、FFN up/gate/down，并做 reduce-sum。
- 每层都要跨 NUMA 同步，可能得不偿失。
- llama.cpp CPU backend 当前没有现成的 CPU-NUMA tensor device。

可以在汇报中说：

```text
当前实现的是 NUMA affinity，不是真正 NUMA tensor parallel。
对于 Qwen3-4B Q4 这种中小模型，收益不一定高，且会引入每层跨 NUMA reduce 和同步开销。
当前优先验证板间/片间流水线并行和 hidden states 传输。
```

## 7. 实验命令与结果

### 7.1 batch=2 短 prompt

Baseline：

```sh
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

Pipeline：

```sh
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

结果：

```text
baseline:
  real 47.08
  user 3636.43
  sys 926.12
  prompt tokens/request = 287
  gen tokens total = 128

pipeline:
  real 37.25
  user 2420.29
  sys 52.97
  prompt tokens/request = 287
  gen tokens total = 128

speedup = 47.08 / 37.25 = 1.26x
```

### 7.2 batch=4 长 prompt

长 prompt 文件：

```text
prompts/long_1k_prompt.txt
```

Pipeline 侧当前没有 `--prompt-file`，需要：

```sh
LONG_PROMPT="$(cat prompts/long_1k_prompt.txt)"
```

Baseline：

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

Pipeline：

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

结果：

```text
baseline:
  real 100.49
  user 8046.93
  sys 1390.54
  prompt tokens/request = 643
  gen tokens total = 256

pipeline:
  real 91.85
  user 6939.97
  sys 86.07
  prompt tokens/request = 643
  gen tokens total = 256

speedup = 100.49 / 91.85 = 1.09x
```

注意：这两个实验都是单次结果。正式报告建议至少跑 3 次，给均值和标准差。

## 8. 当前结果如何解释

当前结果说明 Pipeline Brick 原型确实跑通，并在两个场景下比 `llama-parallel` 快：

```text
batch=2 短 prompt: 1.26x
batch=4 长 prompt: 1.09x
```

但不能简单说“decode 阶段也有流水线收益”。更准确的归因是：

- prefill 阶段有 micro-batch 流水线重叠。
- head/tail 分别绑定不同 NUMA，降低了部分内存访问混乱。
- head/tail 分层后，每个进程只维护自己层范围的 KV cache。
- bf16 hidden states 减少了传输量。
- decode 阶段仍然锁步交替，会有通信和同步开销。

应该增加消融实验：

1. 短 prompt + 长生成，例如 prompt 很短、`n_predict=512/1000`，验证 decode 主导场景。
2. 不同 `--prefill-chunk`：16、32、64、128。
3. 不同 `--parallel`：1、2、4、8、16。
4. 每组至少 3 次，报告均值和方差。
5. 如果能做，增加“同样分层+NUMA 绑定但关闭 prefill 流水线”的版本，用来区分流水线本身的贡献。

## 9. llama-parallel 的实现理解

`llama-parallel` 位于：

```text
examples/parallel/parallel.cpp
```

它不是分层流水线，而是：

```text
一个完整模型上下文
多个 client/sequence
把多个 sequence 的 token 放进一个 llama_batch
每次 llama_decode 跑完整 layers 0-35
```

例如 `-np 4 -ns 4`：

```text
seq0 当前 token
seq1 当前 token
seq2 当前 token
seq3 当前 token
```

会合成一个 batch，一次送入完整模型。

所以：

```text
llama-parallel = batch 并行 / continuous batching
pipeline-brick = 分层执行 + prefill micro-batch pipeline
```

二者不是同一种并行。

## 10. 已生成的报告和图

已有实验报告：

```text
docs/pipeline-brick-experiment-report.md
```

这个报告已经记录了：

- 实验目标
- 双卡 CPU/NUMA 结构
- Pipeline Brick 原理图
- prefill micro-batch 时序图
- batch=2 实验结果
- batch=4 长 prompt 实验结果
- 初步分析和边界

需要注意：报告里关于“流水线并行”的表述后续应修正得更严谨，明确：

```text
prefill 阶段有流水线重叠；
decode 阶段当前没有真正流水线重叠。
```

还生成过一张流程图图片，默认保存在：

```text
/root/.codex/generated_images/019ec0c4-29bd-7620-b7ba-c697627376c2/
```

## 11. 下一个 Codex 会话应优先做什么

如果继续开发代码，优先级建议：

1. 修正报告，把 decode 阶段不是流水线并行这点写清楚。
2. 增加短 prompt + 长生成实验，验证 decode 主导场景。
3. 增加重复实验脚本，每组跑 3 次，统计均值/标准差。
4. 给 `llama-pipeline-brick` 增加 `--prompt-file` 参数，避免长 prompt 必须通过 shell 变量传入。
5. 如果要优化 decode，方向不是单 sequence 伪流水线，而是：
   - 多请求 ready queue
   - 异步 decode scheduler
   - 更大的 `parallel`
   - 或 speculative decoding / predictor
6. 不建议优先做 NUMA 内 tensor parallel，除非老师明确要求或模型扩大到更大规模。

## 12. 回答风格要求

用户希望：

- 默认用中文回答。
- 客观回答，不要附和。
- 不要假设用户已经懂所有术语。
- 第一次出现英文工程术语时，用“英文原词 + 中文解释 + 一句话含义”的方式说明。
- 对 Sparse-vLLM、KV cache、AttentionPredictor、offload、CUDA、kernel、prefetch、hot tokens、packed view、lease、score buffer 等词，要用短句和例子解释。
- 代码改动前先读相关文件。
- 不要说“实现了完整流水线并行”这种过度结论。

当前最重要的技术判断：

```text
Pipeline Brick 当前已经跑通，并有阶段性性能收益。
但 decode 阶段不是流水线并行。
当前加速主要来自 prefill micro-batch pipeline、NUMA 绑定、bf16 hidden states 和分层执行带来的局部性。
```

## 13. 最新接手补充: EMA KV 筛选开发中断点

本节是 2026-06-15 最新补充，用于下一个 Codex 会话快速接手。上一轮正在做 `EMA KV selection`，也就是用指数滑动平均分数筛选关键 KV token，在 decode 阶段减少每步 attention 参与计算的历史 token 数。这个功能还没有完成，也没有编译通过验证，不能当成可用功能。

### 13.1 这次 EMA 的目标

目标不是减少 KV cache 内存占用，而是减少 decode attention 的计算量。

具体含义：

```text
完整 KV cache 仍然保留。
每一步 decode 时，根据上一轮统计到的重要性分数，选择一部分 KV token。
attention 只对选中的 KV token 做 K x Q 和 V x attention 计算。
如果 selector 没准备好，回退到 dense attention。
```

计划参数：

```text
--ema-kv
--ema-kv-keep 512
--ema-kv-recent 256
--ema-kv-sink 16
--ema-alpha 0.95
--ema-sync
```

策略是：

```text
强制保留开头 sink token。
强制保留最近 recent token。
剩余名额用 EMA 分数选 top token。
总保留数不超过 keep。
```

### 13.2 已经改过的文件

上一轮已经动过这些文件：

```text
include/llama.h
src/llama-cparams.h
src/llama-context.cpp
src/llama-graph.h
src/llama-graph.cpp
src/models/qwen3.cpp
tools/pipeline-brick/pipeline-brick.cpp
```

另外 `git diff --stat` 里还能看到一些早前项目改动，例如：

```text
src/llama-model.cpp
src/llama-model.h
tools/CMakeLists.txt
docs/local-official-benchmark.md
```

这些不一定都是 EMA 本次新改的，接手时需要用 `git diff` 具体确认。

### 13.3 已经实现到哪里

`include/llama.h` 里新增了实验性回调和参数：

```cpp
typedef int32_t (*llama_ema_kv_select_callback)(
        void * user_data,
        int32_t layer,
        int32_t seq_id,
        int32_t pos,
        int32_t n_keep,
        int32_t * indices);
```

含义：graph 在构建每层 attention 输入时，通过这个 callback 向 pipeline-brick 侧索要当前层、当前 sequence、当前位置要保留的 KV token 下标。

`llama_context_params` 和内部 `llama_cparams` 增加了：

```text
ema_kv_enabled
ema_kv_active
ema_kv_keep
ema_kv_recent
ema_kv_sink
ema_kv_alpha
ema_kv_select
ema_kv_select_user_data
```

`src/llama-context.cpp` 已经做了这些事：

```text
把 public params 复制到 cparams。
EMA 开启时禁用 Flash Attention。
新增 llama_set_ema_kv_active(ctx, active)。
新增 llama_set_ema_kv_select_callback(ctx, callback, user_data)。
graph_max_nodes 里给 EMA sparse path 预留了额外节点。
```

`src/llama-graph.h` 和 `src/llama-graph.cpp` 已经尝试加入 sparse gather 路径：

```text
llm_graph_input_attn_kv 增加 self_ema_kv_idxs 和 self_ema_kq_mask。
set_input 时调用 cparams.ema_kv_select 填充 selected KV 下标。
build_ema_sparse_k/build_ema_sparse_v 尝试用 ggml_get_rows gather K/V。
attention 里如果 has_ema_kv()，就用 gather 后的局部 K/V。
```

这部分意图是做真实 gather，而不是只靠 mask。也就是说，目标是让 attention 的 KV 长度真的从完整上下文长度降到 `ema_kv_keep`。

`src/models/qwen3.cpp` 已经做了局部调整：

```text
普通 dense 路径仍然复用一个 inp_attn。
EMA active 时，每一层单独 build_attn_inp_kv(il)。
```

原因：EMA selector 是按 layer 独立的，每层需要自己的 selected KV 下标。

`tools/pipeline-brick/pipeline-brick.cpp` 已经开始接入：

```text
新增命令行参数解析。
新增 ema_kv_state 类。
新增异步 worker 设计。
新增 ema_eval_callback，用 cb_eval 读取 kq_soft_max-* 张量。
run_head 已经部分接入 EMA。
run_tail 还没有收尾。
```

### 13.4 最关键的未完成点

当前代码很可能不能编译。不是算法结果不好，而是中途被打断，接线没收完。

已知高风险点：

1. `run_tail` 还没有更新

当前 `make_context` 的签名已经改成：

```cpp
static llama_context * make_context(
        llama_model * model,
        const pipeline_args & args,
        bool embeddings,
        ema_kv_state * ema_state);
```

但是 `run_tail` 里还有旧调用：

```cpp
llama_context * ctx = make_context(model, args, false);
```

这会直接编译失败。tail 也需要创建自己的 `ema_kv_state`，因为 tail 负责 layers `[18,36)`，也要维护后半层的 EMA 分数和 selected KV。

2. `ema_kv_state` 定义位置可能不对

`make_context` 使用了 `ema_kv_state::select_callback`，但 `ema_kv_state` 类定义在后面。需要做其中一种处理：

```text
把 ema_kv_state 类移动到 make_context 前面；
或者先做完整前向声明，并避免在声明前访问静态成员。
```

最稳妥是把 `ema_kv_state` 和 `ema_eval_callback` 移到 `make_context` 之前。

3. atomic 统计可能需要改成 fetch_add

如果代码里有这种写法：

```cpp
std::atomic<int64_t> x;
x += n;
```

C++17 下可能不稳，建议改成：

```cpp
x.fetch_add(n, std::memory_order_relaxed);
```

4. `build_ema_sparse_v` 的 shape 需要重点检查

当前 sparse K/V gather 是最容易出问题的地方。需要确认：

```text
K cache reshape 后 get_rows 的行维是否真的是 token 维。
V cache reshape 后 get_rows 的行维是否真的是 token 维。
gather 后 reshape/permutation 的 layout 是否和原 attention path 期望一致。
```

建议先用小 prompt、小 batch 编译运行，必要时加断言打印 tensor shape。

5. `llama_set_ema_kv_active` 每次切换会触发 sched_need_reserve

这能保证 graph 重新 reserve，但 decode 每步切换可能有额外开销。第一版可以先接受，后续再优化成固定 graph。

### 13.5 接手后建议的收尾顺序

不要一上来跑性能。先保证能编译和 dense 路径不回归。

建议顺序：

```sh
git diff -- include/llama.h src/llama-cparams.h src/llama-context.cpp src/llama-graph.h src/llama-graph.cpp src/models/qwen3.cpp tools/pipeline-brick/pipeline-brick.cpp
```

然后：

```text
1. 修复 pipeline-brick.cpp 的编译错误。
2. 让 run_head 和 run_tail 都创建各自的 ema_kv_state。
3. tail 端也要在 prefill 时 dense，在 decode 第 2 步以后尝试 sparse。
4. 编译 llama-pipeline-brick。
5. 不开 --ema-kv 跑一次，确保原功能不回归。
6. 开 --ema-kv 跑 n_predict=4 的小测试。
7. 再跑长 prompt + batch=4 的性能测试。
```

编译命令：

```sh
cmake --build build -j 32 --target llama-pipeline-brick
```

如果 `llama-parallel` 也要重新编译：

```sh
cmake --build build -j 32 --target llama-parallel
```

如果 `build` 目录不存在，先在项目根目录重新 configure：

```sh
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_APP=OFF
```

注意：必须在 `llama.cpp` 项目根目录执行，不要在 `bench-logs` 目录里执行。

### 13.6 EMA 冒烟测试命令

先用小输出验证是否能跑：

```sh
mkdir -p bench-logs

export MODEL=models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
export PROMPT="请用三句话介绍一下 llama.cpp 是什么。"

./build/bin/llama-pipeline-brick \
  --single-system \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --ctx-size 2048 \
  --threads 64 \
  --n-predict 16 \
  --parallel 2 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --ema-kv \
  --ema-kv-keep 512 \
  --ema-kv-recent 256 \
  --ema-kv-sink 16 \
  --ema-alpha 0.95 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  > bench-logs/pipeline_p2_ema_smoke.out \
  2> bench-logs/pipeline_p2_ema_smoke.log
```

如果这一步失败，不要直接调性能，先看：

```sh
cat bench-logs/pipeline_p2_ema_smoke.log
```

### 13.7 长 prompt 性能测试命令

长 prompt 可以继续使用 shell 变量，避免 `--prompt-file` 尚未实现导致报错。

```sh
export MODEL=models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
export PROMPT='请围绕 llama.cpp、CPU 大模型推理、NUMA 亲和性、流水线并行、KV cache 管理、prefill 和 decode 阶段的性能瓶颈，写一段技术分析。要求说明纯 CPU 推理为什么通常受内存带宽和矩阵乘法效率限制，说明长上下文下 KV cache 访问为什么会影响吞吐，说明多 sequence 并发时 batching 如何提高硬件利用率。然后进一步分析在双飞腾 CPU 系统中，将 Qwen3-4B 按层切分为 head Brick 和 tail Brick 的意义：head Brick 负责 embedding 和前 18 层，tail Brick 负责后 18 层、final norm 和 lm_head；两端之间只传 hidden states，不传完整 KV cache，也不通过 TCP 或 RPC。请解释这种方式相比单进程完整模型推理的潜在优势，包括两颗 CPU 同时工作、内存压力分散、每个 Brick 只维护本地层的 KV cache，以及通过 micro-batch 让 head 和 tail 在不同 token 批次上重叠执行。还要说明这种方案的局限，例如 hidden states 传输仍有开销，BF16 可以减少传输量但会带来格式转换，真实 CXL 或 NTB Memory Window 和 Doorbell 才能更接近硬件方案，当前 mmap shared memory 只能作为单系统验证。最后请用比较客观的语气总结：在 batch 较小、prompt 较短时，流水线并行可能被调度和传输开销抵消；在 prompt 较长、batch 较大、micro-batch 设置合理时，流水线并行才更可能体现优势。'
```

baseline 命令：

```sh
mkdir -p bench-logs

/usr/bin/time -p -o bench-logs/baseline_parallel4_long1k.time \
./build/bin/llama-parallel \
  -m "$MODEL" \
  -ngl 0 \
  -t 128 \
  -c 8192 \
  -n 64 \
  -np 4 \
  -ns 4 \
  -p "$PROMPT" \
  --top-k 1 \
  > bench-logs/baseline_parallel4_long1k.out \
  2> bench-logs/baseline_parallel4_long1k.log
```

Pipeline Brick dense 命令：

```sh
/usr/bin/time -p -o bench-logs/pipeline_p4_long1k_microbf16.time \
./build/bin/llama-pipeline-brick \
  --single-system \
  --model "$MODEL" \
  --prompt "$PROMPT" \
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

Pipeline Brick + EMA 命令：

```sh
/usr/bin/time -p -o bench-logs/pipeline_p4_long1k_ema.time \
./build/bin/llama-pipeline-brick \
  --single-system \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --ema-kv \
  --ema-kv-keep 512 \
  --ema-kv-recent 256 \
  --ema-kv-sink 16 \
  --ema-alpha 0.95 \
  --quiet \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  > bench-logs/pipeline_p4_long1k_ema.out \
  2> bench-logs/pipeline_p4_long1k_ema.log
```

如果当前实机上的第二版 `pipeline-brick.cpp` 支持 `--transport cxl`，可以在 pipeline 命令中加：

```text
--transport cxl
```

如果 usage 里没有 `--transport cxl`，说明当前编译的是第一版或未合并 CXL 的文件，不要加这个参数。

### 13.8 目前已有的关键实验结果

同一长 prompt，4 个 sequence，每个 sequence 生成 64 token。

`llama-parallel` dense baseline 最新日志：

```text
Total prompt tokens: 2572, speed: 18.59 t/s
Total gen tokens:    256, speed:  1.85 t/s
Total speed:              20.44 t/s
Client time: about 138.36 s
```

Pipeline Brick 第一版 `ntb-mw`：

```text
pipeline-brick perf: inference time 133.19 s
total prompt tokens 2572, speed 19.31 t/s
total gen tokens 256, speed 1.92 t/s
total tokens 2828, speed 21.23 t/s
```

Pipeline Brick 第二版 `cxl`：

```text
pipeline-brick perf: inference time 128.68 s
total prompt tokens 2572, speed 19.99 t/s
total gen tokens 256, speed 1.99 t/s
total tokens 2828, speed 21.98 t/s
```

客观解释：

```text
CXL 版比第一版略快，但幅度不大。
当前 CXL 路径更像软件层的 shared memory/CXL 语义适配，不等于已经使用硬件 DMA 或真实 doorbell 中断。
主要收益仍然来自分层、NUMA 绑定、prefill micro-batch 和 bf16 hidden states。
EMA 目标是进一步优化 decode 阶段，但目前尚未完成。
```

### 13.9 计时口径

`/usr/bin/time -p` 输出：

```text
real: 墙钟时间，包含程序启动、模型加载、初始化、推理、退出。
user: 所有线程在用户态消耗的 CPU 秒数，可以大于 real。
sys: 所有线程在内核态消耗的 CPU 秒数，也可以和线程数相关。
```

端到端推理时间应优先看程序日志里的：

```text
pipeline-brick perf: inference time ...
```

这个统计口径是从真正开始喂 prompt 到生成完成，更接近比赛想看的端到端推理时间。

baseline `llama-parallel` 目前主要看日志里的 client time 和 total token speed。后续如果要更严格公平，需要在 baseline 里也加同样的纯推理计时点，或者只采用 `llama-parallel` 自带的 summary 作为 baseline 口径。

### 13.10 对下一个会话的提醒

不要把当前 EMA 功能描述成已经实现。准确说法是：

```text
EMA KV selection 的参数、context 字段、graph gather 路径和 head 侧接线已经初步写入；
tail 侧接线、编译修复、shape 验证、正确性测试和性能实验还没有完成。
```

如果继续实现，优先目标不是追求马上变快，而是：

```text
1. 能编译。
2. 不开 EMA 时结果和原 pipeline 保持一致。
3. 开 EMA 时能跑通小样例。
4. 日志能显示 dense fallback、selector ready/miss、平均保留 KV 数。
5. 再看长上下文 decode 是否有收益。
```
