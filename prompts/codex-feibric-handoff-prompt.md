# Codex 接手提示词: Feibric 飞腾 CPU 推理比赛项目

请先完整阅读这份提示词，再阅读仓库里的代码和文档。你接手的是一个 llama.cpp 私有 fork，用于比赛实机验证，不是准备提交 upstream 的通用 PR。请默认用中文回答。遇到英文工程术语、论文术语、缩写或代码变量名时，第一次出现请用“英文原词/变量名 + 中文解释 + 一句话含义”的格式说明。

## 1. 当前项目目标

项目名建议口径:

```text
Feibric: 基于 CXL/RDMA 分层互联的飞腾 CPU 混合并行推理架构
```

当前比赛目标是在飞腾 CPU 实机上验证大模型推理的混合并行方案:

- Pipeline Parallelism / PP（流水线并行，即按层切分模型，每个 CPU 负责连续一段层）: stage 之间只传 hidden states。
- Tensor Parallelism / TP（张量并行，即同一层矩阵乘法拆给多个 NUMA rank 计算后做 all-reduce sum）: 当前实现是每个 CPU 内 NUMA TP=4。
- CXL / Compute Express Link（计算快速链路，即同 Linux Domain 内两颗 CPU 间通过 MCIO/CXL 承载共享内存访问）: 当前用于同系统内 CPU0->CPU1 或 CPU2->CPU3 的 hidden states 传输。
- RDMA / Remote Direct Memory Access（远程直接内存访问，即跨两台独立 Linux 机器通过 IB 卡直接传内存数据）: 当前用于 Domain A 的 CPU1 -> Domain B 的 CPU2。
- StreamingLLM（流式稀疏注意力，即 decode 后期只保留 sink tokens 和 recent tokens 参与 attention）: 当前替代 EMA，作为稀疏注意力优化路径。

重要项目记忆:

- 后续主实验统一使用 F16 模型，不再默认使用 Q4_K_M。
- 默认模型路径按实机部署位置写。当前 zip 解压后模型在仓库内相对路径:

```text
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf
```

- TP=4 运行需要同目录存在:

```text
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.tp0.gguf
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.tp1.gguf
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.tp2.gguf
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.tp3.gguf
```

如果实机额外挂载了 `/models/qwen3-4b-f16/`，命令可以用 `/models/...`。如果模型就在当前仓库下，必须用 `models/...`，否则会报 `failed to open GGUF file`。

## 2. 实机拓扑和推荐表述

当前最终实机方案是两台独立 Linux 机器，每台机器有两颗飞腾 CPU。每台机器内部两颗 CPU 通过 MCIO/CXL 互联，两个 Linux Domain 之间通过 IB RDMA 连接。

推荐描述:

```text
Domain A / Linux 1:
  CPU A0: embedding + layers [0,9)  + CPU 内 NUMA TP=4
  CPU A1: layers [9,18)             + CPU 内 NUMA TP=4
  CPU A0 -> CPU A1: CXL/MCIO 域内流水线

Domain B / Linux 2:
  CPU B0: layers [18,27)            + CPU 内 NUMA TP=4
  CPU B1: layers [27,36) + lm_head  + CPU 内 NUMA TP=4
  CPU B0 -> CPU B1: CXL/MCIO 域内流水线

Domain A -> Domain B:
  CPU A1 -> CPU B0: IB RDMA 跨系统流水线
```

不要把当前实现说成完整 2D SUMMA TP。当前实现是:

```text
四阶段 pipeline + 每 CPU 内 1D NUMA TP=4
```

2D SUMMA（二维矩阵分块并行算法，即把 rank 排成二维网格并做行/列通信来完成矩阵乘）只属于报告里的理论方向，当前代码没有实现完整 2D SUMMA。

## 3. 当前代码状态

主要入口:

```text
tools/pipeline-brick/pipeline-brick.cpp
```

关键文档:

```text
progress/feibric-current-progress.md
docs/feibric-real-machine-run-commands.md
docs/pipeline-brick-experiment-report.md
```

关键模型切分工具:

```text
tools/gguf-split/gguf-split.py
```

当前已经实现:

- `--domain-mode single|dual`
  - `single`: 单 Linux Domain，两颗 CPU 做两阶段 head/tail pipeline。
  - `dual`: 双 Linux Domain，四阶段 stage0/stage1/stage2/stage3 pipeline。
- 单 Domain 两阶段 pipeline:
  - head: embedding + layers [0,18)
  - tail: layers [18,36) + final norm + lm_head
- 四阶段 pipeline:
  - stage0: embedding + layers [0,9)
  - stage1: layers [9,18)
  - stage2: layers [18,27)
  - stage3: layers [27,36) + final norm + lm_head
- `--tp-size 1|4`
  - `--tp-size 1`: 不启用真实 TP。
  - `--tp-size 4`: 每个 stage 内 fork 4 个 TP rank，每个 rank 绑定该 CPU 的一个 NUMA 节点。
- TP 权重路径:
  - rank0 自动加载 `.tp0.gguf`
  - rank1 自动加载 `.tp1.gguf`
  - rank2 自动加载 `.tp2.gguf`
  - rank3 自动加载 `.tp3.gguf`
- `--stream-kv --stream-kv-sink 16 --stream-kv-recent 128`
  - prefill 阶段保持 dense。
  - decode 后期启用 StreamingLLM 稀疏 attention。
- `ib-rdma` transport 框架:
  - 使用 ibverbs RC QP 的 SEND/RECV 语义。
  - 不依赖 TCP/IP 控制面。
  - 通过本地文件手工交换 QP 信息。
- `zni-rdma` 目前只是接口占位，真实 SDK 适配未完成。
- EMA 运行路径已从 pipeline-brick CLI 移除。`--ema-kv` 会报错并提示使用 `--stream-kv`。

当前没有完成或不要夸大的内容:

- 没有完成完整 2D SUMMA TP。
- 没有完成 CNN AttentionPredictor。
- ZNI RDMA 没有真实 SDK 版本。
- IB RDMA 代码框架已有，但真实链路仍要在实机上验证。
- 当前 `stream-kv` 是固定 sink + recent 策略，不是预测器。

## 4. Prompt 和公平对比注意事项

长 prompt 文件应该在实机创建:

```sh
mkdir -p prompts

cat > prompts/long_1k_prompt.txt <<'EOF'
请围绕 llama.cpp、CPU 大模型推理、NUMA 亲和性、流水线并行、KV cache 管理、prefill 和 decode 阶段的性能瓶颈，写一段技术分析。要求说明纯 CPU 推理为什么通常受内存带宽和矩阵乘法效率限制，说明长上下文下 KV cache 访问为什么会影响吞吐，说明多 sequence 并发时 batching 如何提高硬件利用率。然后进一步分析在双飞腾 CPU 系统中，将 Qwen3-4B 按层切分为 head Brick 和 tail Brick 的意义：head Brick 负责 embedding 和前 18 层，tail Brick 负责后 18 层、final norm 和 lm_head；两端之间只传 hidden states，不传完整 KV cache，也不通过 TCP 或 RPC。请解释这种方式相比单进程完整模型推理的潜在优势，包括两颗 CPU 同时工作、内存压力分散、每个 Brick 只维护本地层的 KV cache，以及通过 micro-batch 让 head 和 tail 在不同 token 批次上重叠执行。还要说明这种方案的局限，例如 hidden states 传输仍有开销，BF16 可以减少传输量但会带来格式转换，真实 CXL 或 NTB Memory Window 和 Doorbell 才能更接近硬件方案，当前 mmap shared memory 只能作为单系统验证。最后请用比较客观的语气总结：在 batch 较小、prompt 较短时，流水线并行可能被调度和传输开销抵消；在 prompt 较长、batch 较大、micro-batch 设置合理时，流水线并行才更可能体现优势。
EOF
```

`llama-pipeline-brick` 不是只喂这段中文 prompt。为了和 `llama-parallel` baseline 对齐，代码里有固定前缀:

```text
PIPELINE_PARALLEL_SYSTEM_PROMPT
```

实际输入格式是:

```text
固定英文 few-shot 前缀
+
User:
中文长 prompt
Assistant:
```

日志里看到 `system tokens = 273` 或 `prompt tokens=643` 是正常的。不要误以为只输入了中文文本。

## 5. 运行命令文档

实机运行命令全部写在:

```text
docs/feibric-real-machine-run-commands.md
```

该文档已经按用户要求改成“除 `LONG_PROMPT` 外，不使用 export 变量”。模型路径应写成仓库内相对路径 `models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf`，除非实机明确把模型挂载到了 `/models/qwen3-4b-f16/`。

里面包括:

- 单 Domain TP 消融:
  - `single_domain_pipeline_tp1`
  - `single_domain_pipeline_tp4`
- 双 Domain full:
  - CXL + IB RDMA + 4-stage pipeline + TP=4 + stream-kv
- 不使用 TP:
  - CXL + IB RDMA + 4-stage pipeline + stream-kv + `--tp-size 1`
- 不使用稀疏:
  - CXL + IB RDMA + 4-stage pipeline + TP=4，不加 `--stream-kv`
- baseline:
  - `llama-parallel`

`--hidden-dtype bf16`（隐藏状态传输数据类型，即 stage 之间传 hidden states 时的格式）只影响 stage 间中间激活 payload，不改变模型权重格式。Qwen3-4B 的 `n_embd=2560`，所以:

```text
F32:  2560 x 4 = 10240 bytes/token
BF16: 2560 x 2 = 5120 bytes/token
```

## 6. 最近打包状态

已经生成过两个包:

```text
/home/zhaowei/feibric-llama-src-20260617.tar.gz
/mnt/e/feibric-llama-src-f16-tp4-20260617.zip
```

Windows E 盘 zip 路径:

```text
E:\feibric-llama-src-f16-tp4-20260617.zip
```

zip 校验结果:

```text
源码文件: 已包含
docs/feibric-real-machine-run-commands.md: 已包含
tools/gguf-split/gguf-split.py: 已包含
models/qwen3-4b-f16/*.tp0-3.gguf: 已包含 4 个
完整 F16 原始 gguf: 未包含
Q4 模型: 未包含
.git/build/bench-logs/.venv/logs: 未包含
```

注意: zip 中模型在:

```text
llama.cpp/models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.tp0.gguf
...
tp3.gguf
```

如果实机命令按 `/models/qwen3-4b-f16/...` 跑，需要把模型目录放到 `/models/qwen3-4b-f16/`，或者改命令里的模型路径为 `models/qwen3-4b-f16/...`。

## 7. 构建和快速检查

构建:

```sh
cmake --build build --target llama-pipeline-brick llama-parallel -j 64
```

如果实机是新解压环境，可能需要先跑 CMake configure。优先保持简单，不要引入复杂新依赖:

```sh
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF
```

单 Domain TP 消融的核心命令见 `docs/feibric-real-machine-run-commands.md` 第 1 节。这个实验只用第一台机器两颗 CPU:

```text
--domain-mode single --tp-size 1
--domain-mode single --tp-size 4
```

双 Domain full 实验见第 3 节，启动顺序:

```text
机器 2: stage3
机器 2: stage2
机器 1: stage1
手工交换 RDMA peer info 文件
机器 1: stage0
```

## 8. 当前实机 TP=4 卡住问题

用户在实机 `/root/yzw-test/llama5/llama.cpp` 上运行单 Domain pipeline + TP=4:

```sh
/usr/bin/time -p -o bench-logs/single_domain_pipeline_tp4.time \
./build/bin/llama-pipeline-brick \
  --domain-mode single \
  --transport cxl \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --prompt "$LONG_PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --quiet \
  > bench-logs/single_domain_pipeline_tp4.out \
  2> bench-logs/single_domain_pipeline_tp4.log
```

现象:

- 运行约 361 秒后用户手动 Ctrl-C。
- 日志显示 head/tail 各自 fork 出 rank 0/4 到 rank 3/4。
- 模型路径已经不是问题。日志中的 `.gguf` 会被代码自动转换为 `.tp0.gguf` 到 `.tp3.gguf`。
- 每个 rank 都初始化了 KV cache，head 是 layers `[0,18)`，tail 是 layers `[18,36)`。
- 日志停在初始化后，没有看到最终 `pipeline-brick perf`。
- 因为命令带 `--quiet`，正常的 `sent prefill` / `recv pos` 进度日志被关闭，所以不能只凭日志判断一定死锁。

当前判断:

- 这不是模型路径错误，也不是 `.gguf` 没有改成 4 个 shard 的问题。
- `numa_tp=1` 出现在 rank 日志里是预期行为。TP 子进程内部把旧的 `numa_tp` 提示参数清成 1，真实 TP 大小看 `tp_size=4` 和 rank 日志。
- 361 秒没有结果偏异常。需要先区分是首个 prefill 计算太慢，还是卡在 TP all-reduce sum（求和规约，即 4 个 rank 把局部结果同步求和）或 transport（传输层，即 hidden states/token 在 head/tail 间传递）。
- 当前 TP all-reduce 是共享内存 busy-wait 原型，不是成熟通信库；如果 4 个 rank 没有完全同序进入同步点，可能死等。

下一步先跑最小 smoke，不要直接跑正式 64 token:

```sh
cd /root/yzw-test/llama5/llama.cpp
mkdir -p bench-logs

/usr/bin/time -p -o bench-logs/tp4_smoke.time \
./build/bin/llama-pipeline-brick \
  --domain-mode single \
  --transport cxl \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --prompt "你好" \
  --ctx-size 512 \
  --threads 64 \
  --n-predict 1 \
  --parallel 1 \
  --prefill-chunk 8 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  > bench-logs/tp4_smoke.out \
  2> bench-logs/tp4_smoke.log
```

另开窗口观察:

```sh
tail -f bench-logs/tp4_smoke.log
```

判断标准:

- 如果 smoke 很快出现 `sent prefill`、`recv pos` 并输出结果，说明大命令可能只是太重且 `--quiet` 没进度。
- 如果 smoke 也停在初始化后，没有 `sent prefill` 或 `recv pos`，优先怀疑 `llama_decode` 内部计算或 TP all-reduce。
- 如果有 `sent prefill` 但没有 `recv pos`，优先看 CXL/shared-window transport。
- 如果有 `recv pos` 但 tail 后续不动，优先看 tail decode 或 tail 侧 all-reduce。

卡住时查看 CPU 和等待点:

```sh
ps -eLo pid,tid,psr,pcpu,stat,wchan:24,comm,args | grep -E 'llama-pipeline-brick|PID'
```

如果允许装/用 gdb，抓所有 rank 的栈:

```sh
for p in $(pgrep -f 'llama-pipeline-brick'); do
  echo "===== PID $p ====="
  gdb -q -p "$p" -ex 'thread apply all bt 6' -ex detach -ex quit
done > bench-logs/tp4_stacks.txt 2>&1
```

重点看栈是否停在:

- `ggml_compute_forward_all_reduce_sum` 或 `ggml_tp_barrier_wait`: TP 同步问题。
- `ggml_compute_forward_mul_mat`: 主要是在矩阵乘，可能是慢。
- `tp_transport_proxy::recv` 或 transport `recv`: head/tail 传输或广播等待问题。

## 9. 下一步优先级

如果用户开新会话后继续比赛工作，优先按这个顺序推进:

1. 先确认实机能解压、构建、找到模型分片。模型如果在仓库内，命令使用 `models/qwen3-4b-f16/...`。
2. 先跑上面的 `tp4_smoke`，判断 TP=4 是慢还是卡死。
3. 如果 smoke 卡住，抓 `ps` 和 `gdb` 栈，优先定位 all-reduce、transport、还是矩阵乘。
4. 如果 smoke 跑通，再跑单 Domain `--tp-size 1` 和 `--tp-size 4`，确认 TP 消融是否能跑通并记录速度。
5. 注意 zip 包只包含 4 个 TP shard，没有完整 `Qwen3-4B-Instruct-2507-F16.gguf`。如果要跑 `--tp-size 1`，需要把完整 F16 `.gguf` 补到同目录。
6. 检查 IB 设备:

```sh
ibv_devinfo
ibstat
rdma link
```

7. 跑双 Domain stage3/stage2/stage1/stage0 smoke，先 `n-predict 1` 或 `8`，不要一开始就跑完整 64 token。
8. 如果 RDMA 出错，优先看:
   - `bench-logs/*stage1.log`
   - `bench-logs/*stage2.log`
   - `/tmp/feibric-rdma/*.local`
   - `/tmp/feibric-rdma/*.peer`
9. 如果用户要写报告，使用谨慎口径:
   - 已实现: 四阶段 pipeline、CXL/MCIO 域内传输、stream-kv、每 CPU 内 1D NUMA TP=4、IB RDMA 代码框架。
   - 待验证: IB RDMA 实机性能、ZNI SDK。
   - 未实现: 完整 2D SUMMA、CNN AttentionPredictor。

## 10. 回答风格要求

- 不要附和用户的错误判断，要客观指出。
- 先读代码再下结论。
- 不要默认改 Q4 相关命令。后续默认 F16。
- 不要把当前 CXL 代码说成“实现了 CXL 协议栈”。更准确说法是: 代码使用共享内存窗口和 NUMA 绑定；在当前飞腾硬件上，跨 CPU 内存访问由 MCIO/CXL 硬件承载。
- 不要把 `--hidden-dtype bf16` 解释成模型权重 BF16。它只是 stage 间 hidden states 传输格式。
- 不要把 StreamingLLM 说成预测器。它是固定 sink + recent 稀疏策略。
- 做代码修改前先给用户简短说明；修改后必须构建或至少说明未能构建。
