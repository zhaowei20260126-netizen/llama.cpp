# Feibric 当前方案差异与实现进展

更新时间：2026-06-16

## 1. 结论

当前项目模型精度口径已经统一：后续主实验、命令和报告默认只使用 F16 模型，不再使用 Q4_K_M 量化模型作为主路径。除非明确做历史对照实验，否则不要再给 Q4 运行命令。

当前建议把主方案从原始的“多 Brick 板间流水线”调整为：

```text
Domain A / Linux 1:
  CPU A0: layers 0-8  + CPU 内 NUMA TP=4
  CPU A1: layers 9-17 + CPU 内 NUMA TP=4
  CPU A0 -> CPU A1: CXL/MCIO 域内流水线

Domain B / Linux 2:
  CPU B0: layers 18-26 + CPU 内 NUMA TP=4
  CPU B1: layers 27-35 + CPU 内 NUMA TP=4
  CPU B0 -> CPU B1: CXL/MCIO 域内流水线

Domain A -> Domain B:
  CPU A1 -> CPU B0: RDMA 跨系统流水线
```

对 Qwen3-4B 这类 36 层模型，更合理的层划分是：

```text
CPU A0: embedding + layers 0-8
CPU A1: layers 9-17
RDMA:   hidden states from layer 17 to layer 18
CPU B0: layers 18-26
CPU B1: layers 27-35 + final norm + lm_head
```

也就是说，当前主路径应表述为“四 CPU 分层流水线 + 每 CPU 内 1D NUMA TP=4”，而不是“完整 2D SUMMA TP + Domain 间流水线”。

原因很直接：当前实机和代码围绕“每个 CPU 负责一段连续层”展开；每个 CPU 内的 4 个 NUMA 节点再做张量并行，通信范围限制在单 CPU 内。完整 2D SUMMA 仍然不作为当前主路径。

## 2. 关键术语

Pipeline Parallelism / PP（流水线并行）：把模型按层切成几段，每个 CPU 负责一段层，段与段之间只传 hidden states。

Tensor Parallelism / TP（张量并行）：同一层的矩阵乘法被拆给多个 CPU/NUMA 一起算，算完需要做 all-reduce 等集合通信。

hidden states（层间隐藏状态）：Transformer 某一层输出的中间向量。本项目中流水线阶段之间只传它，不传完整 KV cache。

KV cache（键值缓存）：每层 attention 保存历史 token 的 K/V 数据。Pipeline Brick 的设计原则是 KV cache 留在本地层，不跨 CPU 或跨机器传输。

CXL / Compute Express Link（计算快速链路）：基于 PCIe 的一致性互联协议。当前同一个 Linux 系统里的两颗飞腾 CPU 通过 MCIO 互联并支持 CXL 语义，可用于共享内存窗口和跨 CPU load/store。

MCIO（高速板级互连接口）：当前机器里两颗飞腾 CPU 之间的物理连接路径。报告里可表述为 MCIO 承载 CXL 域内互联。

RDMA / Remote Direct Memory Access（远程直接内存访问）：两个独立 Linux 系统之间通过网卡直接传输内存数据，绕开普通 TCP/IP 协议栈。

CHI / Coherent Hub Interface（ARM 一致性互联）：飞腾 ARM 平台内部一致性互联应使用 CHI 表述，不建议继续写 QPI。QPI 是 Intel 平台术语。

StreamingLLM（流式稀疏注意力）：decode 阶段只保留开头 sink tokens 和最近 recent tokens 参与注意力计算，减少中间历史 token 的 attention 计算量。

## 3. 与原始方案的区别

原始文档《基于板间桥接的流水线并行推理架构研究方案.docx》的核心是：

```text
多块独立 Brick 板卡
-> 每块 Brick 负责连续若干层
-> Brick 间通过 PCIe NTB / CXL-like Memory Window 传 hidden states
-> 形成线性流水线
```

新实机方案的核心变为：

```text
两套独立 Linux 系统
-> 每套系统内有两颗飞腾 CPU
-> 系统内两 CPU 通过 MCIO/CXL 做小流水线
-> 两套系统之间通过 RDMA 做大流水线
```

保留下来的思想：

- 仍然按层切分模型。
- 仍然只传 hidden states，不传 KV cache。
- 仍然避免普通 TCP/IP 成为主推理通信路径。
- 仍然用 micro-batch / parallel requests 填充流水线。
- 仍然强调国产飞腾 CPU 上的大模型推理。

变化最大的地方：

- 原始方案是多块低功耗 Brick 物理堆叠。
- 当前方案是 4 颗 FT4000 CPU 的两级实机拓扑。
- 原始方案主要讨论 NTB/CXL 板间直连。
- 当前方案需要同时处理 Domain 内 CXL 和 Domain 间 RDMA。
- 原始方案没有必须实现张量并行。
- 当前实现补充了每 CPU 内 NUMA TP=4，但不等价于 report.pdf/HTML 中的完整 2D SUMMA。

## 4. 与 report.pdf 和 HTML 方案的关系

report.pdf 的重点是多级 CPU 张量并行，尤其是把高频集合通信限制在高速互联域，把低频通信放到 CXL 域。这个报告可以作为“后续 TP 扩展”的理论依据。

FEIBRIC_3D_SCHEME_FINAL.html 中提出：

```text
Domain 内：2D TP
Domain 间：RDMA PP
```

但结合当前代码和实机落地，建议修正为：

```text
CPU 内：1D NUMA TP=4
CPU 间：CXL/MCIO 小流水线
Domain 间：RDMA 大流水线
```

这个修正更符合目前 pipeline-brick 的实现：TP 发生在单个 CPU 内部，stage 间仍按层切分并传 hidden states。

## 5. 当前已经实现的内容

### 5.1 单系统内两 CPU 流水线

已实现。

当前 `llama-pipeline-brick` 支持在一个 Linux 系统内 fork 出 head 和 tail 两个进程：

```text
head process:  embedding + 前半层
tail process:  后半层 + final norm + lm_head
```

典型 Qwen3-4B 切分为：

```text
head: layers 0-17
tail: layers 18-35
```

已支持：

- `--single-system`
- `--transport cxl`
- `--head-numa 0-3`
- `--tail-numa 4-7`
- `--hidden-dtype bf16`
- `--prefill-chunk`
- `--parallel`

这对应当前实机中的“一个 Domain 内两 CPU 小流水线”。

### 5.2 CXL / shared-window 传输路径

已实现。

当前代码中有 `cxl_transport`，通过共享窗口传输 hidden states，并对 head-to-tail 和 tail-to-head 窗口做 NUMA 绑定。实际是否走 MCIO/CXL 物理路径，取决于系统的 NUMA 拓扑、内存绑定和硬件路由。

当前可以表述为：

```text
在同一 Linux Domain 内，使用 CXL 语义的共享内存窗口进行 hidden states 传输。
```

不要表述为“已经实现跨独立 Linux 系统的 CXL”，因为两个独立 Linux 之间没有共享地址空间。

### 5.3 稀疏注意力 StreamingLLM

已实现。

当前 `pipeline-brick` 已增加：

```text
--stream-kv
--stream-kv-sink N
--stream-kv-recent N
```

设计逻辑：

- prefill 阶段保持 dense attention。
- decode 早期历史长度不足时仍走 dense。
- decode 后期只保留 sink + recent 的 KV token 参与 attention。
- 目标是减少 attention 访问历史 KV 的计算和访存。

当前默认实验参数：

```text
sink = 16
recent = 128
```

已做过 NEON FP32 方向的稀疏 kernel 优化，实测在 643 token prompt、64 token generation、parallel=4 的场景中，stream-kv 相比 pipeline dense 有小幅提升。

### 5.4 EMA 路径替换

已实现替换。

当前 pipeline-brick 不再暴露 EMA 作为运行路径，`--ema-kv` 相关参数会报错并提示使用 `--stream-kv`。

原因：

- EMA 数学计算本身不大。
- 但为了使用 EMA，需要捕获 attention score、维护状态、选择 KV、重排 KV。
- 在 CPU decode 路径上，这些额外访存和调度开销较高。
- 当前 prompt 和生成长度不够长，attention 被减少的收益不足以抵消这些开销。

因此当前主稀疏方案改为 StreamingLLM。

### 5.5 四阶段 + 每 CPU 内 TP=4

已接通基础版本。

当前代码已有 `--tp-size` 原型，支持：

```text
--tp-size 1
--tp-size 4
```

并且只适合 F16/BF16 切分模型。后续项目已决定统一使用 F16，因此 Q4_K_M 不再作为主实验路径。

已实现内容包括：

- GGUF F16/BF16 权重切分工具原型。
- Qwen3 图中的局部 Q/K/V、FFN 计算。
- `ggml_all_reduce_sum` 原型。
- TP rank 子进程和共享内存屏障原型。
- 与 stream-kv 的基本兼容 smoke test。
- 四阶段 stage 模式下的 `--tp-size 4`。
- 每个 stage 内 fork 4 个 TP rank。
- rank0 负责真实 stage 间 transport，rank1-3 只参与本地计算和共享内存广播。
- stage0/stage1/stage2/stage3 分别负责 9 层。

但它目前不应作为主性能结果，因为：

- TP 仍是 1D TP 原型，不是 report.pdf 中完整 2D SUMMA。
- 只验证过 smoke，不代表稳定性能。
- 不再把 F16 模型和 Q4 模型混合作为公平性能对比。
- 还没有在新的 4 CPU / 2 Domain 实机拓扑上完整验证。

## 6. 当前尚未实现或未完成的内容

### 6.1 跨独立 Linux 的 RDMA transport

代码框架已实现，真实链路未完成验证。

当前 `pipeline-brick` 支持：

```text
--transport ntb-mw
--transport cxl
--transport ib-rdma
```

其中 `ib-rdma` 采用 ibverbs RC QP 的 send/recv 语义，复用现有 hidden state packet 格式；QP 信息通过本地文件手工交换，不依赖 TCP 控制面。

还没有完成的是：

```text
真实 IB 链路上的端到端验证
ZNI SDK 版本的真实适配
```

因此当前可以作为代码路径说明，但还不能把实测结果写成“已经完成跨机器 RDMA 性能验证”。

```text
CPU A1 -> RDMA -> CPU B0
```

风险：

- RDMA 需要网卡、驱动、固件、链路状态都正常。
- 如果 `ibv_devinfo` 和 `ibstat` 看不到 Active 设备，代码实现无法验证。
- 如果没有 IPoIB，QP 信息交换需要手工文件或命令行参数完成。

### 6.2 四阶段流水线

已实现基础版本。

当前 stage 模式支持：

```text
stage0 -> stage1 -> stage2 -> stage3
```

其中 stage1 和 stage2 都是中间阶段：

- stage1 接收 CPU A0 的 hidden states，算 layers 9-17，再通过 RDMA 发给 CPU B0。
- stage2 接收 RDMA hidden states，算 layers 18-26，再通过 CXL 发给 CPU B1。

当前实现已区分：

- stage0 使用 embedding。
- stage1/stage2 是中间 hidden state stage。
- stage3 使用 final norm、lm_head 和 token 采样。

```text
--stage-id N
--stage-count 4
--stage-numa '0-3;4-7;0-3;4-7'
```

本机已用 F16 TP 分片完成四阶段 + `--tp-size 4` 功能 smoke。真实 4 CPU / 2 Domain 性能还需要在实机上验证。

### 6.3 Domain 内 TP 作为主路径

当前采用的是“每 CPU 内 1D NUMA TP=4”，不是“Domain 内两 CPU 2D TP”。

HTML 中写的 Domain 内 2D TP 理论上更完整，但当前实现选择更稳妥的落地版本：

当前更合理的报告表述：

```text
主实现：四 CPU 分层流水线 + 每 CPU 内 NUMA TP=4。
CPU 间：Domain 内 CXL/MCIO，Domain 间 RDMA。
说明：当前 TP 是 1D NUMA TP，不是完整 2D SUMMA。
```

### 6.4 完整 2D SUMMA TP

未实现。

report.pdf 中的 2D SUMMA 需要：

- 二维 rank 网格。
- 行内 all-gather。
- 行内 all-reduce。
- CXL 方向轻量激活分发。
- 与 Qwen3 attention/FFN 权重切分严格匹配。

当前代码的 TP 原型不等价于完整 2D SUMMA。

### 6.5 AttentionPredictor / CNN 预测器

未实现。

当前实现的是 StreamingLLM 固定策略：

```text
保留 sink tokens + recent tokens
```

它不是 CNN predictor，也不是基于下一步注意力模式预测的 KV 筛选。

如果老师要求“预测器”，当前只能说明：

- EMA 曾尝试作为动态分数策略，但负优化。
- StreamingLLM 是当前稳定的稀疏注意力实现。
- CNN AttentionPredictor 仍是未完成项。

## 7. 建议对外汇报口径

建议这样汇报当前实现：

```text
我们已经完成了 Feibric 的核心流水线推理原型：
Qwen3-4B 被切成四个 stage，每个 CPU 负责连续 9 层；
每个 CPU 内部再用 4 个 NUMA rank 做 1D 张量并行，
rank 间通过 all-reduce sum 合成完整 hidden states。

在此基础上，我们实现了 StreamingLLM 稀疏注意力核，
decode 阶段只计算 sink + recent token 的注意力，并针对飞腾 CPU 做了局部向量化优化。

在单个 Domain 内，stage 间通过 MCIO/CXL 共享窗口传输 hidden states；
跨 Domain 链路已加入 IB RDMA transport 代码框架，真实链路性能仍需实机验证。
```

不建议说：

```text
已经完成完整三级 TP-1/TP-2/PP 架构。
已经完成 Domain 间 RDMA 性能验证。
已经完成 2D SUMMA。
已经完成 CNN AttentionPredictor。
```

这些目前都不准确。

## 8. 后续最小开发路线

如果比赛时间很紧，建议按优先级推进：

1. 确认 RDMA 硬件状态。

```bash
ibv_devinfo
ibstat
rdma link
ip link show
lsmod | grep -E 'mlx5|ib_|rdma|ipoib'
```

2. 验证 IB RDMA send/recv transport。

目标不是做完整通信库，而是确认当前 hidden state packet 能在 CPU A1 和 CPU B0 之间通过 RDMA send/recv 跑通。

3. 在真实四 CPU / 两 Domain 机器上跑四阶段 + `--tp-size 4` smoke。

4. 保留 stream-kv 作为稀疏优化亮点。

5. 如果时间允许，再补 ZNI SDK transport；否则报告中写成接口预留和待验证项。

## 9. 当前状态清单

| 模块 | 状态 | 说明 |
|---|---|---|
| 单系统两 CPU pipeline | 已实现 | head/tail 两进程，按层切分 |
| CXL/shared-window transport | 已实现 | Domain 内可用 |
| BF16 hidden state 传输 | 已实现 | 降低阶段间传输量 |
| parallel=4 并发流水线 | 已实现 | 用于吞吐实验 |
| StreamingLLM 稀疏注意力 | 已实现 | `--stream-kv` |
| EMA 稀疏路径 | 已移除主路径 | 负优化，保留底层 dormant 代码 |
| 四阶段 pipeline | 已实现基础版本 | stage0/stage1/stage2/stage3，每段 9 层 |
| 每 CPU 内 TP=4 F16 | 已实现基础版本 | 每 stage fork 4 个 rank，rank 间 all-reduce |
| 完整 2D SUMMA TP | 未实现 | report.pdf 理论方向 |
| IB RDMA transport | 代码框架已实现 | 真实链路待验证 |
| ZNI RDMA transport | 未实现真实 SDK 适配 | 当前只有接口占位 |
| CNN AttentionPredictor | 未实现 | 老师想要的预测器方向 |

## 10. 推荐最终方案命名

建议报告中使用：

```text
Feibric: 基于 CXL/RDMA 分层互联的飞腾 CPU 混合并行推理架构
```

副标题可写：

```text
四阶段流水线 + CPU 内 NUMA TP=4 + StreamingLLM 稀疏注意力优化
```

如果需要保留 TP 叙事，可写成：

```text
当前 TP 是每 CPU 内的 1D NUMA TP，不宣称完整 2D SUMMA。
```
