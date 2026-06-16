# 比赛技术报告与PPT大纲

本文档用于整理当前参赛作品的技术报告和答辩PPT结构。内容基于以下输入：

- 《基于板间桥接的流水线并行推理架构研究方案》
- 《Pipeline_Brick相比GPU的核心优势分析》
- 当前 `llama.cpp` 私有分支中已经实现的 `llama-pipeline-brick`
- 双飞腾CPU实机实验结果

## 1. 当前完成情况总览

### 1.1 已经实现并有实验数据支撑的部分

| 模块 | 当前状态 | 报告中建议表述 |
| --- | --- | --- |
| 双Brick流水线并行 | 已实现 | 将 Qwen3-4B 按层切分为 head Brick 和 tail Brick，head 执行 embedding 和前 18 层，tail 执行后 18 层、final norm 和 lm_head |
| 双飞腾CPU部署 | 已实现 | 在单Linux系统下，两张飞腾CPU分别对应 NUMA 0-3 和 NUMA 4-7，head/tail 进程分别绑核运行 |
| 层范围加载 | 已实现 | 每个 Brick 只加载自身负责的层，减少单侧权重和KV cache压力 |
| 层间hidden states传输 | 已实现 | head 到 tail 只传层间 hidden states，不传完整KV cache |
| bf16 hidden states | 已实现 | hidden states 传输从 f32 降到 bf16，每token从 10240 bytes 降到 5120 bytes |
| prefill micro-batch | 已实现 | prefill 阶段按 chunk 合批，降低小任务调度与通信开销 |
| CXL-like共享窗口传输 | 已实现软件原型 | 通过共享内存窗口和NUMA绑定模拟 CXL/NTB Memory Window 语义 |
| StreamingLLM稀疏注意力 | 已实现 | decode 后期固定保留 sink + recent tokens |
| 飞腾CPU稀疏核优化 | 已实现第一版 | 对 Qwen3 decode 路径的稀疏注意力核加入 ARM NEON FP32 快路径 |
| 端到端性能对比 | 已完成一组关键实验 | baseline、pipeline dense、pipeline stream-kv、no-CXL 均有结果 |

### 1.2 受机器条件限制，当前未完全实现的部分

| 方案目标 | 当前状态 | 报告中建议处理方式 |
| --- | --- | --- |
| 真实CXL设备链路 | 未具备真实设备 | 明确说明当前机器没有真实CXL设备，采用共享内存窗口模拟 CXL Memory Window 语义 |
| 真实PCIe NTB BAR / DMA / Doorbell中断 | 未接真实驱动 | 作为硬件部署方案和未来迁移路径描述；当前用 mmap 共享窗口和轮询模拟 |
| 多块物理Brick堆叠 | 未实现 | 当前实现为双Brick软件原型，报告中作为架构可扩展设计说明 |
| NUMA级张量并行 | 未实现 | 当前是进程绑核和NUMA亲和性，不是真正矩阵切分张量并行 |
| TCP/IP多CPU baseline | 未实现 | 可以在实验设计中列为对比项，但当前结果以 llama-parallel 和共享窗口传输为主 |
| CNN AttentionPredictor | 未实现 | 作为后续工作；当前已完成可复用的稀疏attention kernel |
| EMA预测器 | 尝试过但未作为最终路线 | 说明 EMA 因分数捕获、状态维护、KV选择和gather开销较大，转向 StreamingLLM |

### 1.3 当前最适合对外呈现的核心结论

当前作品可以概括为：

> 面向双飞腾CPU平台，设计并实现了一个基于板间桥接思想的 Pipeline Brick 推理原型。系统将 Qwen3-4B 按层切分到两张飞腾CPU上，通过共享内存窗口模拟 CXL/NTB Memory Window 语义，仅在 Brick 间传输 bf16 hidden states，并在 decode 阶段引入 StreamingLLM 稀疏注意力核。实测在 batch=4、prompt=643 tokens/request、generation=64 tokens/request 场景下，pipeline dense 相比 llama-parallel baseline 生成吞吐提升约 1.31x；加入 StreamingLLM 后生成吞吐进一步提升到 3.10 tokens/s，相比 baseline 总提升约 1.46x。

## 2. 技术报告建议大纲

技术报告建议突出“赛题要求 + 方案设计 + 实现细节 + 实验数据 + 边界说明”。不要把未实现的真实硬件CXL、张量并行、预测器说成已经完成；可以作为“设计目标、模拟验证、后续迁移路径”体现。

### 摘要

建议写法：

- 背景：CPU平台大模型推理受内存带宽、算子效率、长上下文KV cache访问和单节点资源限制影响。
- 目标：在飞腾CPU上探索从“能用”到“好用”、再到“更高效”的推理架构。
- 方法：提出 Pipeline Brick 分层流水线并行架构；通过共享内存窗口模拟 CXL/NTB 板间桥接；引入 bf16 hidden states、prefill micro-batch 和 StreamingLLM 稀疏注意力核。
- 结果：给出关键吞吐提升。
- 边界：当前为双飞腾CPU单Linux系统软件原型，未接真实CXL硬件，未实现真正NUMA张量并行。

### 1. 背景与意义分析

#### 1.1 赛题背景

对应赛题要求：

- 核心AI计算必须在飞腾CPU上完成。
- 需要在CPU平台上实现AI模型推理优化。
- 重点关注推理吞吐、端到端时间、优化方案理论和实现细节。

#### 1.2 CPU大模型推理瓶颈

建议展开：

- 纯CPU推理通常受内存带宽、矩阵乘法效率和线程调度影响。
- 长上下文 decode 阶段需要反复访问 KV cache，容易受缓存局部性和内存带宽限制。
- 多请求并发可以提高硬件利用率，但单进程全模型推理会把所有层、KV cache 和计算压力集中在同一进程和同一内存系统内。

#### 1.3 为什么选择 Pipeline Brick

从方案文档提炼：

- 大模型按层切分，每块 Brick 只负责部分层。
- Brick 间只传 hidden states，不传完整 KV cache。
- 每个 Brick 独立维护本地层的 KV cache。
- 通信是相邻节点点对点，不需要全连接通信。
- 适合低功耗、国产CPU、边缘终端、工业和信创场景。

#### 1.4 相比GPU路线的定位

基于《Pipeline_Brick相比GPU的核心优势分析》：

- GPU绝对吞吐更强，但功耗、成本、部署复杂度高。
- Pipeline Brick 目标不是替代数据中心GPU，而是面向低功耗、可堆叠、国产化和边缘部署。
- Brick 线性拓扑不追求全连接带宽，而是匹配流水线并行的相邻激活值传输需求。

### 2. 基线系统与实验环境

#### 2.1 硬件环境

建议包含表格：

| 项目 | 配置 |
| --- | --- |
| CPU | 双飞腾CPU |
| 系统拓扑 | 单Linux系统可见8个NUMA节点 |
| 第一张卡 | NUMA 0-3，CPU 0-63 |
| 第二张卡 | NUMA 4-7，CPU 64-127 |
| 互连条件 | 两张飞腾CPU通过MCIO线连接，共属一个Linux系统 |
| 真实CXL设备 | 当前无真实CXL设备 |

#### 2.2 软件环境

建议列出：

- 操作系统。
- 编译器和 CMake 版本。
- llama.cpp 私有分支。
- 模型格式：GGUF。
- CPU-only 推理，不使用GPU。
- `llama-parallel` 作为官方CPU并发基线。
- `llama-pipeline-brick` 作为自研流水线原型。

#### 2.3 基线模型

建议列出：

| 项目 | 内容 |
| --- | --- |
| 模型 | Qwen3-4B |
| 量化 | Q4_K_M |
| 推理框架 | llama.cpp |
| 上下文 | 8192 |
| batch/parallel | 4 |
| prompt长度 | 643 tokens/request |
| 生成长度 | 64 tokens/request |

### 3. Pipeline Brick 总体设计

#### 3.1 总体架构

建议图示：

```text
Input tokens
    |
Head Brick: embedding + layers 0-17
    |
bf16 hidden states
    |
Shared Memory Window / CXL-like Memory Window
    |
Tail Brick: layers 18-35 + final norm + lm_head
    |
Output tokens
```

#### 3.2 模型按层切分

当前实现：

- head Brick：embedding + layers `[0,18)`。
- tail Brick：layers `[18,36)` + final norm + lm_head。
- 两侧分别创建 llama context。
- 通过模型加载层过滤，只加载本 Brick 所需层。

报告中要强调：

- 这是层级流水线并行，不是张量并行。
- 每个 Brick 维护本地层的 KV cache。
- 跨 Brick 不传完整 KV cache。

#### 3.3 数据流与控制流

当前实现：

- prefill 阶段：按 `prefill_chunk=32` 切分 prompt。
- decode 阶段：batch=4 时，每步把4个sequence的当前token打成一个微批。
- head -> tail：传 hidden states + token元数据。
- tail -> head：回传采样得到的 token/control 小消息。

#### 3.4 CXL-like 共享窗口传输

必须谨慎表述：

- 当前没有真实CXL设备。
- 代码实现了 `--transport cxl` 路径。
- 底层通过共享内存窗口模拟 CXL/NTB Memory Window 语义。
- head-to-tail 窗口绑定到 tail NUMA，tail读取更接近本地访问。
- tail-to-head 窗口绑定到 head NUMA。

建议用语：

> 当前系统实现的是 CXL/NTB Memory Window 语义的软件原型。在单Linux双飞腾NUMA系统内，通过共享内存窗口和NUMA物理页绑定模拟“发送方写入接收方内存窗口、接收方本地读取”的板间桥接通信模式。该实现为后续替换真实CXL/NTB设备文件、DMA和硬件Doorbell保留了接口边界。

### 4. 优化技术方案与实现细节

#### 4.1 NUMA绑定与局部性优化

当前实现：

- head 进程绑定 CPU 0-63。
- tail 进程绑定 CPU 64-127。
- 共享窗口按照通信方向绑定到接收侧 NUMA。

报告中说明：

- 这是 NUMA affinity 和内存放置优化。
- 当前不是 NUMA 级张量并行。

#### 4.2 bf16 hidden states 传输

当前实现：

- Qwen3-4B hidden size = 2560。
- f32 hidden states 每 token `2560 x 4 = 10240 bytes`。
- bf16 hidden states 每 token `2560 x 2 = 5120 bytes`。
- 通信量减半。

#### 4.3 prefill micro-batch

当前实现：

- `--prefill-chunk 32`。
- batch=4 时每个 micro-batch 最多 `4 x 32 = 128 tokens`。
- 避免每个token都单独跨Brick传输和调度。

报告中说明：

- 对长prompt尤其重要。
- 主要减少图执行次数、进程同步次数和共享窗口轮询次数。

#### 4.4 StreamingLLM 稀疏注意力

当前实现：

- EMA 方案尝试后未作为最终路线。
- 最终采用 StreamingLLM：固定保留开头 `sink=16` 和最近 `recent=128` tokens。
- prefill 仍走 dense/Flash Attention 快路径。
- decode 后期启用稀疏注意力。
- 自定义 sparse attention kernel 只遍历 sink + recent 两段 KV。

报告中说明：

- StreamingLLM 不需要捕获 attention score。
- 不需要预测器状态同步。
- 稀疏集合由位置直接决定，因此开销低、稳定性高。

#### 4.5 飞腾CPU稀疏核优化

当前实现：

- 针对 Qwen3 decode 路径，加入自定义稀疏注意力核。
- QK 点积和 V 加权累加加入 ARM NEON FP32 快路径。
- 针对 `head_dim=128`、F32 Q、F16 KV cache 的常见形状优化。
- 保留 scalar fallback，避免非ARM构建失败。

#### 4.6 EMA 与 CNN AttentionPredictor 的处理

建议报告中这样写：

- 最初尝试 EMA KV selection，通过历史注意力分数筛选关键 KV。
- 但在 CPU decode 路径中，EMA 的负担不在公式本身，而在分数捕获、状态维护、KV选择、gather和同步。
- 实测和分析表明，在当前 643-token prompt 和 64-token generation 场景中，EMA 开销容易超过稀疏收益。
- 因此最终改用 StreamingLLM 作为速度优先的稳定方案。
- CNN AttentionPredictor 作为后续方向：复用当前稀疏attention kernel，将固定规则替换为预测器输出的 KV 下标。

### 5. 实验设计

#### 5.1 对比方案

建议表格：

| 方案 | 含义 | 作用 |
| --- | --- | --- |
| llama-parallel baseline | llama.cpp 官方单进程并发CPU基线 | 衡量原始CPU并发推理能力 |
| Pipeline dense no-CXL | 双Brick流水线 + 默认共享窗口传输 + dense attention | 验证流水线并行与共享窗口通信 |
| Pipeline dense + CXL | 双Brick流水线 + CXL-like共享窗口传输 + dense attention | 验证CXL-like传输路径 |
| Pipeline Stream-KV + CXL | 在CXL-like pipeline基础上加入StreamingLLM稀疏核 | 验证稀疏attention优化 |

#### 5.2 公平性口径

必须写清楚：

- baseline 使用 `-t 128`，单进程使用两张CPU全部线程。
- pipeline 使用 head 64线程 + tail 64线程，总线程数同为128。
- 同一模型、同一prompt、同一batch、同一生成长度。
- pipeline 自研工具不是 llama-parallel 修改版，因此采用“官方CPU并发基线 vs 自研Pipeline Brick原型”的对比口径。

#### 5.3 当前关键实验结果

最新一组结果：

| 方案 | 生成吞吐 tokens/s | 总吞吐 tokens/s | 推理时间 |
| --- | ---: | ---: | ---: |
| llama-parallel baseline | 2.13 | 23.50 | 120.34 s |
| Pipeline dense + CXL | 2.80 | 30.92 | 91.47 s |
| Pipeline Stream-KV + CXL | 3.10 | 34.22 | 82.65 s |
| Pipeline dense no-CXL | 2.78 | 30.74 | 91.99 s |

关键提升：

```text
Pipeline dense + CXL vs baseline:
  2.80 / 2.13 = 1.31x

Pipeline Stream-KV + CXL vs Pipeline dense + CXL:
  3.10 / 2.80 = 1.11x

Pipeline Stream-KV + CXL vs baseline:
  3.10 / 2.13 = 1.46x
```

#### 5.4 消融实验组织方式

当前可完成并报告：

| 消融项 | 对比 | 说明 |
| --- | --- | --- |
| 流水线架构贡献 | baseline vs pipeline dense + CXL | 验证层切分、双CPU分工、micro-batch、bf16传输整体收益 |
| CXL-like传输路径 | pipeline dense no-CXL vs pipeline dense + CXL | 当前两者同属共享窗口软件原型，性能接近 |
| 稀疏注意力贡献 | pipeline dense + CXL vs pipeline Stream-KV + CXL | 验证 StreamingLLM 和稀疏核收益 |

未完成但可作为设计讨论：

| 消融项 | 当前状态 |
| --- | --- |
| NUMA张量并行 | 未实现，作为未来工作 |
| TCP/IP多CPU baseline | 未实现，可作为理论对比或后续实验 |
| CNN AttentionPredictor | 未实现，作为未来工作 |
| EMA同步/异步 | 尝试过，但不作为最终性能路线 |

### 6. 结果分析

#### 6.1 Pipeline dense 为什么快于 baseline

建议分析：

- 两颗CPU同时承担不同层的计算。
- 每侧只维护本地层的KV cache，内存压力分散。
- prefill micro-batch 减少任务粒度过小带来的调度开销。
- bf16 hidden states 减少跨Brick传输量。
- NUMA绑核降低调度迁移和远端内存访问混乱。

#### 6.2 Stream-KV 为什么能进一步加速

建议分析：

- decode 后期不再对完整历史 KV 做 attention。
- 只计算 sink + recent 保留窗口。
- 自定义稀疏核避免扫描完整 mask。
- NEON 快路径加速 QK 点积和 V 累加。

#### 6.3 no-CXL 与 CXL-like 性能接近的解释

建议表述：

> 当前 no-CXL 和 CXL-like 都是共享窗口软件原型，不是真实物理CXL和非CXL硬件链路的对比。CXL-like 版本主要区别在于按方向绑定共享窗口到接收方NUMA，模拟对端内存窗口语义。两者性能接近，说明当前瓶颈主要不在传输路径，而在计算、同步和推理流程本身。

#### 6.4 精度与质量风险

建议说明：

- Pipeline dense 与 baseline 理论上应保持近似输出一致，但由于浮点计算顺序、bf16传输和采样细节可能有微小差异。
- StreamingLLM 是速度优先的稀疏策略，固定丢弃中间历史 token，存在输出质量风险。
- 当前报告可先以吞吐验证为主，质量评估作为后续补充。

### 7. 创新点总结

建议列4点：

1. 面向国产飞腾CPU的双Brick分层推理原型。
2. 基于共享内存窗口的 CXL/NTB Memory Window 语义模拟。
3. 只传 hidden states、不传 KV cache 的跨Brick通信设计。
4. 面向 decode 阶段的 StreamingLLM 稀疏注意力核和飞腾CPU向量化优化。

### 8. 应用前景

结合第二个文档：

- 政务内网和信创终端：国产CPU、无GPU依赖。
- 工业边缘：低功耗、可部署在恶劣环境。
- 便携式/桌面AI设备：通过增加Brick扩展模型容量。
- 教育科研平台：低成本验证流水线并行、CXL-like通信和稀疏推理。

### 9. 局限性与未来工作

必须客观写：

1. 当前无真实CXL设备，CXL为软件共享窗口语义模拟。
2. 当前无真实PCIe NTB BAR、DMA和硬件Doorbell。
3. 当前没有实现 NUMA 张量并行。
4. 当前 StreamingLLM 精度评估不完整。
5. CNN AttentionPredictor 尚未实现。
6. 实验次数需要补充3次均值和标准差。
7. 通信延迟、内存占用、纯decode场景等微基准仍需补齐。

### 10. 附录

建议附：

- 构建命令。
- baseline 运行命令。
- pipeline dense 运行命令。
- pipeline stream-kv 运行命令。
- 关键日志截图。
- lscpu / numactl / NUMA拓扑截图。
- 源码改动文件清单。

## 3. PPT建议大纲

PPT建议控制在 15-18 页，答辩时重点讲清楚“为什么这样设计、实现了什么、数据证明什么、哪些是模拟和未来工作”。

### 第1页：标题页

标题建议：

```text
基于板间桥接的双飞腾CPU流水线并行推理架构
```

副标题：

```text
Pipeline Brick: CXL-like共享窗口传输 + StreamingLLM稀疏注意力优化
```

### 第2页：赛题要求与我们的目标

内容：

- 赛题要求：核心AI计算在飞腾CPU上完成。
- 目标：提升CPU大模型推理吞吐。
- 路线：架构级流水线并行 + 通信路径优化 + decode稀疏注意力。

### 第3页：背景痛点

建议三点：

- CPU推理受内存带宽和矩阵计算效率限制。
- 长上下文decode反复访问KV cache。
- TCP/IP式跨节点推理通信开销过高。

### 第4页：总体方案 Pipeline Brick

画图：

```text
Head Brick -> Shared Window -> Tail Brick
```

标注：

- head：embedding + layers 0-17。
- tail：layers 18-35 + final norm + lm_head。
- 只传 hidden states。
- KV cache 留在本地。

### 第5页：当前硬件环境与方案边界

必须讲清：

- 当前机器：两张飞腾CPU，通过MCIO线连接，共属一个Linux系统。
- 系统可见8个NUMA节点。
- 没有真实CXL设备。
- 当前CXL是 CXL Memory Window 语义的软件模拟。

建议一句话：

```text
我们验证的是面向CXL/NTB板间桥接的通信语义和软件栈，而不是物理CXL链路实测。
```

### 第6页：CXL-like共享窗口传输

内容：

- head-to-tail窗口绑定到tail NUMA。
- tail-to-head窗口绑定到head NUMA。
- 模拟“写入对端内存窗口、接收端本地读取”。
- 不走TCP/RPC。

### 第7页：prefill micro-batch 与 bf16 hidden states

内容：

- prefill chunk = 32。
- batch=4 时每个micro-batch最多128 tokens。
- bf16将每token hidden传输量从10240 bytes降到5120 bytes。

### 第8页：为什么不传KV cache

内容：

- 每层KV cache只和本层attention相关。
- 每个Brick保留自身层KV。
- 跨Brick只需要传层间hidden states。
- 通信量远小于传完整KV cache。

### 第9页：稀疏注意力路线选择

内容：

- 初始尝试：EMA KV selection。
- 问题：score捕获、状态维护、KV选择和gather开销较高。
- 最终路线：StreamingLLM。
- 固定保留 sink + recent，稳定、低开销。

### 第10页：StreamingLLM稀疏核实现

内容：

- prefill保持dense/快路径。
- decode后期启用稀疏注意力。
- 只遍历 sink=16 和 recent=128。
- 针对飞腾ARM CPU加入NEON FP32优化。

### 第11页：实验设置

表格：

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen3-4B Q4_K_M |
| 平台 | 双飞腾CPU |
| batch | 4 |
| prompt | 643 tokens/request |
| generation | 64 tokens/request |
| baseline | llama-parallel |
| pipeline线程 | head 64 + tail 64 |

### 第12页：端到端结果图

放柱状图：

- llama-parallel baseline：2.13 gen tok/s。
- pipeline dense + CXL：2.80 gen tok/s。
- pipeline stream-kv + CXL：3.10 gen tok/s。
- pipeline dense no-CXL：2.78 gen tok/s。

图上标注：

- pipeline dense vs baseline：1.31x。
- stream-kv vs dense：1.11x。
- stream-kv vs baseline：1.46x。

### 第13页：消融实验

表格：

| 对比 | 结论 |
| --- | --- |
| baseline -> pipeline dense | 流水线、双CPU分工、micro-batch和bf16带来整体收益 |
| dense no-CXL -> dense + CXL | 当前两者均为共享窗口软件原型，性能接近 |
| dense + CXL -> stream-kv + CXL | 稀疏注意力核带来约11%增量收益 |

### 第14页：为什么no-CXL和CXL接近

简单解释：

- no-CXL不是单进程，也不是没有通信。
- 它仍是默认共享内存窗口传输。
- CXL-like是把共享窗口按方向绑定到对端NUMA，模拟CXL内存窗口语义。
- 当前没有真实CXL设备，因此两者性能接近是合理的。

### 第15页：与GPU方案的差异化价值

内容：

- GPU吞吐更强，但功耗、成本和部署复杂度高。
- Pipeline Brick 面向低功耗、国产化、边缘部署。
- 线性拓扑匹配流水线通信需求，不需要全连接互联。

### 第16页：当前不足

必须诚实列出：

- 没有真实CXL设备。
- 没有真实NTB/DMA/Doorbell。
- 没有实现NUMA张量并行。
- 没有完成CNN AttentionPredictor。
- StreamingLLM质量评估还需补充。
- 实验还需要多次重复统计。

### 第17页：未来工作

建议四点：

1. 接入真实 CXL/NTB 设备文件，替换共享内存模拟层。
2. 增加通信延迟和内存占用微基准。
3. 在当前稀疏attention kernel基础上接入 CNN AttentionPredictor。
4. 评估更长上下文、更大模型和多Brick扩展。

### 第18页：总结

建议三句话：

```text
1. 我们完成了双飞腾CPU上的 Pipeline Brick 分层流水线推理原型。
2. 通过 CXL-like 共享窗口、bf16 hidden states、prefill micro-batch 和 StreamingLLM 稀疏核，实现了端到端吞吐提升。
3. 当前结果验证了板间桥接流水线推理的可行性，后续可迁移到真实CXL/NTB硬件并接入预测器进一步优化。
```

## 4. 三天内优先补充的材料

按优先级排序：

1. 每组实验至少重复3次，记录均值和标准差。
2. 截图保存 NUMA 拓扑：`lscpu`、`numactl -H`。
3. 保存四组完整日志：baseline、dense+CXL、stream+CXL、dense no-CXL。
4. 补一张代码架构图：`pipeline-brick.cpp`、Qwen3 graph、transport、stream-kv kernel。
5. 补一张“方案目标 vs 当前实现 vs 未来工作”的对照表。
6. 如果时间允许，做一个简单通信微基准：发送固定大小hidden payload，统计平均收发延迟。
7. 不建议最后三天强行做完整NUMA张量并行或CNN预测器，风险高，报告中作为未来工作更稳。

## 5. 报告中建议避免的表述

不建议：

```text
我们实现了真实CXL硬件传输。
我们实现了完整NUMA张量并行。
我们实现了CNN预测器。
StreamingLLM保证精度不下降。
所有收益都来自流水线并行。
```

建议改为：

```text
我们实现了CXL/NTB Memory Window语义的软件原型。
当前实现了NUMA绑核和接收侧内存窗口绑定，尚未实现矩阵级张量并行。
当前完成了可复用的稀疏attention kernel，CNN预测器作为后续接入方向。
StreamingLLM是速度优先的稀疏策略，质量评估仍需补充。
收益来自分层执行、NUMA局部性、bf16传输、prefill micro-batch和稀疏attention的组合。
```
