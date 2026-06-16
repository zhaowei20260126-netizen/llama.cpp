# Pipeline Brick 双飞腾 CPU 流水线并行推理原型实验报告

## 1. 项目背景

本项目对应研电赛飞腾赛道赛题 3（高效模型推理），目标是设计并验证一种基于 **板间流水线并行** 的大模型推理架构。

标书方案《基于板间桥接的流水线并行推理架构研究方案》提出了一种硬件级流水线并行架构——**Pipeline Brick**：

- 将大模型按层切分，每若干层部署在一块独立的低功耗计算板（称为 Brick）上。
- 多块 Brick 通过 **PCIe NTB（Non-Transparent Bridge，非透明桥）** 板间直连，形成线性流水线。
- 请求数据从首块 Brick 流入，逐层计算后从尾块 Brick 流出。
- 板间通信完全绕过 TCP/IP 协议栈，仅通过 Memory Window（内存窗口）+ Doorbell（门铃通知）在 PCIe 物理层直接传输激活值。

最终目标是实现一种**低功耗、可物理堆叠、完全国产化**的边缘 AI 推理硬件形态。

本报告聚焦当前已完成的双飞腾 CPU 软件原型实现，阐述其原理、实验结果，并与标书目标方案进行逐项对照。

---

## 2. 标书目标方案 vs 当前原型的整体定位

标书描述了完整的硬件产品愿景。当前原型是这一愿景的 **软件验证阶段**，核心差异总结如下：

| 层面 | 标书目标方案（硬件产品） | 当前原型（软件验证） |
|---|---|---|
| 硬件形态 | 多块独立 PCB，各自有 CPU/内存，通过 PCIe NTB 金手指物理堆叠 | 共享一台双飞腾 CPU Linux 系统，通过 fork 两个进程模拟双 Brick |
| 板间通信 | PCIe NTB Memory Window（BAR 空间映射） | mmap `/dev/shm` 共享内存文件（Memory Window 语义模拟） |
| 通知机制 | 硬件 Doorbell 寄存器写触发 MSI/MSI-X 中断 | 共享内存中的原子状态标记 + `sleep_us(50)` 轮询（Doorbell 语义模拟） |
| 数据搬运 | DMA 引擎自动搬数据，CPU 不参与 | `memcpy` 写入 shared buffer + `memcpy` 读取 |
| 连接拓扑 | 线性菊花链，每 Brick 只连前后邻居 | 点对点两条单向 ring buffer（head→tail, tail→head） |
| 运行时 | 编译期固化配置，Doorbell 中断驱动，空闲低功耗 | 运行时参数传入，轮询驱动 |

**简言之，当前原型在"传输语义"层面模拟了标书方案的核心行为（Memory Window、Doorbell、单向管道），但在"传输实现"层面仍然是软件模拟（mmap + memcpy + poll），尚未涉及真实硬件。**

---

## 3. 实机硬件环境

原型运行在一套双飞腾 CPU、单 Linux 系统上：

```
                    ┌─────────────────────────────────────────┐
                    │           单 Linux 系统                    │
                    │                                           │
                    │  ┌──────── Card 0 ────────┐               │
                    │  │ NUMA 0-3, CPU 0-63     │               │
                    │  │ 4 NUMA × 16 cores      │               │
                    │  │ 本地内存                │               │
                    │  └────────────────────────┘               │
                    │                                           │
                    │  ┌──────── Card 1 ────────┐               │
                    │  │ NUMA 4-7, CPU 64-127   │               │
                    │  │ 4 NUMA × 16 cores      │               │
                    │  │ 本地内存                │               │
                    │  └────────────────────────┘               │
                    │                                           │
                    │  两张卡通过系统互联总线（MCIO 等）通信        │
                    │  共 128 核，8 个 NUMA node                 │
                    └───────────────────────────────────────────┘
```

关键参数：
- CPU 总量：128 核（card0: 64 核，card1: 64 核）
- 模型：Qwen3-4B-Instruct-2507 GGUF Q4_K_M
- 层数：36 层，hidden size：2560

---

## 4. 软件原型架构原理

### 4.1 模型分层策略

Qwen3-4B（36 层）被按层均分为两段：

```
    Qwen3-4B 完整模型 (36 layers)
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  token_embd  ───  layer[0]  ───  ...  ───  layer[17] │  ← Head Brick (层 0-17)
    │                                                      │
    │  layer[18]  ───  ...  ───  layer[35]  ───  final_norm  ───  lm_head │  ← Tail Brick (层 18-35)
    │                                                      │
    └──────────────────────────────────────────────────────┘
```

- **Head Brick**：负责 token embedding + Transformer 层 `[0, 18)`（前 18 层）。运行结束后将 hidden states 写入传输通道。
- **Tail Brick**：从传输通道读取 hidden states，负责 Transformer 层 `[18, 36)`（后 18 层）+ final_norm + lm_head，采样生成 token。

选择层 18 作为切分点的原因：总层数 36 层均分为 18 + 18，使得两端计算量基本相等（标准 Transformer decoder 各层计算量均匀）。

### 4.2 层权重按需加载

标书方案要求每块 Brick 只存储和加载自己负责的层。当前原型通过修改 llama.cpp 核心代码实现了这一能力：

- `llama_model_params` 中新增 `pipeline_brick_enabled`、`pipeline_brick_role`、`pipeline_brick_layer_start/end` 字段。
- `create_memory()` 中增加层过滤：head 进程只加载层 0-17 的权重张量（如 `blk.0.*` 到 `blk.17.*`），tail 进程只加载层 18-35 和张量。通过 `TENSOR_NOT_REQUIRED | TENSOR_SKIP` 标志跳过非本范围层。
- head 进程不加载 `output_norm` 和 `output`（lm_head 权重），tail 进程不加载 `token_embd`。
- graph 构建中增加 `build_inp_hidden()` 路径：tail 进程的图不从 token embedding 开始，而是直接将接收到的 hidden states 作为第一层输入。head 进程的图在最后一层（层 17）计算完成后提前将 hidden states 导出，跳过 final_norm 和 lm_head。

改动涉及 8 个核心文件：

| 文件 | 改动性质 |
|---|---|
| `include/llama.h` | 新增 `llama_pipeline_brick_role` 枚举、pipeline 参数 |
| `src/llama-model.h` / `src/llama-model.cpp` | 参数传递、层过滤在 `create_memory()` 中的实现 |
| `src/llama-graph.h` / `src/llama-graph.cpp` | 新增 `build_inp_hidden()`，tail 端不走 token embedding |
| `src/models/qwen3.cpp` | 层范围感知的张量加载（`TENSOR_SKIP`）、graph 构建支持 `layer_start/end`、head 端提前导出 hidden、tail 端直接注入 hidden |
| `tools/pipeline-brick/` | 新增工具目标，含 CMakeLists.txt 和主程序 `pipeline-brick.cpp` |

### 4.3 进程模型与 NUMA 绑定

当前 `--single-system` 模式启动两个子进程：

```
    主进程 (run_single_system)
    │
    ├─ fork() → tail 子进程
    │     │
    │     ├─ sched_setaffinity 绑定 CPU 64-127 (NUMA 4-7)
    │     ├─ 加载模型层 [18, 36)
    │     └─ 进入 run_tail() 事件循环
    │
    └─ fork() → head 子进程
          │
          ├─ sched_setaffinity 绑定 CPU 0-63 (NUMA 0-3)
          ├─ 加载模型层 [0, 18)
          └─ 进入 run_head() prefill/decode 循环
```

NUMA（Non-Uniform Memory Access，非统一内存访问）绑定分为两层：

**进程 CPU 亲和性**（通过 `sched_setaffinity`）：
- Head 进程只能调度到 NUMA 0-3 的 CPU 核心，确保计算指令只在第一张卡的 CPU 上执行。
- Tail 进程只能调度到 NUMA 4-7 的 CPU 核心。

**共享内存 NUMA 绑定**（通过 `mbind` 系统调用）：
- `head-to-tail` ring buffer 文件通过 `mbind` 绑定到 **tail 侧 NUMA (4-7)**。这意味着 head 写入的是 tail 的本地内存，tail 读取时是纯本地访问——模拟了"tail 侧 Memory Window"的语义。
- `tail-to-head` ring buffer 文件通过 `mbind` 绑定到 **head 侧 NUMA (0-3)**。

NUMA 绑定的价值：在双卡系统中，跨卡内存访问需要通过系统互联总线，延迟和带宽都不如本地内存。通过 mbind 确保共享内存物理上位于接收方一侧，接收方读数据是零跨卡访问的。

### 4.4 传输层设计

#### 4.4.1 Ring Buffer 结构

传输层实现为 `ntb_mw_transport` 类（即 NTB Memory Window 传输的软件模拟）。每对通信方向使用一个 ring buffer，布局为：

```
    Ring Buffer (在 /dev/shm 文件中通过 mmap 映射)
    ┌──────────────────────────────────────────────────────────┐
    │  Slot 0   │  Slot 1   │  Slot 2   │  ...  │  Slot 63   │
    │           │           │           │       │            │
    │  [state]  │  [state]  │  [state]  │  ...  │  [state]   │
    │  [header] │  [header] │  [header] │  ...  │  [header]  │
    │  [payload]│  [payload]│  [payload]│  ...  │  [payload] │
    │           │           │           │       │            │
    │           共 64 个 slot，slot_size 对齐到 64 字节         │
    └──────────────────────────────────────────────────────────┘
```

每个 slot 的结构：
```
    ┌──────────────┬───────────────────────┬──────────────────┐
    │  state (4B)  │  header (56B)         │  payload (变长)   │
    │  EMPTY/FULL  │  magic/ver/slot/      │  n_tokens ×       │
    │              │  flags/n_tokens/      │  (meta + hidden)  │
    │              │  n_embd/pos/payload   │                   │
    └──────────────┴───────────────────────┴──────────────────┘
```

`state` 字段使用 `__atomic_load_n` / `__atomic_store_n` 实现无锁的 EMPTY/FULL 状态同步。发送方轮询等待当前 slot 状态变为 EMPTY，写入数据后标记为 FULL。接收方轮询等待 FULL，读取数据后标记为 EMPTY。

#### 4.4.2 包格式

head→tail 方向传输 **hidden states 包**，负载包含：

```
    ┌──────────────────────┬─────────────────────────────────────┐
    │  hidden_token_meta[]  │  hidden_states[]                   │
    │  n_tokens × 16B      │  n_tokens × n_embd × dtype_size    │
    │                      │                                    │
    │  每 token 的元信息：    │  每 token 的隐藏状态向量：           │
    │  - seq_id (4B)       │  f32: n_embd × 4B                 │
    │  - pos (4B)          │  bf16: n_embd × 2B                │
    │  - flags (4B)        │                                    │
    └──────────────────────┴─────────────────────────────────────┘
```

tail→head 方向传输 **token 包**，负载很小：
```
    ┌──────────────┬──────────────┐
    │  seq_id (4B) │  token (4B)  │
    └──────────────┴──────────────┘
```

#### 4.4.3 Doorbell 语义模拟

标书方案中，硬件 Doorbell 是写完 Memory Window 后通过写 PCIe 寄存器触发对端 MSI/MSI-X 中断的机制。当前原型用两种模式模拟：

- **poll 模式**（`--single-system` 默认使用）：接收方 `recv()` 循环中每次检查不到 FULL 状态时，`sleep_us(50)` 让出 CPU，然后重新检查。这是一种 busy-wait 的温和形式。
- **write 模式**（`--hardware` 显式模式）：用 `/dev` 下的字符设备文件模拟 Doorbell 寄存器。发送方写通知后 `write(tx_db_fd, ...)`，接收方 `poll(rx_db_fd, ...)` 等待。但需要真实的 NTB 驱动暴露 Doorbell 设备文件才有意义。

**当前 `--single-system` 运行实际使用的是 poll 模式**（`make_single_system_child_args` 中 `child.db_mode = doorbell_mode::poll`）。

#### 4.4.4 Hidden States 的 BF16 传输

Qwen3-4B 的 hidden size 为 2560。每 token 的层间激活值传输量：

| 格式 | 每 token 字节 | batch=2 短 prompt micro-batch=64 传输量 | batch=4 长 prompt micro-batch=128 传输量 |
|---|---|---|---|
| f32（float32） | 2560 × 4 = 10,240 B | 64 × (16 + 10,240) ≈ 641 KB | 128 × (16 + 10,240) ≈ 1.28 MB |
| bf16（bfloat16） | 2560 × 2 = 5,120 B | 64 × (16 + 5,120) ≈ 321 KB | 128 × (16 + 5,120) ≈ 642 KB |

（每 token 另加 16B 的 `hidden_token_meta` 元信息。）

使用 bf16 将每 token 传输量减半。代价是 head 侧需要 `ggml_fp32_to_bf16_row()` 转换，tail 侧需要 `ggml_bf16_to_fp32_row()` 转回 float32 填入 `batch.embd`。这两次格式转换引入了 CPU 计算开销，但减少了共享内存的 memcpy 量。

当前默认使用 bf16（`--hidden-dtype bf16`）。

### 4.5 执行流程与流水线调度

#### 4.5.1 数据流全貌

```
    ┌──────────── Head 进程 ────────────┐     ┌──────────── Tail 进程 ────────────┐
    │                                    │     │                                    │
    │  用户 prompt                        │     │                                    │
    │     │                               │     │                                    │
    │     ▼                               │     │                                    │
    │  tokenize (BPE)                     │     │                                    │
    │     │                               │     │                                    │
    │     ▼                               │     │                                    │
    │  token_embd + layers[0..17]        │     │                                    │
    │     │                               │     │                                    │
    │     ▼                               │     │                                    │
    │  extract hidden states → bf16      │     │                                    │
    │     │                               │     │                                    │
    │     ▼                               │     │                                    │
    │  ┌──────────────────────────┐       │     │                                    │
    │  │ 写入 head-to-tail        │───────┼────→│  读取 head-to-tail                 │
    │  │ shared memory            │       │     │      │                            │
    │  └──────────────────────────┘       │     │      ▼                            │
    │                                     │     │  bf16 → f32 转换                  │
    │                                     │     │      │                            │
    │                                     │     │      ▼                            │
    │                                     │     │  填入 batch.embd                  │
    │                                     │     │      │                            │
    │                                     │     │      ▼                            │
    │                                     │     │  layers[18..35]                  │
    │                                     │     │  + final_norm + lm_head           │
    │                                     │     │      │                            │
    │                                     │     │      ▼                            │
    │                                     │     │  greedy sample → token            │
    │                                     │     │      │                            │
    │                                     │     │      ▼                            │
    │  ┌──────────────────────────┐       │     │  ┌──────────────────────────┐     │
    │  │ 读取 tail-to-head        │◄──────┼─────┤  │ 写入 tail-to-head         │     │
    │  │ shared memory            │       │     │  │ shared memory            │     │
    │  └──────────────────────────┘       │     │  └──────────────────────────┘     │
    │     │                               │     │                                    │
    │     ▼                               │     │                                    │
    │  获取下一个 decode token             │     │                                    │
    │                                    │     │                                    │
    │  循环直至 n_predict 步完成或收到 STOP │     │                                    │
    └────────────────────────────────────┘     └────────────────────────────────────┘
```

#### 4.5.2 Prefill 阶段的 Micro-Batch 流水线

prefill 阶段（处理 prompt token）是当前原型实现流水线重叠的主要阶段。Micro-batch（微批）是流水线中的一次处理和传输单位——不等于整个 batch。

以 batch=2、prompt=287 token/seq、prefill-chunk=32 为例：

```
    整个 prefill 共 287 × 2 = 574 token，被切成 9 个 micro-batch:
      MB0: 2×32=64 token (pos 0-31)
      MB1: 2×32=64 token (pos 32-63)
      MB2: 2×32=64 token (pos 64-95)
      ...
      MB7: 2×32=64 token (pos 224-255)
      MB8: 2×31=62 token (pos 256-286)

    流水线时序（Head 和 Tail 重叠执行）:

    Time ──────────────────────────────────────────────────────────────►

    Head:   [decode MB0]  [send MB0]  [decode MB1]  [send MB1]  [decode MB2]  [send MB2]  ...
               │              ╲            │              ╲            │
               │               ╲           │               ╲           │
    Tail:                      [recv MB0]  [decode MB0]    [recv MB1]  [decode MB1]  ...
                                              ↑                            ↑
                                          Overlap!                     Overlap!
```

Head 发完 MB0 的 hidden states 后**不等待 tail 处理完**，立即开始 MB1 的计算。Tail 收到 MB0 后进行后 18 层的 decode。在同一时刻，Head 在处理 MB1 的前 18 层，Tail 在处理 MB0 的后 18 层——两颗 CPU 同时在工作。

**为什么 prefill 阶段可以实现重叠？**

因为 prefill 阶段多个 micro-batch 之间的数据依赖是单向的：MB0 的结果不需要等 MB1 完成。每个 micro-batch 是独立的 prefill chunk，head 处理完一个 micro-batch 的前 18 层后就可以立即把 hidden states 发给 tail，然后继续处理下一个 micro-batch。

#### 4.5.3 Decode 阶段的锁步交替（无流水线重叠）

decode 阶段（自回归生成 token）与 prefill 有本质不同。以 parallel=2 为例，时序如下：

```
    Step N:
    ┌────────────────────────────┬────────────────────────────┐
    │           Head             │            Tail            │
    │                            │                            │
    │  ① 阻塞等待 tail 发来       │                            │
    │     seq0 和 seq1 的 token  │                            │
    │              ◄──── token ──│  ② 上一步 decode 后采样      │
    │                            │     发送 token 给 head      │
    │  ③ 收到 token 后            │                            │
    │     decode layer [0,18)   │                            │
    │                            │                            │
    │  ④ 发 hidden states ──────►│  ⑤ 收到 hidden states       │
    │                            │     decode layer [18,36)   │
    │                            │     采样 token              │
    │                            │                            │
    │  ⑥ 阻塞等 token ...        │                            │
    └────────────────────────────┴────────────────────────────┘

    任一时刻只有一侧 CPU 在工作，另一侧在阻塞等待。
```

decode 阶段无法实现流水线重叠的根源在于**自回归生成的数据依赖**：

- Head step N 必须等 Tail step N-1 产出的 token（因为需要知道下一个位置输入什么 token）。
- Tail step N 必须等 Head step N 产出的 hidden states（因为后 18 层的输入依赖于前 18 层的输出）。

这是一个无法打破的串行链。层切分在此场景下只是把一次 decode 拆成两次串行的半次 decode，加上通信开销，从理论上讲 decode 阶段的延迟会比单进程更慢。

### 4.6 与 llama-parallel 的 Prompt 模板对齐

为公平对比，当前原型内嵌了与 `llama-parallel` 相同的 multi-turn conversation 系统 prompt。head 进程构造 prompt 的逻辑为：

```cpp
const std::string prompt =
    std::string(PIPELINE_PARALLEL_SYSTEM_PROMPT) +
    "User:\n" + args.prompt + "\nAssistant:\n";
```

其中 `PIPELINE_PARALLEL_SYSTEM_PROMPT` 是一个硬编码的 multi-turn conversation 前缀（含两轮示例对话），与 `llama-parallel` 内置模板完全一致。这确保了两者在 benchmark 时处理的 prompt token 数量相同（短 prompt 场景均为 287 token，长 prompt 场景均为 643 token）。

### 4.7 Build 系统

新增工具目标 `llama-pipeline-brick` 通过 `tools/pipeline-brick/CMakeLists.txt` 加入到构建系统：

```cmake
add_executable(llama-pipeline-brick pipeline-brick.cpp)
target_link_libraries(llama-pipeline-brick PRIVATE common llama)
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

cmake --build build -j 32 --target llama-pipeline-brick
cmake --build build -j 32 --target llama-parallel  # baseline
```

---

## 5. 与标书方案的技术映射

下表逐项对比当前原型的实现方式与标书目标方案，明确哪些已经通过软件模拟验证、哪些仍是设计目标。

| 标书方案的技术点 | 标书目标实现 | 当前原型的模拟方式 | 映射程度 |
|---|---|---|---|
| **Memory Window** | PCIe NTB BAR 空间，对端可像本地内存一样访问 | mmap `/dev/shm` 共享文件，通过 `mbind` 将物理页绑定到接收方 NUMA | 语义等价，性能路径不同 |
| **Doorbell** | 写 PCIe 寄存器触发 MSI/MSI-X 中断，延迟 ~1μs | 共享内存原子状态标记 + `sleep_us(50)` 轮询 | 功能等价（通知语义），延迟远高于硬件（50μs vs 1μs） |
| **DMA 搬运** | DMA 引擎搬运，CPU 不参与 | `memcpy` 写入/读取 shared buffer | 功能等价，但占用 CPU 带宽 |
| **零拷贝** | 发送方 DMA → 接收方 BAR 空间，`mmap` 直读 | head 计算 buffer → shared buffer memcpy → tail memcpy → `batch.embd`（另加 bf16 转换） | 未实现，当前至少 2 次 memcpy + 2 次 bf16 转换 |
| **线性菊花链** | 每 Brick 只连两个邻居，N 块 Brick 线性级联 | 点对点两条单向 ring buffer（head↔tail） | 双 Brick 拓扑等价，N>2 时需扩展 |
| **层切分加载** | 每块 Brick 只存储和加载自己负责的层 | `create_memory()` 层过滤 + `TENSOR_SKIP` 跳过非本范围张量 | **完全实现** |
| **Hidden States 直传** | 层间只传激活值，不传 KV Cache | Head→Tail 只传输 `batch_size × hidden_size` 的 hidden states | **完全实现** |
| **KV Cache 本地化** | 每个 Brick 自有 KV Cache，不跨节点 | 每个进程独立维护本层范围的 KV Cache，不通过共享内存传输 | **完全实现** |
| **Micro-batch 调度** | 打包多请求填满流水线 | `parallel × prefill_chunk` 微批调度，configurable chunk size | **完全实现** |
| **NUMA 绑核** | 飞腾 CPU 亲和性绑定 | `sched_setaffinity` + `mbind` | **完全实现** |
| **BF16 传输** | bf16 格式减少传输量 | `ggml_fp32_to_bf16_row` / `ggml_bf16_to_fp32_row` | **完全实现** |
| **编译期固件** | 配置编译期固化 | 运行时参数 `parse_args()` | 功能等价但工程化程度不同 |
| **多 Brick 堆叠** | 4+ Brick 物理堆叠 | 固定 2 Brick，硬编码 Qwen3-4B 36 层 | 尚未扩展到 N>2 |
| **Doorbell 中断驱动** | 中断驱动，空闲低功耗 | 轮询驱动 | 未实现 |
| **动态层划分** | DP 算法自动分配层 | 硬编码 layer [0,18) / [18,36) | 未实现 |
| **CXL 升级** | 未来通过固件升级迁移至 CXL Fabric | N/A（无硬件） | 未覆盖 |

---

## 6. 实验结果

### 6.1 测试配置

两组实验均在相同的双飞腾 CPU 实机上进行：

| 参数 | batch=2 短 prompt | batch=4 长 prompt |
|---|---|---|
| 每序列 prompt 长度 | 287 token | 643 token |
| 总 prefill token | 574 | 2572 |
| 每序列生成 | 64 token | 64 token |
| 总生成 token | 128 | 256 |
| 上下文大小 | 2048 or 4096 | 8192 |
| 采样方式 | top-k=1 greedy | top-k=1 greedy |

**Baseline (`llama-parallel`)**：
```sh
./build/bin/llama-parallel \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 -t 128 -c 4096 -n 64 -np 2 -ns 2 \
  -p "请用三句话介绍一下 llama.cpp 是什么。" --top-k 1
```

**Pipeline Brick**：
```sh
./build/bin/llama-pipeline-brick \
  --single-system \
  --model models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --prompt "请用三句话介绍一下 llama.cpp 是什么。" \
  --ctx-size 2048 --threads 64 --n-predict 64 --parallel 2 \
  --prefill-chunk 32 --hidden-dtype bf16 --quiet \
  --head-numa 0-3 --tail-numa 4-7
```

### 6.2 结果

#### 场景一：batch=2 短 prompt

| 指标 | llama-parallel | Pipeline Brick |
|---|---|---|
| 墙钟时间 (real) | 47.08 s | 37.25 s |
| 总 prefill token | 574 | 574 |
| 总生成 token | 128 | 128 |
| 生成吞吐量 | 2.72 tok/s | 3.44 tok/s |

```
加速比 = 47.08 / 37.25 = 1.26x
吞吐提升 = 3.44 / 2.72 - 1 = 26%
```

#### 场景二：batch=4 长 prompt

| 指标 | llama-parallel | Pipeline Brick |
|---|---|---|
| 墙钟时间 (real) | 100.49 s | 91.85 s |
| 总 prefill token | 2572 | 2572 |
| 总生成 token | 256 | 256 |
| 生成吞吐量 | 2.55 tok/s | 2.79 tok/s |

```
加速比 = 100.49 / 91.85 = 1.09x
吞吐提升 = 2.79 / 2.55 - 1 = 9%
```

### 6.3 结果分析

**两个场景下 Pipeline Brick 的墙钟时间均优于 baseline。** 但需要注意：

1. **加速随 batch 增大而衰减**：batch=2 场景 1.26x，batch=4 场景降到 1.09x。原因可能是 baseline 的 continuous batching 在更大 batch 下本身效率就更高，流水线并行带来的相对优势被削弱。

2. **加速来源不是单一的"流水线并行"**：当前实验对比的是"分层+NUMA 绑核+bf16 传输+prefill 微批流水线" vs "单进程全层+全 NUMA"。加速来自多个因素的叠加，报告中不能将全部收益笼统归于"流水线并行"。

3. **Decode 阶段无收益**：如第 4.5.3 节分析的，decode 阶段 head 和 tail 严格锁步交替，无重叠。两组实验中 prefill token 占 82%-91%，管道在 prefill 阶段获得的收益掩盖了 decode 阶段的劣势。如果测试一个 decode 主导的纯生成场景（如 prompt=1 token, n_predict=1000），Pipeline Brick 可能反而不如 baseline。

4. **1.09x 的提升在单次实验中不能排除偶然因素**：长 prompt 场景的 8.6 秒差距（91.85 vs 100.49）约 9%。在非隔离的实机环境中（受其他进程、温度、NUMA 迁移等因素影响），需要至少 3 次重复实验并报告方差才能得出可靠结论。

### 6.4 Micro-batch 日志验证

以 batch=2、prefill-chunk=32 的运行为例，日志证实了 micro-batch 和 bf16 传输均已生效：

```
pipeline-brick head: micro-batch max=64 prefill_chunk=32 hidden_dtype=bf16 bytes_per_token=5120
pipeline-brick head: llama-parallel prompt template enabled, prompt tokens=287

sent prefill pos=0   n_tokens=64 bytes=328704 dtype=bf16  (64×5120 + 64×16 = 328704)
sent prefill pos=32  n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=64  n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=96  n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=128 n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=160 n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=192 n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=224 n_tokens=64 bytes=328704 dtype=bf16
sent prefill pos=256 n_tokens=62 bytes=318432 dtype=bf16

sent decode pos=287 n_tokens=2 bytes=10272 dtype=bf16    (2×5120 + 2×16 = 10272)
sent decode pos=288 n_tokens=2 bytes=10272 dtype=bf16
...
```

574 个 prefill token 被压缩为 9 个 micro-batch（而非 574 个单 token 传输），大幅降低了调度和传输开销。

---

## 7. 当前实现的边界与局限

### 已实现

| 能力 | 说明 |
|---|---|
| 双 Brick 分层推理 | Head/Tail 分别处理不同层范围，各自只加载本范围权重 |
| NUMA 绑定 | 进程 CPU 亲和性 + 共享内存 mbind 到接收方 NUMA |
| Prefill micro-batch 流水线 | Parallel × prefill_chunk 微批，head 和 tail 在 prefill 阶段重叠执行 |
| BF16 hidden states 传输 | fp32→bf16 转换后传输，每 token 5120B，tail 侧转回 fp32 |
| Ring buffer 乒乓传输 | 64 slot，无锁 atomic 状态同步 |
| Prompt 模板对齐 | 与 llama-parallel 相同的 multi-turn conversation 模板 |

### 尚未实现

| 目标 | 说明 |
|---|---|
| 真实 PCIe NTB / CXL 驱动 | 当前是 mmap `/dev/shm` 模拟，非真实 NTB BAR 空间映射 |
| 硬件 Doorbell 中断 | 当前是 poll 轮询语义模拟，非硬件 MSI/MSI-X 中断驱动 |
| DMA 搬运 | 当前 head 用 memcpy 写入 shared buffer，tail 用 memcpy 读出 |
| 零拷贝 | 当前至少 2 次 memcpy + 2 次 bf16 格式转换 |
| Decode 阶段流水线重叠 | 受限于自回归依赖，decode 阶段无并行 |
| NUMA 张量并行 | 当前只是进程级绑核，不是把矩阵乘法真实切到多个 NUMA 节点上并行 |
| 多 Brick (N>2) | 当前固定 2 Brick，硬编码 36 层 Qwen3-4B |
| 动态层划分 | 当前硬编码 layer [0,18) / [18,36)，未实现 DP 算法自动分配 |
| 多模型支持 | 当前仅支持 Qwen3-4B dense 版本 |

### 核心理论局限：Decode 阶段无法并行

这是自回归生成的根本约束，不是工程问题。对于 2-Brick 层切分流水线，decode 每步的延迟为：

```
T_step = T_head(0-17层) + T_transport + T_tail(18-35层)
```

而单进程的 decode 延迟为 `T_single(0-35层)`。如果分层后两端各 18 层的计算时间之和等于单进程 36 层的时间，那么加上传输开销，**decode 阶段的 step latency 理论上比单进程更长**。

流水线并行的改进方向是在 decode 阶段做 **speculative decoding**（预测性解码）或 **multi-token prediction**（多 token 预测），而非单纯的 layer pipeline。

---

## 8. 改进路线

按优先级排列的后续工作方向：

### 短期（原型优化）

1. **增加 `--prompt-file` 支持**：避免长 prompt 只能用 shell 变量传入，对齐 `llama-parallel` 的 `-f` 参数。
2. **增加性能统计**：在 head 和 tail 中埋点记录 prefill time、decode time、head wait time（等 token）、tail wait time（等 hidden）、transport time（send/recv 耗时），以拆解各阶段开销。
3. **参数扫描**：测试 `parallel=1/2/4/8` × `prefill-chunk=16/32/64/128` 的 16 组组合，找实机最佳参数点。
4. **减少 memcpy 次数**：探索是否可以在 head 的 compute buffer 中直接以 bf16 格式写入 shared buffer，避免中间一次拷贝。
5. **纯 decode 场景测试**：跑一个 prompt=1、n_predict=500 的测试，量化 decode 阶段 pipeline brick 相对于 baseline 的相对劣势。

### 中期（标书验证）

6. **扩展至 N>2 Brick**：将硬编码的 36 层 2 Brick 改为 `--layer-split` 参数支持的 N 层 M Brick 通用切分。
7. **真实 NTB/CXL 环境适配**：如果拿到真实飞腾 NTB 开发板或 CXL 设备文件路径，替换当前的 mmap `/dev/shm` 路径。

### 长期（硬件对齐）

8. **NUMA 张量并行**：在 ggml 后端中将 Q4_K_M 矩阵乘法真正切分到多个 NUMA 节点上并行（工程量很大）。
9. **中断驱动运行时**：当有真实 Doorbell 设备后，将轮询改为硬件中断驱动的事件循环。

---

## 9. 总结

当前 Pipeline Brick 原型已经实现了标书方案的核心功能路径：

- **层切分 + 双 NUMA 域部署**（通过进程 fork + CPU/NUMA 绑定）
- **Memory Window 语义的层间通信**（通过 mmap `/dev/shm` + `mbind` 模拟）
- **只传 hidden states，KV Cache 本地化**（对标书方案 5.4 节的直接验证）
- **Micro-batch 流水线调度**（prefill 阶段有实际重叠，对标书方案 4.3 节的验证）

在实机上取得了相对于 `llama-parallel` 的 1.09x-1.26x 端到端提升。但需要客观认识到：

- **Decode 阶段无流水线并行**，提升完全来自 prefill 阶段。
- **加速因素不单一**：NUMA 局部性、bf16 传输、prefill 微批各自有贡献，当前实验无法区分。
- **长 batch + 长 prompt 场景下收益衰减**（1.26x → 1.09x），提示 baseline 自身的 continuous batching 有效率优势。
- 1.09x 的结果需要多次重复实验验证。

最稳妥的总体结论：

> 当前 Pipeline Brick 原型已经验证了双卡分层推理、多 sequence micro-batch 流水线、bf16 hidden states 传输和 NUMA 绑核的可行性。它在实机上相对于 `llama-parallel` 取得了 1.09x 到 1.26x 的端到端提升，符合标书方案的主干思路。但当前仍是软件原型——prefill 阶段有流程重叠，decode 阶段无并行；真实 PCIe NTB/CXL、DMA、硬件 Doorbell 和 NUMA 张量并行尚未落地，通信路径本质上仍是 shared memory memcpy + poll。

---

*报告完成时间：2026 年 6 月*
