# Pipeline Brick 双卡 CPU 推理测试命令

本文档用于在实际飞腾双卡 CPU 机器上运行 `llama-pipeline-brick` 推理测试。当前机器拓扑按以下配置：

- 第一张卡：NUMA nodes `0-3`
- 第二张卡：NUMA nodes `4-7`
- 两张卡共属一个 Linux 系统

NUMA node（非统一内存访问节点，即一组 CPU 核和靠近它的内存）用于区分两张卡的计算和内存区域。

## 构建

```sh
cmake -B build -DLLAMA_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build -j 32 --target llama-pipeline-brick
```

参数说明：

- `-DLLAMA_NATIVE=ON`：按本机 CPU 指令集优化编译。
- `-DGGML_OPENMP=ON`：启用 OpenMP 多线程。
- `--target llama-pipeline-brick`：只编译本实验工具。

## 推理测试命令

优先使用普通用户运行：

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

如果出现 `mbind` 权限或内存绑定失败，再使用 `sudo`：

```sh
sudo ./build/bin/llama-pipeline-brick \
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

## 参数解释

- `--single-system`：单 Linux 系统双进程模式。程序自动启动 head 和 tail 两个子进程。
- `--model`：模型权重路径。当前固定使用 Qwen3-4B GGUF。
- `--prompt`：输入提示词。
- `--ctx-size 2048`：每条 sequence 的上下文长度为 2048。
- `--threads 64`：每个 Brick 进程使用 64 个 CPU 线程。
- `--n-predict 64`：最多生成 64 个 token。
- `--parallel 4`：同时维护 4 条 sequence，用于填充流水线。
- `--head-numa 0-3`：head 进程绑定第一张卡的 NUMA nodes 0 到 3。
- `--tail-numa 4-7`：tail 进程绑定第二张卡的 NUMA nodes 4 到 7。

head 进程（头节点，即模型前半段）运行 Qwen3-4B 的 layers `[0,18)`。

tail 进程（尾节点，即模型后半段）运行 Qwen3-4B 的 layers `[18,36)`。

## 当前数据流

```text
head 进程，绑定 NUMA 0-3
  运行 token embedding + layers [0,18)
  输出 hidden states
  memcpy 写入 head-to-tail shared buffer

head-to-tail shared buffer
  通过 mbind 绑定到 NUMA 4-7
  相当于 tail 侧内存窗口

tail 进程，绑定 NUMA 4-7
  从 shared buffer 读取 hidden states
  写入 batch.embd
  运行 layers [18,36) + final norm + lm_head
  采样生成 token

tail-to-head shared buffer
  通过 mbind 绑定到 NUMA 0-3
  回传 token 给 head
```

hidden states（隐藏状态，即模型前半层输出给后半层的中间激活）当前格式为 `float32[1][2560]`，每个 token 传输：

```text
2560 * 4 bytes = 10240 bytes
```

## 预期日志

正常运行时应看到类似日志：

```text
pipeline-brick numa: bound head-to-tail shared window to NUMA nodes 4-7
pipeline-brick numa: bound tail-to-head shared window to NUMA nodes 0-3
pipeline-brick single-system: head NUMA=0-3 tail NUMA=4-7
pipeline-brick head: brick=0 peer=1 layers [0,18), parallel=4, n_embd=2560
pipeline-brick tail: brick=1 peer=0 layers [18,36), parallel=4, n_embd=2560
pipeline-brick head: sent seq=0 ...
pipeline-brick tail: recv seq=0 ...
```

如果 `seq=0,1,2,3` 交错出现，说明多个 sequence 正在填充流水线。

## 重要说明

当前已经实现：

- 双 Brick 流水线并行。
- head/tail 分别绑定不同 NUMA 区域。
- head-to-tail buffer 绑定到 tail NUMA。
- tail-to-head buffer 绑定到 head NUMA。
- 多 sequence 微批流水线。

当前尚未实现：

- 硬件 Doorbell 中断。
- DMA 搬运。
- 严格零拷贝。
- NUMA 级张量并行。
- bfloat16/f16 hidden states 传输。

所以当前版本可以表述为：

```text
单 Linux 系统双 NUMA 域下的双 Brick 流水线并行原型，
使用接收方 NUMA 内存窗口传输 hidden states。
```
