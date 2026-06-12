# 本地 CPU 官方测试工具实验说明

本文档说明如何在当前本地 WSL 环境中，用 llama.cpp 官方工具测试 Qwen3-4B GGUF 模型的 CPU 推理性能。这里不使用自定义脚本，只使用仓库自带的官方可执行程序。

## 1. 当前实验背景

当前模型文件：

```sh
models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

当前目标是先在本机 CPU 上跑通基线性能，为后续修改稀疏注意力代码做对比。

CPU-only（只用 CPU 推理）表示不把模型层 offload（卸载到其他设备运行，即把部分计算放到 GPU 或其他加速器上）到 GPU。这里统一使用：

```sh
-ngl 0
```

`-ngl` 是 `n_gpu_layers`（GPU 层数，即放到 GPU 上执行的 Transformer 层数）。设置为 `0` 表示所有层都在 CPU 上执行。

## 2. 先构建官方测试工具

在仓库根目录执行：

```sh
cmake --build build -j 8 --target llama-bench llama-perplexity llama-batched-bench
```

构建成功后，可执行文件在：

```sh
build/bin/llama-bench
build/bin/llama-perplexity
build/bin/llama-batched-bench
```

注意：CMake target（构建目标，即 cmake --build 里指定要构建的名字）叫 `llama-perplexity` 和 `llama-batched-bench`，不是 `perplexity` 或 `batched-bench`。

## 3. 快速确认模型能跑

先用 `llama-cli`（交互式或单次文本生成工具，即直接让模型根据 prompt 生成文字）做一次最小测试：

```sh
./build/bin/llama-cli \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -c 2048 \
  -n 64 \
  --no-warmup \
  -p "请用三句话介绍一下 llama.cpp 是什么。"
```

关键参数：

- `-m`: model path（模型路径，即要加载的 GGUF 权重文件）。
- `-t`: threads（线程数，即 CPU 并行工作线程数量）。
- `-c`: context size（上下文长度，即一次推理最多能保留多少 token）。
- `-n`: n predict（生成 token 数，即最多继续生成多少个 token）。
- `--no-warmup`: no warmup（跳过预热，即不先跑一次额外推理）。

这一步不是正式 benchmark（基准测试，即可重复比较的性能测试），主要确认模型、路径、CPU 后端都正常。

## 4. 使用 llama-bench 测标准性能

`llama-bench`（官方性能测试工具，即专门统计 prompt 处理和生成速度的 benchmark 程序）是当前最适合做基线对比的工具。

### 4.1 基础 CPU benchmark

```sh
./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -p 512,2048,4096 \
  -n 128 \
  -r 3 \
  -o md
```

关键参数：

- `-m`: model path（模型路径，即要加载的 GGUF 权重文件）。
- `-ngl`: n gpu layers（GPU 层数，即放到 GPU 上执行的模型层数）。`-ngl 0` 表示全部在 CPU 上跑。
- `-t`: threads（线程数，即 CPU 并行计算使用的线程数量）。
- `-p`: n prompt（输入 token 数，即 benchmark 构造多少个 prompt token 来测试提示词处理速度）。
- `-n`: n generation（生成 token 数，即 benchmark 测试逐 token 解码时生成多少个 token）。
- `-r`: repetitions（重复次数，即同一个测试配置重复跑几遍后取平均值和波动）。
- `-o`: output format（输出格式，即结果按什么格式打印）。`md` 表示 Markdown 表格，方便复制到实验记录里。

结果里重点看：

- `pp 512`、`pp 2048`、`pp 4096`: prompt processing（提示词处理，即把输入 token 批量送进模型建立 KV cache）的速度。
- `tg 128`: text generation（文本生成，即模型逐 token 解码生成）的速度。
- `t/s`: tokens per second（每秒 token 数，即吞吐量）。
- `stddev`: standard deviation（标准差，即多次重复测试的波动）。

对于稀疏注意力优化，优先关注长上下文下的 `tg` 速度，因为 decode（解码阶段，即逐 token 生成阶段）会反复访问历史 KV cache，注意力优化更容易体现差异。

### 4.2 测不同线程数

```sh
./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 1,2,4,8 \
  -p 512 \
  -n 128 \
  -r 3 \
  -o md
```

这个实验用于找本机 CPU 的合理线程数。线程数不是越大越好，过多线程可能因为调度、缓存争用导致生成速度下降。

### 4.3 测不同上下文深度

```sh
./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -p 512 \
  -n 128 \
  -d 0,2048,4096,8192 \
  -r 3 \
  -o md
```

`-d` 是 depth（上下文深度，即 benchmark 先预填多少 token 到 KV cache 里，再测试当前位置的速度）。例如 `-d 8192` 表示在已有 8192 token 历史上下文的情况下测试。

这个实验很关键。普通 dense attention（稠密注意力，即每个新 token 都看全部历史 token）在 `-d` 增大时通常会变慢。如果后续做 sparse attention（稀疏注意力，即只看部分重要历史 token），这里就是最直接的对比位置。

### 4.4 保存结果到日志文件

不需要写脚本，可以直接用 `tee` 保存：

```sh
mkdir -p experiment_logs

./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -p 512,2048,4096 \
  -n 128 \
  -d 0,2048,4096,8192 \
  -r 3 \
  -o md | tee experiment_logs/qwen3-4b-cpu-llama-bench.md
```

`tee`（边显示边保存，即终端能看到输出，同时写入文件）适合记录实验结果。

## 5. 使用 llama-batched-bench 测并发吞吐

`llama-batched-bench`（批量解码 benchmark，即模拟多个请求一起推理时的吞吐）适合测试服务型场景。

基础命令：

```sh
./build/bin/llama-batched-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -c 8192 \
  -b 2048 \
  -ub 512 \
  -npp 128,512 \
  -ntg 128 \
  -npl 1,2,4,8
```

关键参数：

- `-c`: KV cache 最大上下文容量。
- `-b`: batch size（批大小，即逻辑上一次处理的最大 token 数）。
- `-ub`: ubatch size（微批大小，即底层实际分块计算的 token 数）。
- `-npp`: prompt tokens per parallel request（每个并发请求的输入 token 数）。
- `-ntg`: generated tokens per request（每个请求生成的 token 数）。
- `-npl`: parallel prompts（并发请求数量）。

结果里重点看：

- `S_PP`: prompt processing speed（输入处理吞吐）。
- `S_TG`: text generation speed（生成吞吐）。
- `N_KV`: required KV cache size（需要的 KV cache token 容量）。

KV cache（键值缓存，即保存历史 token 的 key/value 向量，避免每次生成都重新计算历史）会随着上下文长度和并发数增加而占用更多内存。

## 6. 使用 llama-perplexity 测质量变化

`llama-perplexity`（困惑度测试工具，即衡量模型预测文本下一个 token 的平均难度）主要用于评估量化或推理改动是否明显损害模型质量。

先准备测试文本。官方通常使用 Wikitext-2，可以执行：

```sh
bash scripts/get-wikitext-2.sh
```

然后查看脚本下载出的文件路径：

```sh
find . -iname '*wikitext*' -type f
```

假设测试文件是 `models/wikitext-2-raw/wiki.test.raw`，可运行：

```sh
./build/bin/llama-perplexity \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -c 2048 \
  -f models/wikitext-2-raw/wiki.test.raw
```

PPL（perplexity，困惑度，即模型对测试文本的平均不确定程度）越低通常越好。但不同 tokenizer（分词器，即把文本切成 token 的规则）之间的 PPL 不适合直接横向比较。

对于本项目，`llama-perplexity` 更适合做同一模型、同一测试集、同一参数下的前后对比。例如：

- 原始 dense attention 版本 PPL。
- 修改 sparse attention 后 PPL。
- 比较二者的 PPL 是否明显变差。

## 7. 推荐实验顺序

建议按下面顺序做，避免一上来就跑很长实验：

1. 跑 `llama-cli` 最小生成，确认模型能正常输出中文。
2. 跑 `llama-bench -t 1,2,4,8`，确定本机最佳线程数。
3. 跑 `llama-bench -d 0,2048,4096,8192`，得到长上下文 baseline（基线，即没有优化时的对照数据）。
4. 跑 `llama-batched-bench`，记录并发吞吐。
5. 跑 `llama-perplexity`，记录质量指标。
6. 修改稀疏注意力后，用完全相同命令再跑一遍。

## 8. 建议记录表格

每次实验建议记录这些字段：

```text
日期:
git commit:
模型:
量化:
CPU:
线程数:
上下文长度:
KV cache 类型:
是否开启 flash attention:
命令:
pp 速度:
tg 速度:
PPL:
备注:
```

`flash attention`（Flash Attention，降低注意力计算中显存或内存访问开销的注意力实现）在 CPU 上是否真正有收益要以实测为准。测试时保持参数一致，否则前后结果不能直接比较。

## 9. 当前建议基线命令

本机当前优先跑下面这条，作为第一版正式 baseline：

```sh
mkdir -p experiment_logs

./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -p 512,2048,4096 \
  -n 128 \
  -d 0,2048,4096,8192 \
  -r 3 \
  -o md | tee experiment_logs/qwen3-4b-cpu-baseline.md
```

如果这条太慢，先缩小范围：

```sh
./build/bin/llama-bench \
  -m models/qwen3-4b/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 0 \
  -t 8 \
  -p 512 \
  -n 64 \
  -d 0,2048 \
  -r 1 \
  -o md
```
