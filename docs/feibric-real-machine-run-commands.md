# Feibric 实机运行命令

本文档给出可直接复制的实机命令。默认使用 F16 模型，不再使用 Q4 作为主实验口径。

除 prompt 外，命令中不使用 `export` 环境变量，所有关键参数都直接写在命令里。

## 0. 公共准备

两台机器都进入同一代码目录：

```sh
cd /root/yzw-test/llama5/llama.cpp
mkdir -p bench-logs /tmp/feibric-rdma
```

机器 1 需要读取 prompt：

```sh
LONG_PROMPT="$(cat prompts/long_1k_prompt.txt)"
```

构建命令：

```sh
cmake --build build --target llama-pipeline-brick llama-parallel -j 64
```

如果还没有 TP 分片，在两台机器都准备好 `.tp0.gguf` 到 `.tp3.gguf`：

```sh
python tools/gguf-split/gguf-split.py \
  models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --tp 4

ls -lh models/qwen3-4b-f16/*tp*.gguf
```

参数说明：

`--hidden-dtype bf16` 中的 `hidden-dtype`（隐藏状态传输数据类型，即流水线 stage 之间传 hidden states 时使用的数据格式）只影响 CPU/stage 之间发送的中间激活 payload，不改变模型权重格式。这里的 `bf16` 是 bfloat16（16 位浮点格式），对 Qwen3-4B 的 `n_embd=2560` 来说，每个 token 的 hidden states 传输量从 F32 的 10240 bytes 降到 BF16 的 5120 bytes，用来降低 CXL/RDMA 链路传输压力。

检查 IB 设备名。下面命令默认使用 `mlx5_0`、port `1`、gid index `0`、MTU `1024`，如果实机不同，把后面命令里的对应参数替换掉：

```sh
ibv_devinfo
ibstat
rdma link
```

## 1. 单 Domain TP 消融实验

Domain（域，即一个 Linux 系统内的一组 CPU 和 NUMA 资源）：本节只使用第一台机器的两颗飞腾 CPU，不启动第二个 Domain，也不使用 RDMA。

TP / Tensor Parallelism（张量并行，即把同一层矩阵计算拆给多个 NUMA rank 后再规约）：本节用于单独观察 `--tp-size 4` 是否比 `--tp-size 1` 更快。

### 1.1 单 Domain pipeline, 不使用 TP

在第一台机器执行：

```sh
/usr/bin/time -p -o bench-logs/single_domain_pipeline_tp1.time \
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
  --tp-size 1 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --quiet \
  > bench-logs/single_domain_pipeline_tp1.out \
  2> bench-logs/single_domain_pipeline_tp1.log
```

### 1.2 单 Domain pipeline + TP=4

在第一台机器执行：

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

查看对比结果：

```sh
grep 'pipeline-brick perf' bench-logs/single_domain_pipeline_tp*.log
grep 'pipeline-brick TP:' bench-logs/single_domain_pipeline_tp4.log
cat bench-logs/single_domain_pipeline_tp*.time
```

## 2. 双 Domain 公共准备

机器 1 清理 CPU0 到 CPU1 的 CXL/shared-window 文件：

```sh
rm -f /dev/shm/feibric-a-s0-to-s1 /dev/shm/feibric-a-s1-to-s0
```

机器 2 清理 CPU2 到 CPU3 的 CXL/shared-window 文件：

```sh
rm -f /dev/shm/feibric-b-s2-to-s3 /dev/shm/feibric-b-s3-to-s2
```

推荐启动顺序：

```text
机器 2: 先启动 stage3
机器 2: 再启动 stage2
机器 1: 再启动 stage1
手工交换 RDMA peer info 文件
机器 1: 最后启动 stage0
```

RDMA / Remote Direct Memory Access（远程直接内存访问，即跨机器网卡直接传输内存数据）：`ib-rdma` 会先写本端 info 文件，然后等待对端 peer info 文件。两台机器没有普通网络时，可以通过 KVM、串口、剪贴板或人工方式复制文件内容。

## 3. 场景 A: full, CXL + IB RDMA + 4-stage pipeline + TP=4 + stream-kv

两台机器都先清理本场景 RDMA 文件：

```sh
rm -f /tmp/feibric-rdma/full_cxl_ib_tp4_stream.*
```

### 3.1 机器 2, stage3

```sh
/usr/bin/time -p -o bench-logs/full_cxl_ib_tp4_stream_stage3.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 3 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-b-s2-to-s3 \
  --up-tx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --quiet \
  > bench-logs/full_cxl_ib_tp4_stream_stage3.out \
  2> bench-logs/full_cxl_ib_tp4_stream_stage3.log
```

### 3.2 机器 2, stage2

```sh
/usr/bin/time -p -o bench-logs/full_cxl_ib_tp4_stream_stage2.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 2 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport ib-rdma \
  --up-rdma-dev mlx5_0 \
  --up-rdma-local-info /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage2.local \
  --up-rdma-peer-info /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage1.peer \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-b-s2-to-s3 \
  --down-rx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/full_cxl_ib_tp4_stream_stage2.out \
  2> bench-logs/full_cxl_ib_tp4_stream_stage2.log
```

### 3.3 机器 1, stage1

```sh
/usr/bin/time -p -o bench-logs/full_cxl_ib_tp4_stream_stage1.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 1 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-a-s0-to-s1 \
  --up-tx-mw /dev/shm/feibric-a-s1-to-s0 \
  --down-transport ib-rdma \
  --down-rdma-dev mlx5_0 \
  --down-rdma-local-info /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage1.local \
  --down-rdma-peer-info /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage2.peer \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/full_cxl_ib_tp4_stream_stage1.out \
  2> bench-logs/full_cxl_ib_tp4_stream_stage1.log
```

### 3.4 交换 RDMA peer info

机器 1 查看 stage1 本端信息：

```sh
cat /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage1.local
```

把上面内容复制到机器 2：

```sh
cat > /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage1.peer <<'EOF'
把机器1的 stage1.local 内容粘贴到这里
EOF
```

机器 2 查看 stage2 本端信息：

```sh
cat /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage2.local
```

把上面内容复制到机器 1：

```sh
cat > /tmp/feibric-rdma/full_cxl_ib_tp4_stream.stage2.peer <<'EOF'
把机器2的 stage2.local 内容粘贴到这里
EOF
```

### 3.5 机器 1, stage0

```sh
/usr/bin/time -p -o bench-logs/full_cxl_ib_tp4_stream_stage0.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 0 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --prompt "$LONG_PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-a-s0-to-s1 \
  --down-rx-mw /dev/shm/feibric-a-s1-to-s0 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --quiet \
  > bench-logs/full_cxl_ib_tp4_stream_stage0.out \
  2> bench-logs/full_cxl_ib_tp4_stream_stage0.log
```

查看 full 结果：

```sh
grep 'pipeline-brick perf' bench-logs/full_cxl_ib_tp4_stream_stage0.log
cat bench-logs/full_cxl_ib_tp4_stream_stage3.out
```

## 4. 场景 B: 不使用 TP, 保留 CXL + IB RDMA + 4-stage pipeline + stream-kv

两台机器都先清理本场景 RDMA 文件：

```sh
rm -f /tmp/feibric-rdma/no_tp_cxl_ib_stream.*
```

### 4.1 机器 2, stage3

```sh
/usr/bin/time -p -o bench-logs/no_tp_cxl_ib_stream_stage3.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 3 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 1 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-b-s2-to-s3 \
  --up-tx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --quiet \
  > bench-logs/no_tp_cxl_ib_stream_stage3.out \
  2> bench-logs/no_tp_cxl_ib_stream_stage3.log
```

### 4.2 机器 2, stage2

```sh
/usr/bin/time -p -o bench-logs/no_tp_cxl_ib_stream_stage2.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 2 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 1 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport ib-rdma \
  --up-rdma-dev mlx5_0 \
  --up-rdma-local-info /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage2.local \
  --up-rdma-peer-info /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage1.peer \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-b-s2-to-s3 \
  --down-rx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/no_tp_cxl_ib_stream_stage2.out \
  2> bench-logs/no_tp_cxl_ib_stream_stage2.log
```

### 4.3 机器 1, stage1

```sh
/usr/bin/time -p -o bench-logs/no_tp_cxl_ib_stream_stage1.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 1 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 1 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-a-s0-to-s1 \
  --up-tx-mw /dev/shm/feibric-a-s1-to-s0 \
  --down-transport ib-rdma \
  --down-rdma-dev mlx5_0 \
  --down-rdma-local-info /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage1.local \
  --down-rdma-peer-info /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage2.peer \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/no_tp_cxl_ib_stream_stage1.out \
  2> bench-logs/no_tp_cxl_ib_stream_stage1.log
```

### 4.4 交换 RDMA peer info

```sh
cat /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage1.local
```

把机器 1 的内容复制到机器 2：

```sh
cat > /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage1.peer <<'EOF'
把机器1的 stage1.local 内容粘贴到这里
EOF
```

```sh
cat /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage2.local
```

把机器 2 的内容复制到机器 1：

```sh
cat > /tmp/feibric-rdma/no_tp_cxl_ib_stream.stage2.peer <<'EOF'
把机器2的 stage2.local 内容粘贴到这里
EOF
```

### 4.5 机器 1, stage0

```sh
/usr/bin/time -p -o bench-logs/no_tp_cxl_ib_stream_stage0.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 0 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --prompt "$LONG_PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 1 \
  --stream-kv --stream-kv-sink 16 --stream-kv-recent 128 \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-a-s0-to-s1 \
  --down-rx-mw /dev/shm/feibric-a-s1-to-s0 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --quiet \
  > bench-logs/no_tp_cxl_ib_stream_stage0.out \
  2> bench-logs/no_tp_cxl_ib_stream_stage0.log
```

## 5. 场景 C: 不使用稀疏, 保留 CXL + IB RDMA + 4-stage pipeline + TP=4

两台机器都先清理本场景 RDMA 文件：

```sh
rm -f /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.*
```

### 5.1 机器 2, stage3

```sh
/usr/bin/time -p -o bench-logs/no_sparse_cxl_ib_tp4_stage3.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 3 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-b-s2-to-s3 \
  --up-tx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --quiet \
  > bench-logs/no_sparse_cxl_ib_tp4_stage3.out \
  2> bench-logs/no_sparse_cxl_ib_tp4_stage3.log
```

### 5.2 机器 2, stage2

```sh
/usr/bin/time -p -o bench-logs/no_sparse_cxl_ib_tp4_stage2.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 2 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --up-transport ib-rdma \
  --up-rdma-dev mlx5_0 \
  --up-rdma-local-info /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage2.local \
  --up-rdma-peer-info /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage1.peer \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-b-s2-to-s3 \
  --down-rx-mw /dev/shm/feibric-b-s3-to-s2 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/no_sparse_cxl_ib_tp4_stage2.out \
  2> bench-logs/no_sparse_cxl_ib_tp4_stage2.log
```

### 5.3 机器 1, stage1

```sh
/usr/bin/time -p -o bench-logs/no_sparse_cxl_ib_tp4_stage1.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 1 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 64-127 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --up-transport cxl \
  --up-rx-mw /dev/shm/feibric-a-s0-to-s1 \
  --up-tx-mw /dev/shm/feibric-a-s1-to-s0 \
  --down-transport ib-rdma \
  --down-rdma-dev mlx5_0 \
  --down-rdma-local-info /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage1.local \
  --down-rdma-peer-info /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage2.peer \
  --head-numa 4-7 \
  --tail-numa 0-3 \
  --rdma-port 1 \
  --rdma-gid-index 0 \
  --rdma-mtu 1024 \
  --quiet \
  > bench-logs/no_sparse_cxl_ib_tp4_stage1.out \
  2> bench-logs/no_sparse_cxl_ib_tp4_stage1.log
```

### 5.4 交换 RDMA peer info

```sh
cat /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage1.local
```

把机器 1 的内容复制到机器 2：

```sh
cat > /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage1.peer <<'EOF'
把机器1的 stage1.local 内容粘贴到这里
EOF
```

```sh
cat /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage2.local
```

把机器 2 的内容复制到机器 1：

```sh
cat > /tmp/feibric-rdma/no_sparse_cxl_ib_tp4.stage2.peer <<'EOF'
把机器2的 stage2.local 内容粘贴到这里
EOF
```

### 5.5 机器 1, stage0

```sh
/usr/bin/time -p -o bench-logs/no_sparse_cxl_ib_tp4_stage0.time \
./build/bin/llama-pipeline-brick \
  --domain-mode dual \
  --stage-id 0 \
  --stage-count 4 \
  --stage-numa '0-3;4-7;0-3;4-7' \
  --numa-cpus 0-63 \
  --model models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  --prompt "$LONG_PROMPT" \
  --ctx-size 8192 \
  --threads 64 \
  --n-predict 64 \
  --parallel 4 \
  --prefill-chunk 32 \
  --hidden-dtype bf16 \
  --tp-size 4 \
  --down-transport cxl \
  --down-tx-mw /dev/shm/feibric-a-s0-to-s1 \
  --down-rx-mw /dev/shm/feibric-a-s1-to-s0 \
  --head-numa 0-3 \
  --tail-numa 4-7 \
  --quiet \
  > bench-logs/no_sparse_cxl_ib_tp4_stage0.out \
  2> bench-logs/no_sparse_cxl_ib_tp4_stage0.log
```

## 6. 不使用 pipeline 的同步 baseline

当前代码中“不使用 pipeline”的同步方案就是原始 `llama-parallel` 单进程推理。它不使用 CXL、IB RDMA、Pipeline Brick、TP 和 stream-kv。

在任意一台单 Linux 双 CPU 机器上执行：

```sh
mkdir -p bench-logs

/usr/bin/time -p -o bench-logs/baseline_llama_parallel_f16.time \
./build/bin/llama-parallel \
  -m models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  -ngl 0 \
  -t 128 \
  -c 8192 \
  -n 64 \
  -np 4 \
  -ns 4 \
  -f prompts/long_1k_prompt.txt \
  --top-k 1 \
  > bench-logs/baseline_llama_parallel_f16.out \
  2> bench-logs/baseline_llama_parallel_f16.log
```

## 7. 什么都不使用的 baseline

与第 6 节相同，使用 `llama-parallel`：

```sh
/usr/bin/time -p -o bench-logs/baseline_nothing_llama_parallel_f16.time \
./build/bin/llama-parallel \
  -m models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf \
  -ngl 0 \
  -t 128 \
  -c 8192 \
  -n 64 \
  -np 4 \
  -ns 4 \
  -f prompts/long_1k_prompt.txt \
  --top-k 1 \
  > bench-logs/baseline_nothing_llama_parallel_f16.out \
  2> bench-logs/baseline_nothing_llama_parallel_f16.log
```

说明：第 6 节和第 7 节本质上是同一个 baseline。因为当前没有“单进程同步 + TP + stream-kv”的独立 CLI，取消 pipeline 后就回到原始 llama.cpp 同步推理。

## 8. 结果提取

单 Domain TP 消融结果：

```sh
grep 'pipeline-brick perf' bench-logs/single_domain_pipeline_tp*.log
grep 'pipeline-brick TP:' bench-logs/single_domain_pipeline_tp4.log
cat bench-logs/single_domain_pipeline_tp*.time
```

双 Domain Pipeline Brick 结果看 stage0 日志：

```sh
grep 'pipeline-brick perf' bench-logs/*stage0.log
```

生成文本看最后 stage 输出：

```sh
cat bench-logs/*stage3.out
```

llama-parallel baseline 结果：

```sh
grep -E 'Total prompt tokens|Total gen tokens|Total speed|speed:' bench-logs/baseline*.log
cat bench-logs/baseline*.time
```

## 9. 注意事项

- `--tp-size 4` 时必须存在 `.tp0.gguf` 到 `.tp3.gguf`。
- `--tp-size 1` 时使用完整 F16 模型，不需要 TP 分片。
- `--stage-numa '0-3;4-7;0-3;4-7'` 的含义是四个 stage 分别使用对应 NUMA 节点。机器 2 上的 stage2/stage3 仍使用本机的 NUMA `0-3` 和 `4-7`。
- CXL 链路只用于同一个 Linux 系统内的 CPU0 到 CPU1 或 CPU2 到 CPU3。
- IB RDMA 链路只用于跨 Linux 系统的 stage1 到 stage2。
- `ib-rdma` 当前需要手工交换 peer info 文件；普通 TCP/IP 不参与推理数据传输。
