# Pipeline-brick TP analysis experiment results

Date: 2026-06-18

This document records the tensor-parallel experiments discussed in this Codex session. Most numerical results below come from real-machine logs pasted in the conversation on the Phytium CPU test machine. Some early local `bench-logs/` records in this checkout are listed separately as background, because they are not the latest real-machine TP runs.

## Terminology

- TP, tensor parallelism: tensor parallelism, or splitting one layer's matrix/tensor computation across multiple ranks.
- rank: one TP worker process. In these experiments each rank is pinned to one or more NUMA nodes.
- all-reduce: cross-rank reduction used to combine TP partial outputs.
- `wait_before_reduce`: time a rank waits before the collective reduction can start. If this is large on rank 1/2/3 but near zero on rank 0, rank 0 is the slow compute rank.
- `hidden-dtype bf16`: the inter-stage hidden-state transport type is BF16. It reduces CXL transfer payload compared with F32; it does not mean model weights are BF16.

## Test machine and common setup

The real machine used in the pasted logs has 128 CPU cores and 8 NUMA nodes:

```text
node0: CPUs 0-15
node1: CPUs 16-31
node2: CPUs 32-47
node3: CPUs 48-63
node4: CPUs 64-79
node5: CPUs 80-95
node6: CPUs 96-111
node7: CPUs 112-127
```

Common model:

```text
models/qwen3-4b-f16/Qwen3-4B-Instruct-2507-F16.gguf
```

Common runtime:

```text
./build/bin/llama-pipeline-brick
--domain-mode single
--transport cxl
--hidden-dtype bf16
--prefill-chunk 8
```

Important context: the TP implementation is a Qwen3-4B prototype path. It is not a general Qwen TP implementation.

## Result summary

| ID | Configuration | Main knobs | Inference time | Total speed | Main observation |
|---|---|---:|---:|---:|---|
| E1 | TP2 node smoke | `tp_size=2`, `threads=64`, node pairing before later fixes | 29.14 s | 9.61 t/s | TP2 worked, but rank waiting was still visible. |
| E2 | TP2 neon/node variant | `tp_size=2`, similar smoke | 28.84 s | 9.71 t/s | Similar to E1. |
| E3 | TP2 t32 single-node-per-rank | `tp_size=2`, `threads=32`, `head-numa 0-1`, `tail-numa 4-5` | 21.57 s | 12.98 t/s | Best early TP result, but only uses 2 NUMA nodes per side. |
| E4 | TP2 grouped all 4 NUMA nodes per side | `tp_size=2`, `threads=64`, `head-numa 0-3`, `tail-numa 4-7` | 24.21 s | 11.57 t/s | Uses all NUMA nodes, but slower than E3 because each rank spans two nodes and node1 still hurts. |
| E5 | Uniform TP4 local-memory smoke | `tp_size=4`, one NUMA node per rank | 119.41 s | 2.34 t/s | Very slow. One head rank on node1 becomes the straggler and other ranks wait around 100 s. |
| E6 | Uniform TP4 with head rank/node swap | `head-numa 1,0,2,3`, `tail-numa 5,4,6,7` | 112.66 s | 2.49 t/s | Slowness moves with node1, not with `.tpN` file or rank id. |
| E7 | NUMA maps probe | TP4 with page-placement diagnostics | n/a | n/a | Model pages are mostly local to the intended NUMA node; the problem is not bad first-touch placement. |
| E8 | Asymmetric TP4, rank0 small shard | `Q heads [4,12,8,8]`, `KV heads [1,3,2,2]`, `FFN [256,3200,3136,3136]` | 27.40 s | 10.22 t/s | Around 4.3x faster than uniform TP4, but still slower than TP2 t32 and non-TP pipeline+CXL. |
| E9 | Non-TP pipeline+CXL, batch 4, predict 16 | `parallel=4`, `n-predict=16`, no `--tp-size` | 51.37 s | 27.41 t/s | Current non-TP reference for the longer batch/decode test. |
| E10 | Asymmetric TP4, batch 4, predict 16 | same prompt as E9 plus `tp_size=4` | 62.80 s | 22.42 t/s | Better than old uniform TP4, but still about 22% slower than non-TP pipeline+CXL for this workload. |

## Detailed records

### E1. TP2 node smoke

Observed output:

```text
real 30.08
user 1339.24
sys 26.25
pipeline-brick perf: inference time 29.14 s
pipeline-brick perf: total prompt tokens 279, speed 9.58 t/s
pipeline-brick perf: total gen tokens 1, speed 0.03 t/s
pipeline-brick perf: total tokens 280, speed 9.61 t/s
```

All-reduce summary:

```text
tail rank0 total=5318.743 ms, wait_before_reduce=4248.592 ms
tail rank1 total=1339.693 ms, wait_before_reduce=293.836 ms
head rank0 total=8244.767 ms, wait_before_reduce=6573.921 ms
head rank1 total=1528.875 ms, wait_before_reduce=181.464 ms
```

Interpretation: TP2 was functional but synchronization wait was still significant on one side. It was not yet a clean acceleration path.

### E2. TP2 neon/node variant

Observed output:

```text
real 29.76
user 1312.50
sys 26.99
pipeline-brick perf: inference time 28.84 s
pipeline-brick perf: total prompt tokens 279, speed 9.67 t/s
pipeline-brick perf: total gen tokens 1, speed 0.03 t/s
pipeline-brick perf: total tokens 280, speed 9.71 t/s
```

All-reduce summary:

```text
tail rank0 total=5432.866 ms, wait_before_reduce=4401.836 ms
head rank0 total=8726.935 ms, wait_before_reduce=7120.228 ms
tail rank1 total=1210.483 ms, wait_before_reduce=132.349 ms
head rank1 total=1434.026 ms, wait_before_reduce=131.547 ms
```

Interpretation: similar behavior to E1. The variant was not the main breakthrough.

### E3. TP2 t32, one NUMA node per rank

Command shape:

```text
--threads 32
--tp-size 2
--head-numa 0-1
--tail-numa 4-5
```

Rank binding:

```text
tail rank0: threads=16, CPUs 64-79
tail rank1: threads=16, CPUs 80-95
head rank0: threads=16, CPUs 0-15
head rank1: threads=16, CPUs 16-31
```

Observed output:

```text
real 22.52
user 902.25
sys 5.05
pipeline-brick perf: inference time 21.57 s
pipeline-brick perf: total prompt tokens 279, speed 12.93 t/s
pipeline-brick perf: total gen tokens 1, speed 0.05 t/s
pipeline-brick perf: total tokens 280, speed 12.98 t/s
```

All-reduce summary:

```text
tail rank0 total=6766.166 ms, wait_before_reduce=6718.495 ms
tail rank1 total=97.563 ms, wait_before_reduce=0.596 ms
head rank0 total=9028.970 ms, wait_before_reduce=8372.374 ms
head rank1 total=1205.972 ms, wait_before_reduce=0.842 ms
```

Interpretation: this was the best early TP result. However, it uses only two NUMA nodes per side. The user rejected it as the final direction because the 4-NUMA-per-side Phytium CPU resources should be used.

### E4. TP2 grouped, all 4 NUMA nodes per side

Command shape:

```text
--threads 64
--tp-size 2
--head-numa 0-3
--tail-numa 4-7
```

Rank binding after grouping logic:

```text
tail rank0: CPUs 64-95
tail rank1: CPUs 96-127
head rank0: CPUs 0-31
head rank1: CPUs 32-63
```

Observed output:

```text
real 24.94
user 2250.34
sys 43.42
pipeline-brick perf: inference time 24.21 s
pipeline-brick perf: total prompt tokens 279, speed 11.52 t/s
pipeline-brick perf: total gen tokens 1, speed 0.04 t/s
pipeline-brick perf: total tokens 280, speed 11.57 t/s
```

All-reduce summary:

```text
head rank0 total=2853.251 ms, wait_before_reduce=2673.583 ms
head rank1 total=808.226 ms, wait_before_reduce=588.385 ms
tail rank0 total=3928.113 ms, wait_before_reduce=3140.073 ms
tail rank1 total=5619.925 ms, wait_before_reduce=4874.304 ms
```

Interpretation: all NUMA nodes were used, but a rank spanning node0+node1 still suffered. This pointed toward node1 being a pathological compute participant.

### E5. Uniform TP4 local-memory smoke

Command shape:

```text
--tp-size 4
--head-numa 0-3
--tail-numa 4-7
```

Observed output:

```text
real 120.68
user 2836.20
sys 23.18
pipeline-brick perf: inference time 119.41 s
pipeline-brick perf: total prompt tokens 279, speed 2.34 t/s
pipeline-brick perf: total gen tokens 1, speed 0.01 t/s
pipeline-brick perf: total tokens 280, speed 2.34 t/s
```

All-reduce summary:

```text
head rank0 total=107619.235 ms, wait_before_reduce=106480.501 ms
head rank2 total=118169.692 ms, wait_before_reduce=117218.919 ms
head rank3 total=118188.775 ms, wait_before_reduce=117281.052 ms
head rank1 total=1423.150 ms, wait_before_reduce=11.045 ms
tail ranks total around 58-71 ms
```

Later inspection of head rank1 showed many FFN matmuls around 42 ms average:

```text
head rank1 ffn_gate total about 1470-1480 ms over 35 calls
avg about 42 ms per call
```

Interpretation: rank1, which was bound to node1 in this layout, was the actual slow compute rank. Other head ranks waited around 100 s at all-reduce.

### E6. Uniform TP4 with head rank/node swap

Command shape:

```text
--tp-size 4
--head-numa 1,0,2,3
--tail-numa 5,4,6,7
```

Rank binding:

```text
head rank0: node1, CPUs 16-31
head rank1: node0, CPUs 0-15
head rank2: node2, CPUs 32-47
head rank3: node3, CPUs 48-63
```

Observed output:

```text
real 119.75
user 2817.19
sys 23.98
pipeline-brick perf: inference time 112.66 s
pipeline-brick perf: total prompt tokens 279, speed 2.48 t/s
pipeline-brick perf: total gen tokens 1, speed 0.01 t/s
pipeline-brick perf: total tokens 280, speed 2.49 t/s
```

All-reduce summary:

```text
head rank0 total=844.281 ms, wait_before_reduce=15.678 ms
head rank1 total=103998.671 ms, wait_before_reduce=103084.466 ms
head rank2 total=117193.956 ms, wait_before_reduce=116613.958 ms
head rank3 total=117209.950 ms, wait_before_reduce=116633.206 ms
```

Head rank0, now on node1, became the slow compute rank:

```text
head rank0 ffn_gate/down/up avg about 42 ms
```

Interpretation: the bottleneck follows node1. It is not tied to `.tp0/.tp1/.tp2/.tp3` model files and not tied to a logical rank id.

### E7. NUMA maps probe

The diagnostic printed model page placement after loading TP shards.

Head-side page placement:

```text
head rank0 tp0 model_load_delta: N1=26486 pages, about 1657.3 MiB
head rank1 tp1 model_load_delta: N0=26487 pages, about 1657.3 MiB
head rank2 tp2 model_load_delta: N2=26486 pages, about 1657.3 MiB
head rank3 tp3 model_load_delta: N3=26486 pages, about 1657.3 MiB
```

Tail-side page placement:

```text
tail rank0 tp0 model_load_delta: N5=26487 pages
tail rank1 tp1 model_load_delta: N4=26517 pages
tail rank2 tp2 model_load_delta: N6=26487 pages
tail rank3 tp3 model_load_delta: N7=26487 pages
```

All-reduce from the same probe:

```text
head rank0 total=912.735 ms, wait_before_reduce=0.727 ms
head rank2 total=110816.269 ms, wait_before_reduce=109943.831 ms
head rank3 total=118370.871 ms, wait_before_reduce=117698.565 ms
head rank1 total=117731.344 ms, wait_before_reduce=117082.281 ms
```

Interpretation: first-touch page placement was basically correct. The head-side bottleneck is node1 compute behavior, not wrong NUMA memory placement.

### E8. Asymmetric TP4, rank0 small shard

Implemented split layout:

```text
rank0 -> node1: small shard
rank1 -> node0: large shard
rank2 -> node2: large shard
rank3 -> node3: large shard

Q heads:  [4, 12, 8, 8]
KV heads: [1, 3, 2, 2]
FFN dims: [256, 3200, 3136, 3136]
```

Generated shard sizes:

```text
tp0: 1108.3 MiB
tp1: 3110.8 MiB
tp2: 2852.1 MiB
tp3: 2852.1 MiB
```

Command shape:

```text
--tp-size 4
--head-numa 1,0,2,3
--tail-numa 5,4,6,7
```

Observed output:

```text
real 31.94
user 869.36
sys 15.53
pipeline-brick perf: inference time 27.40 s
pipeline-brick perf: total prompt tokens 279, speed 10.18 t/s
pipeline-brick perf: total gen tokens 1, speed 0.04 t/s
pipeline-brick perf: total tokens 280, speed 10.22 t/s
```

Page placement:

```text
head rank0 tp0 model_load_delta: N1=15506 pages, about 971.1 MiB
head rank1 tp1 model_load_delta: N0=31527 pages, about 1972.3 MiB
head rank2 tp2 model_load_delta: N2=29486 pages, about 1842.9 MiB
head rank3 tp3 model_load_delta: N3=29456 pages, about 1842.9 MiB
```

All-reduce:

```text
head rank0 total=883.930 ms, wait_before_reduce=0.676 ms
head rank1 total=24242.553 ms, wait_before_reduce=23461.560 ms
head rank2 total=29431.389 ms, wait_before_reduce=28881.026 ms
head rank3 total=29357.091 ms, wait_before_reduce=28807.088 ms
```

Head rank0 top compute after asymmetric split:

```text
node_* f16[512,2560] x f32[512,8] -> f32[2560,8], avg about 9.5 ms
Qcur f16[2560,512] x f32[2560,8] -> f32[512,8], avg about 9.1 ms
```

Head rank1 large FFN compute on node0:

```text
ffn_gate/up f16[2560,3200] -> avg about 1.2-1.5 ms
```

Interpretation: asymmetric TP4 fixed the worst uniform TP4 failure and improved from 2.34 t/s to 10.22 t/s, about 4.3x. However, node1 is still slow even with a small shard. The new bottleneck is no longer mainly the huge FFN shard; it is the remaining attention/output projection work on node1.

### E9. Non-TP pipeline+CXL, batch 4, predict 16

Command shape:

```text
--parallel 4
--n-predict 16
no --tp-size
```

Observed output:

```text
real 52.16
user 4671.98
sys 104.49
pipeline-brick perf: inference time 51.37 s
pipeline-brick perf: total prompt tokens 1344, speed 26.16 t/s
pipeline-brick perf: total gen tokens 64, speed 1.25 t/s
pipeline-brick perf: total tokens 1408, speed 27.41 t/s
```

Interpretation: this is the current non-TP reference for the longer batch/decode workload. It should not be compared directly with the 279-token, predict-1 smoke tests.

### E10. Asymmetric TP4, batch 4, predict 16

Same prompt/workload as E9, plus asymmetric TP4.

Observed output:

```text
pipeline-brick perf: inference time 62.80 s
pipeline-brick perf: total prompt tokens 1344, speed 21.40 t/s
pipeline-brick perf: total gen tokens 64, speed 1.02 t/s
pipeline-brick perf: total tokens 1408, speed 22.42 t/s
```

All-reduce summary:

```text
head rank0 wait_before_reduce=5.650 ms
head rank1 wait_before_reduce=47447.757 ms
head rank2 wait_before_reduce=52264.877 ms
head rank3 wait_before_reduce=51837.004 ms
```

Comparison with E9:

```text
non-TP pipeline+CXL: 51.37 s, total 27.41 t/s, gen 1.25 t/s
asym TP4:            62.80 s, total 22.42 t/s, gen 1.02 t/s
```

Interpretation: asymmetric TP4 is still about 22% slower than non-TP pipeline+CXL on this workload. The all-reduce wait pattern still says node1/rank0 is the compute straggler.

## Non-TP path regression concern

The user raised a concern that the TP code changes may have slowed the non-TP pipeline+CXL path.

Code inspection in this session found:

```cpp
if (args.tp_size > 1) {
    return run_hardware_tp(args);
}
```

The non-TP path does not enter the TP runner unless `--tp-size` is greater than 1.

Model loading also keeps the previous mmap behavior for non-TP:

```cpp
if (args.tp_size > 1) {
    mparams.use_mmap = false;
}
```

`qwen3.cpp` and `llama-kv-cache.cpp` also gate the asymmetric TP logic on `tp_size > 1`. Therefore, from code inspection, the TP changes should not directly alter non-TP matrix shapes, KV-cache shape, or model mmap behavior.

The likely explanations for apparent non-TP slowdown are:

1. Different workload: `parallel=4`, `n-predict=16`, and a longer prompt are not comparable with predict-1 smoke tests.
2. Different metric: `total speed`, `prompt speed`, and `gen speed` measure different things. Decode speed is much lower than total speed here.
3. Machine state can matter: node memory pressure and CXL state can change between runs.
4. A strict regression conclusion needs an exact same-command A/B run against the older binary or older commit.

## Main conclusions

1. Uniform TP4 is not suitable on the current machine as-is. It exposes node1 as a severe compute straggler, causing other ranks to wait at all-reduce for around 100 s in short smoke tests.
2. The issue follows node1, not TP rank id or shard file name.
3. NUMA page placement is mostly correct. The model shard pages land on the intended node, so wrong page placement is not the main cause.
4. Asymmetric TP4 is effective as a mitigation. It improves the predict-1 smoke from about 2.34 t/s to 10.22 t/s.
5. Asymmetric TP4 still does not beat the best TP2 t32 result or the non-TP pipeline+CXL reference on the batch=4, predict=16 workload.
6. The current TP bottleneck is node1 compute plus all-reduce synchronization. Reducing only the FFN shard is no longer enough; rank0 attention/output projection remains slow.
7. For the FEIBRIC mainline report, the safer story is still single-domain pipeline+CXL plus F16/BF16 hidden transfer. TP should be described as an exploratory optimization that revealed node-level imbalance and synchronization overhead, not as the current mainline performance winner.

## Recommended next experiments

### 1. Exact non-TP regression check

Run the exact same non-TP command on the current binary and, if available, the older binary/commit. Compare:

```text
inference time
prompt speed
gen speed
total speed
```

Stop rule: if the same workload is not slower on the old binary, do not claim TP code caused a non-TP regression.

### 2. TP stats probe before more TP implementation

Use a short TP4 run with verbose stats and inspect:

```text
TP all-reduce
TP node
TP node-compute
result_output
pipeline-brick perf
```

Stop rule: if all-reduce wait dominates and one rank has near-zero `wait_before_reduce`, the next change must target rank imbalance rather than CXL transport.

### 3. More aggressive asymmetric layout only if TP remains important

Potential next layouts:

```text
FFN dims: [64, 3264, 3200, 3200]
```

or a more aggressive head split if the Q/KV grouping constraints are relaxed carefully.

Stop rule: if rank0 remains dominated by attention/output projection on node1, further FFN shrinking will have limited value.

## Background local logs in this checkout

These are older local files under `bench-logs/`, not the latest pasted real-machine TP results:

```text
bench-logs/f16_pipeline_dense_smoke.log: inference 33.59 s, total 8.34 t/s
bench-logs/f16_pipeline_stream_smoke.log: inference 36.11 s, total 7.75 t/s
bench-logs/tp_smoke_local.log: inference about 34.29-34.41 s, total about 8.14-8.17 t/s
bench-logs/f16_pipeline_tp_stream_smoke.log: multiple child outputs, total about 1.95-3.78 t/s
bench-logs/four_stage_smoke.log: inference 3.70 s, total 75.73 t/s
bench-logs/two_stage_regression.log: inference 5.14 s, total 54.43 t/s
```

These should be treated as historical context only unless rerun with the current real-machine setup.
