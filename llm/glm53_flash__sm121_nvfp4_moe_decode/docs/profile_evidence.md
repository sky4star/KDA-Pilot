# Profile evidence

Source: torch-profiler traces captured on the live tp=2 service (rank 0) via
`POST /start_profile`, plus standalone NCU on the freed GB10.

## A. Whole-decode kernel family shares

Trace: `~/profiles/gemv_ab/decode_overlay/1790822623.890467-TP-0.trace.json.gz`
(DSpark decode + one 42-token extend; per-kernel GPU self-time aggregated).

| family | ms | share |
|---|---:|---:|
| **nvfp4 MoE blockscaled grouped GEMM** | 519.3 | **46.5%** |
| bf16 GEMV `sm120_bf16_gemv` (optimized separately) | 300.4 | 26.9% |
| bf16 cuBLAS fallback `cutlass_80_wmma` | 112.6 | 10.1% |
| nccl all-reduce/broadcast | 47.9 | 4.3% |
| KDA/gated-delta-rule + gated-norm + conv | 23.7 | 2.1% |
| mhc (hyper-connection) tilelang | 12.4 | 1.1% |
| trtllm MoE aux (routing/activation) | 11.7 | 1.1% |
| tiny-N router GEMM | 11.6 | 1.0% |
| deepgemm tf32 prenorm / dsv3 fused-a / attention / rest | ~30 | ~3% |

The MoE is the **largest single remaining operator** after the bf16 GEMV work.

## B. The MoE call is one kernel

`bench/kernel_breakdown.py M=1 E=144 topk=8` inside the serving image:

| kernel | ms/call | share |
|---|---:|---:|
| `device_kernel` (SM120 block-scaled grouped GEMM) | 0.517 | **94.9%** |
| trtllm `finalizeMoeRoutingKernel` | 0.007 | 1.2% |
| trtllm `doActivationKernel` | 0.005 | 1.0% |
| trtllm `expandInputRowsKernel` | 0.004 | 0.8% |
| rest (descale/stride/sort/copy) | ~0.02 | ~2% |

20 kernel launches / 10 calls = 2 grouped GEMMs per call (w13 then w2). So the
whole task is the two grouped GEMM launches.

## C. NCU on the M=1 grouped GEMM

`ncu -k regex:device_kernel -c 2` (needs `--privileged`; `RmProfilingAdminOnly=1`
on this box):

- grid `(1, 48, 1)`, block 384 → **48 CTAs, 1 per SM, `waves_per_multiprocessor=1`**
- 168 registers/thread; `launch__occupancy_limit_shared_mem = 1` block/SM
- 168 regs + 90 KB smem ⇒ only **22% warp occupancy**; ~79% warp time in
  NANOSLEEP (warp-specialized producer/consumer) in the M=512 capture
- local (spill) loads present; DRAM counters are unavailable on sm_121
  (`dram__bytes* = None`), so achieved BW is computed from weight bytes / time.

## D. Roofline (the key number)

Cold-L2, CUDA-graph baseline (`bench/benchmark.py`):

| workload | baseline µs | routed weight MB | achieved GB/s |
|---|---:|---:|---:|
| M=1 E=144 topk=8 | 533.9 | 100.7 | **188** |
| M=2 | 1044.9 | ~201 | ~192 |
| M=4 | 2027.8 | ~403 | ~199 |
| M=8 E=144 topk=8 | 3095.9 | ~718 (57 distinct experts) | ~232 |
| M=1 E=144 topk=1 | 78.4 | 12.6 | 161 |

Time scales linearly with the number of distinct selected experts ⇒ the kernel
is weight-traffic bound. GB10 sustained copy/read ceiling measured at **~228
GB/s** (`glm53…gemv` task `probe_baseline.py`; that task's GEMV reached
250–262 GB/s). At M=1 the MoE is at **~188/228 = 0.82 of ceiling** (≈0.75 of the
250 GB/s the sibling GEMV sustains) → **~1.2–1.35× headroom at M=1**. At M≥2 the
same kernel is already at the ceiling.

## E. KDA small kernels are NOT a target (launch-bound, not GPU-bound)

The trace's per-launch times for the KDA Triton kernels (gated norm ~69 µs,
delta-rule ~77 µs) are profiler artifacts. CUDA-graph timing in the image
(`bench/kda_probe_graph.py`) gives the true cost:

| kernel | rows | graph µs |
|---|---:|---:|
| `layer_norm_gated` D=128 | 512 | 1.61 |
| `layer_norm_gated` D=128 | 4096 | 2.77 |
| `layer_norm_gated` D=128 | 8192 | 4.65 |

244–1351 GB/s (L2-resident) — no headroom. Production runs everything inside
`full_cuda_graph_backend`, so the eager Triton launch overhead (~21 µs) is absent.
