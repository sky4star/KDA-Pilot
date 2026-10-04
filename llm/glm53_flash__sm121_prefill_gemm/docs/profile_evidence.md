# Profile evidence — M=4096 routed MoE on GB10

Two kernels profiled with `ncu --set full` (standalone, idle GPU), both the
**persistent / 1-wave** family.

## A. Baseline: `cutlass_fused_moe` (TRT-LLM) grouped GEMM

Kernel: `device_kernel<GemmUniversal<GroupProblemShape<...>,
MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>,
KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>,
MmaTileShape<128,128,256>, ...>>`
(`bench/_ncu_moe.py`, launch = the w13 GEMM at M=4096).

| metric | value |
|---|---|
| Duration | **8.04 ms** (one of two GEMMs) |
| Grid / Block | **48 / 384** |
| Waves per SM | **1** |
| Theoretical occupancy | 25% (regs 168, smem 89 KB) |
| Achieved occupancy | **20.45%** |
| Compute (SM) / Tensor | 37.0% |
| Memory throughput (DRAM/L2 max) | 36.9% |
| L2 hit rate | 67.8% |
| Local-memory spilling | 960,944 requests (100% overhead) |
| Warp cycles / issued inst | 33.8 |
| Stall "blocked/yield/sleep" | 69.8% (warp-spec NANOSLEEP) |
| Issued warp / scheduler | 0.08 (one issue per 13.1 cycles) |
| Eligible warps / scheduler | 0.10 of 2.57 active |

## B. Candidate: FlashInfer `b12x_fused_moe` (SM12x CuTe-DSL dynamic backend)

Kernel: `MoEDynamicKernel` — one fused kernel for route+FC1+act+FC2+scatter
(`bench/b12x_breakdown.py`). Duration 18.10 ms; torch-profiler confirms it is
**100% of the call** (17.67 ms).

| metric | value |
|---|---|
| Duration | 18.10 ms |
| Grid / Block | **48 / 160** |
| Waves per SM | **1** |
| Theoretical occupancy | **10.42%** (255 regs) |
| Achieved occupancy | 9.89% |
| Compute (SM) / Tensor | 24.6% |
| Memory throughput | 44.2% |
| L2 hit rate | 73.5% |
| Active / eligible warps per scheduler | 1.30 / 0.16 |
| Issue slots busy | 16.3% |

## Interpretation

Both kernels launch **48 CTAs = 1 per SM, 1 wave**, with warp specialization and
NANOSLEEP. At M=4096 (227 rows/expert) the per-CTA work does not hide the
weight-stream + MMA latency; tensor pipe is 25–37% busy and DRAM 37–44% busy.
Neither roof is reached. The dense control (`roofline.md` §4) reaches 333 TFLOPS
on the same FP4 hardware, so the grouped schedule — not the hardware or the data
type — is the limiter.

`b12x` (1.04× vs baseline) does not fix it: its dynamic kernel is even lower
occupancy (10%). Forcing `static`/`micro` at M=4096 fails
(`packed_input` > 2^31-1).

## Raw artifacts

- `bench/roofline.py`, `bench/fp4_peak.py`, `bench/moe_scaling.py`
- `bench/b12x_probe.py` (correctness: max_abs 0.0034 on ref absmax 0.063)
- `bench/b12x_breakdown.py`, `bench/_ncu_moe.py`
- NCU reports: `ncu_moe4096.ncu-rep`, `ncu_b12x.ncu-rep` (in container `/tmp`;
  re-run the commands in `bench/README` notes).
