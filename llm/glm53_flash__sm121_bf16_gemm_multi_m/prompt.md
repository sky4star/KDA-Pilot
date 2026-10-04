# KDA Prompt: glm53_flash__sm121_bf16_gemm_multi_m

Target GPU: **NVIDIA GB10 (Grace-Blackwell, sm_121)**, 48 SMs, ~25 MB L2.

Optimize the bf16 GEMM behind the GLM-5.3-Flash **large-M / prefill-extend**
linears, i.e. the shapes that fall through the M<=8 `sm120_bf16_gemv` fast path
to cuBLAS (`cutlass_80_wmma` / `nvjet`):

- `sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply`
  -> `F.linear` -> cuBLAS, for M in [9, 64].

Motivation: a 42-token extend capture shows many M=42 bf16 linears
(`12576x4096`, `4096x4096`, `4096x2048`, `8192x4096`, ...) plus the M=8 DSpark
draft projections on cuBLAS. The question was whether a native kernel can beat
cuBLAS there.

## Result: **NO-GO** (cuBLAS is already at the roofline)

Cold-L2, CUDA-graph standalone (`bench/benchmark.py`):

| shape (M,K,N) | cuBLAS µs | cuBLAS GB/s |
|---|---:|---:|
| 42,4096,4096 | 155.9 | 219.7 |
| 42,4096,2048 | 83.1 | 208.1 |
| 8,24576,4096 | 861.0 | 234.4 |
| 64,4096,4096 | 163.5 | 211.7 |
| 42,4096,128 | 7.0 | 199.7 |

GB10 practical copy/read ceiling measured in the same run: **~230–240 GB/s**
(64–256 MB), ~185–200 GB/s at 1 GB. cuBLAS is therefore at **~0.9–1.0 of the
ceiling** on every production row → no bandwidth headroom.

The trace's apparently low GB/s (130–157 GB/s) for these extend GEMMs was
**torch-profiler per-launch inflation** (small/medium kernels carry a fixed
~20 µs instrumentation overhead; the same artifact that makes the KDA gated-norm
"69 µs" read 1.6 µs under CUDA graph). Re-measured standalone, cuBLAS is fine.

A hand-written tiled bf16 SIMT GEMM (`solution/bf16_gemm.cuh`) passes
correctness but runs ~12x slower (17 GB/s) — a naive smem layout with bank
conflicts and low occupancy; even a tuned version cannot exceed the DRAM ceiling
cuBLAS already reaches. Not pursued further.

## Conclusion
The large-M bf16 linears are **memory-roofline bound**; there is no standalone
kernel win. The only measured decode headroom remains the **NVFP4 MoE at M=1**
(188 GB/s vs a ~240 GB/s ceiling) — see `glm53_flash__sm121_nvfp4_moe_decode`.
