# Results — GLM-5.3-Flash NVFP4 MoE decode (GB10, sm_121)

## 1. Standalone baseline (cold L2, CUDA-graph, median of 7)

`python3 bench/benchmark.py` (writes `bench/results.jsonl`), inside the image.

| workload (M,E,K,N,topk) | baseline µs | routed weight MB | achieved GB/s |
|---|---:|---:|---:|
| prod M=1, E=144, K=4096, N=2048, topk=8 | 533.9 | 100.7 | 188 |
| prod M=8, E=144, K=4096, N=2048, topk=8 | 3095.9 | ~718 | ~232 |
| reg M=2 | 1044.9 | ~201 | ~192 |
| reg M=4 | 2027.8 | ~403 | ~199 |
| reg M=8, E=16 | 1073.5 | 100.7 | 94 |
| reg M=1, topk=1 | 78.4 | 12.6 | 161 |

## 2. Roofline / active bound

The SM120 block-scaled grouped GEMM is **94.9%** of the call. Runtime is
*identical* per distinct selected expert (M=1→533.9, M=2→2×, M=4→4×), so the
kernel is **DRAM weight-traffic bound**, at ~188 GB/s (M=1) up to ~232 GB/s
(M=8) against a GB10 sustained ceiling of ~228 GB/s (the sibling bf16 GEMV
measured 250–262 GB/s on this box).

- **M≥2:** at/above the copy ceiling → no headroom; the remaining gap is a
  few percent of launch/tail.
- **M=1 (the true single-request decode case):** ~0.75–0.82 of the achievable
  bandwidth, launched as a single wave of **48 CTAs (1/SM)** at 22% occupancy
  with 168 regs and spilling (NCU). Target **≈1.2–1.35×**.

## 3. Candidate status — native CUTLASS SM120 block-scaled GEMM

Built and run on GB10 (`solution/cutlass_fp4.{py,cpp,cuda.cu}`, CUTLASS 4.x from
the FlashInfer bundle; `-gencode=arch=compute_121a`). The dense SM120
block-scaled NVFP4 GEMM compiles and executes.

**Not promoted yet.** Validating the full fused-MoE chain against
`cutlass_fused_moe` (topk=1, single expert) did not reach parity: the residual
error is dominated by three integration details of the *fused* path that a bare
dense GEMM does not reproduce —

1. the exact activation-quantized scale-factor layout the fused kernel feeds to
   `MainloopSm120…BlockScaled` (my `fp4_quantize` SFA layout does not match the
   fused kernel's internal SFA byte-for-byte);
2. the SwiGLU epilogue details (`swiglu_limit=10.0`, gate/up split, and the
   mid-quantization of `act` before the w2 GEMM);
3. the `alpha = input_scale_2 * weight_scale_2` convention for the fused path
   (calibration gave a best-case ~50% residual, i.e. the layout — not the
   scalar — is the blocker).

A correct candidate therefore needs either the fused SFA layout replicated in a
native activation-quant kernel or a grouped block-scaled CUTLASS instantiation
matching FlashInfer's SFB/SFA (`sm120_blockscaled_mma_array_tma.hpp` exists in
the bundled CUTLASS 4.x). That is the next RLCR step; the blocker is named and
the active bound (DRAM weight traffic) is measured.

## 4. What is *not* worth optimizing (evidence)

- KDA gated-norm / delta-rule: 1.6–4.6 µs GPU under CUDA graph (launch-bound in
  eager; profiler per-launch times are artifacts). No headroom.
- bf16 GEMV family: already at/above the DRAM ceiling (sibling task).

## 5. Conclusion

The single-request GLM-5.3-Flash decode is **memory-roofline bound** after the
bf16 GEMV work. The only operator with measured headroom is the **NVFP4 MoE at
M=1 (~1.2–1.35×)**; M≥2 is already saturated. Capturing that headroom requires a
native grouped NVFP4 kernel matching FlashInfer's scale-factor layout and SwiGLU
epilogue — infrastructure is in place (`solution/`), parity is the remaining
blocker.
