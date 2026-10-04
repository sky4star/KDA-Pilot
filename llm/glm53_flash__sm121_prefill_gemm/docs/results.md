# Results — GLM-5.3-Flash prefill GEMM optimization (GB10 ×2, sm_121)

Session 2026-10-02. Container `gemm-dev`, spark1 idle. Sources in `bench/`,
evidence in `docs/`.

## tl;dr

- **bf16 projection GEMMs (24% of prefill): NO-GO.** cuBLAS reaches
  ~100–102 TFLOPS square and the production M=4096 shapes run at 91–99 TFLOPS
  (85–99% MFU). At/above the hardware ceiling.
- **Routed NVFP4 MoE grouped GEMM (21–36% of prefill): ~2.5–3× headroom.**
  At M=4096 it is neither compute- nor memory-bound: 93 TFLOPS / 102 GB/s with a
  **48-CTA, 1-wave, 20% occupancy** cooperative-persistent kernel (NCU). The
  same FLOPs as a dense FP4 GEMM take **5.4 ms vs 17.4 ms**.
- No library drop-in fixes it: `grouped_mm_fp4` (cuDNN ≥9.21) unavailable;
  `cute-dsl` unsupported on cap 121; `b12x_fused_moe` is neutral (1.04×) and
  lower-occupancy (10%).
- Cheap config lever: larger chunk improves MoE utilization
  (93→117→127 TFLOPS at M=4096→8192→16384), i.e. one 8192 chunk ~1.26× the MoE
  throughput of two 4096 chunks.

## Evidence

See `docs/roofline.md` (numbers), `docs/profile_evidence.md` (NCU),
`docs/backend_survey.md` (alternatives).

## Status

- Roofline + NCU + backend survey complete.
- Native grouped block-scaled SM120 GEMM candidate: not started.
- Bench harness (`bench/*.py`) and container recipe in `docs/benchmark_method.md`.

## Next (RLCR)

1. Build a **high-occupancy grouped NVFP4 GEMM** (CUTLASS SM120 block-scaled
   array-TMA, non-cooperative / tile-grid schedule) matching FlashInfer's
   SFB/SFA layout; correctness vs a dequantized oracle then vs
   `cutlass_fused_moe`.
2. Cold-L2, CUDA-graph, symmetric-ABI benchmark vs the baseline at M=4096.
3. Serving-level acceptance: `--moe-runner-backend` path + e2e prefill TTFT.
