# Task — GLM-5.3-Flash prefill GEMM optimization (GB10, sm_121)

## Target

The two GEMM families that dominate prefill GPU time (chunked prefill,
`--chunked-prefill-size 4096`, M=4096):

1. bf16 projections (attention/KDA/dense/shared/lm_head) — **24%**.
2. Routed NVFP4 expert grouped GEMM (E=144/rank, topk=8, K=4096, N=2048) — **21–36%**.

## Frozen production shapes

- MoE: `M=4096, K=4096, N=2048, E=144, topk=8`, NVFP4 E2M1 weights +
  E4M3 block-16 scales (modelopt layout), bf16 in/out, SwiGLU `limit=10`,
  `routed_scaling_factor=2.5`.
- Projections: `M=4096` with the model's (K,N) set (see `docs/roofline.md`).

## Baseline

- Projections: cuBLAS (`nvjet_sm121` / `cutlass_80_*`) via SGLang `F.linear`.
- MoE: `flashinfer.fused_moe.cutlass_fused_moe` (TRT-LLM cooperative-persistent
  CUTLASS SM120 block-scaled grouped GEMM), the online `flashinfer_cutlass`
  runner. Copy source from the exact image; see `docs/baseline_source.md`.

## Evidence / gate

- bf16: baseline is already at the GB10 ~100 TFLOPS bf16 ceiling → **no-go**
  unless a shape is found below 90% MFU.
- MoE: baseline is 48-CTA/1-wave/20%-occupancy, latency-bound, **~2.6× off the
  dense FP4 ceiling** → the candidate must beat 17.7 ms at M=4096 with the same
  numerics.

## Deliverable

Native CUDA/CUTLASS candidate + correctness (oracle + baseline) + cold-L2
CUDA-graph benchmark + NCU/roofline, in `solution/` and `bench/`, with a stated
serving-level path.
