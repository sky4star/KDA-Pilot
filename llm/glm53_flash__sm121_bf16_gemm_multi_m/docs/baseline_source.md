# Baseline source / provenance

Baseline = the production bf16 linear for M>8: `F.linear` -> cuBLAS, reached from
`sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply`. The
SM121 `sm120_bf16_gemv` fast path only covers M<=8 (sibling task
`glm53_flash__sm121_bf16_gemv_multirow`), so the 42-token extend and the M=8
DSpark draft projections issued as M>8 / uncovered rows fall through to cuBLAS
(`cutlass_80_wmma_tensorop_bf16` and cuBLASLt `nvjet_sm121` in the capture).

- Image: `sglang-ppmtp:glm53-ppmtp-dspark-v1`
- Version: `0.0.0.dev1+gf609d677b`; torch 2.13.0+cu130; CUDA 13.0
- Baseline ABI in this task: `bf16_gemm(x[M,K], w[N,K]) -> out[M,N]`, out
  allocated like `F.linear`; `bench/benchmark.py` calls `torch.mm(x, w.t())`,
  which selects the same cuBLAS kernels.

Shapes come from the decode+extend capture (`docs/profile_evidence.md` in the
GEMV task) via the `aten::mm` Input Dims correlation; see `bench/workloads.json`.
