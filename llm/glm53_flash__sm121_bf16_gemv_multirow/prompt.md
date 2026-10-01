# KDA Prompt: glm53_flash__sm121_bf16_gemv_multirow

Target GPU: **NVIDIA GB10 (Grace-Blackwell, sm_121)**, single-node per rank; the
production service is `tp=2` across two GB10 nodes.

Optimize the SGLang kernel path behind the bf16 decode linear:

- `sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply`
- `sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply_into`
- (candidate) `sglang.kernels.ops.gemm.sm120_bf16_gemv.sm120_bf16_gemv`

## Why this task

`nvidia/GLM-5.3-Flash-NVFP4`, served with `--speculative-algorithm DSPARK`, keeps
every attention projection, the dense-MLP gate/up/down, `lm_head` and the DSpark
draft projections in **bf16** (the NVFP4 quant ignore-list); only the MoE experts
are fp4. A real decode capture on the target GPU (see `docs/profile_evidence.md`)
attributes the entire bf16 linear family to `cutlass_80_tensorop_bf16` tiles plus
one cuBLAS `gemvx` (the M=1 `lm_head`), together the largest addressable non-MoE
consumer of decode GPU time.

SGLang ships `hopper_bf16_gemv` (SM90-only) and `sm120_fp8_gemv`, but **no bf16
GEMV for SM120/SM121** — so on GB10 these fall through to cuBLAS. Porting the
weight-streaming bf16 GEMV to SM121 and extending it across the verify rows
`M∈[2,8]` is the task.

## Goal

One native-CUDA bf16 GEMV covering `M∈[1,8]` and the captured `(K, N)` set, never
slower than cuBLAS on a production row, and faster on the shapes where cuBLAS is
off the DRAM ceiling (wide `lm_head`, tiny-N projections).

## Contract

- Candidate API matches the `UnquantizedLinearMethod` ABI: `x[...,K] @ w[N,K]^T`,
  bf16 in/out, fp32 accumulation, output allocated like `F.linear`.
- Numerics: bf16 tolerance `atol=7e-2, rtol=2e-2` (llm_correctness_contract.md).
- Shapes: `bench/workloads.json` (frozen; from the DSpark decode capture).
- Native CUDA (no DSL), single idle target GPU, standard benchmark rules.

## Result

- Standalone: production-row geomean **1.136×** (min 1.013×, max 1.341×).
- No-spec single-request decode: **+15% tok/s**.
- DSpark: verify-step latency **1.15×**, draft-step **1.35×**; raw tok/s is
  content/acceptance-dominated and not a valid single-prompt A/B. See `docs/results.md`.
