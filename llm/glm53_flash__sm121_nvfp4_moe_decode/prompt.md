# KDA Prompt: glm53_flash__sm121_nvfp4_moe_decode

Target GPU: **NVIDIA GB10 (Grace-Blackwell, sm_121)**, 48 SMs, ~25 MB L2,
single-node per rank; the production service is `tp=2` across two GB10 nodes.

Optimize the SGLang kernel path behind the NVFP4 routed-expert MoE in
single-request decode:

- `sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._run_flashinfer_cutlass`
  -> `flashinfer.fused_moe.cutlass_fused_moe` (the production call)
- kernel:
  `cutlass::device_kernel<GemmUniversal<GroupProblemShape<...>,
  CollectiveMma<MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>, ...>>>`
  (SM120 block-scaled NVFP4 grouped GEMM + fused SwiGLU/finalize)

## Why this task

`nvidia/GLM-5.3-Flash-NVFP4` is a 45-layer hybrid (3 dense + 42 MoE, KDA
linear-attention + DSA sparse attention). A real DSpark decode capture on the
target GPU (`docs/profile_evidence.md`) attributes **46.9% of the verify-step
GPU time** to the SM120 block-scaled NVFP4 MoE groupscaled GEMM — by far the
largest single operator after the bf16 decode linears (optimized separately).
Weights are already FP4, so the win is **kernel efficiency at tiny M**, not
weight compression: the decoded NCU profile shows the warp-specialized kernel
launches only **48 CTAs (1 wave)** and spends ~79% of warp time in NANOSLEEP,
with register spilling — a latency/occupancy regime, not a bandwidth ceiling.

## Goal

One native-CUDA grouped NVFP4 (W4A4, block-32/16 scales) expert forward for the
single-request decode regime (`M in [1,8]`, topk 8 of 144 local experts), never
slower than the FlashInfer CUTLASS baseline on a production row and faster where
the CUTLASS single-wave latency leaves the DRAM ceiling unused.

## Contract

- Candidate API matches the production call site:
  given `x` (bf16 `[M,K]`), `topk_ids`/`topk_weights` (`[M,topk]`), packed
  fp4 `w13`/`w2` weights + block scales + global scales, SWIGLU activation and
  finalize, return the combined bf16 `[M,K]` MoE output.
- Numerics: `atol=2e-2, rtol=1e-2` in the **dequant domain** (SGLang
  `test_fp4_moe` / `llm_correctness_contract.md`).
- Shapes: `bench/workloads.json` (frozen; K=4096, N=2048, E=144/rank, topk=8,
  M in {1,8} production + regression grid).
- Native CUDA (no DSL); single idle target GPU; standard benchmark rules
  (`llm/docs/standalone_llm_benchmark.md`, cold L2, CUDA-graph, interleaved A/B).

## Result

<fill in after RLCR: speedup, roofline, NCU evidence, e2e projection>
