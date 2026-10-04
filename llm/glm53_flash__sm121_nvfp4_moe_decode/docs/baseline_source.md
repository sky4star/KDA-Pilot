# Baseline source / provenance

The production baseline is the **in-tree SGLang of the serving image**, copied
verbatim; GLM-5.3-Flash is not yet on a public `main` that ships this exact MoE
runner.

- Image: `sglang-ppmtp:glm53-ppmtp-dspark-v1` (container `dspark-tp2-rank0/1`)
- SGLang tree in image: `/sgl-workspace/sglang/python/sglang`
- Version: `0.0.0.dev1+gf609d677b` (local dev build, git `f609d677b`)
- FlashInfer: `0.6.17`, CUTLASS 4.x bundled at
  `flashinfer/data/cutlass/include`
- torch: `2.13.0+cu130`, CUDA 13.0, driver 580.173.02

## The production call chain

`sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._run_flashinfer_cutlass`
(`moe_runner_backend=flashinfer_cutlass`, ModelOpt NVFP4) calls

```python
flashinfer.fused_moe.cutlass_fused_moe(
    output=..., input=x, token_selected_experts=topk_ids,
    token_final_scales=topk_weights,
    fc1_expert_weights=w13_weight, fc2_expert_weights=w2_weight,
    output_dtype=torch.bfloat16, input_sf=None,
    quant_scales=[a_gs, w1_bs, 1/w1_gs, a_gs, w2_bs, 1/w2_gs],
    activation_type=ActivationType.Swiglu, ...)
```

The hot kernel is the SM120 block-scaled grouped GEMM:

`cutlass::device_kernel<GemmUniversal<GroupProblemShape<...>,
CollectiveMma<MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120>, ...>>>`

The standalone benchmark calls the same `cutlass_fused_moe` through
`bench/probe_baseline.py` / `bench/benchmark.py` (builds the same payload as
`~/perf_glm53/profile/nvfp4_moe/harness/nvfp4_moe_harness.py`). No SGLang server
is needed; the harness imports only `flashinfer` + `torch`.

## Model geometry (config.json, exact)

- 45 layers: 3 dense + 42 MoE; hidden 4096; moe_intermediate 2048;
  n_routed_experts 288 (144/rank at TP=2), topk 8, 1 shared expert;
  routed_scaling_factor 2.5, swiglu_limit 10.0, sigmoid/noaux_tc routing.
- NVFP4 weights, block size 16, per-tensor global scales; activations are
  quantized to NVFP4 inside the fused MoE (a_gs).
