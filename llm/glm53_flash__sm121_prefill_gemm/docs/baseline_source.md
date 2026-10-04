# Baseline source

- **MoE**: `flashinfer.fused_moe.cutlass_fused_moe` from the serving image
  `sglang-ppmtp:glm53-ppmtp-dspark-v1` (flashinfer 0.6.17). Despite the name,
  the compute is **TRT-LLM's** `MoERunner` CUTLASS kernels
  (`tensorrt_llm::kernels::cutlass_kernels::*` + CUTLASS
  `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` /
  `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120`), selected by
  `flashinfer`'s `AutoTuner`. Online path:
  `sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` at image build.
- **Projections**: cuBLAS via SGLang `F.linear`; observed kernels
  `nvjet_sm121_*` and `cutlass_80_tensorop_bf16_*`.

Baseline payload construction mirrors the online
`_run_flashinfer_cutlass` call (see
`glm53_flash__sm121_nvfp4_moe_decode/bench/probe_baseline.py`):
NVFP4 packed weights (E×rows×K/2 uint8), E4M3 block-16 SFB, per-tensor global
scales, `quant_scales=[a_gs, w1_bs, 1/w1_gs, a_gs, w2_bs, 1/w2_gs]`.

No source was copied/modified: the baseline is the installed library call. The
candidate must match it numerically and be timed with the same ABI/wrapper.
