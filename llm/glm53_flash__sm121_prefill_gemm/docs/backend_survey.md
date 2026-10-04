# Backend survey — what exists on GB10 for grouped NVFP4 MoE

Goal: find a drop-in library kernel that beats the current
`cutlass_fused_moe` (TRT-LLM, cooperative persistent) at M=4096.

| backend | availability on sm_121 | result |
|---|---|---|
| `cutlass_fused_moe` (current) | yes | 17.73 ms; cooperative persistent, 1 wave |
| `flashinfer.b12x_fused_moe` (CuTe-DSL SM12x, dynamic) | yes, fused | **17.25 ms (1.04×)**; 10% occupancy, no win |
| `b12x` static/micro at M=4096 | forced | **fails**: `packed_input needs 9.66e9 elements > 2^31-1` |
| `flashinfer.grouped_mm_fp4` (cuDNN MoE) | no | `requires cuDNN >= 92100, found 92000` |
| `cute_dsl_fused_moe_nvfp4` | no | `mm_fp4 cute-dsl` unsupported on cap 121 |
| `trtllm_fp4_block_scale_moe` | untested | handoff notes DSA trtllm crashes on GB10; needs shuffle weight prep |
| DeepGEMM | no | FP8/MXFP8 only, not NVFP4 E4M3 block-16 |
| Triton grouped | no | ROCm-only path in this build |
| **TileLang 0.1.12 (installed)** | no | fp4 `T.gemm` → `Unsupported dtype`; blockscaled = tcgen05/sm100 only |
| **TileLang 0.1.15** | **yes (dense verified)** | `T.mma_gemm_blockscaled` + SM120a NVF4 MMA (ue4m3); needs 1-line SM121A guard patch; no grouped-MoE op yet |

## `tune_max_num_tokens` (baseline autotuner knob)

`bench/moe_tune_probe.py` at M=4096 for 4096/8192/16384/32768/65536 →
**17.6–17.9 ms, no change**. The TRT-LLM `AutoTuner` already picks the same
tactic; the tuning bound is not the limiter (num_tokens=4096 is already within
the default 8192 bucket).

## Conclusion

No library backend on GB10 reaches the dense FP4 ceiling for the grouped
problem. The `b12x` fused kernel is the closest alternative and is neutral
(1.04×). The remaining route is a **native grouped block-scaled SM120 GEMM with
a high-occupancy / non-persistent schedule** (or a modified CUTLASS
instantiation), plus integration to replace the cooperative kernel.

## Chunk-size note

`bench/moe_scaling.py` shows MoE throughput rises with M: 93 TFLOPS at M=4096 vs
117 at 8192 and 127 at 16384. Two 4096-chunks cost 35.5 ms of MoE vs 28.2 ms for
one 8192-chunk (**1.26×** on the MoE). This is a config lever (larger
`--chunked-prefill-size`) but is bounded by ~127 TFLOPS and by activation memory.
