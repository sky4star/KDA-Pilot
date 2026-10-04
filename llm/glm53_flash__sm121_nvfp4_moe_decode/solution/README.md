# solution/ — native CUDA candidate (in progress)

Target: a single native-CUDA grouped NVFP4 (W4A4, block-16) expert forward for
the single-request decode regime (M=1, topk 8 of 144 local experts), replacing
the two FlashInfer CUTLASS block-scaled grouped GEMMs.

| file | role |
|---|---|
| `cutlass_fp4.py` | builds the torch extension against FlashInfer's bundled CUTLASS 4.x headers |
| `cutlass_fp4.cpp` | pybind entry (`cutlass_fp4_gemm(out, x_q, w_q, x_sf, w_sf, alpha)`) |
| `cutlass_fp4_cuda.cu` | SM120 block-scaled NVFP4 GEMM (adapted from KDA-Pilot `qwen38_nvfp4__fp4_w4a4_skinny_gemm`) |
| `validate_fp4.py` | parity harness vs `cutlass_fused_moe` (topk=1 => expert MLP) |

**Status: compiles and runs on GB10 (sm_121), parity not yet reached.**
The dense SM120 block-scaled GEMM executes; matching the *fused* MoE requires
(i) the activation SFA layout the fused kernel feeds its mainloop, (ii) the
SwiGLU epilogue (`swiglu_limit=10.0`, gate/up split, mid-quant for w2), and
(iii) the fused `alpha` convention (`input_scale_2 * weight_scale_2`). See
`docs/draft.md`.

`moe_forward` is intentionally not wired into `solution/entry.py`, so
`bench/benchmark.py` reports baseline-only until parity is achieved.
