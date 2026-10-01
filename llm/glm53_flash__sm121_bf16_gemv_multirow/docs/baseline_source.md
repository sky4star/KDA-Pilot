# Baseline source / provenance

The production baseline is not upstream `main`; it is the in-tree SGLang of the
serving image, copied verbatim.

- Image: `sglang-ppmtp:glm53-ppmtp-dspark-v1`
- Container: `dspark-tp2-rank0` / `dspark-tp2-rank1`
- SGLang tree in image: `/sgl-workspace/sglang/python/sglang`
- Version reported: `0.0.0.dev1+gf609d677b` (local dev build, git `f609d677b`)

Copied files (in `baseline/`):

| file | role |
|---|---|
| `unquant.py` | the dispatch site (`UnquantizedLinearMethod.apply` / `apply_into`); production baseline for the bf16 linear |
| `hopper_bf16_gemv.py` / `.cuh` | SM90 bf16 GEMV reference ported to SM121 |
| `sm120_fp8_gemv.py` / `.cuh` | SM120 fp8 GEMV; wrapper/config template |
| `jit_utils_init.py` | `sglang.kernels.jit.utils` public API (`load_jit`) |

## What the production baseline actually executes

For decode with DSPARK on GB10 the bf16 linears go through
`UnquantizedLinearMethod.apply` → `F.linear` (backend is `torch`, because
`initialize_bf16_gemm_config` only enables `cutedsl` on SM100/SM103 and `gemv`
on SM90). A decode capture shows cuBLAS selecting:

- `cutlass_80_tensorop_bf16_s16816gemm_*` for `M∈[2,8]` (not `gemvx`), and
- `internal::gemvx::kernel<...bf16...>` for the `M=1` `lm_head`.

So the meaningful standalone baseline is `torch.mm` on this GPU (which selects the
same cuBLAS kernels), not a hand-written GEMV.
