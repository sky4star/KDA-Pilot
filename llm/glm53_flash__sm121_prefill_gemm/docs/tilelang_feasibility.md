# TileLang feasibility for the prefill NVFP4 GEMM (GB10, sm_121)

Question: can the routed NVFP4 MoE grouped GEMM be written in TileLang?
Verdict: **yes with TileLang >= 0.1.15 (verified), no with the installed 0.1.12.**

## Installed 0.1.12 — cannot

- FP4 tensor-core path absent: compiling a minimal `T.gemm` with
  `float4_e2m1fnx2` fails with `ValueError: Unsupported dtype: float4_e2m1fnx2`
  (`bench/tl_probe.py`); fp16/bf16/fp8 compile fine.
- Block-scaled lowering is **TCGEN05 only**:
  `AllowTcgen5Mma` requires `TargetIsSm100` (`src/cuda/op/gemm.cc`), and
  `T.tcgen05_gemm_blockscaled` expects **E8M0** (MXFP8/4) scales.
- GB10 has no tcgen05/wgmma (`AllowWgmma` needs Hopper), so it falls back to
  `kCudaMMA` (`mma.h`), which has **zero** `e2m1`/`mxf4` instructions.

## TileLang 0.1.15 — can (dense verified on GB10)

0.1.15 adds the exact instruction GB10 needs:

- `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
  .f32.e2m1.e2m1.f32.ue4m3` (`src/tl_templates/cuda/instruction/mma_block_scale.h`)
  — i.e. **NVFP4 E2M1 weights/acts with UE4M3 (E4M3) block-16 scales**.
- `T.gemm_blockscaled` / `T.mma_gemm_blockscaled` with
  `sf_layout="blockscaled_chunk_kmajor"`, `sf_*_granularity_k=16`.
- Instruction selection: `SelectBlockScaledGemmInst` returns `kCudaMMABlockScaled`
  when `TargetIsSM120` and A shared/fragment, B shared, C fragment
  (`src/cuda/op/gemm_blockscaled.cc`).
- Upstream example: `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`
  (checked in as `solution/tl_sm120_nvfp4.py`), a **non-persistent** tiled GEMM
  (grid = N/128 x M/128) — the high-occupancy design the baseline lacks.

### One needed fix: the SM120A guard

`mma_block_scale.h` guards the asm on
`CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED && CUTLASS_ARCH_MMA_SM120A_ENABLED`.
GB10 is **SM121A**, so CUTLASS defines `..._SM121A_ENABLED`, not `..._SM120A`,
and the kernel hits a device-side assert. CUTLASS's own
`cute/arch/config.hpp` enables `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` for
`SM121A`, so the fix is to accept both:

```c
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED) && \
    (defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))
```

Patch saved at `solution/tilelang/mma_block_scale.h.patched`.

### Reproduction (isolated, does not touch the serving image)

```bash
pip install --no-deps --target /tmp/tl15site tilelang==0.1.15
# apply the guard patch to /tmp/tl15site/tilelang/src/tl_templates/cuda/instruction/mma_block_scale.h
cd /kda
TILELANG_DISABLE_CACHE=1 PYTHONPATH=/tmp/tl15site \
  python3 llm/glm53_flash__sm121_prefill_gemm/solution/tl_sm120_nvfp4.py \
  --m 2048 --n 2048 --k 2048 --verify
# -> "TileLang correctness: passed"
```

`_detect_torch_cuda_arch()` returns `sm_121a` automatically; no target override
is needed.

### Measured dense NVFP4 (TileLang 0.1.15, GB10, tile 128x128x256, 128 thr, 2 stages)

| M,N,K | ms | TFLOPS |
|---|---:|---:|
| 2048,2048,2048 | 0.121 | 141.7 (small-K underfilled) |
| 4096,4096,4096 | 0.526 | 261.2 |
| 16384,4096,4096 | 1.806 | 304.5 |
| **32768,4096,4096 (w13 flat)** | 3.509 | **313.3** |
| **32768,4096,2048 (w2 flat)** | 2.126 | **258.6** |
| sum w13+w2 | 5.64 | — |

Compare: grouped MoE `cutlass_fused_moe` at M=4096 = 17.7 ms; dense TileLang
equivalent = 5.64 ms. **The tensor-core path can reach the roof; only the
grouped dispatch is missing.**

## What is still missing for the MoE

1. **Grouped dispatch.** No ready NVFP4 fused-MoE example; the upstream
   `examples/fusedmoe` is fp16. Must hand-write a TileLang grouped kernel:
   grid over (expert-tile, N-tile), dynamic per-expert token range from an
   offsets array (SGLang's `moe_align_block_size`), per-expert A pointer, and
   `T.mma_gemm_blockscaled`. The fp16 fusedmoe example is a structural template.
2. **Scale-factor layout.** The checkpoint SF (128x4 swizzled) must be converted
   to TileLang's `blockscaled_chunk_kmajor` uint32 packed layout (host helper in
   the example), or generated directly.
3. **Activation quant + SwiGLU + scatter** after each expert GEMM.
4. **Deployment.** 0.1.15 must replace 0.1.12 in the serving image / overlay;
   SGLang and the DSA tilelang kernels currently pin 0.1.12 and must be
   re-validated. (The dev test above used a shadow `PYTHONPATH` install.)

## Verdict

- The NVFP4 block-scaled **GEMM** itself: **writeable in TileLang 0.1.15**; the
  dense path is verified correct and at roofline on GB10.
- The **grouped MoE** wrapper: must be written by us in TileLang (no drop-in),
  but the pieces (blockscaled MMA, dynamic grid, grouped example) exist.
- Cost: tilelang upgrade + 1-line guard patch + grouped kernel + SF conversion
  + serving integration/validation. This is a credible path to the ~2–3x MoE
  headroom with the user-preferred tool.
