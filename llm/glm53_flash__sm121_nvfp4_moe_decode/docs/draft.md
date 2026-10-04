# Draft — RLCR log

## Target and evidence
- Operation: SM120 block-scaled NVFP4 grouped MoE (`cutlass_fused_moe`), the
  kernel behind `flashinfer_cutlass._run_flashinfer_cutlass`.
- Serving share: **46.5%** of decode GPU time (largest remaining operator after
  the bf16 GEMV task).
- Frozen shapes: `bench/workloads.json` (K=4096, N=2048, E=144/rank, topk=8,
  M∈{1,8} production + regression grid).

## Measured bound
- M=1: 533.9 µs, 100.7 MB selected-expert weights → **188 GB/s** vs ~228 GB/s
  sustained ceiling → **~1.2–1.35× headroom**. NCU: 48 CTAs (1/SM), 22%
  occupancy, 168 regs, spilling, warp-specialized NANOSLEEP.
- M≥2: linear in distinct experts, already at ceiling → no headroom.

## Candidate directions (ranked by expected benefit / risk)
1. **Native grouped NVFP4 block-scaled GEMM tuned for M=1** (more CTAs, smaller
   M-tile / split-K, no spill). Reuse the CUTLASS 4.x
   `sm120_blockscaled_mma_array_tma` collective already used by FlashInfer, but
   with a small-M config. Highest payoff; needs scale-factor layout parity.
2. Pre-pack weights so the timed path skips any per-call stride/sort glue
   (only ~2% of the call per the kernel breakdown — low payoff).
3. A different `moe_runner_backend` (config, not a kernel) — out of task scope.

## Kept / rejected ideas
- **Kept:** measure baseline + roofline (done); build a native CUTLASS SM120
  block-scaled GEMM (`solution/cutlass_fp4*`), which compiles and runs on GB10.
- **Rejected:** hand-writing the NVFP4 dequant + SF swizzle from scratch
  (error-prone; CUTLASS already implements `tile_atom_to_shape_SFA/SFB`).
- **Blocker (named):** full fused-MoE parity. The qwen38-derived dense config
  uses `MmaTileShape <256,128,128>`, whereas the fused SM120 MoE uses
  `<128,128,256>` (NCU), so `tile_atom_to_shape_SFB` differs from the checkpoint
  `w1_bs` layout; the activation SFA produced by `fp4_quantize` also differs
  from the fused kernel's internal SFA. Re-instantiating the dense GEMM with the
  fused tile shape (and the SwiGLU epilogue `swiglu_limit=10.0`) is the next
  experiment.

## Diagnostic finding (last run)
`solution/diag_fp4.py`: with the **linear** SF layout the CUTLASS dense GEMM
gives ~1.6 abs error on a 2.6 absmax output; switching the quantizer to the
**swizzled** SF layout (`is_sf_swizzled_layout=True`, the default) drops it to
0.49 — i.e. the GEMM/layout plumbing is in the right family, and the residual is
the SF *atom* (the qwen38 dense config uses a different
`tile_atom_to_shape_SFA/SFB` than the fused MoE mainloop). Note the reference
for a correct claim must be FlashInfer's own fp4 GEMM (`mm_fp4` /
`cutlass_fused_moe`), not the pre-quant bf16 matmul — fp4 quantization alone is
~6% per element.

## Next RLCR step
Change `solution/cutlass_fp4_cuda.cu::Sm120Fp4LargeConfig` to
`MmaTileShape = PerSmTileShapeMNK = Shape<_128,_128,_256>`, rebuild, and compare
the w13 GEMM against the FlashInfer baseline (topk=1); then add the SiLU+mul+
requant and w2, then the routed weighted combine.

**Tried (rejected):** the tile-shape change to `<128,128,256>` did **not** move
parity — the block-scaled SF layout is tile-independent. Best alpha scan still
leaves ~50% residual, so the mismatch is the fused path's activation SFA /
mid-quantization / epilogue, not the weight SFB tile. A correct candidate needs
the grouped block-scaled array collective
(`sm120_blockscaled_mma_array_tma.hpp`) instantiated as FlashInfer does, or a
byte-exact replication of its internal activation-quant SFA layout.

## Cross-task note (profiler inflation)
The extend bf16 GEMMs that looked like 130–157 GB/s in the trace are actually
**200–234 GB/s** standalone (cuBLAS at the GB10 ~230–240 GB/s ceiling) — the
trace's per-launch durations are inflated for small/medium kernels. Only large
kernels (this MoE) have trustworthy trace shares.
