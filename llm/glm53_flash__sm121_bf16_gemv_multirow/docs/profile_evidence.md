# Profile evidence

All traces are torch-profiler JSON gz captured on the live service with
`POST /start_profile` (rank 0).

| trace | arm | path |
|---|---|---|
| decode, DSpark, baseline | pre-change | `~/profiles/gemv_ab/decode_baseline/1790823197.539464-TP-0.trace.json.gz` |
| decode, DSpark, overlay  | with kernel | `~/profiles/gemv_ab/decode_overlay/1790822623.890467-TP-0.trace.json.gz` |
| decode+prefill, baseline | (earlier) | `~/profiles/gemv_ab/baseline/1790820625.37094-TP-0.trace.json.gz` |
| decode+prefill, overlay  | (earlier) | `~/profiles/gemv_ab/overlay/1790819900.2037234-TP-0.trace.json.gz` |

Parse helper: `top_kernels.py` (aggregate by family) and the annotation dump used
for `step[VERIFY/DRAFT bs=1]` latencies.

## Baseline kernel attribution (decode, pre-DSpark capture, `~/profiles/cap`)

- `internal::gemvx::kernel<... bf16 ...>` — 41.8% + 10.0% + 2.1% (885+696+210 launches)
- CUTLASS SM120 block-scaled MoE (fp4 e2m1/ue4m3) — 22.5%
- `cutlass_80_wmma_tensorop_bf16` — 7.1% + 1.6%

## Baseline vs overlay (decode-only, DSpark, 8 steps each)

Same `13714` total launches / `336` router calls; bf16 GEMV family:

- baseline: `cutlass80_wmma` 2247 launches, 501.29 ms + `cuBLAS_gemvx` 3.93 ms
- overlay: `sm120_bf16_gemv` 1477 launches, 300.41 ms + `cutlass80_wmma` 770
  launches, 112.56 ms + `cuBLAS_gemvx` 3.50 ms

The candidate replaces 1055→1477 of the bf16 GEMM calls and cuts that family by
**88.7 ms / 8 steps = 11.1 ms/step (−17.6%)**.

Caveat: run-to-run ambient (clock/thermal) differences on GB10 inflate
cross-start absolute comparisons; the same-launch-count family delta and the
annotation latencies are the robust signals.
