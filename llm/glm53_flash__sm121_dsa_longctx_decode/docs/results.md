# Results — GLM-5.3-Flash 100K-context single-request decode

Target: NVIDIA GB10 ×2 (tp=2, DSpark), single request, decode-only.

## Verdict: **no context-scaling bottleneck** — long-context decode stays in the
same weight-bandwidth-bound regime as short context.

## Evidence
1. **Per-step latency is flat (even lower at 100K).** VERIFY 1.71 → 1.63 → **1.21**
   ms; DRAFT 3.23 → 2.09 → **1.96** ms at 4K/32K/100K.
2. **Decode kernel totals are identical 32K vs 100K** (778.6 vs 774.8 ms) with
   near-identical launch counts (MoE 504, GEMV 1265/1266, ...).
3. **Un-profiled isolated decode is flat-to-faster**: 106 / 113 / **83** ms/token.
4. Only the DSA indexer grows with context, and it stays tiny:
   - `paged_mqa_logits` 0.84 → 1.60 ms (66 launches = 11 DSA layers × 6 steps;
     ~24 µs/layer at 100K);
   - `_gather_index_k_scale_prefix_into_kernel` 0.23 → 0.69 ms (O(context));
   - combined indexer share 0.47% → **0.62%**.
5. **Sparse attention stays bounded** (top-2048): 5.42 → 5.71 ms (0.70 → 0.74%).

## Why
GLM-5.3's attention is **DSA**: each layer scores the context only to pick
`index_topk=2048` tokens (with `index_kpool=4` compression and the indexer shared
across MTP iterations), then attends to those 2048. So decode attention is
**O(2048)**, not O(context). The 34 KDA layers are recurrent (`O(1)` state).
What remains is the fixed weight stream (MoE 43% + projections 33%) — independent
of context.

## Roofline of the one growing kernel
`sm120_fp8_paged_mqa_logits` at 100K: ~25K compressed index keys × 128 dim fp8
≈ 3.2 MB/layer → ~13 µs at the 240 GB/s ceiling; observed ~24 µs → ~1.8× off
roofline, but **~0.15% of decode** total. Not worth a task.

## Note (prefill, not decode)
A **cold** 100K prefill costs ~56 s (chunked-prefill 512 ⇒ ~196 chunks). If
long-context TTFT matters, that is the real long-context cost — a separate
(prefill) question, not decode.

## Conclusion
For single-request decode, 100K is **not** a new bottleneck: it is the same
roofline-bound regime as 4K. No KDA task warranted. The only decode headroom
found anywhere remains the **NVFP4 MoE at M=1** (`…nvfp4_moe_decode`).
