# KDA Prompt: glm53_flash__sm121_dsa_longctx_decode

Target: NVIDIA GB10 ×2 (tp=2, DSpark), single request, 100K context decode.

Question: does long-context decode introduce a new optimizable bottleneck
(presumably the DSA indexer / paged-MQA logits / topk, which are O(context))?

Result: **NO-GO** — decode at 100K is the same weight-bandwidth-bound regime as
4K. Per-step VERIFY/DRAFT latency is flat (1.71/1.63/1.21 ms and 3.23/2.09/1.96 ms
at 4K/32K/100K); decode kernel totals are identical 32K vs 100K. The only
context-scaling kernels (DSA indexer logits, index-key gather) stay at 0.6–0.7%
and are partly overlapped. Sparse attention is O(2048) by construction.

Artifacts: `bench/capture.py`, `bench/latency_probe.py`,
`bench/decode_isolate.py`, `bench/ctxagg.py`; traces in
`~/profiles/longctx/`; write-up in `docs/profile_evidence.md` + `docs/results.md`.
