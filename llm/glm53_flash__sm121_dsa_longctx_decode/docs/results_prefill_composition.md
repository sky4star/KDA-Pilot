# Prefill composition — GLM-5.3-Flash (GB10, tp=2, DSpark)

Method: cold-prefill, `--chunked-prefill-size 4096`, torch-profiler capped to a
small number of forwards (`num_steps`; the box has ~1 GB free so only tiny
captures are safe). Two captures: 2 chunks (~8K ctx) and 4 chunks (~16K ctx).
Traces: `~/profiles/longctx/PF8192_ns2.trace.json.gz`,
`PF16384_ns4.trace.json.gz`.

## Per-chunk breakdown (share of prefill GPU time)

| operator family | 8K | 16K |
|---|---:|---:|
| **bf16 projections (cuBLAS wmma/nvjet)** | **24.3%** | **23.9%** |
| **MoE routed grouped GEMM (NVFP4)** | **21.2%** | **20.8%** |
| MHC / hyper-connection (tilelang) | 12.9% | 12.6% |
| other (elementwise/quant/rope/copy) | 12.2% | 12.0% |
| NCCL | 7.6% | 8.2% |
| MoE aux (routing/sort/activation/finalize) | 7.2% | 7.0% |
| **DSA sparse attention (tilelang)** | **6.9%** | **6.8%** |
| KDA conv + sigmoid-gating delta-rule | 6.4% | 6.2% |
| gated norm | 0.8% | 0.8% |
| DSA indexer logits + topk | 0.5% | 0.8% |
| bf16 GEMV (M<=8 path) | 0 | 0.8% |

## Answer to "is sparse attention + MoE GEMM the biggest?"
- **MoE GEMM yes-ish (21%)**, but the **single biggest is the bf16 projection
  GEMMs (24%)** — all attention/KDA/dense/shared linear layers at M=4096 (they
  fall to cuBLAS, not the M<=8 GEMV fast path).
- **Sparse attention is only ~7%**, not the biggest. The indexer is <1%.
- Other large non-attention/non-MoE slices: **MHC 13%**, NCCL 8%, elementwise 12%.

So prefill ≈ **GEMM-dominated** (projections 24% + MoE 21% + aux 7% ≈ 52%),
plus MHC 13%, comm 8%, attention 8%, KDA 6%.

## Context scaling (timing-only, server per-chunk throughput)
Per-chunk time is ~constant (~2.0 s / 4096 tokens) from ctx≈0 to ctx≈100 K, i.e.
no monotonic growth → prefill is a **fixed per-chunk cost**, not O(N²)-dominated.
This is why DSA (top-2048 sparse attention) keeps attention small; only the
indexer logits grow (0.5%→0.8% over 8K→16K) and stay <~1–2% even at 100K.

## Note on prefill vs decode bound
- Projections at M=4096: ~1.5 ms per [4096]x[4096,4096] bf16 GEMM ≈ **~90 TFLOPS**
  → **compute-bound**, unlike the weight-bound decode.
- MoE grouped fp4 at M=4096: ~5 ms per GEMM → low MFU; headroom to investigate.

## Caveats
- Captures are 2–4 chunks (8–16K ctx) due to the ~1 GB free memory; the
  composition is extrapolated to 100K using the flat per-chunk timing + the
  indexer's measured O(context) growth.
- "proj_cublas" includes attention QKV/KV/o_proj, KDA projections, the 3 dense
  layers, and the (unfused) bf16 shared expert.
