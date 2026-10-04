# Profile evidence — DSA long-context decode (4K / 32K / 100K)

Captured on the live tp=2 DSpark service (rank 0) with `POST /start_profile`
(`detailed_annotations=True`, `record_shapes=False`), decode-only: the radix
prefix cache is warmed with an identical prompt so the measured request does no
prefill. Traces in `~/profiles/longctx/L*_overlay.trace.json.gz`; `g_sk`
(generation sequence length) confirms 4103 / 32775 / 100008.

## Per-step latency (gpu_user_annotation step spans, median)

| context | VERIFY ms | DRAFT ms |
|---:|---:|---:|
| 4 096 | 1.71 | 3.23 |
| 32 768 | 1.63 | 2.09 |
| 100 000 | **1.21** | **1.96** |

## Decode kernel-family totals (same 6 DSpark steps; near-identical launch counts)

| family | 32K | 100K |
|---|---:|---:|
| MoE grouped GEMM | 342.2 ms (43.9%) | 334.3 ms (43.2%) |
| bf16 GEMV projections | 256.7 (33.0%) | 258.8 (33.4%) |
| bf16 cuBLAS fallback | 59.2 (7.6%) | 59.6 (7.7%) |
| NCCL | 32.9 (4.2%) | 32.2 (4.2%) |
| KDA conv + delta-rule | 17.0 (2.2%) | 17.0 (2.2%) |
| gated norm | 15.1 (1.9%) | 15.2 (2.0%) |
| MHC / hyper-connection | 12.3 (1.6%) | 13.4 (1.7%) |
| MoE aux | 8.3 (1.1%) | 8.6 (1.1%) |
| **DSA sparse attention** | 5.42 (0.70%) | 5.71 (0.74%) |
| **DSA indexer / logits / topk** | 3.65 (0.47%) | 4.80 (0.62%) |
| total decode bucket | 778.6 ms | 774.8 ms |

## Context-scaling attention kernels (absolute)

| kernel | 4K | 32K | 100K |
|---|---:|---:|---:|
| `deep_gemm::sm120_fp8_paged_mqa_logits` (indexer logits) | 0.67 ms | 0.84 | 1.60 |
| `_gather_index_k_scale_prefix_into_kernel` (O(context) gather) | — | 0.23 | 0.69 |
| `kpool_topk_transform_kernel<512>` | 1.14 | 1.85 | 1.38 |
| `_fused_mamba_state_scatter` (KDA state) | 6.17 | 4.11 | 4.11 |

## Un-profiled isolated decode (wall(t33)-wall(t1), medians of 5)

| context | ms/token |
|---:|---:|
| 4 096 | 106.4 |
| 32 768 | 113.1 |
| 100 000 | **83.5** |
