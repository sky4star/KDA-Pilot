# Shape dispatch

Single kernel, no shape dispatch yet. The grouped block-scaled GEMM handles all
workload rows; the opportunity is a small-M specialization:

| bucket | planned path | why |
|---|---|---|
| M=1 (single-request decode) | native grouped NVFP4 block-scaled GEMM, small-M tile / more CTAs | measured ~188 GB/s vs ~228 ceiling; baseline uses 48 CTAs, 1 wave, spilling |
| M∈[2,8] (DSpark verify) | keep baseline for now | already at the copy ceiling (232 GB/s) |

`use_native_moe(m, e, topk)` would gate on `m == 1`; all other rows fall back to
the copied baseline.
