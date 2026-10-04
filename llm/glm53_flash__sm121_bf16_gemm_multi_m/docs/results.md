# Results — SM121 bf16 large-M GEMM (GLM-5.3-Flash extend)

## Verdict: **no-go** — cuBLAS is already at the DRAM roofline.

## Measurement
Cold-L2 (rotating weight copies), CUDA-graph replay, interleaved arms, median of
7. `python3 bench/benchmark.py` inside the serving image.

| workload (M,K,N) | cuBLAS µs | cuBLAS GB/s | prod |
|---|---:|---:|---|
| 42,4096,4096 | 155.9 | 219.7 | yes |
| 42,4096,2048 | 83.1 | 208.1 | yes |
| 8,24576,4096 | 861.0 | 234.4 | yes |
| 42,4096,12576 | (see results.jsonl) | ~220 | yes |
| 64,4096,4096 | 163.5 | 211.7 | reg |
| 42,4096,128 | 7.0 | 199.7 | reg |

## Roofline
- GB10 copy ceiling (same run): 230.3 (64 MB), 239.4 (256 MB), 185.1 (1 GB) GB/s.
- GB10 read ceiling: 211.8 (256 MB), 199.2 (1 GB) GB/s.
- cuBLAS sits at **0.87–0.98 of the 64–256 MB copy ceiling** on every row →
  bandwidth-saturated. Active bound = DRAM read bandwidth. No headroom.

## Why the trace looked better
The torch-profiler kernel durations for these M=42 GEMMs implied 130–157 GB/s.
Standalone re-measurement shows 200–234 GB/s. The gap is **profiler per-launch
inflation**, not a real inefficiency (confirmed independently: the KDA gated-norm
reads "69 µs" in the trace but 1.6 µs under a CUDA graph).

## Candidate attempt
`solution/bf16_gemm.cuh` — a tiled SIMT bf16 GEMM (weight reuse across a full
M tile, smem staging, BK=32 double buffer, 4x8 register tile). Correct
(`atol=7e-2, rtol=2e-2` vs cuBLAS + fp64 oracle) but **~12x slower** (17 GB/s)
from smem bank conflicts and low occupancy. A tuned version is bounded by the
same DRAM ceiling cuBLAS already reaches, so the task is closed as a no-go
rather than an open optimization.

## Next operator
`glm53_flash__sm121_nvfp4_moe_decode` (NVFP4 MoE M=1, 188 GB/s vs ~240 ceiling).
