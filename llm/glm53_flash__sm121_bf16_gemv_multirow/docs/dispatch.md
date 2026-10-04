# Shape dispatch

`sm120_bf16_gemv._config(m, n, k)` returns `(kRows, kUnroll, kNumWarps, stageSmem)`:

| bucket | config | why |
|---|---|---|
| `m == 1`, `n ≥ 8192` | `(2, 2, 8, stage)` | wide rows; 2 rows/warp amortizes the smem-staged activation (lm_head) |
| `m == 1`, `n < 8192` | `(1, 2, 8, stage)` | Hopper-tuned M=1 GEMV |
| `m ∈ [2,8]`, `n ≥ 1024` | `(1, 2, 8, stage)` | stream weight once, reuse across M; stage when `M·K·2 ≤ 48KB` |
| `m ∈ [2,8]`, `n < 1024` | `(1, 2, 4, False)` | latency-bound: skip staging, 4 warps ⇒ more blocks |

`stage = (M·K·sizeof(bf16) ≤ 48KB)`: with staging, the activation tile is read
once per block; without it every warp re-reads it from L2 (fine when N is tiny).

`use_sm120_bf16_gemv(m, n, k)` gates `m∈[1,8]`, `k%512==0`, `512≤k≤32768`,
`n%8==0`, `8≤n≤131072`. It additionally **returns False for `m≥6` and
`64≤n≤192`** — the measured latency-bound bucket where cuBLAS still wins at high
M (`M=8,N=128` was 0.83×). Other shapes fall back to `F.linear`.

Per-bucket measured standalone speedups vs cuBLAS (cold L2, GB10) are in
`docs/results.md`.
