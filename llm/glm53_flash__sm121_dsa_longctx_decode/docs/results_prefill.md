# Results — long-context PREFILL / TTFT (cold, single request)

Target: GB10 ×2 (tp=2, DSpark), timing-only (no profiler — see incident doc).

## Verdict: **prefill has a real, large, config-level win.**
`--chunked-prefill-size 512 -> 4096` gives **~2x faster cold prefill/TTFT**,
because chunked prefill re-reads all weights once per chunk.

## Measurement (cold TTFT = wall of a max_new=1 request on a unique prompt)

| context | chunk=512 | chunk=4096 | speedup |
|---:|---:|---:|---:|
| 8 192 | 7.73 s | 4.25 s | 1.82× |
| 32 768 | 34.61 s | 16.66 s | 2.08× |
| 100 000 | 102.34 s | 51.25 s | 2.00× |

Both arms measured on the same box/boot cycle by restarting with
`CHUNKED=<n>` (`~/glm53-gemv-overlay/run-ab.sh`). Throughput rises from
~950–1060 tok/s (512) to ~1950 tok/s (4096), still roughly context-independent.

## Why
Each prefill forward runs every layer and streams all weights; with
`chunked-prefill-size=512` a 100K prompt needs ~196 forwards -> ~196 full weight
passes. At 4096 it needs ~25 -> ~8x less weight traffic. The measured gain is
~2x, not 8x, because a second component (prefill attention / indexer / other,
which does **not** scale down with chunk size) becomes ~half the time:
solving `W + C = 102`, `W/8 + C = 51` gives `W ≈ 57 s`, `C ≈ 45 s` at 100K.
So 4096 is near the knee; going to 8192/16384 only shaves the remaining `W`
(`~7 s` more at most), with more memory risk.

## Cost / caveat
- Larger chunks need more activation memory (avail mem ~10 GB during a 4096
  prefill). Safe for a single request; validate under your concurrency
  (`--max-running-requests 4`) before promoting.
- Boot weights are ~91 GB; the box has ~5 GB headroom, so avoid the profiler.

## Recommendation
Set `--chunked-prefill-size 4096` (from 512) for this workload → **~2x TTFT**.
Next lever if more is needed: the remaining `C ≈ 45 s` at 100K is prefill
attention/indexer (DSA prefill backend `tilelang`), a separate kernel task.
