# Results — SM121 bf16 GEMV for GLM-5.3-Flash-NVFP4 decode

GPU: NVIDIA GB10 (sm_121), 48 SMs, L2 25.2 MB. Baseline = cuBLAS via `F.linear`
(the production path). Candidate = `sm120_bf16_gemv` (native CUDA).

## 1. Standalone A/B (cold L2, CUDA-graph, interleaved, median of 7)

Command: `python3 bench/benchmark.py` (writes `bench/results.jsonl`).

| workload (M,K,N) | cuBLAS cold µs | cand cold µs | speedup | cand GB/s |
|---|---:|---:|---:|---:|
| prod 5,4096,12576 | 472.0 | 416.4 | 1.133 | 247.8 |
| prod 5,4096,2048  | 76.4  | 65.6  | 1.165 | 256.7 |
| prod 5,1024,4096  | 39.8  | 34.1  | 1.168 | 247.8 |
| prod 5,4096,4096  | 149.4 | 131.6 | 1.135 | 255.6 |
| prod 5,4096,128   | 8.35  | 7.44  | 1.122 | 146.6 |
| prod 5,1536,8192  | 115.7 | 97.9  | 1.183 | 258.1 |
| prod 5,8192,4096  | 294.2 | 268.8 | 1.094 | 250.1 |
| prod 5,6144,4096  | 219.4 | 194.4 | 1.128 | 259.4 |
| prod 1,4096,77440 (lm_head) | 3624.9 | 2702.9 | **1.341** | 234.8 |
| prod 8,24576,4096 (draft fc) | 911.3 | 899.5 | 1.013 | 224.3 |
| prod 8,4096,20480 (draft)    | 739.7 | 707.9 | 1.045 | 237.6 |
| reg 1,4096,4096   | 202.8 | 146.7 | 1.382 | 228.8 |
| reg 8,4096,4096   | 151.7 | 140.1 | 1.083 | 240.5 |
| reg 2,4096,4096   | 150.4 | 133.4 | 1.127 | 251.8 |
| reg 8,4096,128    | — | — | fallback (cuBLAS wins 0.83×) | — |

**Production-row geomean = 1.136× (min 1.013×, max 1.341×).** All correctness
checks pass (fp64 oracle + cuBLAS, bf16 tolerance).

## 2. Roofline

- GB10 sustained read/copy ceiling measured at **~228 GB/s** (`bench/probe_baseline.py`,
  `sum` and `copy_` of 256–1024 MB).
- cuBLAS sits at 218–230 GB/s for the mid/large shapes (near ceiling) but only
  **175 GB/s on the M=1 `lm_head`** and **~12–18 GB/s on tiny-N** projections.
- The candidate streams weights once per output row and reuses each loaded weight
  across the M activation rows; it reaches **250–262 GB/s** on the mid/large
  shapes and **234 GB/s** on `lm_head`. Weight traffic (`N·K·2`) dominates
  (activations `M·K·2` and output `M·N·2` are negligible for M≤8).
- Active bound after optimization: DRAM read bandwidth. The remaining gap on
  tiny-N (`n128/n288`) is latency, not bandwidth (few blocks, ~1 MB weight).

## 3. E2E on the live server (tp=2 × 2 GB10, DSPARK)

Deploy: `run-dspark-tp2-overlay.sh` (bind-mounts patched `unquant.py` +
`sm120_bf16_gemv.{py,cuh}`). All 22 required instantiations JIT-built at warmup.

### 3a. No speculation (clean, content-independent)

Single request, real text, 128 new tokens, median of 10.

| arm | decode tok/s |
|---|---:|
| baseline | 15.21 |
| overlay  | **17.49 (+15.0%)** |

### 3b. DSpark — per-step latency (acceptance-independent)

Torch-profiler traces of decode-only steps (`/start_profile`, prefix-cache
warmup so no prefill step); both arms have identical launch counts (13714,
336 router calls = 8 steps):

| stage | baseline | overlay | speedup |
|---|---:|---:|---:|
| `step[VERIFY bs=1]` | 2229.3 µs | 1934.2 µs | **1.153×** |
| `step[DRAFT bs=1]`  | 998.4 µs  | 742.1 µs  | **1.345×** |
| `scheduler.run_batch` | 84.5 ms | 68.9 ms | **1.226×** |

Kernel-family totals over the same 8 steps (identical call counts): bf16 GEMV
`cutlass80_wmma`+`gemvx` 505.2 ms → candidate 300.4 ms + residual `cutlass80_wmma`
112.6 ms + `gemvx` 3.5 ms = **416.5 ms (−88.7 ms, −17.6%; 11.1 ms/step)**.
The DSpark draft projections (`24576×4096`, `4096×20480`) run at M=8 and are the
largest remaining bf16 consumers.

### 3c. DSpark — raw tok/s is confounded (do NOT use a single prompt)

DSpark tokens/s depends on the speculative accept length, which is a function of
the generated content. The candidate's ~1e-2 bf16 rounding differences change the
argmax on a fraction of tokens and therefore the emitted sequence, so the two
arms generate **different content** and have different accept rates. Measured
across 3 prompts × 4 runs, 192 new tokens:

| prompt | baseline tok/s | overlay tok/s |
|---|---:|---:|
| 0 | 23.90 | 21.65 |
| 1 | 16.85 | 16.42 |
| 2 | 29.57 | 42.10 |
| **overall mean / median** | 23.05 / 23.90 | **26.43** / 21.65 |

The per-prompt spread (14–42 tok/s) is far larger than the kernel effect and is
driven by acceptance, not speed. **A single-prompt tok/s A/B is not a valid
measure of this kernel under DSpark.** The per-step latencies in 3b are the
correct metric and both improve substantially.

## 4. Conclusion

- **Promote the kernel**: it is correct, reaches the DRAM ceiling on the wide
  shapes, and cuts verify/draft step latency by ~1.15×/1.35× and no-spec decode
  by +15%.
- **Measurement caveat**: under DSpark, end-to-end tokens/s is content- and
  acceptance-dominated; validate any acceptance-sensitive change with
  profiler step latencies or fixed-content workloads.

## 5. Deployment / rollback

- Overlay files: `~/glm53-gemv-overlay/{unquant.py,sm120_bf16_gemv.py,sm120_bf16_gemv.cuh}`,
  synchronised to spark2.
- Launch: `bash run-dspark-tp2-overlay.sh <0|1>` on each node.
- Rollback: use the original `run-dspark-tp2.sh` (no mounts) — no image change.
