# Benchmark method

## Standalone harness (`bench/benchmark.py`, `bench/timing.py`)

GPU: NVIDIA GB10, sm_121, 48 SMs, L2 = 25.2 MB, CUDA 13.0 / torch 2.13.0+cu130.
Runs inside the serving image (needs `torch` + the `load_jit` toolchain).

Rules (per `nvidia_agent_tasks/docs/measurement_contract.md`):

- **Cold L2** via `R` rotating weight copies, `R = ceil(2·L2 / weight_bytes)`
  (weight > 2·L2 ⇒ `R=1` is already cold). No large flush subtracted from a small
  kernel.
- **CUDA-graph replay** is the timer of record: the graph captures `R` calls with
  distinct buffers; per-call = elapsed/(replays·R). Production runs these ops in
  `full_cuda_graph_backend`, so eager per-launch overhead is excluded.
- **Interleaved A/B arms**, order reversed each trial, median over 7 trials.
- **Correctness first**: candidate vs an fp64 oracle and vs the cuBLAS baseline,
  poisoned outputs, `atol=7e-2, rtol=2e-2`.
- Baseline = `torch.mm(x, w.t(), out=)` (the production cuBLAS path); candidate =
  `sm120_bf16_gemv_out` through the identical signature.

Compile flags: nvcc `-std=c++20 -O3 --expt-relaxed-constexpr`, target
`compute_121a`, no fast-math. Both arms launch through
`at::cuda::getCurrentCUDAStream()`.

Reproduce:

```
python3 bench/probe_baseline.py          # cuBLAS ceiling
python3 bench/benchmark.py               # full A/B, writes bench/results.jsonl
python3 bench/sweep_n128_n288.py         # config sweep for narrow N
```

## E2E harness (`glm53-gemv-overlay/e2e_bench.py`)

Single-request decode via native `/generate` with streaming: reports TTFT, decode
tok/s (`(completion_tokens-1)/decode_time`) and e2e tok/s. Deployed by
bind-mounting the patched files over the image source (`run-ab.sh OVERLAY=1`,
`run-dspark-tp2-overlay.sh`).

Step-level, acceptance-independent latencies are read from torch-profiler traces
captured with `/start_profile` (`step[VERIFY bs=1]`, `step[DRAFT bs=1]`,
`scheduler.run_batch`).
