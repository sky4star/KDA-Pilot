# Benchmark method

GPU: NVIDIA GB10 (sm_121), 48 SMs, L2 25.2 MB, CUDA 13.0, torch 2.13.0+cu130.

- **Cold L2**: `R = ceil(2*L2 / weight_bytes)` rotating weight copies
  (`bench/timing.weight_copies`); weight > 2*L2 rows use R=1.
- **CUDA-graph replay** is the timer of record (`GraphBench`).
- **Interleaved arms**, order reversed per trial, median of 7.
- **Correctness first**: candidate vs cuBLAS and vs an fp64 oracle, poisoned
  outputs, `atol=7e-2, rtol=2e-2`.
- Compile flags (candidate): `-O3 -gencode=arch=compute_121a,code=sm_121a
  --expt-relaxed-constexpr`, no fast math.

Reproduce:
```
python3 bench/benchmark.py                 # A/B, writes bench/results.jsonl
```

Ceiling probe (same run): `torch.Tensor.copy_` / `.sum()` of 64–1024 MB gave
230–239 GB/s (64–256 MB) and 185–212 GB/s (1 GB).
