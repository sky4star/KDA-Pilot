# Benchmark method

GPU: NVIDIA GB10 (sm_121), 48 SMs, L2 25.2 MB, CUDA 13.0, torch 2.13.0+cu130.
Runs inside the serving image (needs torch + flashinfer; no server).

Rules (per `nvidia_agent_tasks/docs/measurement_contract.md`):

- **Cold L2 by default.** The selected-expert weight set (≥100 MB) is >2×L2, so
  every call already reads it from DRAM; no flush buffer is needed or subtracted.
- **CUDA-graph replay is the timer of record** (`bench/timing.GraphBench`).
  Production runs the MoE inside `full_cuda_graph_backend`.
- **Interleaved A/B arms**, order reversed each trial, median over 7 trials.
- **Correctness first** (baseline vs candidate, dequant-domain tolerance
  `atol=2e-2, rtol=1e-2`, poison + NaN/Inf checks) before any speedup counts.

Baseline arm = `flashinfer.fused_moe.cutlass_fused_moe` (`bench/probe_baseline.py`).
Candidate arm = `solution/entry.py::moe_forward` (native CUDA), same ABI.

Compile flags (candidate): nvcc `-O3 -gencode=arch=compute_121a,code=sm_121a
--expt-relaxed-constexpr`, CUTLASS headers from `flashinfer/data/cutlass/include`,
no fast-math.

Reproduce:

```
# (service stopped, dev container from the same image)
python3 bench/benchmark.py            # A/B, writes bench/results.jsonl
python3 bench/kernel_breakdown.py 1 144 8
python3 bench/kda_probe_graph.py
ncu --privileged -k regex:device_kernel -c 2 python3 bench/_ncu_moe.py 1 144 8
```
