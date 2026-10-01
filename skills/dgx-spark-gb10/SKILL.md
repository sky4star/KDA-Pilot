---
name: dgx-spark-gb10
description: Use when working on NVIDIA DGX Spark / GB10 (sm_121, arm64) GPU kernels, SGLang serving, or bandwidth/occupancy tuning on this box. Covers GB10 hardware facts (48 SMs, 25MB L2, ~228 GB/s DRAM, ~121GB unified memory), sm_121 kernel gotchas, which cuBLAS/SGLang JIT paths exist or are missing, and the shared-memory/live-service constraints. Triggers include "GB10", "DGX Spark", "sm_121", "spark1/spark2", "为什么 GB10 上 ...".
---

# DGX Spark / GB10 (sm_121) environment and kernel notes

Measured as of 2026-10 on this box (`spark1`, second node `spark2`). Re-verify with
the snippets below; do not trust memory of a different generation.

## Hardware / software

```
nvidia-smi:  NVIDIA GB10, driver 580.173.02, CUDA 13.0
torch 2.13.0+cu130; capability (12, 1); arch target compute_121a / sm_121a
```

| property | value |
|---|---|
| SMs | **48** |
| L2 | **25.2 MB** (`L2_cache_size = 25165824`) |
| DRAM read ceiling | **~228–232 GB/s** (measured; `sum`/`copy_` of ≥256MB) |
| total memory | ~121–130 GB **unified** (shared CPU+GPU) |
| max threads/SM | 1536 |
| regs/SM | 65536 |
| shared mem/block (opt-in) | 101376 B (~99 KB); **static limit 48 KB** |

Probe:
```bash
python3 -c "import torch;p=torch.cuda.get_device_properties(0);\
print(p.name,p.multi_processor_count,p.L2_cache_size/1e6);\
print(torch.cuda.get_device_capability(0))"
```

## Implications you must design around

- **Bandwidth is low (~228 GB/s), L2 is small (25MB).** A GEMV/GEMM weight larger
  than 25MB is cold on every call; do not use hot-L2 numbers. Weights < 25MB can
  stay L2-resident across decode steps — that is where cuBLAS wins and a
  `__ldcs` (evict-first) streaming kernel can lose.
- **Unified memory + a live service.** A `tp=2` server holds ~103/121 GB. There
  is normally **no idle GPU**: isolated kernel benchmarks require stopping the
  service, or a dedicated `--gpus all` dev container after freeing memory.
  `free -g` "free" ≈ GPU-free memory here.
- **Static smem ≤ 48KB.** For an M-row activation tile `M*K*2` bytes exceed 48KB
  as soon as `M≥5, K=4096`; either read x from L2/global or use dynamic smem
  (`cudaFuncAttributeMaxDynamicSharedMemorySize`, up to ~99KB).

## What exists / is missing on sm_121 in SGLang

- **No `cutedsl` bf16 GEMM** — `initialize_bf16_gemm_config()` only enables it on
  `get_platform().is_sm100`; `hopper_bf16_gemv` is gated to SM90. So bf16 decode
  linears on GB10 fall through to **cuBLAS**.
- `sm120_fp8_gemv` (fp8) exists; there was **no bf16 GEMV** for SM12x before the
  work this session added.
- cuBLAS kernel selection on sm_121 (observed in torch-profiler captures):
  - `M=1` → `internal::gemvx::kernel<... __nv_bfloat16 ...>`
  - `M∈[2,8]` → `cutlass_80_tensorop_bf16_s16816gemm_*` (not gemvx)
  - larger M / other → `nvjet_sm121_*`
  - MoE router (N=288) → `sglang::tiny_n_gemm_kernel` / `tiny_gemm_bf16`
  - SM120 fp4 MoE → CUTLASS `SM120ArrayTmaWarpSpecializedBlockScaled`
- **Do not assume the baseline is `gemvx`.** Profile first; a change that beats
  gemvx may still lose to the `cutlass_80_wmma` tile cuBLAS actually picks.

## SGLang JIT build

- `load_jit(name, *args, cuda_files=["gemm/foo.cuh"], cuda_wrappers=[("run","ns::Cls<...>::run")])`.
  `cuda_files` relative names resolve against `sglang/kernels/jit/csrc/`; **absolute
  paths work too**. `KERNEL_PATH/include` holds the `sgl_kernel/*` headers.
- Build flags: `-std=c++20 -O3 --expt-relaxed-constexpr -gencode=arch=compute_121a,code=sm_121a -DSGL_CUDA_ARCH=1210`; no fast-math.
- Cache: `/root/.cache/sglang/jit/sm121a/<module>_<args>/` (content-keyed, so edits
  recompile). `SGLANG_JIT_FORCE_RECOMPILE=1` forces a rebuild;
  `SGLANG_CRASH_ON_JIT_COMPILE` guards capture-time compiles.
- `@cache_once` on the module getter; `load_jit` is `torch.compiler.disable`d.
- First call compiles (~10–20 s/shape). During server warmup this is fine; if a
  shape is first hit **inside CUDA-graph capture** it errors — pre-warm or
  pre-compile.

## Measuring on GB10

- Cold-L2: rotate `R = ceil(2*L2 / weight_bytes)` weight copies (≥2× L2 total);
  `R=1` is already cold when `weight_bytes > 2*L2`.
- Time with CUDA-graph replay, not eager (kernels are 4–300 µs; eager launch cost
  dominates).
- The `dgx-spark-gb10` box is arm64 + GB10; treat anything built for x86/sm_100 as
  non-portable until re-measured.
