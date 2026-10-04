# Roofline — GLM-5.3-Flash prefill GEMMs on GB10 (sm_121)

Task: `glm53_flash__sm121_prefill_gemm`. Host spark1, idle, `gemm-dev` container
(image `sglang-ppmtp:glm53-ppmtp-dspark-v1`, torch 2.13+cu130, CUDA 13.0,
flashinfer 0.6.17, capability (12,1), 48 SM, 25.2 MB L2).

All numbers are standalone on this box with the GPU otherwise idle. Sources:
`bench/roofline.py`, `bench/fp4_peak.py`, `bench/moe_scaling.py`,
`bench/b12x_probe.py`.

## 1. bf16 projections (cuBLAS) — NO-GO

`bench/roofline.py` (GPU events, median of N; square GEMMs are cold by size):

| GEMM | µs | TFLOPS |
|---|---:|---:|
| 4096³ | 1470 | **93.5** |
| 8192³ | 11086 | 99.2 |
| 12288³ | 37191 | 99.8 |
| 16384³ | 86049 | **102.2** |

M=4096 projection-like shapes (`torch.mm`):

| M,K,N | µs | TFLOPS |
|---|---:|---:|
| 4096,4096,4096 | 1507 | 91.2 |
| 4096,4096,2048 | 746 | 92.2 |
| 4096,4096,12288 | 4406 | 93.6 |
| 4096,4096,24576 | 8436 | 97.7 |
| 4096,4096,154880 (lm_head) | 52708 | 98.6 |
| 4096,512,4096 | 292 | 58.9 |
| 4096,4096,512 | 275 | 62.4 |

**GB10 bf16 cuBLAS peak is ~100–102 TFLOPS.** The trace's ~90 TFLOPS
projections are at **88–99% MFU**; wide shapes are at peak. There is no
meaningful headroom for a native bf16 kernel — consistent with the earlier
`glm53_flash__sm121_bf16_gemm_multi_m` no-go. (The small-K/N rows are
latency-bound, but they are a tiny share; quantizing the ignore-list to FP4 would
change numerics and is out of scope.)

## 2. NVFP4 dense peak (control)

`bench/fp4_peak.py`, `flashinfer.mm_fp4`:

| backend | M,K,N | µs | TFLOPS |
|---|---|---:|---:|
| cudnn | 8192,8192,8192 | 3035 | **362.2** |
| cudnn | 16384,8192,8192 | 6384 | 344.5 |
| cutlass | 16384,4096,4096 | 1635 | **336.2** |
| cutlass | 4096,4096,4096 | 454 | 302–325 |
| b12x | 4096,4096,4096 | 426 | 322.4 |
| b12x | 8192,8192,8192 | 12137 | 90.6 (bad tactic at large M) |
| cute-dsl | any | — | unsupported on cap 121 |

**Dense NVFP4 peak ≈ 330–360 TFLOPS.**

## 3. Routed MoE (`cutlass_fused_moe`, TRT-LLM) — the target

K=4096, N=2048, E=144, topk=8 (`bench/moe_scaling.py`, weights built once,
activations vary; median of 15):

| M | M_e=rows/expert | base µs | TFLOPs | expert-wt GB/s |
|---:|---:|---:|---:|---:|
| 512 | 28 | 9846 | 20.9 | 184 |
| 1024 | 56 | 10529 | 39.2 | 172 |
| 2048 | 113 | 12268 | 67.2 | 148 |
| **4096 (prefill chunk)** | **227** | **17733** | **93.0** | **102** |
| 8192 | 455 | 28215 | 116.9 | 64 |
| 16384 | 910 | 51757 | 127.5 | 35 |

Regimes:
- **M≤1024**: weight-streaming bound (~180 GB/s of 211–228 ceiling).
- **M=4096**: neither roof — 93 TFLOPS (27% of FP4 peak) / 102 GB/s (45%).
- **M≥8192**: compute-bound plateau at ~127 TFLOPS (only ~36% of peak).

Kernel breakdown at M=4096 (`kernel_breakdown.py`): grouped CUTLASS GEMM
13.88 ms (77%), activation 1.41 ms, finalize 1.37 ms, expand/quant 1.21 ms;
total 17.9 ms.

## 4. Equivalent *dense* control — the gap

The MoE problem is equivalent to flat GEMMs over all `cum_m = M*topk = 32768`
routed rows:

| path | shape | µs | TFLOPS |
|---|---|---:|---:|
| cutlass `mm_fp4` | 32768,4096,4096 (w13) | 3294 | **333.8** |
| cutlass `mm_fp4` | 32768,2048,4096 (w2) | 2140 | 256.9 |
| **sum dense** | | **5434** | |
| **grouped MoE GEMM** | same FLOPs | ~13880 | 119 |

→ **The grouped kernel is ~2.6× off the dense FP4 ceiling** (17.4 ms vs 5.4 ms
when aux is added). This is the headroom.

## 5. Verdict

- bf16 projections: **no-go** (cuBLAS at hardware peak).
- Routed NVFP4 MoE at M=4096: **~2.5–3× theoretical headroom**, currently
  latency/occupancy-bound, not at either roof.
- Next: native grouped block-scaled SM120 GEMM with a high-occupancy
  (non-persistent) schedule, or a config/parallelism change that raises the
  grouped kernel's wave count. See `profile_evidence.md`, `backend_survey.md`.
