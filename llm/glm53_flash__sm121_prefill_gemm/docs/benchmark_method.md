# Benchmark method / reproduction

## Environment

```bash
docker rm -f gemm-dev
docker run -d --name gemm-dev --gpus all --privileged --network host --ipc host \
  --ulimit memlock=-1:-1 -v $HOME/kda/KDA-Pilot:/kda \
  sglang-ppmtp:glm53-ppmtp-dspark-v1 sleep infinity
```

Idle spark1 (48 SM, 25.2 MB L2, cap (12,1), torch 2.13+cu130, flashinfer 0.6.17).

## Probes (all in `bench/`)

| script | purpose |
|---|---|
| `roofline.py` | bf16 peak + M=4096 projection shapes |
| `fp4_peak.py` | dense NVFP4 peak per backend (mm_fp4) |
| `moe_scaling.py` | routed MoE time/TFLOPS/GB/s vs M (weights built once) |
| `b12x_probe.py` | b12x fused MoE correctness+timing vs `cutlass_fused_moe` |
| `b12x_backend_probe.py` | force b12x micro/static/dynamic |
| `b12x_breakdown.py` | torch-profiler kernel breakdown of b12x |
| `moe_tune_probe.py` | baseline `tune_max_num_tokens` sweep |
| `_ncu_moe.py` | NCU target for the baseline grouped GEMM |

Run inside the container, e.g.:

```bash
docker exec gemm-dev python3 /kda/llm/glm53_flash__sm121_prefill_gemm/bench/roofline.py
docker exec gemm-dev python3 /kda/llm/glm53_flash__sm121_prefill_gemm/bench/moe_scaling.py
```

NCU (baseline grouped GEMM, w13 launch at M=4096):

```bash
docker exec gemm-dev bash -lc "cd /kda/llm/glm53_flash__sm121_nvfp4_moe_decode/bench && \
  ncu --target-processes all --kernel-name regex:device_kernel --launch-skip 6 \
  --launch-count 1 --set full --export /tmp/ncu_moe4096 python3 _ncu_moe.py 4096 144 8"
docker exec gemm-dev bash -lc "ncu --import /tmp/ncu_moe4096.ncu-rep --page details"
```

NCU (b12x dynamic):

```bash
docker exec gemm-dev bash -lc "cd /kda/llm/glm53_flash__sm121_prefill_gemm/bench && \
  ncu --target-processes all --kernel-name regex:MoEDynamicKernel --launch-skip 3 \
  --launch-count 1 --set full --export /tmp/ncu_b12x python3 b12x_breakdown.py 4096 144 8 --only"
```

## Timing rules

- Standalone timing uses CUDA events, median of N; M=4096 MoE weights (1.8 GB)
  are cold by size.
- For the promoted candidate: cold-L2 (rotate ≥2× L2), CUDA-graph replay,
  interleaved A/B, preallocated outputs — per the KDA measurement contract.
