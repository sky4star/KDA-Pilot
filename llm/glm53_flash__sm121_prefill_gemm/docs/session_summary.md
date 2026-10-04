# Session summary — GLM-5.3-Flash prefill optimization (continue-here)

Date: 2026-10-02/03. Work root: `~/kda/KDA-Pilot`.
Task: `llm/glm53_flash__sm121_prefill_gemm/`. See also the umbrella
`~/kda/KDA-Pilot/llm/GLM53_FLASH_NVFP4_OPTIMIZATION_HANDOFF.md` (sections 7b–7d).

## 0. Environment / how to run

- Box: spark1 + spark2, **GB10 sm_121**, 48 SM, 25 MB L2, ~121 GB unified,
  ~228–240 GB/s. Serving = tp=2, DSpark, port 8940, chunked=4096. **Server is
  stopped (GPUs idle).**
- Dev container: `gemm-dev` (`sglang-ppmtp:glm53-ppmtp-dspark-v1`),
  mount `-v $HOME/kda/KDA-Pilot:/kda`.
- **TileLang 0.1.15 shadow install** in the container at `/tmp/tl15site`
  (the image has 0.1.12, which has no FP4). Run everything with:
  ```bash
  docker exec gemm-dev bash -lc "cd /kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe \
    && TILELANG_DISABLE_CACHE=1 PYTHONPATH=/tmp/tl15site python3 <script>"
  ```
  It requires the **SM121A guard patch** applied to
  `/tmp/tl15site/tilelang/src/tl_templates/cuda/instruction/mma_block_scale.h`
  (`patches/mma_block_scale_sm121a.patch`); if `/tmp/tl15site` is gone, reinstall
  `pip install --no-deps --target /tmp/tl15site tilelang==0.1.15` and re-apply.

## 1. Headline results

- **bf16 projections (24% of prefill): NO-GO.** GB10 cuBLAS bf16 ~100–102 TFLOPS;
  M=4096 shapes at 91–99 TFLOPS. Caveat: NCU shows the tensor pipe at **85%** →
  true silicon peak ≈120 TFLOPS, so up to ~1.2x theoretical headroom (open).
- **Routed NVFP4 MoE**: a fully-fused TileLang MoE **beats** the production
  `flashinfer.cutlass_fused_moe` baseline:

  | M | TileLang fused | baseline | speedup |
  |---:|---:|---:|---:|
  | 4096 | **15.31 ms** | 17.73 | **1.16x** |
  | 8192 | 24.04 | 28.20 | 1.17x |
  | 16384 | 44.18 | 51.76 | 1.17x |

  Correctness: exact vs a dequant-operand oracle (`max_rel = 0`).
  Breakdown at M=4096: K0 gather+quant 0.76 + K1 w13+SwiGLU 7.49 + K2 requant
  0.90 + K3 w2 GEMM 4.70 + combine 1.46 = 15.31 ms.

## 2. PR

- **https://github.com/sky4star/sglang/pull/7** — branch
  `tilelang-nvfp4-sm120-moe`, base `main` (fork `sky4star/sglang`), 10 files,
  +1456. Package: `python/sglang/srt/layers/moe/tilelang_nvfp4/`.
- Clean worktree used: `/tmp/opencode/sglang-tilelang`. Push with
  `GIT_SSH_COMMAND='ssh -o BatchMode=yes'` (see `spark-github-pr` skill; the
  stored PAT fails for git-over-HTTPS).

## 3. Final design (the promoted one)

Four/five kernels, all SM120 NVFP4 block-scaled (`T.mma_gemm_blockscaled`,
`sf_layout="blockscaled_chunk_kmajor"`), **warp specialization DISABLED**:

| kernel | file | role |
|---|---|---|
| K0 | `act_quant_kernel.py` | gather-by-expert + NVFP4 block-16 (E4M3) quant, tile `[128,128]` (3D fragment + `reduce_absmax(dim=2)`) |
| K1 | `w13_swiglu_kernel.py` | grouped w13 with **gate+up dual MMA** + fused SwiGLU(limit) |
| K2 | `act_quant_kernel.py` | requant the activated intermediate |
| K3 | `w2_gemm_kernel.py` | grouped w2, expert-ordered contiguous output |
| C | `moe_combine_kernel.py` | **gather-reduction** finalize (no atomics): `y[t]=Σ_s out2[slot_rows[t,s]]*scale` |

Key wins vs baseline: fused gather+quant, fused w13+SwiGLU, and (crucially) a
**gather-reduction combine instead of an atomic scatter** (atomic scatter was
3.35 ms; combine 1.46 ms).

## 4. Negative results (do NOT redo blindly)

All attempts to push beyond 15.31 ms failed:
1. **Packed "no empty blocks"** (Mega-MoE actual-block-count idea,
   `w13_swiglu_packed.py`): **neutral** (7.80 vs 7.88 ms). NCU: K1 is
   **latency/occupancy bound** (Tensor 51%, occupancy 16.8%, 0 spill), so
   removing hidden empty-block work doesn't help.
2. **Auto warp specialization + `setmaxnreg` hints** (`w13_swiglu_ws.py`):
   **compile/run hangs >30 min** — TileLang's auto-WS doesn't scale to the
   dual-MMA blockscaled kernel.
3. **Manual `T.ws` producer/consumer** (`ws_w13.py`, H100 recipe): **correct
   (max_rel 0.31%) but 1.5x slower** (12.86 vs 8.0–8.5 ms). The 128/128 role
   split + SIMT scale loads reduce parallelism; TileLang's `T.Pipelined`
   (WS-off) already overlaps TMA+MMA well.

Root cause: sm_121 is like H100 (register accumulator, no TMEM, register-based
MMA) but **worse** — `mma.sync` is **synchronous** (H100 has async `wgmma`) and
smem is **100 KB** (H100 228 KB). The occupancy bottleneck is intrinsic and not
fixable via TileLang's available scheduling controls. The baseline faces the
same (20% occ, 168 regs, spill) and we are already faster.

## 5. Open directions for the next session

1. **Custom SM120 block-scaled atom with a smaller output tile** to cut register
   pressure (TileLang's SM120 path is hard-fixed at `block_M==block_N==128`; the
   dual gate+up accumulator is ~128–256 regs/thread). Smaller tile → 2 blocks/SM
   → higher occupancy. Largest lever; a real kernel-writing effort.
2. **`T.Persistent` + `num_stages=3`** on the current kernels (cheap, untested;
   `examples/gemm/example_gemm_persistent.py`).
3. **Settle the bf16 peak**: raw MMA microbenchmark + a tuned native bf16 square
   GEMM vs cuBLAS (is the silicon peak ~102 or ~120 TFLOPS?). See
   `docs/open_questions.md`.
4. **chunked-prefill config A/B** on the live server (4096 vs 8192/16384) — MoE
   utilization rises with tokens; cheap config lever (see handoff §7b).
5. **Parity + integration**: match the baseline's activation-quant convention
   and wire `--moe-runner-backend`; also validate the tilelang 0.1.15 upgrade
   against SGLang/DSA.

## 6. Artifacts

- `solution/tilelang_moe/`: `grouped_nvfp4_gemm.py`, `act_quant_kernel.py`,
  `w13_swiglu_kernel.py`, `w2_gemm_kernel.py`, `moe_combine_kernel.py`,
  `fused_moe.py` (validated pipeline + bench); experiments:
  `w13_swiglu_packed.py`, `w13_swiglu_ws.py`, `ws_w13.py`, `ws_w13_s1.py`,
  `scatter_kernel.py`, `swiglu_kernel.py`, `_tune.py`, `_ws_real.py`.
- `solution/tilelang/`: `mma_block_scale_sm121a.patch`, `tl_sm120_nvfp4.py`.
- `bench/`: `roofline.py`, `fp4_peak.py`, `moe_scaling.py`, `b12x_probe.py`,
  `tl_probe.py`.
- `docs/`: `roofline.md`, `profile_evidence.md`, `backend_survey.md`,
  `tilelang_feasibility.md`, `tilelang_moe_milestone.md`,
  `fused_moe_tilelang_draft.md`, `open_questions.md`, `results.md`,
  `session_summary.md` (this file).
- Benchmarks: run `fused_moe.py --M 4096 --E 144 --K 4096 --N 2048`.

## 7. One-line status

TileLang can express a production-grade SM120 NVFP4 fused MoE and **beat the
CUDA baseline by 1.16x** (PR #7). Pushing further is blocked by sm_121's
register-accumulator / synchronous-MMA / 100 KB-smem occupancy wall; the next
real lever is a custom smaller-tile SM120 kernel.
