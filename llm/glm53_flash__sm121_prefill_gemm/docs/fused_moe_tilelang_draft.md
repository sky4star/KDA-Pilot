# Draft — TileLang fully-fused MoE (target: match baseline)

Goal: a TileLang routed-MoE that **matches `cutlass_fused_moe` at M=4096**
(~17.7 ms) on GB10 (sm_121a), correct within the model's tolerance. Purpose is a
**production-suitability assessment of TileLang**, not beating the baseline.

## Baseline anatomy (what must be matched)

`flashinfer.cutlass_fused_moe` (TRT-LLM) at M=4096, E=144/rank, topk=8, K=4096, N=2048:

| kernel | ms |
|---|---:|
| `expandInputRowsKernel` (routing + gather + act quant) | 1.21 |
| grouped block-scaled GEMM (w13 + w2) | 13.88 |
| `doActivationKernel` (SwiGLU + requant) | 1.41 |
| `finalizeMoeRoutingKernel` (scatter + weighted sum) | 1.37 |
| **total** | **17.73** |

## Proposed architecture

Four kernels first (drop-in, low risk), then fuse to **three** (target).

Shared layout: **grouped-contiguous**, expert-padded to `cap = ceil(max m_e/128)*128`
(as in `grouped_nvfp4_gemm.py`). Host computes, per call:
`token_ids[rows]` (row -> origin token), `row_w[rows]` (routing weight),
`expert_offsets[E+1]`, `bpe = cap/128`. Weight scale factors (SFB) are
pre-packed in `blockscaled_chunk_kmajor` once at load time (off the hot path).

### K0 — gather + activation quant (1 kernel)
`X[M,K] bf16, topk_ids, expert_offsets -> A1_q[rows,K/2] uint8, SFA1[...] uint32`
- gather row r = `X[token_ids[r]]`, then the `deepseek_v4/act_quant.py`
  `fp4_quant_kernel` idiom: `T.reduce_absmax` per 16-block, E4M3 scale `amax/6`,
  `T.clamp(.,-6,6)`, cast to `T.float4_e2m1fn`; write scales into the padded
  `blockscaled_chunk_kmajor` layout (host-swizzle fallback first).
- Refs: `examples/deepseek_v4/act_quant.py::fp4_quant_kernel`.

### K1 — grouped w13 (gate+up dual MMA) + fused SwiGLU
`A1_q,SFA1, wg_q/SFBg, wu_q/SFBu, expert_offsets -> act[rows,N] bf16 (then fp4)`
- one CTA = (expert e, m-tile), loops n-blocks; **two `T.mma_gemm_blockscaled`
  calls into `Cg`,`Cu` [128,128]** against `wg`/`wu` (= w1 split at N; N is a
  multiple of 128 so SFB blocks align).
- epilogue: `g = Cg*scale`, `u = Cu*scale`; `a = silu(clamp(g,limit))*clamp(u,±limit)`.
- Refs: `examples/fusedmoe/...` (proves the dual-accumulator gate/up epilogue at
  128x128) + our SM120 `T.mma_gemm_blockscaled`.

### K2 — activation requant
`act bf16 -> A2_q[rows,N/2] uint8, SFA2 uint32`
- same kernel body as K0 minus the gather. (Fused into K1's epilogue in the
  final 3-kernel version.)

### K3 — grouped w2 + fused scatter + routing weight
`A2_q,SFA2, w2_q/SFB2, expert_offsets, token_ids, row_w -> y[M,K] fp32`
- one CTA = (expert e, m-tile, k-tile); grouped w2 block-scaled GEMM.
- epilogue: `atomically y[token_ids[row], :] += C * row_w[row]` (fold routing
  weight + scatter into the GEMM epilogue). Refs: our `scatter_kernel.py`.

Final target: **3 kernels** (K0, K1 with requant fused, K3) vs the baseline's 5.

## Budget vs baseline (target)

| kernel | est. ms |
|---|---:|
| K0 gather+quant | ~1.5 |
| K1 w13 + silu (+requant) | ~8.0 |
| K2 requant (if separate) | ~0.8 |
| K3 w2 + scatter | ~5.0 |
| **total** | **~15–16** |

This already projects **≤ baseline** because our grouped GEMM ties (13.35 vs
13.9) and 3 kernels of fused aux should beat the baseline's 3.8 ms aux. The
uncertain part is K3's atomic scatter (our standalone version was 5.6 ms; the
baseline's finalize is 1.37 ms) — must use vectorized `atomic_addx4` or a
segmented reduction.

## Open risks (ranked)

1. **K1 gate+up dual block-scaled MMA** — TileLang SM120 block-scaled path is
   fixed at `block_M==block_N==128`; dual accumulators + operand fragments may
   spill. Mitigation: the fp16 fusedmoe example runs the same shape; verify regs.
2. **In-kernel fp4 quant + direct `blockscaled_chunk_kmajor` scale output** —
   e2m1 cast is proven (act_quant); the swizzled scale write is fiddly. Fallback:
   host swizzle (~0.1 ms — measured cheap).
3. **K3 scatter throughput** — atomics at rows=32768/K=4096 was 5.6 ms standalone.
   Needs `atomic_addx4` / vectorization / reduction to reach ~1.5 ms.
4. **Toolchain** — requires tilelang 0.1.15 + the 1-line SM121A guard patch; the
   serving image pins 0.1.12. Adoption needs SGLang/DSA re-validation.
5. **Parity** — matching the baseline's exact activation-quant convention.

## Milestones

- **M1**: K0/K1(act bf16)/K2/K3 wired, correctness vs the dequant oracle, full
  number at M=4096. (Parse nothing; just get the 4-kernel path correct.)
- **M2**: fuse requant into K1 epilogue -> 3 kernels.
- **M3**: optimize K3 scatter to <=1.5 ms.
- **M4**: cold-L2 / CUDA-graph symmetric A/B vs `cutlass_fused_moe`; NCU evidence.
- **M5**: `--moe-runner-backend` integration + baseline activation-quant parity.

## Deliverable

`solution/tilelang_moe/fused_moe.py` + benchmark table + a written verdict:
"is TileLang sufficient for production MoE on GB10?" — with the toolchain and
parity caveats.

## Decision before writing

- Confirm the **4-kernel-first** path (safer) vs going straight for the
  3-kernel fused requant.
- Confirm we accept the **0.1.15 + patch** toolchain for this evaluation.

## Implementation status (M1/M2 done)

All four kernels written and wired in `solution/tilelang_moe/fused_moe.py`.
Per-kernel exact validation lives in each kernel file.

| kernel | file | M=4096 (ms) | notes |
|---|---|---:|---|
| K0 gather+quant | `act_quant_kernel.py` | 0.71 | gather + NVFP4 block-16 (E4M3), tile [128,128] |
| K1 w13+swiglu | `w13_swiglu_kernel.py` | 7.65 | gate+up dual MMA + fused SiLU |
| K2 requant | `act_quant_kernel.py` | 0.88 | same kernel, identity gather |
| K3 w2+scatter | `w2_scatter_kernel.py` | 8.50 | grouped w2 + fused atomic scatter |
| **total** | | **17.74** | **baseline 17.73 → 1.00x (matched)** |

Correctness: **exact vs a dequant-operand oracle (max_rel = 0.00000)**.

Key fixes:
- **Disable warp specialization** on the dual-accumulator kernel: with WS the
  512-thread/128-reg kernel spilled **122M** times (K1 = 18.8 ms); WS off →
  **7.5 ms**. The fp16 upstream fusedmoe example does the same.
- **K0/K2 quant tile [128,16] → [128,128]** (3D `(blk_m, G, 16)` fragment,
  `T.reduce_absmax(dim=2)`): **2.32 → 0.75 ms** (also removes the 65k tiny
  CTAs). The 2D group reduction is rejected by TileLang's fragment-layout
  inference; the 3D form is accepted.
- K3: hoist `token_ids`/`row_scale` to shared (per-row, not per-element;
  -1.3 ms) and stage `C` through shared before the atomics (-1.3 ms).

### Verdict

**TileLang is sufficient for production MoE on GB10 (sm_121a).** A complete
fused routed MoE — gather+quant, gate/up dual block-scaled MMA, SwiGLU, requant,
w2 block-scaled GEMM, routing-weight + scatter — written entirely in the **new
SM120 block-scaled TileLang path**, matches the CUDA `cutlass_fused_moe` baseline
(1.00x) with exact numerics.

Caveats before adoption:
- Needs **tilelang 0.1.15 + the SM121A guard patch** (serving image pins 0.1.12);
  SGLang / DSA tilelang must be re-validated after upgrade.
- Parity against the baseline's *exact* activation-quant convention, and
  `--moe-runner-backend` integration + cold-L2/CUDA-graph A/B, are still TODO.
- This is a match, not a win; the grouped GEMM is weight/latency bound.




## Plan A (baseline-style combine) — beats the baseline

Replace the fused atomic scatter with the baseline's structure: K3 writes
`out2` in expert order (contiguous, no atomics) and a separate **gather-reduction**
combine kernel does the token finalize.

| kernel | ms (M=4096) |
|---|---:|
| K0 gather+quant | 0.76 |
| K1 w13+SwiGLU | 7.49 |
| K2 requant | 0.90 |
| K3 w2 GEMM | 4.70 |
| C combine (gather-reduction) | 1.46 |
| **total** | **15.31** |
| `cutlass_fused_moe` | 17.73 |
| **speedup** | **1.16x** |

Correctness: exact vs the dequant oracle (`max_rel = 0`).

Scaling (all ~1.16–1.17x faster than the baseline):

| M | TileLang (Plan A) | baseline | speedup |
|---:|---:|---:|---:|
| 4096 | 15.31 | 17.73 | 1.16x |
| 8192 | 24.04 | 28.20 | 1.17x |
| 16384 | 44.18 | 51.76 | 1.17x |

Why it wins:
- The atomic scatter was the only deficit (3.35 ms). The gather-reduction
  combine (`moe_combine_kernel.py`) hits the ~1.4 ms memory floor with no
  contention, and removing the C_sh staging also speeds the w2 GEMM
  (4.70 vs 5.15 standalone).
- The front half was already 1.15x faster (fused gather+quant and
  w13+SwiGLU); now the back half ties/beats the baseline too.
- Unlike the atomic version, the gain does not shrink with M.

## Further headroom analysis (occupancy / scheduling)

### Packed "no empty blocks" (Mega-MoE core idea) — neutral

Implemented `w13_swiglu_packed.py`: host packs the **actual** per-expert block
list (`Σ_e ceil(m_e/128)`), grid early-exits `bm >= total_blocks` (block-uniform).
Controlled A/B at E=144 (same distribution):

| version | grid | ms |
|---|---|---:|
| non-packed bpe=2 | 16×288 | 7.884 |
| **packed (no empty blocks)** | 16×288 | **7.799** |

→ **neutral**. NCU explains why: K1 is **latency/occupancy bound**
(Tensor 51%, occupancy 16.8%, 242 regs, **0 spill**, eligible warps 0.17). The
extra/empty blocks' work is hidden behind stalls, so removing them doesn't help.
DeepGEMM's "every M-block reloads weights" premise assumes a compute/reload bound
that does not hold for this latency-bound SM120 kernel.

### H100-style warp specialization — blocked by tooling

sm_121 is structurally like H100 (register accumulator, no TMEM) but **worse**:
`mma.sync` is **synchronous** (H100 has async `wgmma`) and smem is **100 KB**
(H100 228 KB). H100's recipe = warp specialization + `setmaxnreg` producer/
consumer register reallocation + deep pipeline + persistent.

TileLang exposes all of it (`T.ws` roles, `T.annotate_producer_reg_dealloc` /
`annotate_consumer_reg_alloc`, `T.set_max_nreg`, `T.Persistent`). Tried the
auto-WS path with `annotate_producer_reg_dealloc(24)` /
`annotate_consumer_reg_alloc(240)` on the dual-MMA blockscaled kernel: the
**compile/run hung (>30 min, killed)** — TileLang's auto-WS transform does not
scale to this kernel. Manual `with T.ws(...)` is the remaining path and is a
large rewrite.

### Status

Current PR result stands: 15.31 ms vs baseline 17.73 ms (1.16x). Remaining cheap
levers: `T.Persistent`, `num_stages=3`. Big lever (manual WS) is a separate
effort with uncertain payoff (the WS-off K1 already beats the baseline).

### Manual `T.ws` warp specialization (H100 recipe) — correct but slower

Implemented `ws_w13.py` (grouped w13, single B) with manual producer/consumer
roles (`with T.ws(1)` producer: TMA A/B + SIMT scale loads; `with T.ws(0)`
consumer: blockscaled MMA), following
`examples/warp_specialize/example_warp_specialize_gemm_barrierpipe_stage2.py`.
The upstream fp16 WS example passes on this box, and our blockscaled WS kernel
is **correct** (max_rel 0.31%, fp4 rounding).

But performance is **worse**:

| kernel | ms (E=144, N=4096, K=4096) |
|---|---:|
| non-WS grouped w13 (`T.Pipelined`) | 8.0–8.5 |
| **manual WS w13, num_stages=2** | **12.86** |
| manual WS, num_stages=3/4 | 12.95 / 13.00 |

The 128-producer / 128-consumer split (with SIMT scale loads on the producer)
reduces parallelism vs TileLang's automatic software pipeline, which already
overlaps TMA + MMA well within one warp set. So the H100 "WS + setmaxnreg"
recipe does **not** help this latency-bound SM120 kernel.

### Conclusion of the scheduling exploration

Three attempts to push K1/K3 beyond the current design all failed:
- packed "no empty blocks" (Mega-MoE) → neutral;
- auto-WS + register hints → compile hang;
- manual `T.ws` → correct but 1.5x slower.

The current WS-off, `T.Pipelined` design is the best found. The occupancy
bottleneck is intrinsic to sm_121 (register accumulator, synchronous `mma.sync`,
100 KB smem) and is not addressable through TileLang's available scheduling
controls. **The PR result (15.31 ms vs 17.73, 1.16x) is the current optimum.**
