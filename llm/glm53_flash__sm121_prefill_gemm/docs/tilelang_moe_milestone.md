# TileLang grouped NVFP4 MoE GEMM — milestone

Status (2026-10-02): **working grouped NVFP4 block-scaled GEMM, correct, matches
the FlashInfer baseline at M=4096.** Code: `solution/tilelang_moe/grouped_nvfp4_gemm.py`.
Runs with tilelang 0.1.15 (`PYTHONPATH=/tmp/tl15site`) + the SM121A guard patch.

## Design

- SM120a `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale...ue4m3` via
  `T.mma_gemm_blockscaled(sf_layout="blockscaled_chunk_kmajor")`, tile
  128x128x256, 2 stages, warp-specialized producer/consumer (TMA for A/B,
  `cp.async` for packed uint32 scale words).
- Non-persistent grid `(n_blocks, total_m_blocks)`, `total_m_blocks = E *
  blocks_per_expert`. Expert = `bm // blocks_per_expert`; **no routing loop, no
  recompilation on routing changes** (rows handled by `expert_offsets`).
- Activation gathered per expert into `A[cum_m, K]` (expert-contiguous);
  weight `B[E, N, K]`; scale factors in `blockscaled_chunk_kmajor` uint32 words.

### The one optimization that mattered: grid order

`T.Kernel(n_blocks, total_m_blocks)` — **N varies fastest**. The first version
used `(total_m_blocks, n_blocks)` and re-streamed the 128 MB activation 32x
(L2 hit 53%). Swapping to N-fastest gives the activation-tile L2 reuse the dense
kernel enjoys:

| kernel | first version | n-fastest |
|---|---:|---:|
| E=1, N=4096, K=4096, rows=32768 | 12.28 ms | **3.57 ms** |
| E=144 (w13), same | 18.35 ms | **8.03 ms** |

## Results (GB10, tilelang 0.1.15, cold by size)

| GEMM | shape (cum_m, N, K) | baseline (grouped) | TileLang | TFLOPS | correctness |
|---|---|---:|---:|---:|---|
| w13 | 32768, 4096, 4096 | ~8.0–9.25 ms | **8.03 ms** | 137 | max_rel 0.26% |
| w2 | 32768, 4096, 2048 | ~4.6–5.8 ms | **4.80 ms** | 115 | max_rel 0.33% |
| total | | 13.88 ms | **12.83 ms** | | **1.08x** |

Correctness = vs an independent fp32 dequant+block-scaled-matmul oracle
(`reference()`), not the baseline (activation-quant parity with the fused path
is a separate step).

Batch scaling (w13): M=4096 136 TFLOPS, M=8192 179, M=16384 **206** — the
grouped kernel improves sharply with per-expert M, so larger prefill chunks help
the MoE more than the baseline does.

## Bound (NCU, w13, M=4096)

| metric | value |
|---|---|
| Compute (SM) / Memory | 36.4% / 37.6% |
| Grid / waves | 9216 / 192 |
| Achieved occupancy | 15.5% |
| Block Limit Registers / Shared | **1 / 1** (197 regs, 73.7 KB smem) |
| L2 hit rate | 74.6% |
| Eligible warps/scheduler | 0.08 |

Latency-bound. `block_K` (128/256), `stages` (2/3), `threads` (128/256) sweeps
all land at ~8.0–8.7 ms. The TileLang SM120 block-scaled path is fixed at
`block_M == block_N == 128`, which prevents a single CTA from covering an
expert's ~227 rows with one weight stream (the reason the dense roof — 5.6 ms
for w13+w2 — is not reachable for the grouped problem at M=4096).

## Still missing for a drop-in MoE replacement

1. Epilogue fusion or separate kernels for **SwiGLU(limit=10) + fp4 requant**
   of the w13 output into the w2 A operand (+ its scale factors).
2. **w2** call (kernel is done/verified) wired on the activated input.
3. **Scatter + routing-weight combine**.
4. **Parity** vs `cutlass_fused_moe` incl. the pointwise activation-quant
   convention (the historical blocker).

The two grouped GEMMs already match the baseline GEMM cost; the remaining win is
mostly removing the baseline's ~3.8 ms `expand/activation/finalize` aux
overhead via fusion.

## Full MoE pipeline (added)

`solution/tilelang_moe/full_moe.py` implements the **complete routed MoE**:
gather → w13 grouped GEMM → SwiGLU(limit=10) → dynamic NVFP4 requant →
w2 grouped GEMM → scatter + routing weights, validated end-to-end against a
torch fp32 reference built from the same dequantized operands.

| M | rows | bpe | both GEMMs | verify |
|---:|---:|---:|---:|---:|
| 256 | 2048 | 1 | 8.98 ms (weight-bound) | max_rel 0.27% |
| 4096 | 32768 | 3 | **13.4 ms** (123 TFLOPS) | max_rel 0.26% |

Baseline `cutlass_fused_moe`: GEMMs 13.9 ms, **full 17.73 ms** (aux ~3.8 ms).

- The two TileLang grouped GEMMs = 13.4 ms → already ~equal to the baseline GEMM
  portion (1.04x).
- **The prototype torch glue is not viable** (full forward 87 ms): it quantizes
  activations with 288 separate `flashinfer.fp4_quantize` calls (144 experts x 2).
  A real implementation needs a **single fused activation-requant kernel** (and
  ideally fold SwiGLU + requant into the w13 epilogue and scatter into w2).
- `bpe=3` here (max expert size 261 > 256) adds one mostly-empty m-block; keeping
  the cap at 256 or handling the tail better trims w13.

### Honest status

- **Achieved:** a correct, roofline-appropriate TileLang grouped NVFP4 MoE GEMM
  matching the baseline; a correct full-MoE pipeline; tilelang 0.1.15 viability
  on GB10 proven with a 1-line patch.
- **Not yet a serving win:** the GEMMs only tie the baseline, and the glue needs
  fusion. To beat 17.73 ms the work is: fused SwiGLU+requant epilogue, fused
  gather/quant of activations, fused scatter, then parity vs the baseline's
  activation-quant convention. Expected ceiling ~14 ms (≈1.25x) if the aux is
  fully fused, bounded by the grouped weight-traffic floor.

### Glue profile (M=4096, after switching to whole-tensor quant)

`full_moe.py` uses one `fp4_quantize` per activation tensor (not 288) and still
measures **~69 ms total** (GEMMs 13.3 ms → glue ~55 ms). Per-piece:

| glue step | ms | note |
|---|---:|---|
| `X[tok]` gather | 2.0 | |
| `global_scale` (`x.abs().max()`) | 10.7 → ~1 after bf16-amax fix | was fp32 cast of 512 MB |
| `fp4_quantize` whole tensor | 1.6 | x2 |
| `pack_scale_words` + `swizzle_chunk_kmajor` | 4.6 | x2, int64 + permute on host |
| `o2.float()*scale` + `index_add` scatter | 20.0 → **5.6 fused** | `scatter_kernel.py` |
| `o1.float()*scale` + SwiGLU torch | 20.5 → **1.85 fused** | `swiglu_kernel.py` |

`scatter_kernel.py` (TileLang, global atomicAdd) replaces torch `index_add_`:
**5.62 ms vs 14.94 ms (2.66x), bit-exact**. `swiglu_kernel.py` (fused
dequant-scale + clamp + silu + mul) replaces the torch chain:
**1.85 ms vs 20.5 ms (11.05x), bit-exact**. Plus `pack_scale_words` int32 and
`repeat_interleave` scale vectors.

Full-forward progression: 87 → 68.7 → 59.2 → 41.7 → **35.1 ms** (GEMMs 13.35 ms
fixed), still correct (max_rel 0.27%).

Every one of these is a trivially-parallel memory op that the baseline does in
fused C++ (`expandInputRows`, `doActivation`, `finalizeMoeRouting`) for ~3.8 ms.
**Torch glue cannot close the gap**; the path to a win is kernel fusion:

1. **w13 epilogue:** compute gate+up in-CTA (two blockscaled MMAs against
   `w_gate`/`w_up` tiles), apply `silu(clamp(gate,limit))*clamp(up,±limit)`, and
   emit `act` directly in NVFP4 + `blockscaled_chunk_kmajor` words. Open issue:
   TileLang's SM120 block-scaled path is fixed at `block_M == block_N == 128`, so
   the two accumulators cost ~128 extra regs/thread.
2. **Gather+quant:** one kernel to write gathered `A1` fp4 + padded/swizzled SFA
   from `X` and `topk_ids` (kills gather + global_scale + pack/swizzle).
3. **Fused scatter:** one kernel to do `y[tok] += w * out2` (kills the 20 ms
   `index_add`).
4. **Parity** against the fused activation-quant convention.

With 1–3 the pipeline projects to ≈ GEMMs (13.3 ms) + ~1 ms fused aux ≈ **14 ms**,
i.e. ~1.25x over the 17.73 ms baseline; the grouped weight-traffic floor keeps it
from the dense 5.6 ms.

### Update after fusion + a floor analysis (negative result)

Fusing SwiGLU (11x) and the scatter (2.66x) plus host-scale fixes brought the
full forward to 35.1 ms, but a **hard floor analysis** shows the baseline is not
beatable at M=4096 on GB10 with this approach:

- GEMMs: **13.35 ms** (w13 8.3 + w2 5.0), already ≈ the baseline GEMM 13.9 ms.
  The grouped kernel streams expert weights `bpe` times (bpe=3 here because
  max expert size is 261). Sweeps of `block_K/stages/threads` and a
  "one-CTA-per-expert reuses B" variant (`grouped_nvfp4_gemm_reuse`, grid
  `(n_blocks, E)` + inner `lm` loop) did **not** help — the reuse variant is
  slower (10.6 vs 8.5 ms) because the serial `lm` loop removes parallelism.
- Even a **perfectly fused** aux needs ≥ gather 1.4 + swiglu/quant 1.4 + scatter
  3.4 ≈ **6.2 ms** of unavoidable traffic, giving a total floor ≈ **19.5 ms**,
  above the 17.73 ms baseline.

So at chunk=4096 the routed NVFP4 MoE is not where the prefill win is. The
TileLang kernel does scale much better with batch, though: w13 reaches
**206 TFLOPS at M=16384 vs the baseline's ~127**, so the lever is a **larger
chunk / more tokens per MoE call**, not a faster single-call kernel.

### Delivered artifacts

- `grouped_nvfp4_gemm.py` — correct grouped NVFP4 blockscaled GEMM (+ a
  slower `_reuse` variant, kept for the record).
- `full_moe.py` — complete MoE, oracle-validated.
- `swiglu_kernel.py` — fused scale+SwiGLU(limit), 11x over torch.
- `scatter_kernel.py` — fused routed scatter, 2.66x over torch.
- `profile_glue.py` — per-step glue profile.

### Full-MoE vs baseline across M (the batch lever does not save it either)

| M | TileLang GEMMs | TileLang GEMM TFLOPS | baseline full MoE | TileLang full (torch glue) |
|---:|---:|---:|---:|---:|
| 4096 | 13.35 ms | 124 | 17.7 ms | 35.1 ms |
| 8192 | 19.43 ms | 170 | 28.2 ms | 61.6 ms |
| 16384 | 33.41 ms | 197 | 51.8 ms | 116.9 ms |

The TileLang **GEMMs beat the baseline's full MoE** at M=8192/16384 (19.4 < 28.2,
33.4 < 51.8), but the **torch glue scales linearly with M** (42 ms at 8192, 83 ms
at 16384) and swamps it. A fully-fused glue floor (~15–20 ms at M=16384) brings
the total to ≈ the baseline, i.e. **break-even, not a win**.

### Bottom line

- The grouped NVFP4 MoE GEMM is **matchable** in TileLang (≈ baseline) and scales
  well with batch (197 TFLOPS at M=16384), but it is **weight/latency bound** and
  far from the dense 313 TFLOPS; the earlier "2.5x headroom" was a dense-vs-grouped
  artifact.
- The activation-side glue must be as fused as FlashInfer's C++ (3.8 ms at M=4096)
  to be competitive, and even then the projected total is ≈ break-even.
- **Recommendation:** do not promote a TileLang MoE replacement on this evidence.
  If pursued, the only credible path is a single fully-fused kernel (gather+quant
  + w13 + SwiGLU+requant + w2 + scatter) and parity, and the expected upside is
  small. The prefill win is more likely in the **chunked-prefill config** (larger
  chunk helps the MoE utilization on both kernels) and in the non-GEMM operators.


