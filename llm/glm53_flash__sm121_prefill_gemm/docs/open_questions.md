# Open questions / follow-up experiments

Threads left open across the GLM-5.3-Flash prefill investigation. None started.

## 1. Is cuBLAS actually at the GB10 bf16 silicon peak? (challenged)

First-round judgement "bf16 projections = NO-GO, cuBLAS at peak (~100–102 TFLOPS)"
was **too strong**. NCU on the 8192³ bf16 cuBLAS GEMM shows the **Tensor (FP)
pipe at 85.1%**, NCU itself calling it *"over-utilized and likely a performance
bottleneck"*. That implies the silicon bf16 dense peak is **~120 TFLOPS**
(`102 / 0.85`), so cuBLAS leaves **~15%** on the table, and the production M=4096
projections (91–99 TFLOPS) sit at ~76–83% of the true peak → up to **~1.2x**
theoretical headroom, not 0 and not 2x.

This is a real (small) opening, not yet disproven. Decide with:

1. **Raw MMA microbenchmark** — a kernel that issues back-to-back
   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` from shared/registers
   with **no global traffic**, to measure the pure tensor-pipe FLOPs. This gives
   the true per-SM ceiling independent of cuBLAS.
2. **Tuned native bf16 square GEMM** — CUTLASS SM80 collective or TileLang, tile
   sweep (128x128/128x256/256x128), stages 3–4, CUDA-graph timing, compared to
   cuBLAS at 8192³/16384³ and at the production M=4096 shapes.

Success bar: beat 102 TFLOPS on 16384³ (or 99 TFLOPS on the production wide
shapes) by >5–10%. If the raw MMA ceiling is ~120 and both cuBLAS and the native
kernel top out ~102, the "no-go (small headroom, hard)" verdict stands.
Caveat: for 8192³/16384³, DRAM is only ~58% utilized and the stall is "waiting on
the tensor pipe", so **memory/overlap cannot explain a >1.2x gap** — any win must
come from tensor-pipe utilization (tile shape, MMA atom, pipeline depth,
occupancy).

## 2. chunked-prefill config A/B (cheap, unresolved)

MoE utilization rises with tokens/call: 93 → 117 → 127 TFLOPS (baseline) and
136 → 179 → 206 (TileLang w13) at M=4096 → 8192 → 16384. Test
`--chunked-prefill-size 4096` vs 8192 vs 16384 end-to-end on the live 2-node
service (needs restart ~15 min). Bounded by activation memory.

## 3. Non-GEMM prefill headroom (never scoped)

MHC 13%, elementwise/quant/rope/copy 12%, NCCL 8%, DSA sparse attention 7%,
KDA 6%. Small-kernel trace times are profiler artifacts (see
`glm53_flash__sm121_dsa_longctx_decode/docs/incident_long_prefill_profile.md`),
so any of these needs a standalone re-measure before it counts.
