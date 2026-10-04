#!/usr/bin/env python3
"""Arbitrate Part C: compare both paths against an UNQUANTIZED fp32 oracle."""
import math, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
sys.path.insert(0, "/tmp")
from verify_emaj import run_pipeline, stats, DEV, E4M3_MAX, E2M1_MAX

LIM = 10.0
M, E, K, N, topk = 4096, 144, 4096, 2048, 8
g = torch.Generator(device=DEV).manual_seed(7)
X = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
w1 = torch.randn(E, 2 * N, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
w2 = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16, generator=g) / 10
score = torch.randn(M, E, device=DEV, dtype=torch.bfloat16, generator=g)
tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
tw = torch.softmax(tw.float(), dim=-1)

# fp32 oracle from ORIGINAL bf16 weights (no quantization anywhere)
y_ref = torch.zeros(M, K, device=DEV, dtype=torch.float32)
for e in range(E):
    tok, slot = (ti == e).nonzero(as_tuple=True)
    if tok.numel() == 0:
        continue
    a = X[tok].float()
    g_ = (a @ w1[e, :N].t().float()).clamp(max=LIM)
    u_ = (a @ w1[e, N:].t().float()).clamp(-LIM, LIM)
    act = F.silu(g_) * u_
    o = act.to(torch.bfloat16).float() @ w2[e].t().float()
    y_ref.index_add_(0, tok, o * tw[tok, slot].unsqueeze(-1))
torch.cuda.synchronize()

# path 1: flashinfer baseline (same call as verify_emaj.part_c)
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
def round_up(x, y): return (x + y - 1) // y * y
def qfi(w):
    E_, R, C = w.shape
    q = torch.empty((E_, R, C // 2), dtype=torch.uint8, device=DEV)
    bs = torch.empty((E_, round_up(R, 128), round_up(C // 16, 4)), dtype=torch.float8_e4m3fn, device=DEV)
    gs = torch.empty(E_, dtype=torch.float32, device=DEV)
    for e in range(E_):
        gv = (E4M3_MAX * E2M1_MAX) / w[e].float().abs().max().clamp_min(1e-8)
        q[e], bs[e] = fp4_quantize(w[e], gv.reshape(1))
        gs[e] = gv
    return q, bs, gs
w1_q, w1_bs, w1_gs = qfi(w1)
w2_q, w2_bs, w2_gs = qfi(w2)
a_gs = torch.ones(E, device=DEV, dtype=torch.float32)
y_fi = cutlass_fused_moe(X, ti.to(torch.int), tw, w1_q.view(torch.long), w2_q.view(torch.long),
                         X.dtype, quant_scales=[a_gs, w1_bs.view(torch.int32), (1 / w1_gs),
                                                a_gs, w2_bs.view(torch.int32), (1 / w2_gs)])[0]
torch.cuda.synchronize()

# path 2: emaj pipeline
y_e = run_pipeline(M, E, K, N, topk, True, X, w1, w2, score)

st_ref_fi = stats(y_fi, y_ref)
st_ref_e = stats(y_e, y_ref)
st_e_fi = stats(y_e, y_fi)
print(f"flashinfer vs fp32-oracle : max_rel={st_ref_fi['max_rel']:.5f} mean_rel={st_ref_fi['mean_rel']:.6f} ref_absmax={st_ref_fi['ref_absmax']}")
print(f"emaj-pipe  vs fp32-oracle : max_rel={st_ref_e['max_rel']:.5f} mean_rel={st_ref_e['mean_rel']:.6f}")
print(f"emaj-pipe  vs flashinfer  : max_rel={st_e_fi['max_rel']:.5f} mean_rel={st_e_fi['mean_rel']:.6f}")
# scale-factor diagnostic: median of y_fi/y_ref where |y_ref| significant
mask = y_ref.abs() > 0.05 * y_ref.abs().max()
ratio = (y_fi[mask] / y_ref[mask].clamp_min(1e-9)).median().item()
print(f"median(y_fi/y_ref) on significant elems = {ratio:.4f}  (≈1 → same convention; ≈gs² or 1/gs² → convention bug)")
