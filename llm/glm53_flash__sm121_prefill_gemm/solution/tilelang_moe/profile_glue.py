#!/usr/bin/env python3
"""Per-step profile of the full_moe pipeline glue (M=4096)."""
import math
import torch
import torch.nn.functional as F
import full_moe as FM
from full_moe import pack_acts_whole, build_weights, global_scale
from grouped_nvfp4_gemm import grouped_nvfp4_gemm
from scatter_kernel import moe_scatter
from flashinfer import fp4_quantize

dev = "cuda"
M, E, K, N, topk, BLOCK_K = 4096, 144, 4096, 2048, 8, 256
rows = M * topk
words = BLOCK_K // 64
g = torch.Generator(device=dev).manual_seed(0)
X = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
w2 = torch.randn(E, K, N, device=dev, dtype=torch.bfloat16, generator=g) / 10
score = torch.randn(M, E, device=dev, dtype=torch.bfloat16, generator=g)
tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
tw = torch.softmax(tw.float(), dim=-1)
flat = ti.reshape(-1).to(torch.int64)
order = torch.argsort(flat, stable=True)
tok = (order // topk)
counts = torch.bincount(flat, minlength=E)
sizes = counts.cpu().tolist()
cap = math.ceil(max(sizes) / 128) * 128
bpe = cap // 128
offsets = torch.zeros(E + 1, device=dev, dtype=torch.int32)
offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
W = build_weights(w1, w2, BLOCK_K, words)
w13 = grouped_nvfp4_gemm(E, 2 * N, K, rows, bpe, block_K=BLOCK_K)
w2k = grouped_nvfp4_gemm(E, K, N, rows, bpe, block_K=BLOCK_K)
scat = moe_scatter(rows, K, M)
o1 = torch.empty(rows, 2 * N, device=dev, dtype=torch.bfloat16)
o2 = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
y = torch.zeros(M, K, device=dev, dtype=torch.float32)
A1, SFA1, gs1, sf1 = pack_acts_whole(X[tok], sizes, cap, K, BLOCK_K, words, dev)
sc1 = torch.cat([(1.0 / (gs1 * W["w1_gs"][e])).repeat(sizes[e]) for e in range(E)])
o1.copy_(torch.empty_like(o1)); w13(A1, W["w1_q"], SFA1, W["SFB1"], offsets, o1)


def ev(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / n


import time
def wall(fn, n=10):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def step(name, fn, n=20):
    print(f"  {name:36s} {ev(fn, n):8.3f} ms")


print("glue step profile (M=4096)")
step("gather X[tok]", lambda: X[tok])
step("global_scale(bf16 amax)", lambda: global_scale(X[tok]))
step("fp4_quantize(whole)", lambda: fp4_quantize(X[tok], global_scale(X[tok]), sf_vec_size=16, is_sf_swizzled_layout=False))
step("pack_acts_whole", lambda: pack_acts_whole(X[tok], sizes, cap, K, BLOCK_K, words, dev))
step("cat sc1 (144 iters)", lambda: torch.cat([(1.0 / (gs1 * W["w1_gs"][e])).repeat(sizes[e]) for e in range(E)]))
step("o1.float()*sc1", lambda: o1.float().mul_(sc1.unsqueeze(1)))
f1 = o1.float().mul_(sc1.unsqueeze(1))
step("swiglu", lambda: (F.silu(f1[:, :N].clamp(max=10.0)) * f1[:, N:].clamp(-10.0, 10.0)).to(torch.bfloat16))
act = (F.silu(f1[:, :N].clamp(max=10.0)) * f1[:, N:].clamp(-10.0, 10.0)).to(torch.bfloat16)
step("pack_acts_whole(act)", lambda: pack_acts_whole(act, sizes, cap, N, BLOCK_K, words, dev))
rowscale = torch.rand(rows, device=dev)
step("fused scatter", lambda: (y.zero_(), scat(o2, rowscale, tok.to(torch.int32), y)))
