#!/usr/bin/env python3
"""Fused dequant-scale + SwiGLU(limit): out1[rows,2N] bf16 + sc[rows] -> act[rows,N] bf16.

Replaces the torch chain (`o1.float()*sc` then clamp/silu/mul) which costs
~20 ms at rows=32768, N=2048 due to ~10 fp32 passes over 512 MB.
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def swiglu_limit(rows, N, lim, block_R=64, block_C=256, threads=256):
    @T.prim_func
    def main(
        o1: T.Tensor((rows, 2 * N), "bfloat16"),
        sc: T.Tensor((rows,), "float32"),
        act: T.Tensor((rows, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, block_C), T.ceildiv(rows, block_R), threads=threads) as (bc, br):
            for i, j in T.Parallel(block_R, block_C):
                r = br * block_R + i
                c = bc * block_C + j
                if r < rows:
                    s = sc[r]
                    g = T.Cast("float32", o1[r, c]) * s
                    u = T.Cast("float32", o1[r, N + c]) * s
                    g = T.min(g, lim)
                    u = T.max(T.min(u, lim), -lim)
                    act[r, c] = T.Cast("bfloat16", g / (1.0 + T.exp(-g)) * u)
    return main


def main():
    dev = "cuda"
    rows, N, lim = 32768, 2048, 10.0
    g = torch.Generator(device=dev).manual_seed(0)
    o1 = torch.randn(rows, 2 * N, device=dev, dtype=torch.bfloat16, generator=g) * 3
    sc = torch.rand(rows, device=dev, dtype=torch.float32) + 0.5
    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    kern = swiglu_limit(rows, N, lim)
    kern(o1, sc, act)
    torch.cuda.synchronize()

    f1 = o1.float() * sc.unsqueeze(1)
    gg = f1[:, :N].clamp(max=lim)
    uu = f1[:, N:].clamp(-lim, lim)
    ref = (F.silu(gg) * uu).to(torch.bfloat16)
    err = (act.float() - ref.float()).abs().max().item()
    print(f"swiglu verify: max_abs={err:.5f}")
    assert err < 1e-3

    t_k = do_bench(lambda: kern(o1, sc, act))
    def torch_impl():
        f = o1.float() * sc.unsqueeze(1)
        return (F.silu(f[:, :N].clamp(max=lim)) * f[:, N:].clamp(-lim, lim)).to(torch.bfloat16)
    t_t = do_bench(torch_impl)
    print(f"TileLang swiglu={t_k:.3f} ms   torch={t_t:.3f} ms   speedup={t_t/t_k:.2f}x")


if __name__ == "__main__":
    main()
