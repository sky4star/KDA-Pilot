#!/usr/bin/env python3
"""Fused routed-MoE scatter: y[tok[r], :] += scale[r] * out2[r, :].

Replaces torch `index_add_` (measured ~20 ms at rows=32768, K=4096) with a single
TileLang kernel using global atomicAdd.
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def moe_scatter(rows, K, M, block_R=64, block_K=256, threads=256):
    @T.prim_func
    def main(
        out2: T.Tensor((rows, K), "bfloat16"),
        scale: T.Tensor((rows,), "float32"),
        tok: T.Tensor((rows,), "int32"),
        y: T.Tensor((M, K), "float32"),
    ):
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(rows, block_R), threads=threads) as (bk, br):
            for i, j in T.Parallel(block_R, block_K):
                r = br * block_R + i
                c = bk * block_K + j
                if r < rows:
                    T.atomic_add(y[tok[r], c], T.Cast("float32", out2[r, c]) * scale[r])
    return main


def main():
    dev = "cuda"
    rows, K, M = 32768, 4096, 4096
    g = torch.Generator(device=dev).manual_seed(0)
    out2 = torch.randn(rows, K, device=dev, dtype=torch.bfloat16, generator=g)
    scale = torch.rand(rows, device=dev, dtype=torch.float32)
    tok = torch.randint(0, M, (rows,), device=dev, dtype=torch.int32)

    kern = moe_scatter(rows, K, M)
    y = torch.zeros(M, K, device=dev, dtype=torch.float32)
    kern(out2, scale, tok, y)
    torch.cuda.synchronize()

    y_ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
    y_ref.index_add_(0, tok, out2.float() * scale.unsqueeze(1))
    err = (y - y_ref).abs()
    rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"scatter verify: max_abs={err.max().item():.5f} max_rel={rel:.6f}")
    assert rel < 1e-4

    def run():
        y.zero_()
        kern(out2, scale, tok, y)

    t_k = do_bench(run)
    t_t = do_bench(lambda: y_ref.zero_().index_add_(0, tok, out2.float() * scale.unsqueeze(1)))
    print(f"TileLang scatter={t_k:.3f} ms   torch index_add={t_t:.3f} ms   speedup={t_t/t_k:.2f}x")


if __name__ == "__main__":
    main()
