#!/usr/bin/env python3
"""TileLang bf16 dense GEMM on GB10 — is the true silicon peak above cuBLAS ~102 TFLOPS?"""
import statistics
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def tl_gemm(M, N, K, BM=128, BN=128, BK=64, stages=4, threads=256):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.bfloat16), B: T.Tensor((N, K), T.bfloat16),
             C: T.Tensor((M, N), T.bfloat16)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bn, bm):
            A_s = T.alloc_shared((BM, BK), T.bfloat16)
            B_s = T.alloc_shared((BN, BK), T.bfloat16)
            C_l = T.alloc_fragment((BM, BN), T.float32)
            T.clear(C_l)
            for ko in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[bm * BM:(bm + 1) * BM, ko * BK:(ko + 1) * BK], A_s)
                T.copy(B[bn * BN:(bn + 1) * BN, ko * BK:(ko + 1) * BK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                C[bm * BM + i, bn * BN + j] = C_l[i, j]
    return main


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / iters)
    return statistics.median(ts)


if __name__ == "__main__":
    dev = "cuda"
    for (M, N, K) in [(4096, 4096, 4096), (8192, 4096, 4096), (4096, 12288, 4096)]:
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16) / 10
        b = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / 10
        c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        fl = 2.0 * M * N * K
        t0 = bench(lambda: a @ b.t())
        print(f"({M},{N},{K}) torch.mm: {t0:8.2f}us  {fl/t0/1e-6/1e12:6.1f} TFLOPS")
        for (BK, st, thr) in [(64, 2, 256), (32, 4, 256), (64, 3, 256), (128, 2, 256), (64, 4, 128)]:
            try:
                k = tl_gemm(M, N, K, BK=BK, stages=st, threads=thr)
                k(a, b, c)
                torch.cuda.synchronize()
                ref = (a.float() @ b.t().float())
                rel = (c.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                t = bench(lambda: k(a, b, c))
                print(f"  tl BK={BK} st={st} thr={thr}: {t:8.2f}us  {fl/t/1e-6/1e12:6.1f} TFLOPS  rel={rel:.4f}")
            except Exception as ex:
                print(f"  tl BK={BK} st={st} thr={thr}: FAIL {str(ex).splitlines()[0][:90]}")
        del a, b, c
        torch.cuda.empty_cache()
