#!/usr/bin/env python3
"""Probe: can TileLang emit a tensor-core GEMM for these dtypes on sm_121?

Compiles a minimal tiled GEMM for fp16/bf16 and for float4_e2m1fnx2, prints
whether lowering/compile succeeds.
"""
from __future__ import annotations
import sys
import tilelang
import tilelang.language as T


def make(M, N, K, block_M, block_N, block_K, dtype, accum="float"):
    @tilelang.jit(out_idx=[2])
    def kernel(M, N, K, block_M, block_N, block_K):
        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
                 C: T.Tensor((M, N), accum)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C_l = T.alloc_fragment((block_M, block_N), accum)
                T.clear(C_l)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_l)
                T.copy(C_l, C[by * block_M, bx * block_N])
        return main
    return kernel(M, N, K, block_M, block_N, block_K)


def main():
    torch = __import__("torch")
    print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(0))
    try:
        tgt = tilelang.cuda.get_target() if hasattr(tilelang, "cuda") else None
    except Exception:
        tgt = None
    for dtype in ["float16", "bfloat16", "float8_e4m3fn", "float4_e2m1fnx2"]:
        try:
            k = make(1024, 1024, 1024, 128, 128, 64, dtype)
            # inspect generated source if available
            src = None
            for attr in ("get_kernel_source",):
                if hasattr(k, attr):
                    src = getattr(k, attr)()
            has_mma = bool(src and ("mma.sync" in src or "tcgen" in src or "wgmma" in src))
            print(f"  dtype={dtype:20s} COMPILE OK   tensor-core-asm={has_mma}")
        except Exception as ex:
            print(f"  dtype={dtype:20s} FAILED {type(ex).__name__}: {str(ex)[:200]}")


if __name__ == "__main__":
    main()
