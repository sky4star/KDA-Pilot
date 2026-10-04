#!/usr/bin/env python3
"""Manual warp-specialized grouped w13 (single B, no SwiGLU) — H100-style vehicle.

Producer role (T.ws(1)): TMA-load A/B + SIMT-load scale words into multi-stage
shared, then arrive.
Consumer role (T.ws(0)): wait, blockscaled MMA, arrive back.

Based on tilelang examples/warp_specialize/example_warp_specialize_gemm_barrierpipe_stage2.py
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

try:
    from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words
except ImportError:
    from grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words


@tilelang.jit
def ws_w13(E: int, N: int, K: int, rows: int, bpe: int,
           block_K: int = 128, num_stages: int = 2, threads: int = 256):
    assert N % 128 == 0 and K % 256 == 0 and block_K % 64 == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // 128
    BM = 128
    BN = 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        B: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        offsets: T.Tensor((E + 1,), T.int32),
        C: T.Tensor((rows, N), T.bfloat16),
    ):
        with T.Kernel(n_blocks, E * bpe, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((num_stages, BM, block_K), in_dtype)
            B_s = T.alloc_shared((num_stages, BN, block_K), in_dtype)
            SFA_s = T.alloc_shared((num_stages, BM, words), T.uint32)
            SFB_s = T.alloc_shared((num_stages, BN, words), T.uint32)
            C_l = T.alloc_fragment((BM, BN), T.float32)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            e = bm // bpe
            local = bm % bpe
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))

            with T.ws(0):
                T.clear(C_l)

            for ko in range(k_blocks):
                with T.ws(1):
                    T.mbarrier_wait_parity(mbarrier=mbars[ko % num_stages + num_stages],
                                           parity=((ko // num_stages) % 2) ^ 1)
                    T.tma_copy(A[m_start:m_start + BM, ko * block_K:(ko + 1) * block_K],
                               A_s[ko % num_stages, :, :], barrier=mbars[ko % num_stages])
                    T.tma_copy(B[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K],
                               B_s[ko % num_stages, :, :], barrier=mbars[ko % num_stages])
                    T.copy(SFA[((e * bpe + local) * k_blocks + ko) * BM:((e * bpe + local) * k_blocks + ko) * BM + BM, :],
                           SFA_s[ko % num_stages, :, :])
                    T.copy(SFB[((e * n_blocks + bn) * k_blocks + ko) * BN:((e * n_blocks + bn) * k_blocks + ko) * BN + BN, :],
                           SFB_s[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(mbarrier=mbars[ko % num_stages],
                                           parity=(ko // num_stages) % 2)
                    T.mma_gemm_blockscaled(A_s[ko % num_stages, :, :], B_s[ko % num_stages, :, :], C_l,
                                           SFA_s[ko % num_stages, :, :], SFB_s[ko % num_stages, :, :],
                                           transpose_B=True, clear_accum=False, k_start=ko * block_K,
                                           sf_a_granularity_k=16, sf_b_granularity_k=16,
                                           sf_layout="blockscaled_chunk_kmajor")
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])

            with T.ws(0):
                for i, j in T.Parallel(BM, BN):
                    if i < actual:
                        C[m_start + i, bn * BN + j] = T.Cast(T.bfloat16, C_l[i, j])

    return main


def build_test(E, N, K, rows, bpe, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cap = bpe * 128
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64, device=dev); sizes[: rows - base * E] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)
    A = torch.randint(-128, 128, (rows, K // 2), device=dev, dtype=torch.int8, generator=g)
    A_sf_unpad = torch.randint(56, 64, (rows, K // 16), device=dev, dtype=torch.uint8, generator=g)
    sfa_sem = torch.zeros((E * cap, K // 16), device=dev, dtype=torch.uint8)
    for e in range(E):
        s0, s1 = int(offsets[e]), int(offsets[e + 1])
        sfa_sem[e * cap:e * cap + (s1 - s0)] = A_sf_unpad[s0:s1]
    words = 128 // 64
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, words)
    B = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)
    sfb_sem = torch.randint(56, 64, (E * N, K // 16), device=dev, dtype=torch.uint8, generator=g)
    SFB = swizzle_chunk_kmajor(pack_scale_words(sfb_sem)).reshape(-1, words)
    return dict(A=A, B=B, SFA=SFA, SFB=SFB, offsets=offsets, sfa_sem=sfa_sem, sfb_sem=sfb_sem, A_sf_unpad=A_sf_unpad)


def main():
    E, N, K, rows, bpe = 8, 256, 512, 1024, 2
    dev = "cuda"
    d = build_test(E, N, K, rows, bpe)
    C = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    k = ws_w13(E, N, K, rows, bpe, block_K=128)
    k(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], C)
    torch.cuda.synchronize()
    Ad = decode_fp4(d["A"], rows, K) * decode_scale_words(pack_scale_words(d["A_sf_unpad"]), K).repeat_interleave(16, 1)
    Bd = decode_fp4(d["B"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfb_sem"]), K).reshape(E, N, K // 16).repeat_interleave(16, 2)
    ref = torch.empty(rows, N, device=dev, dtype=torch.float32)
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 > s0:
            ref[s0:s1] = Ad[s0:s1] @ Bd[e].t()
    err = (C.float() - ref).abs().max().item()
    print(f"ws_w13 verify: max_abs={err:.5f} ref_absmax={ref.abs().max().item():.3f} max_rel={err/(ref.abs().max().item()+1e-6):.5f}")
    ms = do_bench(lambda: k(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], C))
    print(f"ws_w13: {ms:.4f} ms")


if __name__ == "__main__":
    main()
