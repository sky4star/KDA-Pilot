#!/usr/bin/env python3
"""K1 (packed): grouped w13 gate+up dual block-scaled MMA + fused SwiGLU.

Mega-MoE-style scheduling: the host builds a **packed list of actual blocks**
(``Σ_e ceil(m_e / 128)``), so there are no empty blocks (the previous fixed
``bpe = ceil(max_m_e/128)`` wasted ~1/3 of the work at M=4096).  The grid keeps a
compile-time ``MAX_BLOCKS`` upper bound and CTAs with ``bm >= total_blocks`` exit
immediately (block-uniform branch, no warp divergence).
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

try:
    from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words
except ImportError:
    from grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w13_swiglu_packed(
    E: int,
    N: int,
    K: int,
    rows: int,
    MAX_BLOCKS: int,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    lim: float = 10.0,
    out_dtype=T.bfloat16,
):
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
        Bg: T.Tensor((E, N, K), in_dtype),
        Bu: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((MAX_BLOCKS * BM * k_blocks, words), T.uint32),
        SFBg: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        SFBu: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        block_expert: T.Tensor((MAX_BLOCKS,), T.int32),
        block_mstart: T.Tensor((MAX_BLOCKS,), T.int32),
        block_actual: T.Tensor((MAX_BLOCKS,), T.int32),
        total_blocks: T.Tensor((1,), T.int32),
        scale: T.Tensor((E,), T.float32),
        act: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(n_blocks, MAX_BLOCKS, threads=threads) as (bn, bm):
            if bm < total_blocks[0]:
                A_s = T.alloc_shared((BM, block_K), in_dtype)
                Bg_s = T.alloc_shared((BN, block_K), in_dtype)
                Bu_s = T.alloc_shared((BN, block_K), in_dtype)
                SFA_s = T.alloc_shared((BM, words), T.uint32)
                SFBg_s = T.alloc_shared((BN, words), T.uint32)
                SFBu_s = T.alloc_shared((BN, words), T.uint32)
                Cg = T.alloc_fragment((BM, BN), T.float32)
                Cu = T.alloc_fragment((BM, BN), T.float32)

                e = block_expert[bm]
                m_start = block_mstart[bm]
                actual = block_actual[bm]
                sc = scale[e]

                T.clear(Cg)
                T.clear(Cu)
                for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                    T.copy(A[m_start:m_start + BM, ko * block_K:(ko + 1) * block_K], A_s)
                    T.copy(Bg[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bg_s)
                    T.copy(Bu[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bu_s)
                    for r, w in T.Parallel(BM, words):
                        SFA_s[r, w] = SFA[(bm * k_blocks + ko) * BM + r, w]
                    for r, w in T.Parallel(BN, words):
                        SFBg_s[r, w] = SFBg[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                    for r, w in T.Parallel(BN, words):
                        SFBu_s[r, w] = SFBu[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                    T.mma_gemm_blockscaled(A_s, Bg_s, Cg, SFA_s, SFBg_s, transpose_B=True,
                                           clear_accum=False, k_start=ko * block_K,
                                           sf_a_granularity_k=16, sf_b_granularity_k=16,
                                           sf_layout="blockscaled_chunk_kmajor")
                    T.mma_gemm_blockscaled(A_s, Bu_s, Cu, SFA_s, SFBu_s, transpose_B=True,
                                           clear_accum=False, k_start=ko * block_K,
                                           sf_a_granularity_k=16, sf_b_granularity_k=16,
                                           sf_layout="blockscaled_chunk_kmajor")

                for i, j in T.Parallel(BM, BN):
                    if i < actual:
                        g = Cg[i, j] * sc
                        u = Cu[i, j] * sc
                        g = T.min(g, lim)
                        u = T.max(T.min(u, lim), -lim)
                        act[m_start + i, bn * BN + j] = T.Cast(out_dtype, g / (1.0 + T.exp(-g)) * u)

    return main


def build_packed(E, N, K, rows, bpe_cap, seed=0):
    """Packed per-expert blocks (no empty blocks). bpe_cap is the compile-time max."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64, device=dev); sizes[: rows - base * E] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)

    nb = ((sizes + 127) // 128)
    MAX_BLOCKS = int(nb.max().item()) * E  # compile-time upper bound
    total = int(nb.sum().item())
    block_expert = torch.repeat_interleave(torch.arange(E, device=dev), nb).to(torch.int32)
    local = torch.cat([torch.arange(int(n), device=dev) for n in nb])
    block_mstart = offsets[block_expert.long()] + (local * 128).to(torch.int32)
    block_actual = torch.clamp(offsets[block_expert.long() + 1] - block_mstart, 0, 128).to(torch.int32)
    # pad block arrays to MAX_BLOCKS
    def pad(t):
        out = torch.zeros(MAX_BLOCKS, dtype=t.dtype, device=dev); out[:total] = t; return out
    block_expert = pad(block_expert); block_mstart = pad(block_mstart); block_actual = pad(block_actual)

    # packed SFA: per-expert scales padded to ceil(m_e/128)*128, concatenated by block
    sfa_sem = torch.zeros((MAX_BLOCKS * 128, K // 16), device=dev, dtype=torch.uint8)
    A_sf_unpad = torch.randint(56, 64, (rows, K // 16), device=dev, dtype=torch.uint8, generator=g)
    p = 0
    for e in range(E):
        n = int(nb[e]); m = int(sizes[e])
        s0 = int(offsets[e])
        sfa_sem[p * 128: p * 128 + m] = A_sf_unpad[s0:s0 + m]
        p += n
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, 128 // 64)

    A = torch.randint(-128, 128, (rows, K // 2), device=dev, dtype=torch.int8, generator=g)
    Bg = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)
    Bu = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)
    words = 128 // 64
    sfbg_sem = torch.randint(56, 64, (E * N, K // 16), device=dev, dtype=torch.uint8, generator=g)
    sfbu_sem = torch.randint(56, 64, (E * N, K // 16), device=dev, dtype=torch.uint8, generator=g)
    SFBg = swizzle_chunk_kmajor(pack_scale_words(sfbg_sem)).reshape(-1, words)
    SFBu = swizzle_chunk_kmajor(pack_scale_words(sfbu_sem)).reshape(-1, words)
    scale = torch.rand(E, device=dev, dtype=torch.float32) + 0.05
    total_t = torch.tensor([total], device=dev, dtype=torch.int32)
    return dict(A=A, Bg=Bg, Bu=Bu, SFA=SFA, SFBg=SFBg, SFBu=SFBu,
                block_expert=block_expert, block_mstart=block_mstart, block_actual=block_actual,
                total=total_t, scale=scale, sizes=sizes, offsets=offsets,
                A_sf_unpad=A_sf_unpad, sfbg_sem=sfbg_sem, sfbu_sem=sfbu_sem)


def main():
    E, N, K, rows = 8, 256, 512, 1024
    dev = "cuda"
    d = build_packed(E, N, K, rows, bpe_cap=3)
    MB = int(d["block_expert"].numel())
    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    kern = w13_swiglu_packed(E, N, K, rows, MB, block_K=128)
    kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"],
         d["block_expert"], d["block_mstart"], d["block_actual"], d["total"], d["scale"], act)
    torch.cuda.synchronize()

    K16 = K // 16
    Ad = decode_fp4(d["A"], rows, K) * decode_scale_words(pack_scale_words(d["A_sf_unpad"]), K).repeat_interleave(16, 1)
    Bgd = decode_fp4(d["Bg"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfbg_sem"]), K).reshape(E, N, K16).repeat_interleave(16, 2)
    Bud = decode_fp4(d["Bu"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfbu_sem"]), K).reshape(E, N, K16).repeat_interleave(16, 2)
    ref = torch.zeros(rows, N, device=dev, dtype=torch.float32)
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 <= s0:
            continue
        g_ = (Ad[s0:s1] @ Bgd[e].t()) * d["scale"][e]
        u_ = (Ad[s0:s1] @ Bud[e].t()) * d["scale"][e]
        ref[s0:s1] = F.silu(g_.clamp(max=10.0)) * u_.clamp(-10.0, 10.0)
    err = (act.float() - ref).abs().max().item()
    print(f"w13_packed verify: max_abs={err:.5f} MAX_BLOCKS={MB}")

    ms = do_bench(lambda: kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"],
                               d["block_expert"], d["block_mstart"], d["block_actual"], d["total"], d["scale"], act))
    print(f"w13_packed: {ms:.4f} ms")


if __name__ == "__main__":
    main()
