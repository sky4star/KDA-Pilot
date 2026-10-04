#!/usr/bin/env python3
"""Tile-size sweep for SM120 NVFP4 grouped GEMM in TileLang 0.1.15.

SF packing built in the emitter's kernel-order K-major layout:
  flat = kb*BM + (idx%32)*(BM//32) + idx//32 ; stored at (flat//words, flat%words)
matching TensorCoreIntrinEmitterSM120._tile_kmajor_scale_word(idx, kb, tile_rows).
"""
import sys
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
from grouped_nvfp4_gemm import pack_scale_words, decode_fp4, decode_scale_words  # noqa
from flashinfer import fp4_quantize  # noqa

E4M3_MAX, E2M1_MAX = 448.0, 6.0
DEV = "cuda"


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def grouped_gemm(E, N, K, rows, blocks_per_expert, block_M=128, block_N=128,
                 block_K=256, num_stages=2, threads=128, out_dtype=T.bfloat16):
    assert block_M % 32 == 0 and block_N % 32 == 0
    assert block_K % 64 == 0 and N % block_N == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // block_N
    in_dtype = T.float4_e2m1fn
    total_m_blocks = E * blocks_per_expert

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        B: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((total_m_blocks * k_blocks * block_M, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * block_N, words), T.uint32),
        expert_offsets: T.Tensor((E + 1,), T.int32),
        C: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(n_blocks, total_m_blocks, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_s = T.alloc_shared((block_M, words), T.uint32)
            SFB_s = T.alloc_shared((block_N, words), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)

            e = bm // blocks_per_expert
            local = bm % blocks_per_expert
            m_start = expert_offsets[e] + local * block_M
            actual = T.max(0, T.min(block_M, expert_offsets[e + 1] - m_start))

            T.clear(C_l)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K], A_s)
                T.copy(B[e, bn * block_N:(bn + 1) * block_N, ko * block_K:(ko + 1) * block_K], B_s)
                for r, w in T.Parallel(block_M, words):
                    SFA_s[r, w] = SFA[(bm * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, words):
                    SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * block_N + r, w]
                T.mma_gemm_blockscaled(
                    A_s, B_s, C_l, SFA_s, SFB_s,
                    transpose_B=True, clear_accum=False,
                    k_start=ko * block_K,
                    sf_a_granularity_k=16, sf_b_granularity_k=16,
                    sf_layout="blockscaled_chunk_kmajor",
                )
            for i, j in T.Parallel(block_M, block_N):
                if i < actual:
                    C[m_start + i, bn * block_N + j] = C_l[i, j]

    return main


def pack_kernel_order(sem_pad, tile_rows, words):
    """sem_pad: (tiles*tile_rows, K/16) uint8 semantic, tile-aligned.
    Returns (tiles*k_chunks*tile_rows, words) uint32 in the emitter's layout."""
    T_ = sem_pad.shape[0] // tile_rows
    K16 = sem_pad.shape[1]
    n_chk = (K16 * 16) // (words * 64)  # chunks of block_K over K
    pk = pack_scale_words(sem_pad)  # (tiles*tile_rows, K/64) uint32
    t_ar = torch.arange(T_, device=DEV)
    idx = torch.arange(tile_rows, device=DEV)
    intra = (idx % 32) * (tile_rows // 32) + idx // 32
    out32 = torch.zeros((T_ * n_chk * tile_rows, words), dtype=torch.int32, device=DEV)
    pki = pk.view(torch.int32)
    for ko in range(n_chk):
        for kb in range(words):
            flat = kb * tile_rows + intra
            r, w = flat // words, flat % words
            gl = (t_ar[:, None] * n_chk + ko) * tile_rows + r[None, :]
            out32[gl, w] = pki[t_ar[:, None] * tile_rows + idx[None, :], ko * words + kb]
    return out32.view(torch.uint32)


def quant_w(w):
    E, R, K = w.shape
    q = torch.empty((E, R, K // 2), dtype=torch.uint8, device=DEV)
    sf = torch.empty((E, R, K // 16), dtype=torch.uint8, device=DEV)
    gs = torch.empty((E,), dtype=torch.float32, device=DEV)
    for e in range(E):
        g = (E4M3_MAX * E2M1_MAX) / w[e].float().abs().max().clamp_min(1e-8)
        qe, sfe = fp4_quantize(w[e], g.reshape(1), sf_vec_size=16, is_sf_swizzled_layout=False)
        q[e] = qe; sf[e] = sfe; gs[e] = g
    return q, sf, gs


def build(E, N, K, rows, BM, BN, bk=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64)
    sizes[: rows - base * E] += 1
    bpe = (int(sizes.max()) + BM - 1) // BM
    cap = bpe * BM
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=DEV)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)
    rows_pad = int(offsets[E]) + cap  # headroom for tail-tile overread

    A = torch.randint(-120, 120, (rows_pad, K // 2), device=DEV, dtype=torch.int8, generator=g)
    # semantic A scales, per-expert padded to cap rows
    sfa_pad = torch.randint(56, 64, (E * cap, K // 16), device=DEV, dtype=torch.uint8, generator=g)
    words = bk // 64
    SFA = pack_kernel_order(sfa_pad, BM, words)
    Wq, Wsf, Wgs = quant_w(torch.randn((E, N, K), device=DEV, dtype=torch.bfloat16, generator=g) / 10)
    sfb_pad = Wsf.reshape(E * N, K // 16)
    SFB = pack_kernel_order(sfb_pad, BN, words)  # N divisible by BN
    return dict(A=A, B=Wq, SFA=SFA, SFB=SFB, offsets=offsets, scale=Wgs,
                sfa_pad=sfa_pad, sfb_pad=sfb_pad, sizes=sizes, bpe=bpe, cap=cap,
                rows_pad=rows_pad)


def verify(d, E, N, K, rows, BM, BN, out):
    K16 = K // 16
    # decode per real row from per-expert padded semantic scales
    sfa_real = torch.zeros((rows, K16), device=DEV, dtype=torch.uint8)
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        sfa_real[s0:s1] = d["sfa_pad"][e * d["cap"]: e * d["cap"] + (s1 - s0)]
    Ad = decode_fp4(d["A"][:rows], rows, K) * decode_scale_words(pack_scale_words(sfa_real), K).repeat_interleave(16, 1)
    Wd = decode_fp4(d["B"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfb_pad"]), K).reshape(E, N, K16).repeat_interleave(16, 2)
    worst = 0.0
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 <= s0:
            continue
        ref = (Ad[s0:s1] @ Wd[e].t())  # no global alpha in this kernel
        got = out[s0:s1].float()
        rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        worst = max(worst, rel)
    return worst


def run_case(BM, BN, threads, block_K=256, stages=2):
    tag = f"BM={BM:3d} BN={BN:3d} thr={threads:3d} BK={block_K}"
    try:
        E2, N2, K2, rows2 = 4, 256, 1024, 512
        d2 = build(E2, N2, K2, rows2, BM, BN, bk=block_K)
        kern2 = grouped_gemm(E2, N2, K2, d2["rows_pad"], d2["bpe"], block_M=BM, block_N=BN,
                             block_K=block_K, num_stages=stages, threads=threads)
        out2 = torch.full((d2["rows_pad"], N2), float("nan"), device=DEV, dtype=torch.bfloat16)
        kern2(d2["A"], d2["B"], d2["SFA"], d2["SFB"], d2["offsets"], out2)
        torch.cuda.synchronize()
        rel = verify(d2, E2, N2, K2, rows2, BM, BN, out2)
        if rel > 2e-2:
            print(f"[{tag}] CORRECTNESS FAIL rel={rel:.4f}")
            return None
        print(f"[{tag}] correctness OK (rel={rel:.5f})")

        E3, N3, K3, rows3 = 144, 2048, 4096, 32768
        d3 = build(E3, N3, K3, rows3, BM, BN, bk=block_K)
        kern3 = grouped_gemm(E3, N3, K3, d3["rows_pad"], d3["bpe"], block_M=BM, block_N=BN,
                             block_K=block_K, num_stages=stages, threads=threads)
        out3 = torch.empty((d3["rows_pad"], N3), device=DEV, dtype=torch.bfloat16)
        kern3(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out3)
        torch.cuda.synchronize()
        ms = do_bench(lambda: kern3(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out3))
        grid = (N3 // BN) * (E3 * d3["bpe"])
        print(f"[{tag}] w13-shape: {ms:.3f} ms   grid={grid} CTAs   bpe={d3['bpe']}")
        return ms
    except Exception as ex:
        import traceback; print(f"[{tag}] FAILED:"); traceback.print_exc()
        return None


if __name__ == "__main__":
    torch.manual_seed(0)
    results = {}
    for BM, BN, thr, bk in [
        (128, 128, 128, 256),
        (64, 128, 128, 256),
        (128, 64, 128, 256),
        (64, 128, 256, 256),
        (64, 64, 128, 256),
        (128, 128, 128, 128),
        (64, 128, 128, 128),
    ]:
        ms = run_case(BM, BN, thr, block_K=bk)
        results[f"{BM}x{BN}x{thr}x{bk}"] = ms
    print("\n==== summary (w13 shape: E=144 N=2048 K=4096 rows=32768) ====")
    for k, v in results.items():
        print(f"{k:24s} {'FAIL' if v is None else f'{v:.3f} ms'}")
