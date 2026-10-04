#!/usr/bin/env python3
"""K1 (w13+SwiGLU dual-MMA) and K3 (w2) ported to expert-major rasterization.

Grid (local, bn, e) instead of (bn, bm): consecutive CTAs share the same
(expert, bn) B tiles across the expert's M-tiles -> B stays in L2.
SF packing unchanged (BM=128 -> 128-row atom layout identical).
"""
import sys
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
sys.path.insert(0, "/tmp")
from w13_swiglu_kernel import w13_swiglu, build_test as k1_build  # noqa
from w2_gemm_kernel import w2_gemm, build_test as k3_build  # noqa
from grouped_nvfp4_gemm import decode_fp4, decode_scale_words, pack_scale_words  # noqa

DEV = "cuda"


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w13_swiglu_emaj(E, N, K, rows, bpe, block_K=128, num_stages=2, threads=256,
                    lim=10.0, out_dtype=T.bfloat16):
    assert N % 128 == 0 and K % block_K == 0 and block_K % 64 == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // 128
    BM, BN = 128, 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        Bg: T.Tensor((E, N, K), in_dtype),
        Bu: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
        SFBg: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        SFBu: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        offsets: T.Tensor((E + 1,), T.int32),
        scale: T.Tensor((E,), T.float32),
        act: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(bpe, n_blocks, E, threads=threads) as (local, bn, e):
            A_s = T.alloc_shared((BM, block_K), in_dtype)
            Bg_s = T.alloc_shared((BN, block_K), in_dtype)
            Bu_s = T.alloc_shared((BN, block_K), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFBg_s = T.alloc_shared((BN, words), T.uint32)
            SFBu_s = T.alloc_shared((BN, words), T.uint32)
            Cg = T.alloc_fragment((BM, BN), T.float32)
            Cu = T.alloc_fragment((BM, BN), T.float32)

            bm = e * bpe + local
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))
            sc = scale[e]

            T.clear(Cg)
            T.clear(Cu)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + BM, ko * block_K:(ko + 1) * block_K], A_s)
                T.copy(Bg[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bg_s)
                T.copy(Bu[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bu_s)
                for r, w in T.Parallel(BM, words):
                    SFA_s[r, w] = SFA[((e * bpe + local) * k_blocks + ko) * BM + r, w]
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


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w2_emaj(E, K, N, rows, bpe, block_N=128, threads=128, num_stages=2,
            out_dtype=T.bfloat16):
    assert K % block_N == 0 and block_N % 64 == 0 and N % 128 == 0
    k_blocks = N // block_N
    words = block_N // 64
    n_blocks = K // 128
    BM, BN = 128, 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, N), in_dtype),
        B: T.Tensor((E, K, N), in_dtype),
        SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        offsets: T.Tensor((E + 1,), T.int32),
        out2: T.Tensor((rows, K), out_dtype),
    ):
        with T.Kernel(bpe, n_blocks, E, threads=threads) as (local, bn, e):
            A_s = T.alloc_shared((BM, block_N), in_dtype)
            B_s = T.alloc_shared((BN, block_N), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFB_s = T.alloc_shared((BN, words), T.uint32)
            C = T.alloc_fragment((BM, BN), T.float32)

            bm = e * bpe + local
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))

            T.clear(C)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + BM, ko * block_N:(ko + 1) * block_N], A_s)
                T.copy(B[e, bn * BN:(bn + 1) * BN, ko * block_N:(ko + 1) * block_N], B_s)
                for r, w in T.Parallel(BM, words):
                    SFA_s[r, w] = SFA[((e * bpe + local) * k_blocks + ko) * BM + r, w]
                for r, w in T.Parallel(BN, words):
                    SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                T.mma_gemm_blockscaled(A_s, B_s, C, SFA_s, SFB_s, transpose_B=True,
                                       clear_accum=False, k_start=ko * block_N,
                                       sf_a_granularity_k=16, sf_b_granularity_k=16,
                                       sf_layout="blockscaled_chunk_kmajor")
            for i, j in T.Parallel(BM, BN):
                if i < actual:
                    out2[m_start + i, bn * BN + j] = T.Cast(out_dtype, C[i, j])

    return main


def verify_k1(d, E, N, K, rows, act, experts, lim=10.0, cap=None):
    cap = cap or (d["SFA"].shape[0] // (E * (K // 512) if False else 0)) if False else cap
    import torch.nn.functional as F
    worst = 0.0
    cap = cap or 128
    for e in experts:
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        sfa_e = d["sfa_sem"][e * cap: e * cap + (s1 - s0)]
        Ad = decode_fp4(d["A"][s0:s1], s1 - s0, K) * \
            decode_scale_words(pack_scale_words(sfa_e), K).repeat_interleave(16, 1)
        outs = []
        for Wq, Wsf in [(d["Bg"], d["sfbg_sem"]), (d["Bu"], d["sfbu_sem"])]:
            b = decode_fp4(Wq[e].reshape(N, K // 2), N, K)
            bsf = decode_scale_words(pack_scale_words(Wsf.reshape(E * N, K // 16)[e * N:(e + 1) * N]), K)
            outs.append((b.reshape(N, K // 16, 16) * bsf.unsqueeze(-1)).reshape(N, K))
        g = (Ad @ outs[0].t()) * d["scale"][e]
        u = (Ad @ outs[1].t()) * d["scale"][e]
        ref = F.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
        rel = (act[s0:s1].float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        worst = max(worst, rel)
    return worst


def verify_k3(d, E, K, N, rows, out2, experts, cap=None):
    worst = 0.0
    cap = cap or 128
    for e in experts:
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        sfa_e = d["sfa_sem"][e * cap: e * cap + (s1 - s0)]
        Ad = decode_fp4(d["A"][s0:s1], s1 - s0, N) * \
            decode_scale_words(pack_scale_words(sfa_e), N).repeat_interleave(16, 1)
        b = decode_fp4(d["B"][e].reshape(K, N // 2), K, N)
        bsf = decode_scale_words(pack_scale_words(d["sfb_sem"].reshape(E * K, N // 16)[e * K:(e + 1) * K]), N)
        Wd = (b.reshape(K, N // 16, 16) * bsf.unsqueeze(-1)).reshape(K, N)
        ref = Ad @ Wd.t()
        rel = (out2[s0:s1].float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        worst = max(worst, rel)
    return worst


def main():
    torch.manual_seed(0)
    # small correctness first (both kernels, emaj vs oracle)
    E, N, K, rows = 8, 256, 512, 1024
    d = k1_build(E, N, K, rows, 1)
    act = torch.empty(rows, N, device=DEV, dtype=torch.bfloat16)
    k = w13_swiglu_emaj(E, N, K, rows, 1, block_K=128)
    k(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
    torch.cuda.synchronize()
    print(f"K1-emaj small verify rel={verify_k1(d, E, N, K, rows, act, range(4), cap=128):.5f}")

    d3 = k3_build(E, K, N, rows, 1)
    out2 = torch.empty(rows, K, device=DEV, dtype=torch.bfloat16)
    k3 = w2_emaj(E, K, N, rows, 1, block_N=128)
    k3(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out2)
    torch.cuda.synchronize()
    print(f"K3-emaj small verify rel={verify_k3(d3, E, K, N, rows, out2, range(4), cap=128):.5f}")

    # production shapes
    for rows, mtag in [(32768, "M4096"), (65536, "M8192")]:
        E, N, K = 144, 2048, 4096
        bpe = 2 if rows == 32768 else 4
        print(f"\n#### K1 rows={rows} ({mtag}) — PR#7 bn-major ref 7.49ms @M4096; flashinfer w13+act 9.45", flush=True)
        d = k1_build(E, N, K, rows, bpe)
        act = torch.empty(rows, N, device=DEV, dtype=torch.bfloat16)
        for name, fn in [
            ("K1 bnmaj(BK128,ref)", lambda: w13_swiglu(E, N, K, rows, bpe, block_K=128)),
            ("K1 emaj (BK128)", lambda: w13_swiglu_emaj(E, N, K, rows, bpe, block_K=128)),
            ("K1 emaj (BK256)", lambda: w13_swiglu_emaj(E, N, K, rows, bpe, block_K=256)),
        ]:
            try:
                kern = fn()
                kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
                torch.cuda.synchronize()
                rel = verify_k1(d, E, N, K, rows, act, [0, 70, 143], cap=bpe*128)
                if rel > 2e-2:
                    print(f"  [{name}] CORRECTNESS FAIL rel={rel:.4f}", flush=True); continue
                ms = do_bench(lambda: kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act))
                print(f"  [{name}] {ms:6.2f} ms  rel={rel:.4f}", flush=True)
            except Exception as ex:
                print(f"  [{name}] FAILED: {str(ex).splitlines()[0][:120]}", flush=True)
        del d; torch.cuda.empty_cache()

        print(f"#### K3 rows={rows} — PR#7 bn-major ref 4.70ms @M4096; flashinfer w2 5.8", flush=True)
        d3 = k3_build(E, K, N, rows, bpe)
        out2 = torch.empty(rows, K, device=DEV, dtype=torch.bfloat16)
        for name, fn in [
            ("K3 bnmaj(BN128,ref)", lambda: w2_gemm(E, K, N, rows, bpe, block_N=128)),
            ("K3 emaj (BN128)", lambda: w2_emaj(E, K, N, rows, bpe, block_N=128)),
            ("K3 emaj (BN256)", lambda: w2_emaj(E, K, N, rows, bpe, block_N=256)),
        ]:
            try:
                kern = fn()
                kern(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out2)
                torch.cuda.synchronize()
                rel = verify_k3(d3, E, K, N, rows, out2, [0, 70, 143], cap=bpe*128)
                if rel > 2e-2:
                    print(f"  [{name}] CORRECTNESS FAIL rel={rel:.4f}", flush=True); continue
                ms = do_bench(lambda: kern(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out2))
                print(f"  [{name}] {ms:6.2f} ms  rel={rel:.4f}", flush=True)
            except Exception as ex:
                print(f"  [{name}] FAILED: {str(ex).splitlines()[0][:120]}", flush=True)
        del d3; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
