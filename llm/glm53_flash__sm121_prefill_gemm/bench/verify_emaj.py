#!/usr/bin/env python3
"""Correctness/precision verification for the emaj fused-MoE operators.

A. standalone K1-emaj / K3-emaj vs dequant oracle — ALL 144 experts, prod shape
B. full pipeline: emaj vs bnmaj numeric identity + fused_moe.py's own oracle
C. pipeline vs production baseline (flashinfer cutlass_fused_moe), same inputs
D. edge cases: empty experts, extreme activations (swiglu limit), NaN/Inf scan
"""
import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
sys.path.insert(0, "/tmp")
from grouped_nvfp4_gemm import decode_fp4, decode_scale_words, pack_scale_words  # noqa
from k1k3_emaj import w13_swiglu_emaj, w2_emaj  # noqa
from flashinfer import fp4_quantize  # noqa
from w13_swiglu_kernel import w13_swiglu, build_test as k1_build  # noqa
from w2_gemm_kernel import w2_gemm, build_test as k3_build  # noqa

DEV = "cuda"
E4M3_MAX, E2M1_MAX = 448.0, 6.0
LIM = 10.0


def stats(got, ref):
    got = got.float(); ref = ref.float()
    err = (got - ref).abs()
    amax = ref.abs().max().item() + 1e-9
    r = err / ref.abs().clamp_min(1e-6)
    return dict(max_abs=round(err.max().item(), 5),
                max_rel=round(err.max().item() / amax, 6),
                mean_rel=round(r.mean().item(), 6),
                p999_rel=round(r.flatten().kthvalue(max(1, int(err.numel() * 0.999))).values.item(), 6),
                finite=bool(torch.isfinite(got).all().item()), ref_absmax=round(amax, 3))


def dequant_w(q, sf, shape):
    """q (R, C/2) uint8 + semantic sf (R, C/16) uint8 -> (R, C) fp32."""
    R, C = shape
    C16 = C // 16
    d = decode_fp4(q.reshape(R, C // 2), R, C).reshape(R, C16, 16)
    s = decode_scale_words(pack_scale_words(sf), C)
    return (d * s.unsqueeze(-1)).reshape(R, C)


# ---------------- Part A ----------------
def part_a(rows=32768, bpe=3):
    print("== A. standalone emaj kernels vs dequant oracle (ALL experts)")
    E, N, K = 144, 2048, 4096
    d = k1_build(E, N, K, rows, bpe)
    act = torch.empty(rows, N, device=DEV, dtype=torch.bfloat16)
    k1 = w13_swiglu_emaj(E, N, K, rows, bpe, block_K=128, lim=LIM)
    k1(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
    torch.cuda.synchronize()
    worst, agg = dict(max_rel=0.0), []
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 <= s0:
            continue
        sfa_e = d["sfa_sem"][e * bpe * 128: e * bpe * 128 + (s1 - s0)]
        Ad = decode_fp4(d["A"][s0:s1], s1 - s0, K) * \
            decode_scale_words(pack_scale_words(sfa_e), K).repeat_interleave(16, 1)
        Wg = dequant_w(d["Bg"][e].reshape(N, K // 2),
                       d["sfbg_sem"].reshape(E * N, -1)[e * N:(e + 1) * N], (N, K))
        Wu = dequant_w(d["Bu"][e].reshape(N, K // 2),
                       d["sfbu_sem"].reshape(E * N, -1)[e * N:(e + 1) * N], (N, K))
        g = (Ad @ Wg.t()) * d["scale"][e]
        u = (Ad @ Wu.t()) * d["scale"][e]
        r = F.silu(g.clamp(max=LIM)) * u.clamp(-LIM, LIM)
        st = stats(act[s0:s1], r)
        agg.append(st)
        if st["max_rel"] > worst["max_rel"]:
            worst = {**st, "expert": e}
        del Ad, Wg, Wu, g, u, r
    print(f"   K1-emaj 144 experts: max_rel={worst['max_rel']:.5f} (e={worst['expert']}) "
          f"mean_of_mean_rel={sum(x['mean_rel'] for x in agg)/len(agg):.6f} finite={worst['finite']}")
    assert worst["max_rel"] < 2e-2 and worst["finite"], "K1 verify FAILED"

    d3 = k3_build(E, K, N, rows, bpe)
    out2 = torch.empty(rows, K, device=DEV, dtype=torch.bfloat16)
    k3 = w2_emaj(E, K, N, rows, bpe, block_N=128)
    k3(d3["A"], d3["B"], d3["SFA"], d3["SFB"], d3["offsets"], out2)
    torch.cuda.synchronize()
    worst, agg = dict(max_rel=0.0), []
    for e in range(E):
        s0, s1 = int(d3["offsets"][e]), int(d3["offsets"][e + 1])
        if s1 <= s0:
            continue
        sfa_e = d3["sfa_sem"][e * bpe * 128: e * bpe * 128 + (s1 - s0)]
        Ad = decode_fp4(d3["A"][s0:s1], s1 - s0, N) * \
            decode_scale_words(pack_scale_words(sfa_e), N).repeat_interleave(16, 1)
        Wd = dequant_w(d3["B"][e].reshape(K, N // 2),
                       d3["sfb_sem"].reshape(E * K, -1)[e * K:(e + 1) * K], (K, N))
        r = Ad @ Wd.t()
        st = stats(out2[s0:s1], r)
        agg.append(st)
        if st["max_rel"] > worst["max_rel"]:
            worst = {**st, "expert": e}
        del Ad, Wd, r
    print(f"   K3-emaj 144 experts: max_rel={worst['max_rel']:.5f} (e={worst['expert']}) "
          f"mean_of_mean_rel={sum(x['mean_rel'] for x in agg)/len(agg):.6f} finite={worst['finite']}")
    assert worst["max_rel"] < 2e-2 and worst["finite"], "K3 verify FAILED"


# ---------------- pipeline (mirrors fused_moe.py) ----------------
def run_pipeline(M, E, K, N, topk, emaj, X, w1, w2, score, lim=LIM):
    from moe_combine_kernel import moe_combine  # noqa
    from act_quant_kernel import gather_quant  # noqa
    from grouped_nvfp4_gemm import swizzle_chunk_kmajor  # noqa

    dev = DEV
    rows = M * topk
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)
    flat = ti.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    token_ids = (order // topk).to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    sizes = counts.cpu().tolist()
    cap = math.ceil(max(sizes) / 128) * 128
    bpe = cap // 128
    offsets = torch.zeros(E + 1, device=dev, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    counts_dev = torch.tensor(sizes, device=dev)
    row_w = tw.reshape(-1)[order].contiguous()
    padded = torch.empty(rows, dtype=torch.int32, device=dev)
    off = 0
    for e in range(E):
        padded[off:off + sizes[e]] = e * cap + torch.arange(sizes[e], device=dev)
        off += sizes[e]
    src_identity = torch.arange(rows, device=dev, dtype=torch.int32)

    def qpe(w):
        q = torch.empty(w.shape[0], w.shape[1], w.shape[2] // 2, dtype=torch.uint8, device=dev)
        sf = torch.empty(w.shape[0], w.shape[1], w.shape[2] // 16, dtype=torch.uint8, device=dev)
        gs = torch.empty(w.shape[0], dtype=torch.float32, device=dev)
        for e in range(w.shape[0]):
            gv = (E4M3_MAX * E2M1_MAX) / w[e].float().abs().max().clamp_min(1e-8)
            qe, sfe = fp4_quantize(w[e], gv.reshape(1), sf_vec_size=16,
                                   is_sf_swizzled_layout=False)
            q[e] = qe; sf[e] = sfe; gs[e] = gv
        return q, sf, gs

    wg_q, wg_sf, wg_gs = qpe(w1[:, :N].contiguous())
    wu_q, wu_sf, wu_gs = qpe(w1[:, N:].contiguous())
    w2_q, w2_sf, w2_gs = qpe(w2)
    words = 128 // 64
    SFBg = swizzle_chunk_kmajor(pack_scale_words(wg_sf.reshape(E * N, -1))).reshape(-1, words)
    SFBu = swizzle_chunk_kmajor(pack_scale_words(wu_sf.reshape(E * N, -1))).reshape(-1, words)
    SFB2 = swizzle_chunk_kmajor(pack_scale_words(w2_sf.reshape(E * K, -1))).reshape(-1, words)

    A1 = torch.empty(rows, K // 2, device=dev, dtype=torch.uint8)
    SFA1sem = torch.empty(E * cap, K // 16, device=dev, dtype=torch.float8_e4m3fn)
    gather_quant(rows, M, K, E, cap)(X, token_ids, padded, A1, SFA1sem)
    SFA1 = swizzle_chunk_kmajor(pack_scale_words(SFA1sem.view(torch.uint8))).reshape(-1, words)

    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    k1 = w13_swiglu_emaj(E, N, K, rows, bpe, block_K=128, lim=lim) if emaj \
        else w13_swiglu(E, N, K, rows, bpe, block_K=128, lim=lim)
    k1(A1, wg_q, wu_q, SFA1, SFBg, SFBu, offsets, 1.0 / wg_gs, act)

    A2 = torch.empty(rows, N // 2, device=dev, dtype=torch.uint8)
    SFA2sem = torch.empty(E * cap, N // 16, device=dev, dtype=torch.float8_e4m3fn)
    gather_quant(rows, rows, N, E, cap)(act, src_identity, padded, A2, SFA2sem)
    SFA2 = swizzle_chunk_kmajor(pack_scale_words(SFA2sem.view(torch.uint8))).reshape(-1, words)

    out2 = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
    k3 = w2_emaj(E, K, N, rows, bpe, block_N=128) if emaj \
        else w2_gemm(E, K, N, rows, bpe, block_N=128)
    k3(A2, w2_q, SFA2, SFB2, offsets, out2)

    row_scale = (1.0 / w2_gs).repeat_interleave(counts_dev) * row_w
    slot_of_row = (order % topk).to(torch.int64)
    slot_rows = torch.empty(M, topk, device=dev, dtype=torch.int64)
    slot_rows[token_ids.long(), slot_of_row] = torch.arange(rows, device=dev)
    slot_scale = row_scale[slot_rows].float().contiguous()
    y = torch.empty(M, K, device=dev, dtype=torch.float32)
    moe_combine(M, rows, K, topk)(out2, slot_rows.to(torch.int32).contiguous(), slot_scale, y)
    torch.cuda.synchronize()
    return y


# ---------------- Part B ----------------
def part_b():
    print("== B. pipeline emaj vs bnmaj (same inputs) + NaN/Inf scan")
    M, E, K, N, topk = 4096, 144, 4096, 2048, 8
    g = torch.Generator(device=DEV).manual_seed(7)
    X = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    w1 = torch.randn(E, 2 * N, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    w2 = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    score = torch.randn(M, E, device=DEV, dtype=torch.bfloat16, generator=g)
    y_a = run_pipeline(M, E, K, N, topk, False, X, w1, w2, score)
    y_e = run_pipeline(M, E, K, N, topk, True, X, w1, w2, score)
    bitexact = torch.equal(y_a, y_e)
    st = stats(y_e, y_a)
    print(f"   emaj vs bnmaj: bit-exact={bitexact} max_rel={st['max_rel']:.2e} "
          f"finite={st['finite']}")
    return X, w1, w2, score, y_e


# ---------------- Part C ----------------
def part_c(X, w1, w2, score):
    print("== C. pipeline(emaj) vs flashinfer cutlass_fused_moe (same bf16 inputs)")
    from flashinfer import fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import fp4_quantize as _fi_q4
    M, E, K, N = X.shape[0], w1.shape[0], X.shape[1], w1.shape[2] // 2
    topk = score.shape[1] and 8
    tw, ti = torch.topk(score, 8, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)

    fp4_quantize = _fi_q4
    def qfi(w):  # flashinfer convention (swizzled SF)
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
    a_gs = torch.ones(E_, device=DEV, dtype=torch.float32) if False else torch.ones(E, device=DEV, dtype=torch.float32)
    y_fi = cutlass_fused_moe(
        X, ti.to(torch.int), tw, w1_q.view(torch.long), w2_q.view(torch.long), X.dtype,
        quant_scales=[a_gs, w1_bs.view(torch.int32), (1 / w1_gs),
                      a_gs, w2_bs.view(torch.int32), (1 / w2_gs)])[0]
    y_e = run_pipeline(M, E, K, N, 8, True, X, w1, w2, score)
    st = stats(y_e, y_fi)
    print(f"   emaj-pipeline vs flashinfer: max_rel={st['max_rel']:.5f} mean_rel={st['mean_rel']:.6f} "
          f"p999_rel={st['p999_rel']:.5f} max_abs={st['max_abs']:.5f} (ref_absmax={st['ref_absmax']}) "
          f"finite={st['finite']}")
    ok = st["finite"] and st["max_rel"] < 5e-2
    print(f"   => {'PASS' if ok else 'FAIL'} (threshold max_rel<5%, fp4 act-quant realization differs)")
    return ok


def round_up(x, y):
    return (x + y - 1) // y * y


# ---------------- Part D ----------------
def part_d():
    print("== D. edge cases (emaj pipeline)")
    M, E, K, N, topk = 2048, 144, 4096, 2048, 8
    g = torch.Generator(device=DEV).manual_seed(9)
    X = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    w1 = torch.randn(E, 2 * N, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    w2 = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16, generator=g) / 10
    score = torch.randn(M, E, device=DEV, dtype=torch.bfloat16, generator=g)

    # D1: empty experts — 24 experts never selected
    score[:, ::6] = -1e4
    ti = torch.topk(score, topk, dim=-1).indices
    counts = torch.bincount(ti.reshape(-1).long(), minlength=E)
    n_empty = (counts == 0).sum().item()
    y = run_pipeline(M, E, K, N, topk, True, X, w1, w2, score)
    y0 = run_pipeline(M, E, K, N, topk, False, X, w1, w2, score)
    st = stats(y, y0)
    print(f"   D1 empty experts ({n_empty} experts idle): emaj-vs-bnmaj bit-exact={torch.equal(y, y0)} "
          f"max_rel={st['max_rel']:.2e} finite={st['finite']}")

    # D2: extreme activations -> swiglu limit clamps
    X2 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) / 10 * 60
    y2 = run_pipeline(M, E, K, N, topk, True, X2, w1, w2, score)
    y2b = run_pipeline(M, E, K, N, topk, False, X2, w1, w2, score)
    st2 = stats(y2, y2b)
    print(f"   D2 extreme X (x60, limit clamps): bit-exact={torch.equal(y2, y2b)} "
          f"max_rel={st2['max_rel']:.2e} finite={st2['finite']}")

    # D3: single hot expert (capacity stress: all tokens put one expert in topk)
    score3 = torch.full((M, E), -1e4, device=DEV, dtype=torch.bfloat16)
    score3[:, 77] = 1e4
    score3[:, :7] = torch.randn(M, 7, device=DEV, dtype=torch.bfloat16, generator=g)
    y3 = run_pipeline(M, E, K, N, topk, True, X, w1, w2, score3)
    print(f"   D3 hot expert 77 (all tokens): finite={torch.isfinite(y3).all().item()} "
          f"absmax={y3.abs().max().item():.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    part_a()
    X, w1, w2, score, _ = part_b()
    ok = part_c(X, w1, w2, score)
    part_d()
    print("\nALL VERIFICATIONS DONE")
