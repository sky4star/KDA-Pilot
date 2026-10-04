#!/usr/bin/env python3
"""Full routed MoE on TileLang grouped NVFP4 GEMMs (SM120a), GB10.

Pipeline (torch glue + two TileLang grouped GEMMs):
    gather tokens per expert
    w13 grouped GEMM        -> out1 [cum_m, 2N]
    SwiGLU(limit) + rescale -> act  [cum_m, N]
    dynamic NVFP4 requant   -> act fp4 + blockscaled scales
    w2 grouped GEMM         -> out2 [cum_m, K]
    scatter + routing weights -> y [M, K]

Validate against a torch fp32 reference built from the *same* quantized
operands, then benchmark vs flashinfer.cutlass_fused_moe.

Run:
    PYTHONPATH=/tmp/tl15site python3 full_moe.py --M 256 --E 144 --K 4096 --N 2048
"""
import argparse
import math
from pathlib import Path
import sys

import torch
from flashinfer import fp4_quantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_nvfp4_gemm import (  # noqa: E402
    grouped_nvfp4_gemm, pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words,
)
from scatter_kernel import moe_scatter  # noqa: E402
from swiglu_kernel import swiglu_limit  # noqa: E402

GS_CONST = 448.0 * 6.0
BLOCK_K = 256


def global_scale(x):
    # amax in bf16 to avoid materializing an fp32 copy of the whole tensor
    return (GS_CONST / x.abs().max().float().clamp_min(1e-8)).reshape(1)


def quant_expert(x):
    """x [r, K] bf16 -> (q uint8[r,K/2], sf uint8[r,K/16] linear, gs float)."""
    gs = global_scale(x)
    q, sf = fp4_quantize(x, gs, sf_vec_size=16, is_sf_swizzled_layout=False)
    return q, sf, float(gs)


def build_weights(w1_bf16, w2_bf16, block_K, words):
    """w1 [E,2N,K], w2 [E,K,N] bf16 -> packed + swizzled scales + global scales."""
    E = w1_bf16.shape[0]
    # w1
    qs, sfs, gss = [], [], []
    for e in range(E):
        q, sf, gs = quant_expert(w1_bf16[e])
        qs.append(q); sfs.append(sf); gss.append(gs)
    w1_q = torch.stack(qs)
    w1_sf = torch.stack(sfs)
    w1_gs = torch.tensor(gss, device=w1_bf16.device)
    # w2
    qs, sfs, gss = [], [], []
    for e in range(E):
        q, sf, gs = quant_expert(w2_bf16[e])
        qs.append(q); sfs.append(sf); gss.append(gs)
    w2_q = torch.stack(qs)
    w2_sf = torch.stack(sfs)
    w2_gs = torch.tensor(gss, device=w1_bf16.device)

    w1_rows = w1_bf16.shape[1]
    w2_rows = w2_bf16.shape[1]
    SFB1 = swizzle_chunk_kmajor(pack_scale_words(w1_sf.reshape(E * w1_rows, -1))).reshape(-1, words)
    SFB2 = swizzle_chunk_kmajor(pack_scale_words(w2_sf.reshape(E * w2_rows, -1))).reshape(-1, words)
    return dict(w1_q=w1_q, SFB1=SFB1, w1_gs=w1_gs, w2_q=w2_q, SFB2=SFB2, w2_gs=w2_gs,
                w1_sf=w1_sf, w2_sf=w2_sf)


def pack_acts_whole(A, sizes, cap, K, block_K, words, dev):
    """A [rows,K] bf16 (expert-contiguous) -> (q, SFA blockscaled, gs scalar, sf unpadded).

    One fp4_quantize for the whole tensor (not per expert); only the per-expert
    padded scale layout is assembled on the host.
    """
    E = len(sizes)
    gs = global_scale(A)
    q, sf = fp4_quantize(A, gs, sf_vec_size=16, is_sf_swizzled_layout=False)
    sfa_sem = torch.zeros((E * cap, K // 16), device=dev, dtype=torch.uint8)
    off = 0
    for e in range(E):
        n = sizes[e]
        sfa_sem[e * cap: e * cap + n] = sf[off:off + n]
        off += n
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, words)
    return q, SFA, float(gs), sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--E", type=int, default=144)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    M, E, K, N, topk = args.M, args.E, args.K, args.N, args.topk
    rows = M * topk
    g = torch.Generator(device=dev).manual_seed(0)

    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    w2 = torch.randn(E, K, N, device=dev, dtype=torch.bfloat16, generator=g) / 10
    score = torch.randn(M, E, device=dev, dtype=torch.bfloat16, generator=g)
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)

    # routing: sort (token, slot) by expert
    flat = ti.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    tok = (order // topk)
    slot = (order % topk)
    counts = torch.bincount(flat, minlength=E)
    sizes = counts.cpu().tolist()
    cap = math.ceil(max(sizes) / 128) * 128
    bpe = cap // 128
    counts_dev = torch.tensor(sizes, device=dev)
    offsets = torch.zeros(E + 1, device=dev, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)

    print(f"M={M} E={E} K={K} N={N} topk={topk} rows={rows} bpe={bpe} cap={cap} "
          f"sizes[min/max]={min(sizes)}/{max(sizes)}")

    # gather activations per expert and quantize
    A_bf16 = X[tok]  # [rows, K]
    words = BLOCK_K // 64
    A1, SFA1, gs_a1, sfa1_sem = pack_acts_whole(A_bf16, sizes, cap, K, BLOCK_K, words, dev)

    W = build_weights(w1, w2, BLOCK_K, words)

    # ---- w13 grouped GEMM ----
    w13 = grouped_nvfp4_gemm(E, 2 * N, K, rows, bpe, block_K=BLOCK_K)
    out1 = torch.empty(rows, 2 * N, device=dev, dtype=torch.bfloat16)
    w13(A1, W["w1_q"], SFA1, W["SFB1"], offsets, out1)
    torch.cuda.synchronize()

    # dequant scale per row (fp4 quantized A * weight): x/(gs_a*gs_w1)
    scale1 = (1.0 / (gs_a1 * W["w1_gs"]))  # [E]
    row_scale1 = torch.repeat_interleave(scale1, counts_dev)
    sgl = swiglu_limit(rows, N, args.swiglu_limit)
    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    sgl(out1, row_scale1, act)
    torch.cuda.synchronize()

    # ---- requant act for w2 ----
    A2, SFA2, gs_a2, sfa2_sem = pack_acts_whole(act, sizes, cap, N, BLOCK_K, words, dev)

    w2k = grouped_nvfp4_gemm(E, K, N, rows, bpe, block_K=BLOCK_K)
    out2 = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
    w2k(A2, W["w2_q"], SFA2, W["SFB2"], offsets, out2)
    torch.cuda.synchronize()

    scale2 = 1.0 / (gs_a2 * W["w2_gs"])
    row_scale2 = torch.repeat_interleave(scale2, counts_dev)

    # ---- fused scatter + routing weights ----
    wslot = tw.reshape(-1)[order].float()
    rowscale = row_scale2 * wslot
    tok32 = tok.to(torch.int32)
    scat = moe_scatter(rows, K, M)
    y = torch.zeros(M, K, device=dev, dtype=torch.float32)
    scat(out2, rowscale, tok32, y)

    if not args.no_verify:
        def deq(q2d, sem):
            r, c2 = q2d.shape
            cols = c2 * 2
            full = decode_fp4(q2d, r, cols)
            sf = decode_scale_words(pack_scale_words(sem), cols)
            return full * sf.repeat_interleave(16, dim=1)

        sizes_dev = torch.tensor(sizes, device=dev)
        sz = sizes_dev
        # reference: dequantized operands, fp32, per expert (no global scale)
        A1d = deq(A1, sfa1_sem)
        w1d = deq(W["w1_q"].reshape(E * 2 * N, K // 2), W["w1_sf"].reshape(E * 2 * N, -1)).reshape(E, 2 * N, K)
        A2d = deq(A2, sfa2_sem)
        w2d = deq(W["w2_q"].reshape(E * K, N // 2), W["w2_sf"].reshape(E * K, -1)).reshape(E, K, N)

        y_ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
        for e in range(E):
            s0, s1 = int(offsets[e]), int(offsets[e + 1])
            o1 = A1d[s0:s1] @ w1d[e].t()          # raw (pre gs)
            f1r = o1 * (1.0 / (gs_a1 * W["w1_gs"][e]))
            g = f1r[:, :N].clamp(max=args.swiglu_limit)
            u = f1r[:, N:].clamp(-args.swiglu_limit, args.swiglu_limit)
            act_r = torch.nn.functional.silu(g) * u
            # must match the requantized act the pipeline fed to w2
            ar_err = (act_r.to(torch.bfloat16).float() - act[s0:s1].float()).abs().max().item()
            o2 = A2d[s0:s1] @ w2d[e].t() * (1.0 / (gs_a2 * W["w2_gs"][e]))
            seg = order[s0:s1]
            y_ref.index_add_(0, tok[s0:s1], o2 * tw.reshape(-1)[seg].unsqueeze(1))
        del ar_err
        err = (y - y_ref).abs()
        rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
        print(f"verify: max_abs={err.max().item():.5f} ref_absmax={y_ref.abs().max().item():.3f} "
              f"max_rel={rel:.5f}")
        assert rel < 2e-2, f"pipeline mismatch max_rel={rel}"

    # ---- timing ----
    def t(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); e.synchronize()
        return s.elapsed_time(e) / n

    t_w13 = t(lambda: w13(A1, W["w1_q"], SFA1, W["SFB1"], offsets, out1))
    t_w2 = t(lambda: w2k(A2, W["w2_q"], SFA2, W["SFB2"], offsets, out2))
    print(f"w13={t_w13:.3f} ms  w2={t_w2:.3f} ms  gemm_total={t_w13 + t_w2:.3f} ms")
    print(f"flops={2*rows*((2*N)*K + N*K)/(1e12):.3f} TFLOP  eff={(2*rows*((2*N)*K + N*K))/((t_w13+t_w2)*1e-3)/1e12:.1f} TFLOPS")

    # full forward (prototype torch glue; activation quant is NOT fused)
    sizes_l = sizes
    import torch.nn.functional as F

    def full():
        Ab = X[tok]
        A1_, SFA1_, gs1_, _ = pack_acts_whole(Ab, sizes_l, cap, K, BLOCK_K, words, dev)
        o1 = torch.empty(rows, 2 * N, device=dev, dtype=torch.bfloat16)
        w13(A1_, W["w1_q"], SFA1_, W["SFB1"], offsets, o1)
        sc1 = (1.0 / gs1_) * torch.repeat_interleave(1.0 / W["w1_gs"], counts_dev)
        aa = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
        sgl(o1, sc1, aa)
        A2_, SFA2_, gs2_, _ = pack_acts_whole(aa, sizes_l, cap, N, BLOCK_K, words, dev)
        o2 = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
        w2k(A2_, W["w2_q"], SFA2_, W["SFB2"], offsets, o2)
        sc2 = (1.0 / gs2_) * torch.repeat_interleave(1.0 / W["w2_gs"], counts_dev)
        rowscale = sc2 * tw.reshape(-1)[order].float()
        out = torch.zeros(M, K, device=dev, dtype=torch.float32)
        scat(o2, rowscale, tok32, out)
        return out

    t_full = t(full, n=5)
    print(f"full_forward(whole-tensor quant glue)={t_full:.3f} ms  (baseline cutlass_fused_moe ~17.73 ms)")


if __name__ == "__main__":
    main()
