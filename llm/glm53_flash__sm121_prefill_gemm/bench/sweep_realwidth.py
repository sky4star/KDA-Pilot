#!/usr/bin/env python3
"""Real-width tile sweep for the SM120 NVFP4 grouped GEMM (server stopped).

Shapes = production per-rank widths:
  w13: N=4096 (gate+up), K=4096   flashinfer ref: 8.03 ms @rows=32768, 13.05 @65536
  w2 : N=4096 (hidden),  K=2048   flashinfer ref: ~4.6-5.8 @32768,        7.26 @65536
rows = M*topk gathered token-expert rows (M=4096 -> 32768, M=8192 -> 65536).
"""
import gc
import sys
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
sys.path.insert(0, "/tmp")
from grouped_nvfp4_gemm import pack_scale_words, decode_fp4, decode_scale_words  # noqa
from flashinfer import fp4_quantize  # noqa
from tl_tile_sweep import grouped_gemm, pack_kernel_order  # noqa

E4M3_MAX, E2M1_MAX = 448.0, 6.0
DEV = "cuda"


def quant_w_stream(E, N, K, seed=11):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.empty((E, N, K // 2), dtype=torch.uint8, device=DEV)
    sf = torch.empty((E, N, K // 16), dtype=torch.uint8, device=DEV)
    gs = torch.empty((E,), dtype=torch.float32, device=DEV)
    for e in range(E):
        w = torch.randn((N, K), device=DEV, dtype=torch.bfloat16, generator=g) / 10
        gv = (E4M3_MAX * E2M1_MAX) / w.float().abs().max().clamp_min(1e-8)
        qe, sfe = fp4_quantize(w, gv.reshape(1), sf_vec_size=16, is_sf_swizzled_layout=False)
        q[e] = qe; sf[e] = sfe; gs[e] = gv
        del w
    return q, sf, gs


def build(E, N, K, rows, BM, BN, bk, seed=5):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64)
    sizes[: rows - base * E] += 1
    bpe = (int(sizes.max()) + BM - 1) // BM
    cap = bpe * BM
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=DEV)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)
    rows_pad = int(offsets[E]) + cap
    words = bk // 64

    A = torch.randint(-120, 120, (rows_pad, K // 2), device=DEV, dtype=torch.int8, generator=g)
    sfa_pad = torch.randint(56, 64, (E * cap, K // 16), device=DEV, dtype=torch.uint8, generator=g)
    SFA = pack_kernel_order(sfa_pad, BM, words)
    q, sf, gs = quant_w_stream(E, N, K)
    SFB = pack_kernel_order(sf.reshape(E * N, K // 16), BN, words)
    return dict(A=A, B=q, SFA=SFA, SFB=SFB, offsets=offsets, gs=gs,
                sfa_pad=sfa_pad, sf=sf, bpe=bpe, cap=cap, rows_pad=rows_pad)


def verify_sample(d, E, N, K, rows, out, experts):
    K16 = K // 16
    worst = 0.0
    for e in experts:
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        sfa_e = d["sfa_pad"][e * d["cap"]: e * d["cap"] + (s1 - s0)]
        Ad = decode_fp4(d["A"][s0:s1], s1 - s0, K) * \
            decode_scale_words(pack_scale_words(sfa_e), K).repeat_interleave(16, 1)
        b = decode_fp4(d["B"][e], N, K)
        bsf = decode_scale_words(pack_scale_words(d["sf"][e]), K)
        Wd = (b.reshape(N, K16, 16) * bsf.unsqueeze(-1)).reshape(N, K)
        ref = Ad @ Wd.t()
        rel = (out[s0:s1].float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        worst = max(worst, rel)
        del Ad, b, Wd, ref
    return worst


def run(tag, N, K, rows, BM, BN, thr, bk, stages=2):
    E = 144
    try:
        # correctness at production shape on 3 sampled experts (also warms compile)
        d = build(E, N, K, rows, BM, BN, bk)
        kern = grouped_gemm(E, N, K, d["rows_pad"], d["bpe"], block_M=BM, block_N=BN,
                            block_K=bk, num_stages=stages, threads=thr)
        out = torch.empty((d["rows_pad"], N), device=DEV, dtype=torch.bfloat16)
        kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], out)
        torch.cuda.synchronize()
        rel = verify_sample(d, E, N, K, rows, out, [0, 70, 143])
        if rel > 2e-2:
            print(f"[{tag}] CORRECTNESS FAIL rel={rel:.4f}", flush=True)
            return None
        ms = do_bench(lambda: kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], out))
        fl = 2.0 * rows * N * K
        byt = E * (N * K // 2 + N * (K // 16)) + rows * (K // 2) + rows * (K // 64) \
            + rows * N * 2 + (rows // (E * 0) if False else 0)
        print(f"[{tag}] {ms:7.2f} ms  {fl/ms/1e9:6.0f} TFLOPS  ~{byt/ms/1e6:4.0f} GB/s  "
              f"grid={(N//BN)*(E*d['bpe'])} CTAs  rel={rel:.4f}", flush=True)
        del d, kern, out
        gc.collect(); torch.cuda.empty_cache()
        return ms
    except Exception as ex:
        print(f"[{tag}] FAILED: {str(ex).splitlines()[0][:140]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        return None


if __name__ == "__main__":
    torch.manual_seed(0)
    refs = {(4096, 4096): {32768: 8.03, 65536: 13.05}, (4096, 2048): {32768: 5.8, 65536: 7.26}}
    for (N, K) in [(4096, 4096), (4096, 2048)]:
        gname = "w13" if K == 4096 else "w2"
        for rows in (32768, 65536):
            print(f"\n#### {gname} N={N} K={K} rows={rows} (M={rows//8}) "
                  f"— flashinfer ref {refs[(N,K)][rows]} ms", flush=True)
            for BM, BN, thr, bk, st in [
                (128, 128, 128, 256, 2),
                (128, 128, 128, 128, 2),
                (64, 128, 128, 128, 2),
                (64, 128, 256, 128, 2),
                (128, 64, 128, 128, 2),
                (64, 64, 128, 128, 2),
                (64, 128, 128, 128, 3),
            ]:
                run(f"{gname} r{rows} BM={BM:3d} BN={BN:3d} thr={thr:3d} BK={bk} st={st}",
                    N, K, rows, BM, BN, thr, bk, st)
