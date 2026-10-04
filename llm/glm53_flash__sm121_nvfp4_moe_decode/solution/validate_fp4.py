#!/usr/bin/env python3
"""Validate native CUTLASS SM120 block-scaled NVFP4 dense GEMM against the
FlashInfer production MoE (single active expert => the expert MLP + SwiGLU).

Shapes: GLM-5.3-Flash hidden K=4096, moe_intermediate N=2048.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutlass_fp4 import cutlass_fp4_gemm  # noqa: E402

K = 4096
N = 2048
E = 8
BLK = 16


def round_up(x, y):
    return (x + y - 1) // y * y


def main():
    dev = "cuda"
    torch.manual_seed(0)
    gs = 448.0 * 6.0 / 0.2

    w1 = torch.randn(2 * N, K, device=dev, dtype=torch.bfloat16) / 10
    w2 = torch.randn(K, N, device=dev, dtype=torch.bfloat16) / 10
    w1_q, w1_bs = fp4_quantize(w1, torch.tensor([gs], device=dev))
    w2_q, w2_bs = fp4_quantize(w2, torch.tensor([gs], device=dev))

    a = torch.randn(1, K, device=dev, dtype=torch.bfloat16) / 10
    a_gs = torch.ones((), device=dev)

    w1_all = torch.zeros(E, 2 * N, K // 2, device=dev, dtype=torch.uint8)
    w2_all = torch.zeros(E, K, N // 2, device=dev, dtype=torch.uint8)
    w1_bs_all = torch.zeros(E, round_up(2 * N, 128), round_up(K // BLK, 4), device=dev, dtype=torch.float8_e4m3fn)
    w2_bs_all = torch.zeros(E, round_up(K, 128), round_up(N // BLK, 4), device=dev, dtype=torch.float8_e4m3fn)
    w1_all[0] = w1_q; w2_all[0] = w2_q; w1_bs_all[0] = w1_bs; w2_bs_all[0] = w2_bs
    w1_gs = torch.full((E,), gs, device=dev)
    w2_gs = torch.full((E,), gs, device=dev)
    ti = torch.zeros(1, 1, device=dev, dtype=torch.int)
    tw = torch.ones(1, 1, device=dev, dtype=torch.float32)
    out_moe = cutlass_fused_moe(
        a, ti, tw, w1_all.view(torch.long), w2_all.view(torch.long), a.dtype,
        quant_scales=[a_gs, w1_bs_all.view(torch.int32), 1 / w1_gs,
                      a_gs, w2_bs_all.view(torch.int32), 1 / w2_gs],
    )[0]
    torch.cuda.synchronize()

    # fp32 reference on the *checkpoint fp4* weights: dequant is not attempted;
    # compare candidate chain vs baseline MoE.
    import math
    a_q, a_sf = fp4_quantize(a, a_gs)
    ref = out_moe.float()
    for k in (1.0, 1/6, 6.0, 1/3, 1/2, 1/12, 1/36, 1/4, 1/8):
        alpha1 = (a_gs * (1 / w1_gs[0:1]) * k).contiguous()
        gu = torch.empty(1, 2 * N, device=dev, dtype=torch.bfloat16)
        cutlass_fp4_gemm(gu, a_q.view(torch.uint8), w1_q, a_sf.view(torch.uint8), w1_bs.view(torch.uint8), alpha1)
        gate, up = gu[:, :N], gu[:, N:]
        act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        a2_q, a2_sf = fp4_quantize(act, a_gs)
        alpha2 = (a_gs * (1 / w2_gs[0:1]) * k).contiguous()
        out_c = torch.empty(1, K, device=dev, dtype=torch.bfloat16)
        cutlass_fp4_gemm(out_c, a2_q.view(torch.uint8), w2_q, a2_sf.view(torch.uint8), w2_bs.view(torch.uint8), alpha2)
        err = (out_c.float() - ref).abs().max().item()
        print(f"k={k:8.5f} absmax_err={err:.4e} ref_absmax={ref.abs().max().item():.4f}")
    print("baseline out[:4]", out_moe[0, :4].float().tolist())


if __name__ == "__main__":
    main()
