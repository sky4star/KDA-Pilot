#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from flashinfer import fp4_quantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutlass_fp4 import cutlass_fp4_gemm  # noqa: E402

K, N = 4096, 4096
gs = 448.0 * 6.0 / 0.2


def main():
    import flashinfer
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16) / 10
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / 10
    ref = (x.float() @ w.float().t())
    print("ref absmax", ref.abs().max().item())
    alpha = torch.tensor([1.0 / (gs * gs)], device=dev, dtype=torch.float32)

    # 1) mm_fp4 reference (linear sf), weight given as [K/2, N] view
    try:
        xq, xsf = fp4_quantize(x, torch.tensor([gs], device=dev), is_sf_swizzled_layout=False)
        wq, wsf = fp4_quantize(w, torch.tensor([gs], device=dev), is_sf_swizzled_layout=False)
        mm = flashinfer.mm_fp4(xq, wq.t().contiguous().view(torch.uint8), xsf, wsf, alpha,
                               out_dtype=torch.bfloat16, block_size=16, use_8x4_sf_layout=False)
        torch.cuda.synchronize()
        print("mm_fp4(linear)  err_vs_fp32", (mm.float()-ref).abs().max().item())
    except Exception as e:
        print("mm_fp4 failed:", repr(e)[:400])

    # 2) candidate with SWIZZLED sf layout
    try:
        xq, xsf = fp4_quantize(x, torch.tensor([gs], device=dev), is_sf_swizzled_layout=True)
        wq, wsf = fp4_quantize(w, torch.tensor([gs], device=dev), is_sf_swizzled_layout=True)
        print("swizzled xsf", tuple(xsf.shape), "wsf", tuple(wsf.shape))
        out = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
        cutlass_fp4_gemm(out, xq.view(torch.uint8), wq, xsf.view(torch.uint8), wsf.view(torch.uint8), alpha)
        torch.cuda.synchronize()
        print("cand(swizzled)  err_vs_fp32", (out.float()-ref).abs().max().item())
    except Exception as e:
        print("cand(swizzled) failed:", repr(e)[:400])


if __name__ == "__main__":
    main()
