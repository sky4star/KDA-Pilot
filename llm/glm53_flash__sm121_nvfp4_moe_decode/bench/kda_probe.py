#!/usr/bin/env python3
"""Micro-benchmark the GLM-5.3 KDA decode elementwise/recurrent kernels.

Confirms whether the torch-profiler per-launch times for FusedRMSNormGated and
the fused gating delta-rule update reflect real GPU cost. Run in the image.
"""
import sys
import time

import torch

T_OK = True
try:
    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
except Exception as ex:
    print("import failed:", ex)
    T_OK = False


def bench(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def main():
    if not T_OK:
        return
    dev = "cuda"
    for rows, D in [(256, 128), (512, 128), (2048, 128), (4096, 128)]:
        x = torch.randn(rows, D, device=dev, dtype=torch.bfloat16)
        g = torch.randn(rows, D, device=dev, dtype=torch.bfloat16)
        m = FusedRMSNormGated(D, eps=1e-5, device=dev, dtype=torch.bfloat16)
        out = m(x, g)
        us = bench(lambda: m(x, g))
        # bandwidth: read x + g, write y (each rows*D*2 bytes)
        by = rows * D * 2 * 3
        print(f"layer_norm_gated rows={rows:5d} D={D:4d}  {us:8.2f} us  "
              f"({by/(us*1e-6)/1e9:6.1f} GB/s)  out={tuple(out.shape)}")
        # correctness vs torch ref
        xf = x.float(); gf = g.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        ref = xf * torch.rsqrt(var + 1e-5) * torch.sigmoid(gf) * m.weight.float()
        err = (out.float() - ref).abs().max().item()
        print(f"    max_abs_err vs torch ref = {err:.2e}")


if __name__ == "__main__":
    main()
