#!/usr/bin/env python3
"""CUDA-graph timing of the KDA decode elementwise kernels (removes Triton
Python launch overhead, which dominates the eager number)."""
import torch
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated


def graph_us(fn, iters=100, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / (reps * iters)


def main():
    for rows, D in [(512, 128), (2048, 128), (4096, 128), (8192, 128)]:
        x = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
        gg = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
        m = FusedRMSNormGated(D, eps=1e-5, device="cuda", dtype=torch.bfloat16)
        us = graph_us(lambda: m(x, gg))
        by = rows * D * 2 * 3
        print(f"layer_norm_gated rows={rows:5d} D={D}  graph {us:7.3f} us  "
              f"({by/(us*1e-6)/1e9:6.1f} GB/s)")


if __name__ == "__main__":
    main()
