#!/usr/bin/env python3
"""Kernel-level breakdown of the FlashInfer cutlass_fused_moe call (M=1 / M=8)."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_baseline import build, call  # noqa: E402


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 144
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    p = build(M, E, topk)
    for _ in range(5):
        call(p)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            call(p)
        torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.count, e.key) for e in prof.key_averages()
            if e.self_device_time_total > 0]
    tot = sum(r[0] for r in rows)
    print(f"M={M} E={E} topk={topk}  total={tot/10/1000:.1f} ms/call over 10 calls")
    for t, c, k in sorted(rows)[::-1][:20]:
        print(f"{t/10/1000:9.3f} ms {100*t/tot:5.1f}%  n={c:4d}  {k[:120]}")


if __name__ == "__main__":
    main()
