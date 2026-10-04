#!/usr/bin/env python3
"""Kernel breakdown + optional single-call for NCU of the b12x fused MoE."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12x_probe import b12x, baseline  # noqa: E402
sys.path.insert(0, str(Path("/kda/llm/glm53_flash__sm121_nvfp4_moe_decode/bench")))
from probe_baseline import build  # noqa: E402


def main():
    only = "--only" in sys.argv
    M = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4096
    E = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 144
    topk = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8
    p = build(M, E, topk)
    for _ in range(3):
        b12x(p, E, topk)
    torch.cuda.synchronize()
    if only:
        b12x(p, E, topk)
        torch.cuda.synchronize()
        return
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            b12x(p, E, topk)
        torch.cuda.synchronize()
    rows = [(e.self_device_time_total, e.count, e.key) for e in prof.key_averages()
            if e.self_device_time_total > 0]
    tot = sum(r[0] for r in rows)
    print(f"b12x M={M} E={E} topk={topk}  total={tot/10/1000:.2f} ms/call")
    for t, c, k in sorted(rows)[::-1][:12]:
        print(f"{t/10/1000:9.3f} ms {100*t/tot:5.1f}%  n={c:4d}  {k[:110]}")


if __name__ == "__main__":
    main()
