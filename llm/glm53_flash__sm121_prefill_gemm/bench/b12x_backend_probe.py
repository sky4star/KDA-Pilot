#!/usr/bin/env python3
"""Force b12x backend (micro/static/dynamic) at M=4096 and time each."""
from __future__ import annotations
import statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12x_probe import b12x  # noqa: E402
sys.path.insert(0, str(Path("/kda/llm/glm53_flash__sm121_nvfp4_moe_decode/bench")))
from probe_baseline import build  # noqa: E402

import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as md  # noqa: E402


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    E = 144; topk = 8
    p = build(M, E, topk)
    for backend in ["dynamic", "static", "micro"]:
        md._FORCED_BACKEND = backend
        try:
            b12x(p, E, topk); torch.cuda.synchronize()
            med = timed(lambda: b12x(p, E, topk))
            print(f"backend={backend:8s} M={M}  median={med:.1f} us")
        except Exception as ex:
            print(f"backend={backend:8s} M={M}  FAILED {type(ex).__name__}: {str(ex)[:150]}")


if __name__ == "__main__":
    main()
