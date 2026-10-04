#!/usr/bin/env python3
"""Time flashinfer cutlass_fused_moe (TRT-LLM MoE) at M=4096 vs tuning knobs."""
from __future__ import annotations
import argparse, statistics, sys
from pathlib import Path

import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe

sys.path.insert(0, str(Path("/kda/llm/glm53_flash__sm121_nvfp4_moe_decode/bench")))
from probe_baseline import build, N, K, BLK  # noqa: E402


def call(p, **kw):
    return cutlass_fused_moe(
        p["a"], p["ti"], p["tw"],
        p["w1_q"].view(torch.long), p["w2_q"].view(torch.long), p["a"].dtype,
        quant_scales=[p["a_gs"], p["w1_bs"].view(torch.int32), (1 / p["w1_gs"]),
                      p["a_gs"], p["w2_bs"].view(torch.int32), (1 / p["w2_gs"])],
        **kw,
    )[0]


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--E", type=int, default=144)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tune-max-tokens", type=int, default=8192)
    ap.add_argument("--fused-finalize", type=int, default=1)
    ap.add_argument("--activation", type=int, default=3)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0))
    p = build(args.M, args.E, args.topk)
    kw = dict(tune_max_num_tokens=args.tune_max_tokens,
              use_fused_finalize=bool(args.fused_finalize),
              activation_type=args.activation)
    try:
        out = call(p, **kw); torch.cuda.synchronize()
    except Exception as ex:
        print(f"M={args.M} kw={kw} FAILED {type(ex).__name__}: {str(ex)[:200]}")
        return
    med, lo = timed(lambda: call(p, **kw))
    print(f"M={args.M} E={args.E} topk={args.topk} kw={kw}  median={med:.1f} us  min={lo:.1f} us")


if __name__ == "__main__":
    main()
