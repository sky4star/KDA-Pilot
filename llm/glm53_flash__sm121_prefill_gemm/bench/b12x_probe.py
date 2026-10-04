#!/usr/bin/env python3
"""Test FlashInfer b12x fused MoE (SM120/121 CuTe-DSL) at M=4096 vs the
current cutlass_fused_moe (TRT-LLM) baseline.  Correctness + timing.
"""
from __future__ import annotations
import argparse, statistics, sys
from pathlib import Path

import torch
from flashinfer import b12x_fused_moe
sys.path.insert(0, str(Path("/kda/llm/glm53_flash__sm121_nvfp4_moe_decode/bench")))
from probe_baseline import build  # noqa: E402


def baseline(p):
    from flashinfer.fused_moe import cutlass_fused_moe
    return cutlass_fused_moe(
        p["a"], p["ti"], p["tw"],
        p["w1_q"].view(torch.long), p["w2_q"].view(torch.long), p["a"].dtype,
        quant_scales=[p["a_gs"], p["w1_bs"].view(torch.int32), (1 / p["w1_gs"]),
                      p["a_gs"], p["w2_bs"].view(torch.int32), (1 / p["w2_gs"])],
    )[0]


def b12x(p, E, topk, quant_mode="nvfp4"):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    E = p["w1_q"].shape[0]
    w1rows = p["w1_q"].shape[1]
    w2rows = p["w2_q"].shape[1]
    w1_sf = convert_sf_to_mma_layout(p["w1_bs"].reshape(E * w1rows, -1),
                                     m=w1rows, k=4096, num_groups=E, sf_vec_size=16)
    w2_sf = convert_sf_to_mma_layout(p["w2_bs"].reshape(E * w2rows, -1),
                                     m=w2rows, k=2048, num_groups=E, sf_vec_size=16)
    return b12x_fused_moe(
        x=p["a"],
        w1_weight=p["w1_q"], w1_weight_sf=w1_sf,
        w2_weight=p["w2_q"], w2_weight_sf=w2_sf,
        token_selected_experts=p["ti"], token_final_scales=p["tw"],
        num_experts=E, top_k=topk,
        w1_alpha=(1 / p["w1_gs"]), w2_alpha=(1 / p["w2_gs"]),
        fc2_input_scale=p["a_gs"], input_global_scale=p["a_gs"],
        quant_mode=quant_mode,
    )


def timed(fn, iters, warmup=5):
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
    ap.add_argument("--quant-mode", default="nvfp4")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0))
    p = build(args.M, args.E, args.topk)
    try:
        ref = baseline(p); torch.cuda.synchronize()
    except Exception as ex:
        print("baseline FAILED", repr(ex)); return
    try:
        got = b12x(p, args.E, args.topk, args.quant_mode); torch.cuda.synchronize()
    except Exception as ex:
        print(f"b12x FAILED {type(ex).__name__}: {str(ex)[:400]}"); return
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    print(f"b12x out {tuple(got.shape)} ref {tuple(ref.shape)}")
    print(f"  max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} "
          f"max_rel={(diff/denom).max().item():.4f} ref_absmax={ref.abs().max().item():.3f}")
    bmed, _ = timed(lambda: baseline(p), args.iters)
    gmed, gmin = timed(lambda: b12x(p, args.E, args.topk, args.quant_mode), args.iters)
    print(f"  baseline (cutlass_fused_moe) median={bmed:.1f} us")
    print(f"  b12x                         median={gmed:.1f} us  min={gmin:.1f} us  speedup={bmed/gmed:.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
