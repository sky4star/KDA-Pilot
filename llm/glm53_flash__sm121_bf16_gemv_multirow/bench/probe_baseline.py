#!/usr/bin/env python3
"""Measure the cuBLAS (F.linear) baseline per workload with cold-L2 graph timing.

Reports achieved DRAM bandwidth vs the GB10 copy ceiling so the headroom is
explicit before any candidate exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing import GraphBench, bandwidth_gbps, weight_copies  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_workloads(path: Path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", type=Path, default=HERE / "workloads.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "baseline_probe.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.set_grad_enabled(False)
    bench = GraphBench(dev)
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU {props.name} cc={torch.cuda.get_device_capability(dev)} "
          f"sm={props.multi_processor_count} L2={props.L2_cache_size/1e6:.1f}MB")

    rows = []
    for wl in load_workloads(args.workloads):
        wid = wl["id"]
        if args.only and wid not in args.only:
            continue
        M, K, N = wl["shapes"]["M"], wl["shapes"]["K"], wl["shapes"]["N"]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.1
        base_w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.05
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        R = weight_copies(N * K * 2, dev)
        weights = [base_w.clone() for _ in range(R)]

        wt = [w.t() for w in weights]  # [K, N] views, built once (not timed)

        def fn(i, x=x, wts=wt, out=out):
            torch.mm(x, wts[i % R], out=out)

        fn(0)
        res = bench.time(fn, reps=args.reps, trials=args.trials)
        weight_bytes = N * K * 2
        total_bytes = weight_bytes + M * K * 2 + M * N * 2
        bps = bandwidth_gbps(total_bytes, res["cold_us"])
        rec = {
            "id": wid, "production": wl.get("production", True),
            "M": M, "K": K, "N": N, "weight_MB": weight_bytes / 1e6,
            "copies": R, **res, "weight_gbps_cold": bps,
        }
        rows.append(rec)
        print(f"{wid:34s} M{M} K{K} N{N:6d} w={weight_bytes/1e6:7.1f}MB "
              f"R={R:3d} cold={res['cold_us']:9.2f}us hot={res['hot_us']:9.2f}us "
              f"BW={bps:6.1f}GB/s")

    args.out.write_text(json.dumps({"gpu": props.name, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
