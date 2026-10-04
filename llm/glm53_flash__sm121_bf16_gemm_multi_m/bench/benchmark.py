#!/usr/bin/env python3
"""A/B benchmark: cuBLAS (torch.mm) baseline vs native bf16 GEMM candidate for
the GLM-5.3-Flash large-M / extend shapes (M in [9,64]).

Cold-L2 by default (weight > 2*L2 for most rows; R rotating copies otherwise),
CUDA-graph replay timer of record, interleaved arms, median of N trials.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "solution"))

from timing import GraphBench, bandwidth_gbps, weight_copies  # noqa: E402
import entry as cand  # noqa: E402  (solution/entry.py)


def build_case(wl, dev):
    M, K, N = wl["shapes"]["M"], wl["shapes"]["K"], wl["shapes"]["N"]
    g = torch.Generator(device=dev).manual_seed(wl.get("seed", 0))
    x = (torch.randn(M, K, dtype=torch.bfloat16, device=dev, generator=g) * 0.1)
    w0 = (torch.randn(N, K, dtype=torch.bfloat16, device=dev, generator=g) * 0.05)
    R = weight_copies(N * K * 2, dev)
    weights = [w0.clone() for _ in range(R)]
    wts = [w.t() for w in weights]
    out_b = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    out_c = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    return dict(M=M, K=K, N=N, x=x, weights=weights, wts=wts, R=R, out_b=out_b, out_c=out_c)


def check(wl, case):
    M, K, N = case["M"], case["K"], case["N"]
    x, out_b, out_c = case["x"], case["out_b"], case["out_c"]
    atol = wl.get("atol", 7e-2); rtol = wl.get("rtol", 2e-2)
    out_b.fill_(float("nan")); torch.mm(x, case["wts"][0], out=out_b)
    out_c.fill_(float("nan")); cand.bf16_gemm_out(x, case["weights"][0], out_c)
    torch.cuda.synchronize()
    ref = (x.float() @ case["weights"][0].float().t())
    b, c = out_b.float(), out_c.float()
    if torch.isnan(c).any() or torch.isinf(c).any():
        return {"ok": False, "msg": "candidate NaN/Inf"}
    tol = atol + rtol * ref.abs()
    return {"ok": bool(torch.all((c - ref).abs() <= tol).item()),
            "max_abs_vs_oracle": float((c - ref).abs().max().item()),
            "max_abs_vs_cublas": float((c - b).abs().max().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", type=Path, default=HERE / "workloads.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "results.jsonl")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev); torch.set_grad_enabled(False)
    bench = GraphBench(dev)
    workloads = json.loads(args.workloads.read_text())
    if args.only:
        workloads = [w for w in workloads if w["id"] in set(args.only)]

    rows = []
    with args.out.open("w") as f:
        for wl in workloads:
            case = build_case(wl, dev)
            M, K, N, R = case["M"], case["K"], case["N"], case["R"]
            corr = check(wl, case)
            if not corr["ok"]:
                rec = {"id": wl["id"], "status": "INCORRECT", **corr}
                rows.append(rec); f.write(json.dumps(rec) + "\n"); print("INCORRECT", wl["id"], corr); continue
            x, weights, wts, out_b, out_c = (case["x"], case["weights"], case["wts"], case["out_b"], case["out_c"])

            def fb(i): torch.mm(x, wts[i % R], out=out_b)
            def fc(i): cand.bf16_gemm_out(x, weights[i % R], out_c)
            fb(0); fc(0); torch.cuda.synchronize()
            gs = {"b_cold": bench.build(fb, R, True), "c_cold": bench.build(fc, R, True)}
            for g in gs.values(): g.replay()
            torch.cuda.synchronize()
            samp = {k: [] for k in gs}
            for t in range(args.trials):
                order = list(gs.items())
                if t % 2: order = order[::-1]
                for name, g in order:
                    samp[name].append(bench.measure(g, R))
            med = {k: statistics.median(v) for k, v in samp.items()}
            total = N * K * 2 + M * K * 2 + M * N * 2
            rec = {"id": wl["id"], "status": "PASSED", "production": wl.get("production", False),
                   "M": M, "K": K, "N": N, "copies": R,
                   "baseline_cold_us": med["b_cold"], "candidate_cold_us": med["c_cold"],
                   "speedup_cold": med["b_cold"] / med["c_cold"],
                   "baseline_gbps": bandwidth_gbps(total, med["b_cold"]),
                   "candidate_gbps": bandwidth_gbps(total, med["c_cold"]),
                   "correctness": corr}
            rows.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{wl['id']:30s} M{M} K{K} N{N:6d} | cuBLAS {med['b_cold']:8.2f}us "
                  f"({rec['baseline_gbps']:5.1f}GB/s) cand {med['c_cold']:8.2f}us "
                  f"({rec['candidate_gbps']:5.1f}GB/s) | x{rec['speedup_cold']:.3f}")
        prod = [r for r in rows if r.get("status") == "PASSED" and r.get("production")]
        if prod:
            gm = math.exp(sum(math.log(r["speedup_cold"]) for r in prod) / len(prod))
            print(f"\nproduction geomean={gm:.4f} min={min(r['speedup_cold'] for r in prod):.3f} "
                  f"max={max(r['speedup_cold'] for r in prod):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
