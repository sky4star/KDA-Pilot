#!/usr/bin/env python3
"""A/B benchmark: cuBLAS (F.linear) baseline vs sm120_bf16_gemv candidate.

Runs inside the sglang container (needs torch + the JIT toolchain).

Deviations from llm/docs/standalone_llm_benchmark_template.py, per the
measurement contract:
  * timing is CUDA-graph replay with rotating weight copies (cold L2), not an
    eager inner loop -- production runs these ops inside a full CUDA graph.
  * baseline and candidate arms are interleaved within each trial.
  * correctness is checked against an fp64 oracle before timing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "solution"))

from timing import GraphBench, bandwidth_gbps, weight_copies  # noqa: E402

import sm120_bf16_gemv as cand  # noqa: E402


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
    return dict(M=M, K=K, N=N, x=x, weights=weights, wts=wts, R=R,
                out_b=out_b, out_c=out_c)


def check(wl, case):
    M, K, N = case["M"], case["K"], case["N"]
    x, out_b, out_c = case["x"], case["out_b"], case["out_c"]
    atol = wl.get("atol", 7e-2)
    rtol = wl.get("rtol", 2e-2)

    out_b.fill_(float("nan"))
    torch.mm(x, case["wts"][0], out=out_b)
    out_c.fill_(float("nan"))
    cand.sm120_bf16_gemv_out(x, case["weights"][0], out_c)
    torch.cuda.synchronize()

    ref = (x.float() @ case["weights"][0].float().t())
    b = out_b.float()
    c = out_c.float()
    if torch.isnan(c).any() or torch.isinf(c).any():
        return {"ok": False, "msg": "candidate has NaN/Inf"}
    db = (c - b).abs()
    dr = (c - ref).abs()
    tol = atol + rtol * ref.abs()
    ok = bool(torch.all(dr <= tol).item()) and bool(torch.all(db <= tol).item())
    return {
        "ok": ok,
        "max_abs_vs_oracle": float(dr.max().item()),
        "max_abs_vs_cublas": float(db.max().item()),
        "mean_abs_ref": float(ref.abs().mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", type=Path, default=HERE / "workloads.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "results.jsonl")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.set_grad_enabled(False)
    bench = GraphBench(dev)

    workloads = json.loads(args.workloads.read_text())
    if args.only:
        workloads = [w for w in workloads if w["id"] in set(args.only)]

    rows = []
    with args.out.open("w") as f:
        for wl in workloads:
            wid = wl["id"]
            case = build_case(wl, dev)
            M, K, N, R = case["M"], case["K"], case["N"], case["R"]
            if not cand.use_sm120_bf16_gemv(M, N, K):
                print(f"SKIP {wid}: not covered by candidate")
                continue
            corr = check(wl, case)
            if not corr["ok"]:
                rec = {"id": wid, "status": "INCORRECT", **corr}
                rows.append(rec); f.write(json.dumps(rec) + "\n")
                print(f"INCORRECT {wid}: {corr}")
                continue

            x, weights, wts, out_b, out_c = (case["x"], case["weights"],
                                             case["wts"], case["out_b"], case["out_c"])

            def fb(i, x=x, wts=wts, out_b=out_b, R=R):
                torch.mm(x, wts[i % R], out=out_b)

            def fc(i, x=x, weights=weights, out_c=out_c, R=R):
                cand.sm120_bf16_gemv_out(x, weights[i % R], out_c)

            fb(0); fc(0); torch.cuda.synchronize()
            gs = {
                "b_cold": bench.build(fb, R, True),
                "c_cold": bench.build(fc, R, True),
                "b_hot": bench.build(fb, R, False),
                "c_hot": bench.build(fc, R, False),
            }
            for g in gs.values():
                g.replay()
            torch.cuda.synchronize()

            samp = {k: [] for k in gs}
            for t in range(args.trials):
                order = list(gs.items())
                if t % 2:
                    order = order[::-1]
                for name, g in order:
                    samp[name].append(bench.measure(g, R))

            import statistics
            med = {k: statistics.median(v) for k, v in samp.items()}
            wb = N * K * 2
            total = wb + M * K * 2 + M * N * 2
            rec = {
                "id": wid, "status": "PASSED", "production": wl.get("production", True),
                "M": M, "K": K, "N": N, "weight_MB": wb / 1e6, "copies": R,
                "baseline_cold_us": med["b_cold"], "candidate_cold_us": med["c_cold"],
                "baseline_hot_us": med["b_hot"], "candidate_hot_us": med["c_hot"],
                "speedup_cold": med["b_cold"] / med["c_cold"],
                "baseline_gbps": bandwidth_gbps(total, med["b_cold"]),
                "candidate_gbps": bandwidth_gbps(total, med["c_cold"]),
                "correctness": corr,
            }
            rows.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{wid:34s} M{M} K{K} N{N:6d} R={R:3d} | "
                  f"cuBLAS {med['b_cold']:8.2f}us ({rec['baseline_gbps']:5.1f}GB/s) "
                  f"cand {med['c_cold']:8.2f}us ({rec['candidate_gbps']:5.1f}GB/s) "
                  f"| x{rec['speedup_cold']:.3f}")

        prod = [r for r in rows if r["status"] == "PASSED" and r["production"]]
        if prod:
            gm = math.exp(sum(math.log(r["speedup_cold"]) for r in prod) / len(prod))
            print(f"\nproduction rows={len(prod)} geomean_speedup={gm:.4f} "
                  f"min={min(r['speedup_cold'] for r in prod):.3f} "
                  f"max={max(r['speedup_cold'] for r in prod):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
