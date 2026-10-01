#!/usr/bin/env python3
"""Config sweep for the small-N GEMV shapes (n128, n288, n32)."""
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "solution"))
from timing import GraphBench, weight_copies, bandwidth_gbps  # noqa: E402
import sm120_bf16_gemv as cand  # noqa: E402
from sm120_bf16_gemv import _jit_module  # noqa: E402

CFGS = [
    (1, 2, 4, False), (2, 2, 4, False), (2, 4, 4, False), (4, 2, 4, False),
    (4, 4, 4, False), (2, 2, 2, False), (2, 4, 2, False), (4, 2, 2, False),
    (4, 4, 2, False), (2, 2, 8, False), (4, 2, 8, False), (8, 2, 2, False),
    (4, 1, 2, False), (2, 1, 2, False), (8, 2, 4, False),
]


def bench_shape(M, K, N, dev, trials=5, reps=8):
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.bfloat16, device=dev, generator=g) * 0.1)
    w0 = (torch.randn(N, K, dtype=torch.bfloat16, device=dev, generator=g) * 0.05)
    R = max(4, weight_copies(N * K * 2, dev))
    weights = [w0.clone() for _ in range(R)]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    bench = GraphBench(dev)

    # baseline
    wts = [w.t() for w in weights]
    fb = lambda i: torch.mm(x, wts[i % R], out=out)
    fb(0); torch.cuda.synchronize()  # init cuBLAS handle before capture
    gb = bench.build(fb, R, True)
    for _ in range(3):
        gb.replay()
    torch.cuda.synchronize()
    base = statistics.median([bench.measure(gb, R) for _ in range(trials)])

    print(f"== M{M} K{K} N{N} R={R} cuBLAS {base:8.2f}us ==")
    results = []
    for (rows, unroll, warps, stage) in CFGS:
        try:
            mod = _jit_module(N, K, M, rows, unroll, warps, stage)
        except Exception as e:
            print(f"  cfg r{rows} u{unroll} w{warps} sm{int(stage)}: COMPILE_FAIL {e}")
            continue
        def fc(i, mod=mod, weights=weights, out=out, R=R):
            mod.run(x.view(M, K), weights[i % R], out)
        try:
            fc(0); torch.cuda.synchronize()
            cg = bench.build(fc, R, True)
            for _ in range(3):
                cg.replay()
            torch.cuda.synchronize()
            t = statistics.median([bench.measure(cg, R) for _ in range(trials)])
            results.append((t, (rows, unroll, warps, stage)))
            print(f"  cfg r{rows} u{unroll} w{warps} sm{int(stage)}: {t:8.2f}us "
                  f"x{base/t:.3f} ({bandwidth_gbps(N*K*2, t):5.1f}GB/s)")
        except Exception as e:
            print(f"  cfg r{rows} u{unroll} w{warps} sm{int(stage)}: RUN_FAIL {e}")
    results.sort()
    if results:
        print(f"  BEST {results[0][1]} {results[0][0]:.2f}us x{base/results[0][0]:.3f}")


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.set_grad_enabled(False)
    for (M, K, N) in [(5, 4096, 128), (5, 4096, 288), (8, 4096, 128), (8, 4096, 288)]:
        bench_shape(M, K, N, dev)


if __name__ == "__main__":
    main()
