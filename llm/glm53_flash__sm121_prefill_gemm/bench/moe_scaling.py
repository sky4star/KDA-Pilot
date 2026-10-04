#!/usr/bin/env python3
"""MoE throughput vs M: build weights once, vary activations/topk."""
from __future__ import annotations
import statistics, sys
from pathlib import Path
import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12x_probe import b12x  # noqa: E402

N, K, BLK = 2048, 4096, 16
FLOAT8_E4M3_MAX, FLOAT4_E2M1_MAX = 448.0, 6.0


def round_up(x, y):
    return (x + y - 1) // y * y


def build_weights(E, seed=7):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    w1_q = torch.empty((E, 2 * N, K // 2), device=dev, dtype=torch.uint8)
    w2_q = torch.empty((E, K, N // 2), device=dev, dtype=torch.uint8)
    w1_bs = torch.empty((E, round_up(2 * N, 128), round_up(K // BLK, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w2_bs = torch.empty((E, round_up(K, 128), round_up(N // BLK, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w1_gs = torch.empty((E,), device=dev, dtype=torch.float32)
    w2_gs = torch.empty((E,), device=dev, dtype=torch.float32)
    gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (2.0 / 10.0)
    for e in range(E):
        w1 = torch.randn((2 * N, K), device=dev, dtype=torch.bfloat16, generator=g) / 10
        w2 = torch.randn((K, N), device=dev, dtype=torch.bfloat16, generator=g) / 10
        w1_gs[e] = gs; w2_gs[e] = gs
        w1_q[e], w1_bs[e] = fp4_quantize(w1, w1_gs[e])
        w2_q[e], w2_bs[e] = fp4_quantize(w2, w2_gs[e])
        del w1, w2
    a_gs = torch.ones((E,), device=dev, dtype=torch.float32)
    return dict(w1_q=w1_q, w2_q=w2_q, w1_bs=w1_bs, w2_bs=w2_bs,
                w1_gs=w1_gs, w2_gs=w2_gs, a_gs=a_gs)


def build_act(M, E, topk, seed):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    a = (torch.randn((M, K), device=dev, dtype=torch.bfloat16, generator=g) / 10)
    score = torch.randn((M, E), device=dev, dtype=torch.bfloat16, generator=g)
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)
    return a, tw, ti.to(torch.int)


def baseline(p):
    return cutlass_fused_moe(
        p["a"], p["ti"], p["tw"], p["w1_q"].view(torch.long), p["w2_q"].view(torch.long),
        p["a"].dtype,
        quant_scales=[p["a_gs"], p["w1_bs"].view(torch.int32), (1 / p["w1_gs"]),
                      p["a_gs"], p["w2_bs"].view(torch.int32), (1 / p["w2_gs"])],
    )[0]


def timed(fn, iters, warmup=3):
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
    E = 144; topk = 8
    W = build_weights(E)
    print(f"{torch.cuda.get_device_name(0)} E={E} topk={topk}")
    print(f"{'M':>7} {'M_e':>6} {'base_us':>10} {'TFLOPs':>8} {'GB/s_w':>8} {'b12x_us':>9} {'TFLOPs':>8}")
    for M in [512, 1024, 2048, 4096, 8192, 16384]:
        a, tw, ti = build_act(M, E, topk, 100 + M)
        p = dict(W, a=a, tw=tw, ti=ti)
        try:
            base_us = timed(lambda: baseline(p), 15)
        except Exception as ex:
            print(f"{M:7d} baseline FAILED {str(ex)[:60]}"); continue
        flops = 2.0 * M * topk * (K * (2 * N) + N * K)
        wt = M * topk * (2 * N * (K // 2) + K * (N // 2)) / E  # only if all experts used; approx
        # expert weight bytes actually resident E*(w1+w2)
        wt_bytes = E * (2 * N * (K // 2) + K * (N // 2))
        try:
            g_us = timed(lambda: b12x(p, E, topk), 15)
            gfl = f"{flops/(g_us*1e-6)/1e12:8.1f}"
        except Exception as ex:
            g_us = float('nan'); gfl = f"FAIL {str(ex)[:20]}"
        print(f"{M:7d} {M*topk//E:6d} {base_us:10.1f} {flops/(base_us*1e-6)/1e12:8.1f} "
              f"{wt_bytes/(base_us*1e-6)/1e9:8.0f} {g_us:9.1f} {gfl}")


if __name__ == "__main__":
    main()
