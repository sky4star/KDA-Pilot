"""Correctness gate for the NVFP4 MoE task.

Primary comparison is candidate vs the copied production baseline
(`baseline/entry.py`) on identical inputs, in the dequant domain, with the
tolerance from SGLang's `test_fp4_moe` (atol=2e-2, rtol=1e-2). Also checks NaN/Inf
and that poisoning the output buffer is overwritten.
"""
from __future__ import annotations

import torch


def check_outputs(ref, cand, atol=2e-2, rtol=1e-2):
    assert ref.shape == cand.shape, (ref.shape, cand.shape)
    if torch.isnan(cand).any() or torch.isinf(cand).any():
        return {"ok": False, "msg": "candidate NaN/Inf"}
    r = ref.float()
    c = cand.float()
    d = (c - r).abs()
    tol = atol + rtol * r.abs()
    ok = bool(torch.all(d <= tol).item())
    return {"ok": ok, "max_abs": float(d.max().item()),
            "max_rel": float((d / (r.abs() + 1e-6)).max().item()),
            "ref_absmax": float(r.abs().max().item()),
            "n_violations": int((d > tol).sum().item()), "n": d.numel()}
