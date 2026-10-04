"""Baseline wrapper: the production bf16 path for M>8 (cuBLAS via torch.mm).

For the GLM-5.3-Flash 42-token extend the bf16 linears fall through
`UnquantizedLinearMethod.apply` -> `F.linear` -> cuBLAS, since the SM121 bf16
GEMV fast path only covers M<=8. This wrapper exposes the same ABI as the
candidate: `bf16_gemm(x, w) -> out`, out = x[M,K] @ w[N,K]^T, bf16 in/out.
"""
from __future__ import annotations

import torch


def bf16_gemm(x, w):
    return x @ w.t()


def bf16_gemm_out(x, w, out):
    torch.mm(x, w.t(), out=out)
    return out
