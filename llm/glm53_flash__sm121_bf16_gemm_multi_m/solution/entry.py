"""Candidate: native CUDA tiled bf16 GEMM for the large-M / extend shapes."""
from __future__ import annotations

import torch

from bf16_gemm import bf16_gemm as _gemm


def bf16_gemm(x, w):
    out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    _gemm(out, x.contiguous(), w.contiguous())
    return out


def bf16_gemm_out(x, w, out):
    _gemm(out, x.contiguous(), w.contiguous())
    return out
