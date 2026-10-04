"""Baseline wrapper: the production FlashInfer CUTLASS fused-MoE call.

Same ABI as `solution/entry.py::moe_forward` so the benchmark can call both arms
with identical inputs. Mirrors
`sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._run_flashinfer_cutlass`
for the ModelOpt NVFP4 path (SWIGLU, finalize, routed scaling applied outside).
"""
from __future__ import annotations

import torch
from flashinfer.fused_moe import cutlass_fused_moe


def moe_forward(a, topk_ids, topk_weights, w13, w2, out_dtype, quant_scales, out=None):
    res = cutlass_fused_moe(
        a,
        topk_ids.to(torch.int),
        topk_weights,
        w13.view(torch.long),
        w2.view(torch.long),
        out_dtype,
        quant_scales=quant_scales,
    )[0]
    if out is not None:
        out.copy_(res)
        return out
    return res
