from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120/SM121 bf16 GEMV (see csrc/gemm/sm120_bf16_gemv.cuh).
#
# Mirrors the Hopper bf16 GEMV ABI but is enabled on the sm120/sm121 generation
# and covers M in [1, 8] (DSpark speculative verify rows). cuBLAS reaches the
# DRAM ceiling for the mid/large-M dense shapes on GB10, so the wins are the
# shapes where it does not: the wide lm_head (M=1, N=77440 -> ~176 of ~230 GB/s)
# and the tiny-N projections (N in {32,128,288}) where a fixed per-call cost
# dominates.

_CUH = os.path.join(os.path.dirname(__file__), "sm120_bf16_gemv.cuh")
_MAX_K = 32768
_MAX_N = 131072
_SMEM_LIMIT = 48 * 1024  # static shared memory budget for the staged activation


def _config(m: int, n: int, k: int) -> tuple[int, int, int, bool]:
    """(rows_per_warp, k_unroll, num_warps, stage_smem).

    Measured on GB10 (sm121), see docs/dispatch.md:
      * small N (<1024) is latency-bound: skip smem staging, 4 warps -> more
        blocks; staging overhead + low occupancy loses to cuBLAS otherwise.
      * wide N: stage the activation in smem (cuts its L2 reread by #warps).
    """
    if n < 1024:
        return (1, 2, 4, False)
    stage = m * k * 2 <= _SMEM_LIMIT
    if m == 1:
        return (2 if n >= 8192 else 1, 2, 8, stage)
    return (1, 2, 8, stage)


@cache_once
def _jit_module(n: int, k: int, m: int, k_rows: int, k_unroll: int, k_warps: int, stage: bool) -> Module:
    args = make_cpp_args(n, k, m, k_rows, k_unroll, k_warps, stage)
    return load_jit(
        "sm120_bf16_gemv",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("run", f"sglang::Sm120Bf16GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_sm120_bf16_gemv(m: int, n: int, k: int) -> bool:
    if not (
        1 <= m <= 8
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 8 <= n <= _MAX_N
    ):
        return False
    # Latency-bound narrow-N bucket where cuBLAS still wins at high M (measured
    # 0.83x at M=8, N=128): keep the baseline there rather than regress.
    if m >= 6 and 64 <= n <= 192:
        return False
    return True


def sm120_bf16_gemv_out(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """Write out[M, N] = x[M, K] @ w[N, K]^T into a preallocated buffer."""
    m = x.numel() // x.shape[-1]
    k = w.shape[1]
    rows, unroll, warps, stage = _config(m, w.shape[0], k)
    module = _jit_module(w.shape[0], k, m, rows, unroll, warps, stage)
    module.run(x.view(m, k), w, out)


def sm120_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[..., N] = x[..., K] @ w[N, K]^T, all bf16, fp32 accumulation."""
    m = x.numel() // x.shape[-1]
    n = w.shape[0]
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    sm120_bf16_gemv_out(x, w, out.view(m, n))
    return out
