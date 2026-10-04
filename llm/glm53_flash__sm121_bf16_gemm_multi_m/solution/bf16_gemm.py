"""Candidate: native CUDA tiled bf16 GEMM for the large-M / extend shapes."""
from __future__ import annotations

import hashlib
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_SRC = [_ROOT / "bf16_gemm_ext.cu", _ROOT / "bf16_gemm.cuh"]
_H = hashlib.sha256()
for _s in _SRC:
    _H.update(_s.name.encode()); _H.update(_s.read_bytes())

_ext = load(
    name=f"glm53_bf16_gemm_{_H.hexdigest()[:12]}",
    sources=[str(_SRC[0])],
    extra_include_paths=[str(_ROOT)],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "--expt-relaxed-constexpr"],
)

bf16_gemm = _ext.bf16_gemm
