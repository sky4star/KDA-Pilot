"""Candidate-owned CUTLASS SM120 bf16 GEMM extension (native CUDA C++)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_spec = importlib.util.find_spec("flashinfer")
if _spec is None or _spec.origin is None:
    raise ImportError("flashinfer's bundled CUTLASS headers are required")
_DATA = Path(_spec.origin).resolve().parent / "data"
_SRC = (_ROOT / "cutlass_bf16.cpp", _ROOT / "cutlass_bf16_cuda.cu")
_H = hashlib.sha256()
for _s in _SRC:
    _H.update(_s.name.encode()); _H.update(_s.read_bytes())

_ext = load(
    name=f"glm53_bf16_sm121_{_H.hexdigest()[:12]}",
    sources=[str(s) for s in _SRC],
    extra_include_paths=[
        str(_DATA / "cutlass/include"),
        str(_DATA / "cutlass/tools/util/include"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_121a,code=sm_121a",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
)

cutlass_bf16_gemm = _ext.cutlass_bf16_gemm
