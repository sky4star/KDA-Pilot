"""Candidate-owned CUTLASS SM120 NVFP4 GEMM extension (adapted from KDA-Pilot
qwen38_nvfp4__fp4_w4a4_skinny_gemm; the CUTLASS instantiation is native CUDA).
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent
_spec = importlib.util.find_spec("flashinfer")
if _spec is None or _spec.origin is None:
    raise ImportError("flashinfer's CUTLASS headers are required")
_DATA = Path(_spec.origin).resolve().parent / "data"
_LOCAL_SOURCES = (_ROOT / "cutlass_fp4.cpp", _ROOT / "cutlass_fp4_cuda.cu")
_H = hashlib.sha256()
for _s in _LOCAL_SOURCES:
    _H.update(_s.name.encode()); _H.update(_s.read_bytes())

_extension = load(
    name=f"glm53_cutlass_fp4_sm121_{_H.hexdigest()[:12]}",
    sources=[str(s) for s in _LOCAL_SOURCES],
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

cutlass_fp4_gemm = _extension.cutlass_fp4_gemm
