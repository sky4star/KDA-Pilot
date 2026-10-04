import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang.profiler import do_bench
from w13_swiglu_kernel import w13_swiglu, build_test
from w13_swiglu_packed import w13_swiglu_packed, build_packed

E, N, K, rows = 144, 2048, 4096, 32768
# uniform distribution, bpe=2 (max m_e ~228 <= 256)
d1 = build_test(E, N, K, rows, bpe=2)
act = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
k1 = w13_swiglu(E, N, K, rows, 2, block_K=128)
ms1 = do_bench(lambda: k1(d1["A"], d1["Bg"], d1["Bu"], d1["SFA"], d1["SFBg"], d1["SFBu"], d1["offsets"], d1["scale"], act))
print(f"non-packed bpe=2 (grid 16x288): {ms1:.3f} ms", flush=True)

d2 = build_packed(E, N, K, rows, 3)
MB = int(d2["block_expert"].numel())
k2 = w13_swiglu_packed(E, N, K, rows, MB, block_K=128)
ms2 = do_bench(lambda: k2(d2["A"], d2["Bg"], d2["Bu"], d2["SFA"], d2["SFBg"], d2["SFBu"],
                          d2["block_expert"], d2["block_mstart"], d2["block_actual"], d2["total"], d2["scale"], act))
print(f"packed MAX={MB} total={int(d2['total'][0])} (grid 16x{MB}): {ms2:.3f} ms", flush=True)
