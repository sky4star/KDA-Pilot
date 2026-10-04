import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang.profiler import do_bench
from w13_swiglu_packed import w13_swiglu_packed, build_packed

E, N, K, rows = 144, 2048, 4096, 32768
d = build_packed(E, N, K, rows, bpe_cap=3)
MB = int(d["block_expert"].numel())
total = int(d["total"][0])
nb = ((d["sizes"] + 127) // 128)
print(f"E={E} MAX_BLOCKS={MB} total_blocks={total} old_E*bpe={E*3} "
      f"nb[max]={int(nb.max())} saved_frac={1-total/(E*3):.2f}", flush=True)
act = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
k = w13_swiglu_packed(E, N, K, rows, MB, block_K=128)
ms = do_bench(lambda: k(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"],
                        d["block_expert"], d["block_mstart"], d["block_actual"], d["total"], d["scale"], act))
print(f"w13_packed (E=144): {ms:.3f} ms   (non-packed K1 was 7.49 ms)")
