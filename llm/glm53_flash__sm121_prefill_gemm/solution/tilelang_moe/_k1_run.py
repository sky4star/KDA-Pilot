import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w13_swiglu_kernel import w13_swiglu, build_test

E, N, K, rows, bpe = 144, 2048, 4096, 32768, 3
d = build_test(E, N, K, rows, bpe)
act = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
k = w13_swiglu(E, N, K, rows, bpe, block_K=128)
for _ in range(3):
    k(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
torch.cuda.synchronize()
k(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
torch.cuda.synchronize()
