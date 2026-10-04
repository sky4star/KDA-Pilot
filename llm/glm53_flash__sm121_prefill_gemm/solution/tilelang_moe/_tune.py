import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang.profiler import do_bench
from w13_swiglu_kernel import w13_swiglu, build_test as bt13
from w2_gemm_kernel import w2_gemm, build_test as bt2

E, N, K, rows, bpe = 144, 2048, 4096, 32768, 3

print("=== K1 (w13+swiglu) configs ===")
d = bt13(E, N, K, rows, bpe)
act = torch.empty(rows, N, device="cuda", dtype=torch.bfloat16)
for st in (2, 3):
    for th in (128, 256):
        try:
            k = w13_swiglu(E, N, K, rows, bpe, block_K=128, num_stages=st, threads=th)
            ms = do_bench(lambda: k(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act))
            print(f"  stages={st} threads={th}: {ms:.3f} ms")
        except Exception as ex:
            print(f"  stages={st} threads={th}: FAIL {str(ex)[:80]}")

print("=== K3 (w2) configs ===")
d2 = bt2(E, K, N, rows, bpe)
out2 = torch.empty(rows, K, device="cuda", dtype=torch.bfloat16)
for st in (2, 3):
    for th in (128, 256):
        try:
            k = w2_gemm(E, K, N, rows, bpe, block_N=128, num_stages=st, threads=th)
            ms = do_bench(lambda: k(d2["A"], d2["B"], d2["SFA"], d2["SFB"], d2["offsets"], out2))
            print(f"  stages={st} threads={th}: {ms:.3f} ms")
        except Exception as ex:
            print(f"  stages={st} threads={th}: FAIL {str(ex)[:80]}")
