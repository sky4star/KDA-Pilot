import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang.profiler import do_bench
from w13_swiglu_ws import w13_swiglu_ws, build_test
E,N,K,rows,bpe=144,2048,4096,32768,3
d=build_test(E,N,K,rows,bpe); act=torch.empty(rows,N,device='cuda',dtype=torch.bfloat16)
k=w13_swiglu_ws(E,N,K,rows,bpe,block_K=128)
ms=do_bench(lambda:k(d['A'],d['Bg'],d['Bu'],d['SFA'],d['SFBg'],d['SFBu'],d['offsets'],d['scale'],act))
print(f"K1 auto-WS + reg hints: {ms:.3f} ms   (WS-off was 7.49)", flush=True)
