import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tilelang.profiler import do_bench
from ws_w13 import ws_w13, build_test
E,N,K,rows,bpe=144,4096,4096,32768,3
d=build_test(E,N,K,rows,bpe)
C=torch.empty(rows,N,device='cuda',dtype=torch.bfloat16)
for st in (2,3,4):
  try:
    k=ws_w13(E,N,K,rows,bpe,block_K=128,num_stages=st)
    ms=do_bench(lambda:k(d['A'],d['B'],d['SFA'],d['SFB'],d['offsets'],C))
    print(f"ws_w13 (manual WS) num_stages={st}: {ms:.3f} ms   (non-WS w13 was 8.0-8.5)", flush=True)
  except Exception as ex:
    print(f"num_stages={st} FAIL {str(ex)[:100]}")
