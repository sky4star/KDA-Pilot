import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_baseline import build, call  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
E = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 144
topk = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8
p = build(M, E, topk)
for _ in range(3):
    call(p)
torch.cuda.synchronize()
call(p)
torch.cuda.synchronize()
