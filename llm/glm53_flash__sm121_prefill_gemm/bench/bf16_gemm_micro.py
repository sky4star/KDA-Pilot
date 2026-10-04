import torch, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=8192)
a_=ap.parse_args(); n=a_.n
a=torch.randn(n,n,device="cuda",dtype=torch.bfloat16)
b=torch.randn(n,n,device="cuda",dtype=torch.bfloat16)
for _ in range(3): torch.mm(a,b)
torch.cuda.synchronize()
torch.mm(a,b); torch.cuda.synchronize()
