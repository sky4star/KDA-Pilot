import torch, tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

rows, K, M = 32768, 4096, 4096


def build(atomic):
    @tilelang.jit
    def kern(rows, K, M, block_R=64, block_K=256, threads=256):
        @T.prim_func
        def main(vals: T.Tensor((rows, K), "float32"), tok: T.Tensor((rows,), "int32"),
                 y: T.Tensor((M, K), "float32")):
            with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(rows, block_R), threads=threads) as (bk, br):
                for i, j in T.Parallel(block_R, block_K):
                    r = br * block_R + i
                    c = bk * block_K + j
                    if r < rows:
                        if atomic:
                            T.atomic_add(y[tok[r], c], vals[r, c])
                        else:
                            y[tok[r], c] = vals[r, c]
        return main
    return kern(rows, K, M)


dev = "cuda"
vals = torch.randn(rows, K, device=dev)
tok = torch.randint(0, M, (rows,), device=dev, dtype=torch.int32)
y = torch.zeros(M, K, device=dev)
ka = build(True)
kn = build(False)
print("atomic    ", round(do_bench(lambda: ka(vals, tok, y)), 3), "ms")
print("non-atomic", round(do_bench(lambda: kn(vals, tok, y)), 3), "ms")
