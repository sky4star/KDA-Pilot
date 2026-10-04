import torch, tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_w13 import build_test
from grouped_nvfp4_gemm import pack_scale_words, decode_fp4, decode_scale_words


@tilelang.jit
def ws_w13_s1(E: int, N: int, K: int, rows: int, bpe: int, block_K: int = 128, threads: int = 256):
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // 128
    BM = 128; BN = 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(A: T.Tensor((rows, K), in_dtype), B: T.Tensor((E, N, K), in_dtype),
             SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
             SFB: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
             offsets: T.Tensor((E + 1,), T.int32), C: T.Tensor((rows, N), T.bfloat16)):
        with T.Kernel(n_blocks, E * bpe, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((BM, block_K), in_dtype)
            B_s = T.alloc_shared((BN, block_K), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFB_s = T.alloc_shared((BN, words), T.uint32)
            C_l = T.alloc_fragment((BM, BN), T.float32)
            data_ready = T.alloc_barrier(arrive_count=128)
            compute_done = T.alloc_barrier(arrive_count=128)
            e = bm // bpe
            local = bm % bpe
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))
            with T.ws(0):
                T.clear(C_l)
            for ko in range(k_blocks):
                with T.ws(1):
                    T.barrier_wait(compute_done, (ko + 1) % 2)
                    T.tma_copy(A[m_start:m_start + BM, ko * block_K:(ko + 1) * block_K], A_s, barrier=data_ready)
                    T.tma_copy(B[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], B_s, barrier=data_ready)
                    for r, w in T.Parallel(BM, words):
                        SFA_s[r, w] = SFA[((e * bpe + local) * k_blocks + ko) * BM + r, w]
                    for r, w in T.Parallel(BN, words):
                        SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                    T.barrier_arrive(data_ready)
                with T.ws(0):
                    T.barrier_wait(data_ready, ko % 2)
                    T.mma_gemm_blockscaled(A_s, B_s, C_l, SFA_s, SFB_s, transpose_B=True,
                                           clear_accum=False, k_start=ko * block_K,
                                           sf_a_granularity_k=16, sf_b_granularity_k=16,
                                           sf_layout="blockscaled_chunk_kmajor")
                    T.barrier_arrive(compute_done)
            with T.ws(0):
                for i, j in T.Parallel(BM, BN):
                    if i < actual:
                        C[m_start + i, bn * BN + j] = T.Cast(T.bfloat16, C_l[i, j])
    return main


E, N, K, rows, bpe = 8, 256, 512, 1024, 2
dev = "cuda"
d = build_test(E, N, K, rows, bpe)
C = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
k = ws_w13_s1(E, N, K, rows, bpe)
k(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], C)
torch.cuda.synchronize()
Ad = decode_fp4(d["A"], rows, K) * decode_scale_words(pack_scale_words(d["A_sf_unpad"]), K).repeat_interleave(16, 1)
Bd = decode_fp4(d["B"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
    decode_scale_words(pack_scale_words(d["sfb_sem"]), K).reshape(E, N, K // 16).repeat_interleave(16, 2)
ref = torch.empty(rows, N, device=dev, dtype=torch.float32)
for e in range(E):
    s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
    if s1 > s0:
        ref[s0:s1] = Ad[s0:s1] @ Bd[e].t()
e_=(C.float()-ref).abs(); print("ws_w13_s1 verify max_abs:", e_.max().item(), "ref_absmax:", ref.abs().max().item(), "max_rel:", (e_.max()/(ref.abs().max()+1e-6)).item())
