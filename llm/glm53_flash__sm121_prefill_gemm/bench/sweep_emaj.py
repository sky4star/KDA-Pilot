#!/usr/bin/env python3
"""Expert-major rasterization variant: grid (local, bn, e) so consecutive CTAs
reuse the same (expert, bn) B tile across the expert's M-tiles -> B stays in L2.
Bench at production shapes, correctness vs dequant oracle.
"""
import gc
import sys
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

sys.path.insert(0, "/kda/llm/glm53_flash__sm121_prefill_gemm/solution/tilelang_moe")
sys.path.insert(0, "/tmp")
from tl_tile_sweep import pack_kernel_order  # noqa
from sweep_realwidth import build, verify_sample  # noqa

DEV = "cuda"


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def grouped_gemm_emaj(E, N, K, rows, blocks_per_expert, block_M=128, block_N=128,
                      block_K=128, num_stages=2, threads=128, out_dtype=T.bfloat16):
    assert block_M % 32 == 0 and block_N % 32 == 0 and block_K % 64 == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // block_N
    in_dtype = T.float4_e2m1fn
    total_m_blocks = E * blocks_per_expert

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        B: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((total_m_blocks * k_blocks * block_M, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * block_N, words), T.uint32),
        expert_offsets: T.Tensor((E + 1,), T.int32),
        C: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(blocks_per_expert, n_blocks, E, threads=threads) as (local, bn, e):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_s = T.alloc_shared((block_M, words), T.uint32)
            SFB_s = T.alloc_shared((block_N, words), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)

            bm = e * blocks_per_expert + local
            m_start = expert_offsets[e] + local * block_M
            actual = T.max(0, T.min(block_M, expert_offsets[e + 1] - m_start))

            T.clear(C_l)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K], A_s)
                T.copy(B[e, bn * block_N:(bn + 1) * block_N, ko * block_K:(ko + 1) * block_K], B_s)
                for r, w in T.Parallel(block_M, words):
                    SFA_s[r, w] = SFA[(bm * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, words):
                    SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * block_N + r, w]
                T.mma_gemm_blockscaled(
                    A_s, B_s, C_l, SFA_s, SFB_s,
                    transpose_B=True, clear_accum=False,
                    k_start=ko * block_K,
                    sf_a_granularity_k=16, sf_b_granularity_k=16,
                    sf_layout="blockscaled_chunk_kmajor",
                )
            for i, j in T.Parallel(block_M, block_N):
                if i < actual:
                    C[m_start + i, bn * block_N + j] = C_l[i, j]

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    refs = {(4096, 4096): {32768: 8.03, 65536: 13.05}, (4096, 2048): {32768: 5.8, 65536: 7.26}}
    for (N, K) in [(4096, 4096), (4096, 2048)]:
        gname = "w13" if K == 4096 else "w2"
        for rows in (65536, 32768):
            print(f"\n#### EMAJ {gname} N={N} K={K} rows={rows} — flashinfer ref {refs[(N,K)][rows]} ms", flush=True)
            for BM, BN, thr, bk in [(128, 128, 128, 128), (64, 128, 128, 128), (128, 128, 128, 256)]:
                try:
                    E = 144
                    d = build(E, N, K, rows, BM, BN, bk)
                    kern = grouped_gemm_emaj(E, N, K, d["rows_pad"], d["bpe"], block_M=BM, block_N=BN,
                                             block_K=bk, num_stages=2, threads=thr)
                    out = torch.empty((d["rows_pad"], N), device=DEV, dtype=torch.bfloat16)
                    kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], out)
                    torch.cuda.synchronize()
                    rel = verify_sample(d, E, N, K, rows, out, [0, 70, 143])
                    if rel > 2e-2:
                        print(f"[BM={BM} BN={BN} BK={bk}] CORRECTNESS FAIL rel={rel:.4f}", flush=True)
                        continue
                    ms = do_bench(lambda: kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], out))
                    fl = 2.0 * rows * N * K
                    print(f"[BM={BM:3d} BN={BN:3d} thr={thr} BK={bk}] {ms:7.2f} ms  "
                          f"{fl/ms/1e9:5.0f} TFLOPS  rel={rel:.4f}", flush=True)
                    del d, kern, out
                    gc.collect(); torch.cuda.empty_cache()
                except Exception as ex:
                    print(f"[BM={BM} BN={BN} BK={bk}] FAILED: {str(ex).splitlines()[0][:120]}", flush=True)
                    gc.collect(); torch.cuda.empty_cache()
