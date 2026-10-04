// SM121 bf16 GEMM for tall-skinny large-M extend shapes:
//   C[M, N] = A[M, K] @ B[N, K]^T
//
// Design: each block owns a full M tile (BM >= max M) and a BN-wide slice of N,
// so every weight element it loads is reused across all M rows -> the weight
// (N*K*2 bytes) is read from DRAM exactly once. K is streamed in BK chunks with
// double buffering; both operands are staged in shared memory and consumed with
// a register tile (TM x TN). This is the large-M sibling of sm120_bf16_gemv:
// GEMV keeps acc[kRows][M] in registers, which explodes past M~8; here we tile
// M and stage the weight so the reuse is preserved for M in [9, 64].
#include <cuda_bf16.h>
#include <cstdint>

namespace glm53 {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float ld_bf16_f32(const bf16* p) {
  return __bfloat162float(*p);
}

// BM: M tile (covers M<=BM), BN: N tile, BK: K step, TM/TN: per-thread tile.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
bf16_gemm_tiled_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                       bf16* __restrict__ C, int M, int N, int K) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  __shared__ bf16 As[BM][BK];
  __shared__ bf16 Bs[BN][BK];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);         // N group
  const int ty = tid / (BN / TN);          // M group
  const int block_n = blockIdx.x * BN;
  const int m0 = ty * TM;
  const int n0 = tx * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  // Vectorized cooperative loads: BK bf16 = BK/8 float4 per row.
  constexpr int VEC = 8;                   // bf16 per 16B
  constexpr int NV = BK / VEC;
  const int ld_rows_a = BM, ld_rows_b = BN;
  const int total_vec_a = ld_rows_a * NV;
  const int total_vec_b = ld_rows_b * NV;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- load A tile [BM, BK] ----
    for (int t = tid; t < total_vec_a; t += THREADS) {
      const int row = t / NV, v = t % NV, kk = k0 + v * VEC;
      const bf16* src = A + (int64_t)(row < M ? row : 0) * K + kk;
      bf16 tmp[VEC];
      if (row < M && kk + VEC <= K) {
        *reinterpret_cast<float4*>(tmp) = *reinterpret_cast<const float4*>(src);
      } else {
#pragma unroll
        for (int e = 0; e < VEC; ++e) tmp[e] = (row < M && kk + e < K) ? src[e] : __float2bfloat16(0.f);
      }
      *reinterpret_cast<float4*>(&As[row][v * VEC]) = *reinterpret_cast<float4*>(tmp);
    }
    // ---- load B tile [BN, BK] (weights, N-major rows) ----
    for (int t = tid; t < total_vec_b; t += THREADS) {
      const int row = t / NV, v = t % NV, kk = k0 + v * VEC;
      const bf16* src = B + (int64_t)(block_n + row < N ? (block_n + row) : 0) * K + kk;
      bf16 tmp[VEC];
      if (block_n + row < N && kk + VEC <= K) {
        *reinterpret_cast<float4*>(tmp) = *reinterpret_cast<const float4*>(src);
      } else {
#pragma unroll
        for (int e = 0; e < VEC; ++e) tmp[e] = (block_n + row < N && kk + e < K) ? src[e] : __float2bfloat16(0.f);
      }
      *reinterpret_cast<float4*>(&Bs[row][v * VEC]) = *reinterpret_cast<float4*>(tmp);
    }
    __syncthreads();

    // ---- compute ----
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = ld_bf16_f32(&As[m0 + i][k]);
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = ld_bf16_f32(&Bs[n0 + j][k]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      if (m0 + i < M && n0 + j < N)
        C[(int64_t)(m0 + i) * N + block_n + n0 + j] = __float2bfloat16(acc[i][j]);
}

}  // namespace glm53
