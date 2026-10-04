// Native bf16 GEMM extension for the GLM-5.3-Flash large-M / extend shapes.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "bf16_gemm.cuh"

namespace {
constexpr int BM = 64, BN = 128, BK = 32, TM = 4, TN = 8;

int launch(const void* A, const void* B, void* C, int M, int N, int K, cudaStream_t stream) {
  if (K % 8 != 0 || N % 2 != 0) return 1;
  dim3 grid((N + BN - 1) / BN);
  dim3 block((BM / TM) * (BN / TN));
  glm53::bf16_gemm_tiled_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
      reinterpret_cast<const glm53::bf16*>(A), reinterpret_cast<const glm53::bf16*>(B),
      reinterpret_cast<glm53::bf16*>(C), M, N, K);
  return static_cast<int>(cudaGetLastError() != cudaSuccess);
}
}  // namespace

void bf16_gemm(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && w.is_cuda());
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16 &&
              w.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous());
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(w.size(1) == K && out.size(0) == M && out.size(1) == N);
  const int rc = launch(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K,
                        c10::cuda::getCurrentCUDAStream(x.get_device()));
  TORCH_CHECK(rc == 0, "bf16_gemm launch failed rc=", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("bf16_gemm", &bf16_gemm);
}
