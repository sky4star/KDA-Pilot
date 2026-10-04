#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

int launch_cutlass_bf16_gemm(void* output, const void* input, const void* weight,
                             int m, int n, int k, cudaStream_t stream);

void cutlass_bf16_gemm(torch::Tensor output, torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda());
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && output.is_contiguous());
  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(weight.size(0));
  TORCH_CHECK(weight.size(1) == k);
  TORCH_CHECK(output.size(0) == m && output.size(1) == n);
  const int status = launch_cutlass_bf16_gemm(
      output.data_ptr(), input.data_ptr(), weight.data_ptr(), m, n, k,
      c10::cuda::getCurrentCUDAStream(input.get_device()));
  TORCH_CHECK(status == 0, "CUTLASS SM120 BF16 GEMM failed with status ", status);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cutlass_bf16_gemm", &cutlass_bf16_gemm);
}
