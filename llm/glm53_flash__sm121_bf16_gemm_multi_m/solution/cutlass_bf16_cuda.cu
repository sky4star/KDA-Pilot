/* Copyright 2026 SGLang Team. All Rights Reserved.
   Native CUTLASS SM120 bf16 GEMM for the GLM-5.3-Flash large-M/extend shapes.
   Adapted from the SM120 block-scaled template in KDA-Pilot
   qwen38_nvfp4__fp4_w4a4_skinny_gemm (same CUTLASS 4.x bundled with FlashInfer). */
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace glm53 {

struct Sm120Bf16Config {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _64>;
  using PerSmTileShapeMNK = Shape<_128, _128, _64>;
};

template <typename Config>
struct Bf16GemmSm120 {
  using ElementA = cutlass::bfloat16_t;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::bfloat16_t;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = cutlass::bfloat16_t;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShapeMNK = typename Config::PerSmTileShapeMNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShapeMNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          void, LayoutDTag, AlignmentD,
          ElementD, LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementA, LayoutATag, AlignmentA,
          ElementB, LayoutBTag, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
int run(void* output, const void* input, const void* weight, int m, int n, int k,
        cudaStream_t stream) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<ElementA const*>(input), stride_a,
       static_cast<ElementB const*>(weight), stride_b},
      {{}, nullptr, stride_d, static_cast<ElementD*>(output), stride_d}};
  arguments.scheduler.max_swizzle_size = 4;

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (ws != 0 && cudaMallocAsync(&workspace, ws, stream) != cudaSuccess) return 1000;
  auto status = gemm.can_implement(arguments);
  if (status == cutlass::Status::kSuccess) status = gemm.initialize(arguments, workspace, stream);
  if (status == cutlass::Status::kSuccess) status = gemm.run(arguments, workspace, stream);
  if (workspace) cudaFreeAsync(workspace, stream);
  return static_cast<int>(status);
}

}  // namespace glm53

int launch_cutlass_bf16_gemm(void* output, const void* input, const void* weight,
                             int m, int n, int k, cudaStream_t stream) {
  using Gemm = glm53::Bf16GemmSm120<glm53::Sm120Bf16Config>::Gemm;
  return glm53::run<Gemm>(output, input, weight, m, n, k, stream);
}
