#ifndef SM120_BLOCKSCALED_GEMM_FP4_SHIFT_UTILS_H
#define SM120_BLOCKSCALED_GEMM_FP4_SHIFT_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/container/array.hpp>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/arch/mma_sm120.hpp"
using namespace cute;
namespace FP4Shift
{
//Transform if needed

template<class MMA_Op, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(MMA_Op const& op, Tensor&& tensor) {
    //print("I am fp4 shift A, I just do nothing");
}
template<class MMA_Op, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(MMA_Op const& op, Tensor&& tensor) {
    //print("I am fp4 shift B, I just do nothing");
}

// For SM120 MMA F8F6F4 input fp4, the operand A/B are load from ld.matrix. 
// ld.matrix b4x16_p64 places FP4 data at the first four bits in each
// eight-bit container, whereas MMA F8F6F4 expects the four-bit data to be in 
// the middle of the eight-bit container. Thus, e2m1 operands being fed
// to MMA F8F6F4 must be shifted left by two bits.
// 0b0000ABCD --> 0b00ABCD00
// NOTE: Same transformation is NOT needed for FP6 and FP8.
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    auto tensor_shift = recast<RegisterTypeA>(tensor);
    for(int i = 0; i < size(tensor_shift); i++) tensor_shift(i) <<= 2;
    // cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
template<class AType, class BType, class CType, class SFType, int VS, class Tensor>
CUTLASS_DEVICE void
fp4_shift_B(SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS> const&, Tensor&& tensor) {
  using RegisterTypeB = typename remove_extent<typename
                        SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<AType, BType, CType, SFType, VS>::BRegisters>::type;
  if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
    auto tensor_shift = recast<RegisterTypeB>(tensor);
    for(int i = 0; i < size(tensor_shift); i++) tensor_shift(i) <<= 2;
    // cute::transform(recast<RegisterTypeB>(tensor), [](RegisterTypeB& v){ return v << 2; });
  }
}
}// namespace FP4Shift
#endif //SM120_BLOCKSCALED_GEMM_FP4_SHIFT_UTILS_H