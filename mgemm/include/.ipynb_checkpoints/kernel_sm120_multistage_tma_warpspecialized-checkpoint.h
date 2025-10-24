#ifndef SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_WARPSPECIALIZED_H
#define SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_WARPSPECIALIZED_H

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

using namespace cute;
template<
        class LayoutA, class SmemLayoutA, class TmaLoadA, class SmemCopyAtomA,
        class LayoutB, class SmemLayoutB, class TmaLoadB, class SmemCopyAtomB,
        class LayoutSFA, class SmemLayoutSFA, class TmaLoadSFA, class SmemCopyAtomSFA,
        class LayoutSFB, class SmemLayoutSFB, class TmaLoadSFB, class SmemCopyAtomSFB,
        class LayoutC, class SmemLayoutC, class TmaStoreC, class SmemCopyAtomC,
        class MMA, class ProbShape_MNK, class TileShape_MNK, class MainloopSharedStorage, class EpilogueSharedStorage,
        class MainloopPipeline, class PipelineState, int N_STAGE, int TmaTransactionBytes>
//__launch_bounds__(256, 1)
__global__
void
gemm_device_multistage_warpspecialized(__grid_constant__ const TmaLoadA tma_load_a,
                                       LayoutA layout_A,
                                       __grid_constant__ const TmaLoadB tma_load_b,
                                       LayoutB layout_B,
                                       __grid_constant__ const TmaLoadSFA tma_load_sfa,
                                       LayoutSFA layout_SFA,
                                       __grid_constant__ const TmaLoadSFB tma_load_sfb,
                                       LayoutSFB layout_SFB,
                                       __grid_constant__ const TmaStoreC tma_store_c,
                                       LayoutC layout_C) {

    enum class WarpGroupRole {
        Producer = 0,
        Consumer = 1,
    };
    enum class ProducerWarpRole {
        MainloopEpilogue = 0,
        Warp1 = 1,
        Warp2 = 2,
        Warp3 = 3
    };

    //Smem Alloc
    extern __shared__ uint8_t smem_ptr[];
    MainloopSharedStorage &mainloop_shared_storage = *reinterpret_cast<MainloopSharedStorage *>(smem_ptr);
    EpilogueSharedStorage &epilogue_shared_storage = *reinterpret_cast<EpilogueSharedStorage *>(smem_ptr);
    auto const block_thread_num = blockDim.x;
    assert(block_thread_num == 2 * cutlass::NumThreadsPerWarpGroup);

    int thread_idx = int(threadIdx.x);
    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();

    auto const block_x = blockIdx.x;
    auto const block_y = blockIdx.y;

    if ((warp_idx == 0) && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_sfb.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_c.get_tma_descriptor());
    }

    Tensor mA = tma_load_a.get_tma_tensor(shape(layout_A));             // (M, K)
    Tensor mB = tma_load_b.get_tma_tensor(shape(layout_B));             // (N, K)
    Tensor mSFA = tma_load_sfa.get_tma_tensor(shape(layout_SFA));       // (M, K)
    Tensor mSFB = tma_load_sfb.get_tma_tensor(shape(layout_SFB));       // (N, K)

    Tensor mC = tma_store_c.get_tma_tensor(shape(layout_C));

    auto const block_coord = make_coord(block_x, block_y, _);
    Tensor gA = local_tile(mA, TileShape_MNK{}, block_coord, Step < _1, X, _1 > {});        // (BLK_M, BLK_K, k_tile)
    Tensor gB = local_tile(mB, TileShape_MNK{}, block_coord, Step < X, _1, _1 > {});        // (BLK_N, BLK_K, k_tile)
    Tensor gSFA = local_tile(mSFA, TileShape_MNK{}, block_coord, Step < _1, X, _1 > {});    // (BLK_M, BLK_K, k_tile)
    Tensor gSFB = local_tile(mSFB, TileShape_MNK{}, block_coord, Step < X, _1, _1 > {});    // (BLK_N, BLK_K, k_tile)
    Tensor gC = local_tile(mC, TileShape_MNK{}, block_coord, Step < _1, _1, X > {});        // (BLK_M, BLK_N)


    auto &mainloop_shared_tensors = mainloop_shared_storage.tensors;
    auto &epilogue_shared_tensors = epilogue_shared_storage.tensors;
    Tensor sA = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});    // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});    // (BLK_N,BLK_K,PIPE)
    Tensor sC = make_tensor(make_smem_ptr(epilogue_shared_tensors.smem_C.begin()), SmemLayoutC{});          // (BLK_M, BLK_N)

    /*if(1 && thread0() && block_x == 0 && block_y == 0)
    {

        print("tma_load_a:      "); print(tma_load_a); print('\n');
        print("tma_load_b:      "); print(tma_load_b); print('\n');
        print("tma_load_sfa:    "); print(tma_load_sfa); print('\n');
        print("tma_load_sfb:    "); print(tma_load_sfb); print('\n');
        print('\n');

        print("LayoutA:      "); print(layout_A); print('\n');
        print("LayoutB:      "); print(layout_B); print('\n');
        print("LayoutSFA:    "); print(layout_SFA); print('\n');
        print("LayoutSFB:    "); print(layout_SFB); print('\n');
        print("LayoutC:      "); print(layout_C); print('\n');
        print('\n');

        print("SmemLayoutA:      "); print(SmemLayoutA{}); print('\n');
        print("SmemLayoutB:      "); print(SmemLayoutB{}); print('\n');
        print("SmemLayoutSFA:    "); print(SmemLayoutSFA{}); print('\n');
        print("SmemLayoutSFB:    "); print(SmemLayoutSFA{}); print('\n');
        print('\n');

        print("mA:      "); print(mA); print('\n');
        print("mB:      "); print(mB); print('\n');
        print("mSFA:    "); print(mSFA); print('\n');
        print("mSFB:    "); print(mSFB); print('\n');
        print('\n');

        print("gA:      "); print(gA); print('\n');
        print("gB:      "); print(gB); print('\n');
        print("gSFA:    "); print(gSFA); print('\n');
        print("gSFB:    "); print(gSFB); print('\n');
        print('\n');

        print("sA:      "); print(sA); print('\n');
        print("sB:      "); print(sB); print('\n');
        print("sSFA:    "); print(sSFA); print('\n');
        print("sSFB:    "); print(sSFB); print('\n');
        print('\n');

        print("tAgA:    "); print(tAgA); print('\n');
        print("tBgB:    "); print(tBgB); print('\n');
        print("tAgSFA:  "); print(tAgSFA); print('\n');
        print("tBgSFB:  "); print(tBgSFB); print('\n');
        print('\n');

        print("tAsA:    "); print(tAsA); print('\n');
        print("tBsB:    "); print(tBsB); print('\n');
        print("tAsSFA:  "); print(tAsSFA); print('\n');
        print("tBsSFB:  "); print(tBsSFB); print('\n');
        print('\n');

        print("tCsA:    "); print(tCsA); print('\n');
        print("tCsB:    "); print(tCsB); print('\n');
        print("tCsSFA:  "); print(tCsSFA); print('\n');
        print("tCsSFB:  "); print(tCsSFB); print('\n');
        print('\n');

        print("tCrA_copy_view:    "); print(tCrA_copy_view); print('\n');
        print("tCrB_copy_view:    "); print(tCrB_copy_view); print('\n');
        print("tCrSFA_copy_view:  "); print(tCrSFA_copy_view); print('\n');
        print("tCrSFB_copy_view:  "); print(tCrSFB_copy_view); print('\n');
        print('\n');

        print("tCrA:    "); print(tCrA); print('\n');
        print("tCrB:    "); print(tCrB); print('\n');
        print("tCrC:    "); print(tCrC); print('\n');
        print("tCrSFA:  "); print(tCrSFA); print('\n');
        print("tCrSFB:  "); print(tCrSFB); print('\n');
        print('\n');

    }*/

    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
    // TMA Params set
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.transaction_bytes = TmaTransactionBytes;
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup; //size(MMA{});

    //SM120 No Cluster
    auto cluster_shape = Shape < _1, _1, _1>{};
    MainloopPipeline pipeline(mainloop_shared_storage.pipeline_storage, mainloop_pipeline_params,
                              cluster_shape);

    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read;

    int num_k_tile = size<2>(gA);

    // Block sync before pipeline start, ensure all thread's pipeline has initialized
    // Or may happen that the consumer calls the consumer_wait
    // before the producer's pipeline initialized
    __syncthreads(); // <------- There must be a block barrier

    // Producer TMA Load
    if (warp_group_role == WarpGroupRole::Producer) {

        if(producer_warp_role == ProducerWarpRole::MainloopEpilogue) {

            auto block_tma_a = tma_load_a.get_slice(0);
            auto block_tma_b = tma_load_b.get_slice(0);

            auto block_tma_sfa = tma_load_sfa.get_slice(0);
            auto block_tma_sfb = tma_load_sfb.get_slice(0);

            // Partition source and destination tensors for tma copies
            Tensor tAgA = block_tma_a.partition_S(gA);                   // (TMA,TMA_M,TMA_K,k)
            Tensor tAsA = block_tma_a.partition_D(sA);                   // (TMA,TMA_M,TMA_K,PIPE)

            Tensor tBgB = block_tma_b.partition_S(gB);                   // (TMA,TMA_N,TMA_K,k)
            Tensor tBsB = block_tma_b.partition_D(sB);                   // (TMA,TMA_N,TMA_K,PIPE)

            Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);             // (TMA,TMA_M,TMA_K,k)
            Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);             // (TMA,TMA_M,TMA_K,PIPE)

            Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);             // (TMA,TMA_N,TMA_K,k)
            Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);             // (TMA,TMA_N,TMA_K,PIPE)


            static_assert(rank(gSFA) == Int<3>{}, "gSFA rank should be 3 (BLK_M, BLK_K, k_tile)");
            static_assert(rank(sSFA) == Int<3>{}, "sSFA rank should be 3 (BLK_M, BLK_K, PIPE)");
            static_assert(rank(tAgSFA) == Int<4>{}, "tAgSFA rank should be 4 (TMA, TMA_M, TMA_K, k)");
            static_assert(rank(tAsSFA) == Int<4>{}, "tAsSFA rank should be 4 (TMA, TMA_M, TMA_K, PIPE)");

            static_assert(rank(gSFB) == Int<3>{}, "gSFB rank should be 3 (BLK_N, BLK_K, k_tile)");
            static_assert(rank(sSFB) == Int<3>{}, "sSFB rank should be 3 (BLK_N, BLK_K, PIPE)");
            static_assert(rank(tBgSFB) == Int<4>{}, "tBgSFB rank should be 4 (TMA, TMA_N, TMA_K, k)");
            static_assert(rank(tBsSFB) == Int<4>{}, "tBsSFB rank should be 4 (TMA, TMA_N, TMA_K, PIPE)");

            CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sA));                 // PIPE
            CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sB));                 // PIPE
            CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));                  // PIPE
            CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));                  // PIPE

            if (lane_predicate) {
                // Mainloop tiles load
                for (int i = 0; i < num_k_tile; i++) {

                    pipeline.producer_acquire(smem_pipe_write);

                    BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);

                    int write_stage = smem_pipe_write.index();
                    copy(tma_load_a.with(*tmaBar), tAgA(_, _, _, i), tAsA(_, _, _, write_stage));
                    copy(tma_load_b.with(*tmaBar), tBgB(_, _, _, i), tBsB(_, _, _, write_stage));
                    copy(tma_load_sfa.with(*tmaBar), tAgSFA(_, _, _, i), tAsSFA(_, _, _, write_stage));
                    copy(tma_load_sfb.with(*tmaBar), tBgSFB(_, _, _, i), tBsSFB(_, _, _, write_stage));

                    ++smem_pipe_write;
                }
            }
            __syncwarp();
        }
    }
    else if(warp_group_role == WarpGroupRole::Consumer) // Consumer MMA
    {
        MMA tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

        // Allocate fragments and descriptors
        Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));      // (MMA,MMA_M,MMA_K)
        Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));      // (MMA,MMA_N,MMA_K)
        Tensor tCrC = thread_mma.partition_fragment_C(gC);                      // (MMA,MMA_M,MMA_N)
        Tensor tCgC = thread_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

        // clear accumulator
        clear(tCrC);

        Tensor tCrSFA = partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);   // (MMA,MMA_M,MMA_K)
        Tensor tCrSFB = partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);   // (MMA,MMA_N,MMA_K)

        // A
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);
        Tensor tCsA = smem_thr_copy_A.partition_S(
                as_position_independent_swizzle_tensor(sA));        // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);     // (CPY,CPY_M,CPY_K)

        // B
        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_group_thread_idx);
        Tensor tCsB = smem_thr_copy_B.partition_S(
                as_position_independent_swizzle_tensor(sB));        // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);     // (CPY,CPY_M,CPY_K)

        // SFA
        auto tile_shape_mnk = tile_shape(tiled_mma);
        auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                        get_layoutSFA_TV(tiled_mma),
                                                        make_shape(size<0>(tile_shape_mnk),
                                                                   size<2>(tile_shape_mnk))
        );
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(warp_group_thread_idx);
        Tensor tCsSFA = smem_thr_copy_SFA.partition_S(
                as_position_independent_swizzle_tensor(sSFA));           // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);    // (CPY,CPY_M,CPY_K)

        // SFB
        auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                        get_layoutSFB_TV(tiled_mma),
                                                        make_shape(size<1>(tile_shape_mnk),
                                                                   size<2>(tile_shape_mnk))
        );
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(warp_group_thread_idx);
        Tensor tCsSFB = smem_thr_copy_SFB.partition_S(
                as_position_independent_swizzle_tensor(sSFB));            // (CPY,CPY_N,CPY_K,PIPE)
        Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);     // (CPY,CPY_N,CPY_K)

        // C
        auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(warp_group_thread_idx);
        Tensor tCrC_copy_view = smem_thr_copy_C.retile_S(tCrC);
        Tensor tCsC_r2s = smem_thr_copy_C.partition_D(sC);

        auto block_tma_c = tma_store_c.get_slice(0);
        Tensor tCsC_s2g = block_tma_c.partition_S(sC);
        Tensor tCgC_s2g = block_tma_c.partition_D(gC);
        if(0 && warp_group_thread_idx == 0)
        {
            print("gA:      "); print(gA); print('\n');
            print("gB:      "); print(gB); print('\n');
            print("gSFA:    "); print(gSFA); print('\n');
            print("gSFB:    "); print(gSFB); print('\n');
            print('\n');

            print("sA:      "); print(sA); print('\n');
            print("sB:      "); print(sB); print('\n');
            print("sSFA:    "); print(sSFA); print('\n');
            print("sSFB:    "); print(sSFB); print('\n');
            print('\n');

            print("tCsA:    "); print(tCsA); print('\n');
            print("tCsB:    "); print(tCsB); print('\n');
            print("tCsSFA:  "); print(tCsSFA); print('\n');
            print("tCsSFB:  "); print(tCsSFB); print('\n');
            print('\n');

            print("tCrA_copy_view:    "); print(tCrA_copy_view); print('\n');
            print("tCrB_copy_view:    "); print(tCrB_copy_view); print('\n');
            print("tCrSFA_copy_view:  "); print(tCrSFA_copy_view); print('\n');
            print("tCrSFB_copy_view:  "); print(tCrSFB_copy_view); print('\n');
            print('\n');

            print("tCrA:    "); print(tCrA); print('\n');
            print("tCrB:    "); print(tCrB); print('\n');
            print("tCrC:    "); print(tCrC); print('\n');
            print("tCrSFA:  "); print(tCrSFA); print('\n');
            print("tCrSFB:  "); print(tCrSFB); print('\n');
            print('\n');
        }
/*        if (warp_group_thread_idx == 0) {
            print_latex(smem_tiled_copy_A);
            print_latex(smem_tiled_copy_B);
            print_latex(smem_tiled_copy_SFA);
            print_latex(smem_tiled_copy_SFB);
            print_latex(smem_tiled_copy_C);
        }*/
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));      // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));      // CPY_K
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));                // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));                // MMA_N
        CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(tCrC));              // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(tCrC));              // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // CPY_K
        CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                // PIPE
        CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));            // CPY_K
        CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));            // PIPE
        CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));  // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));  // CPY_K

        // xxx_stage: point to current read stage tensor
        int read_stage = smem_pipe_read.index();
        auto tCsA_stage   = tCsA(_,_,_,read_stage);
        auto tCsB_stage   = tCsB(_,_,_,read_stage);
        auto tCsSFA_stage = tCsSFA(_,_,_,read_stage);
        auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

/*        cutlass::arch::NamedBarrier::sync(
                thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);*/
        pipeline.consumer_wait(smem_pipe_read);
        auto fp4_shift = [&](int k_block)
        {
            using MMAOp = typename MMA::MMA_Op;
            FP4Shift::fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
            FP4Shift::fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));
        };
        // Load the first block to reg before start
        copy(smem_tiled_copy_A, tCsA_stage(_, _, 0), tCrA_copy_view(_, _, 0));
        copy(smem_tiled_copy_B, tCsB_stage(_, _, 0), tCrB_copy_view(_, _, 0));
        fp4_shift(0);
        copy(tCsSFA_stage(_, _, 0), tCrSFA_copy_view(_, _, 0));
        copy(tCsSFB_stage(_, _, 0), tCrSFB_copy_view(_, _, 0));

        auto const num_k_block = size<2>(tCrA);        // MMA_K

        for (int k_tile = 0; k_tile < num_k_tile; k_tile++) {

            for (int k_block = 0; k_block < num_k_block; k_block++) {
                auto k_block_next = ((k_block + 1) == num_k_block) ? 0 : (k_block + 1);
                // The last block
                if (k_block == num_k_block - 1) {
                    cutlass::arch::NamedBarrier::sync(
                            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                    // release current read buffer
                    pipeline.consumer_release(smem_pipe_read);
                    //  not the last tile, preload the next tile's first block
                    if(k_tile < num_k_tile - 1)
                    {
                        ++smem_pipe_read;
                        // Read next tile
                        read_stage = smem_pipe_read.index();
                        tCsA_stage = tCsA(_, _, _, read_stage);
                        tCsB_stage = tCsB(_, _, _, read_stage);
                        tCsSFA_stage = tCsSFA(_, _, _, read_stage);
                        tCsSFB_stage = tCsSFB(_, _, _, read_stage);

                        // Make sure the next tile's data has arrived
                        pipeline.consumer_wait(smem_pipe_read);
                    }
                }
                if(0 && warp_group_thread_idx == 0)
                {
                    print("A Befor:\n"); print_tensor(tCrA_copy_view(_, _, k_block_next));
                    print("B Befor:\n"); print_tensor(tCrB_copy_view(_, _, k_block_next));
                }
                // copy the next block
                copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block_next), tCrA_copy_view(_, _, k_block_next));
                copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block_next), tCrB_copy_view(_, _, k_block_next));
                fp4_shift(k_block_next);
                copy(tCsSFA_stage(_, _, k_block_next), tCrSFA_copy_view(_, _, k_block_next));
                copy(tCsSFB_stage(_, _, k_block_next), tCrSFB_copy_view(_, _, k_block_next));
                if(0 && warp_group_thread_idx == 0)
                {
                    print("A After:\n"); print_tensor(tCrA_copy_view(_, _, k_block_next));
                    print("B After:\n"); print_tensor(tCrB_copy_view(_, _, k_block_next));
                }
                
                // auto tCrA_shift = recast<uint32_t>(tCrA(_, _, k_block));
                // auto tCrB_shift = recast<uint32_t>(tCrB(_, _, k_block));
                // for(int i=0; i < size(tCrA_shift); i++) tCrA_shift(i) <<= 2;
                // for(int i=0; i < size(tCrB_shift); i++) tCrB_shift(i) <<= 2;
                

                //gemm
                gemm(tiled_mma,
                     make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                     make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
                     tCrC);
            }

        }
        using ElementC = typename EpilogueSharedStorage::TypeC;
        cutlass::NumericConverter<ElementC, typename MMA::ValTypeC> converterC;
        auto tCrC_copy_view_D = make_tensor<ElementC>(layout(tCrC_copy_view));
        for(int i = 0; i < size(tCrC_copy_view_D); i++)
        {
            // convert accumulator from float to bfloat16
            tCrC_copy_view_D(i) = converterC(tCrC_copy_view(i));
        }

        //Epilogue writeback

        // step 1: Copy from register to shared memory
        copy(smem_tiled_copy_C, tCrC_copy_view_D, tCsC_r2s);

        // async proxy fence: make sure the shared memory writing is visible to TMA
        cutlass::arch::fence_view_async_shared();
        // bar thread sync: make sure the customer thread has arrived
        cutlass::arch::NamedBarrier::sync(size(tiled_mma), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // step 2: Launch TMA Store
        if(warp_group_thread_idx == 0)
        {
            copy(tma_store_c, tCsC_s2g, tCgC_s2g);
        }

    }

}

#endif //SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_WARPSPECIALIZED_H
