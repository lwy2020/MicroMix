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

#include "sm120_fp4_shift.h"

using namespace cute;
template<
        class ElementA, class LayoutA, class SmemLayoutA, class TmaLoadA, class SmemCopyAtomA,
        class ElementB, class LayoutB, class SmemLayoutB, class TmaLoadB, class SmemCopyAtomB,
        class ElementSFA, class LayoutSFA, class SmemLayoutSFA, class TmaLoadSFA, class SmemCopyAtomSFA,
        class ElementSFB, class LayoutSFB, class SmemLayoutSFB, class TmaLoadSFB, class SmemCopyAtomSFB,
        class ElementC, class LayoutC, class SmemLayoutC, class TmaLoadC, class SmemCopyAtomC,
        class ElementD, class LayoutD, class SmemLayoutD, class TmaStoreD, class SmemCopyAtomD,
        class MMA, class ProbShape_MNK, class TileShape_MNK, 
        class MainloopSharedStorage, class EpilogueSharedStorage,
        class MainloopPipeline, class MainloopPipelineState, 
        class EpiloguePipeline, class EpiloguePipelineState, 
        int N_STAGE, int MainloopTmaTransactionBytes, int EpilogueTmaTransactionBytes>
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
                                       __grid_constant__ const TmaLoadC tma_load_c,
                                       LayoutC layout_C,
                                       __grid_constant__ const TmaStoreD tma_store_d,
                                       LayoutD layout_D) {

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
        cute::prefetch_tma_descriptor(tma_load_c.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_store_d.get_tma_descriptor());
    }

    Tensor mA = tma_load_a.get_tma_tensor(shape(layout_A));             // (M, K)
    Tensor mB = tma_load_b.get_tma_tensor(shape(layout_B));             // (N, K)
    Tensor mSFA = tma_load_sfa.get_tma_tensor(shape(layout_SFA));       // (M, K)
    Tensor mSFB = tma_load_sfb.get_tma_tensor(shape(layout_SFB));       // (N, K)
    Tensor mC = tma_load_c.get_tma_tensor(shape(layout_C));             // (M, N)
    Tensor mD = tma_store_d.get_tma_tensor(shape(layout_D));            // (M, N)

    auto const block_coord = make_coord(block_x, block_y, _);
    Tensor gA = local_tile(mA, TileShape_MNK{}, block_coord, Step < _1, X, _1 > {});        // (BLK_M, BLK_K, k_tile)
    Tensor gB = local_tile(mB, TileShape_MNK{}, block_coord, Step < X, _1, _1 > {});        // (BLK_N, BLK_K, k_tile)
    Tensor gSFA = local_tile(mSFA, TileShape_MNK{}, block_coord, Step < _1, X, _1 > {});    // (BLK_M, BLK_K, k_tile)
    Tensor gSFB = local_tile(mSFB, TileShape_MNK{}, block_coord, Step < X, _1, _1 > {});    // (BLK_N, BLK_K, k_tile)
    Tensor gC = local_tile(mC, TileShape_MNK{}, block_coord, Step < _1, _1, X > {});        // (BLK_M, BLK_N)
    Tensor gD = local_tile(mD, TileShape_MNK{}, block_coord, Step < _1, _1, X > {});        // (BLK_M, BLK_N)


    auto &mainloop_shared_tensors = mainloop_shared_storage.tensors;
    auto &epilogue_shared_tensors = epilogue_shared_storage.tensors;
    Tensor sA = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});    // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});    // (BLK_N,BLK_K,PIPE)
    Tensor sC = make_tensor(make_smem_ptr(mainloop_shared_tensors.smem_C.begin()), SmemLayoutC{});          // (BLK_M, BLK_N)
    Tensor sD = make_tensor(make_smem_ptr(epilogue_shared_tensors.smem_D.begin()), SmemLayoutD{});          // (BLK_M, BLK_N)

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

    
    // Mainloop TMA Params set
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.transaction_bytes = MainloopTmaTransactionBytes;
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup; //size(MMA{});

    // Epilogue TMA Params set
    typename EpiloguePipeline::Params epilogue_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        epilogue_pipeline_params.role = EpiloguePipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
        epilogue_pipeline_params.role = EpiloguePipeline::ThreadCategory::Consumer;
    }
    epilogue_pipeline_params.transaction_bytes = EpilogueTmaTransactionBytes;
    epilogue_pipeline_params.is_leader = warp_group_thread_idx == 0;
    epilogue_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup; //size(MMA{});

    //SM120 No Cluster
    auto cluster_shape = Shape < _1, _1, _1>{};

    MainloopPipeline mainloop_pipeline(mainloop_shared_storage.mainloop_pipeline_storage, mainloop_pipeline_params,
                              cluster_shape);

    EpiloguePipeline epilogue_pipeline(mainloop_shared_storage.epilogue_pipeline_storage, epilogue_pipeline_params,
                              cluster_shape);

    MainloopPipelineState mainloop_smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    MainloopPipelineState mainloop_smem_pipe_read;

    EpiloguePipelineState epilogue_smem_pipe_write = cutlass::make_producer_start_state<EpiloguePipeline>();
    EpiloguePipelineState epilogue_smem_pipe_read;

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

            auto block_tma_c = tma_load_c.get_slice(0);

            // Partition source and destination tensors for tma copies
            Tensor tAgA = block_tma_a.partition_S(gA);                   // (TMA,TMA_M,TMA_K,k)
            Tensor tAsA = block_tma_a.partition_D(sA);                   // (TMA,TMA_M,TMA_K,PIPE)

            Tensor tBgB = block_tma_b.partition_S(gB);                   // (TMA,TMA_N,TMA_K,k)
            Tensor tBsB = block_tma_b.partition_D(sB);                   // (TMA,TMA_N,TMA_K,PIPE)

            Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);             // (TMA,TMA_M,TMA_K,k)
            Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);             // (TMA,TMA_M,TMA_K,PIPE)

            Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);             // (TMA,TMA_N,TMA_K,k)
            Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);             // (TMA,TMA_N,TMA_K,PIPE)

            Tensor tCgC = block_tma_c.partition_S(gC);                   // (TMA,TMA_M,TMA_N)
            Tensor tCsC = block_tma_c.partition_D(sC);                   // (TMA,TMA_M,TMA_N)

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

                    mainloop_pipeline.producer_acquire(mainloop_smem_pipe_write);

                    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                    BarrierType *tmaBar = mainloop_pipeline.producer_get_barrier(mainloop_smem_pipe_write);

                    int write_stage = mainloop_smem_pipe_write.index();
                    copy(tma_load_a.with(*tmaBar), tAgA(_, _, _, i), tAsA(_, _, _, write_stage));
                    copy(tma_load_b.with(*tmaBar), tBgB(_, _, _, i), tBsB(_, _, _, write_stage));
                    copy(tma_load_sfa.with(*tmaBar), tAgSFA(_, _, _, i), tAsSFA(_, _, _, write_stage));
                    copy(tma_load_sfb.with(*tmaBar), tBgSFB(_, _, _, i), tBsSFB(_, _, _, write_stage));

                    ++mainloop_smem_pipe_write;
                }

                //Epilogue C load
                epilogue_pipeline.producer_acquire(epilogue_smem_pipe_write);
                using BarrierType = typename EpiloguePipeline::ProducerBarrierType;
                BarrierType *tmaBar = epilogue_pipeline.producer_get_barrier(epilogue_smem_pipe_write);
                copy(tma_load_c.with(*tmaBar), tCgC, tCsC);
                ++epilogue_smem_pipe_write;

            }
            __syncwarp();
        }
    }
    else if(warp_group_role == WarpGroupRole::Consumer) // Consumer MMA
    {
        MMA tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

        // Allocate fragments and descriptors
        Tensor tDrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));      // (MMA,MMA_M,MMA_K)
        Tensor tDrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));      // (MMA,MMA_N,MMA_K)
        Tensor tDrD = thread_mma.partition_fragment_C(gD);                      // (MMA,MMA_M,MMA_N)

        // clear accumulator
        clear(tDrD);

        Tensor tDrSFA = partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);   // (MMA,MMA_M,MMA_K)
        Tensor tDrSFB = partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);   // (MMA,MMA_N,MMA_K)

        // A: S->R
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);
        Tensor tDsA = smem_thr_copy_A.partition_S(
                as_position_independent_swizzle_tensor(sA));        // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tDrA_copy_view = smem_thr_copy_A.retile_D(tDrA);     // (CPY,CPY_M,CPY_K)

        // B: S->R
        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(warp_group_thread_idx);
        Tensor tDsB = smem_thr_copy_B.partition_S(
                as_position_independent_swizzle_tensor(sB));        // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tDrB_copy_view = smem_thr_copy_B.retile_D(tDrB);     // (CPY,CPY_M,CPY_K)

        // SFA: S->R
        auto tile_shape_mnk = tile_shape(tiled_mma);
        auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                        get_layoutSFA_TV(tiled_mma),
                                                        make_shape(size<0>(tile_shape_mnk),
                                                                   size<2>(tile_shape_mnk))
        );
        auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(warp_group_thread_idx);
        Tensor tDsSFA = smem_thr_copy_SFA.partition_S(
                as_position_independent_swizzle_tensor(sSFA));           // (CPY,CPY_M,CPY_K,PIPE)
        Tensor tDrSFA_copy_view = smem_thr_copy_SFA.retile_D(tDrSFA);    // (CPY,CPY_M,CPY_K)

        // SFB: S->R
        auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                        get_layoutSFB_TV(tiled_mma),
                                                        make_shape(size<1>(tile_shape_mnk),
                                                                   size<2>(tile_shape_mnk))
        );
        auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(warp_group_thread_idx);
        Tensor tDsSFB = smem_thr_copy_SFB.partition_S(
                as_position_independent_swizzle_tensor(sSFB));            // (CPY,CPY_N,CPY_K,PIPE)
        Tensor tDrSFB_copy_view = smem_thr_copy_SFB.retile_D(tDrSFB);     // (CPY,CPY_N,CPY_K)


        CUTE_STATIC_ASSERT_V(size<1>(tDsA) == size<1>(tDrA_copy_view));      // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tDsA) == size<2>(tDrA_copy_view));      // CPY_K
        CUTE_STATIC_ASSERT_V(size<1>(tDrA) == size<1>(tDrD));                // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tDrB) == size<2>(tDrD));                // MMA_N
        CUTE_STATIC_ASSERT_V(size<1>(tDrSFA) == size<1>(tDrD));              // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tDrSFB) == size<2>(tDrD));              // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tDsA) == size<2>(tDsB));                // CPY_K
        CUTE_STATIC_ASSERT_V(size<3>(tDsA) == size<3>(tDsB));                // PIPE
        CUTE_STATIC_ASSERT_V(size<2>(tDsSFA) == size<2>(tDsSFB));            // CPY_K
        CUTE_STATIC_ASSERT_V(size<3>(tDsSFA) == size<3>(tDsSFB));            // PIPE
        CUTE_STATIC_ASSERT_V(size<1>(tDsSFA) == size<1>(tDrSFA_copy_view));  // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tDsSFA) == size<2>(tDrSFA_copy_view));  // CPY_K

        // xxx_stage: point to current read stage tensor
        int read_stage = mainloop_smem_pipe_read.index();
        auto tDsA_stage   = tDsA(_,_,_,read_stage);
        auto tDsB_stage   = tDsB(_,_,_,read_stage);
        auto tDsSFA_stage = tDsSFA(_,_,_,read_stage);
        auto tDsSFB_stage = tDsSFB(_,_,_,read_stage);

        mainloop_pipeline.consumer_wait(mainloop_smem_pipe_read);


        auto load_block = [&](int k_block)
        {
            copy(smem_tiled_copy_A, tDsA_stage(_, _, k_block), tDrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_B, tDsB_stage(_, _, k_block), tDrB_copy_view(_, _, k_block));

            using MMAOp = typename MMA::MMA_Op;
            FP4Shift::fp4_shift_A(MMAOp{}, tDrA_copy_view(_, _, k_block));
            FP4Shift::fp4_shift_B(MMAOp{}, tDrB_copy_view(_, _, k_block));

            copy(tDsSFA_stage(_, _, k_block), tDrSFA_copy_view(_, _, k_block));
            copy(tDsSFB_stage(_, _, k_block), tDrSFB_copy_view(_, _, k_block));
        };
        
        // Load the first block to reg before start
        load_block(0);

        auto const num_k_block = size<2>(tDrA);        // MMA_K

        for (int k_tile = 0; k_tile < num_k_tile; k_tile++) {

            for (int k_block = 0; k_block < num_k_block; k_block++) {
                auto k_block_next = ((k_block + 1) == num_k_block) ? 0 : (k_block + 1);
                // The last block
                if (k_block == num_k_block - 1) {
                    cutlass::arch::NamedBarrier::sync(
                            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                    // release current read buffer
                    mainloop_pipeline.consumer_release(mainloop_smem_pipe_read);
                    //  not the last tile, preload the next tile's first block
                    if(k_tile < num_k_tile - 1)
                    {
                        ++mainloop_smem_pipe_read;
                        // Read next tile
                        read_stage = mainloop_smem_pipe_read.index();
                        tDsA_stage = tDsA(_, _, _, read_stage);
                        tDsB_stage = tDsB(_, _, _, read_stage);
                        tDsSFA_stage = tDsSFA(_, _, _, read_stage);
                        tDsSFB_stage = tDsSFB(_, _, _, read_stage);

                        // Make sure the next tile's data has arrived
                        mainloop_pipeline.consumer_wait(mainloop_smem_pipe_read);
                    }
                }

                // copy the next block
                load_block(k_block_next);
                
                //gemm
                gemm(tiled_mma,
                     make_zip_tensor(tDrA(_, _, k_block), tDrSFA(_, _, k_block)),
                     make_zip_tensor(tDrB(_, _, k_block), tDrSFB(_, _, k_block)),
                     tDrD);
            }

        }


        // Epilogue

        epilogue_pipeline.consumer_wait(epilogue_smem_pipe_read);

        // C: S -> R
        auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(warp_group_thread_idx);
        Tensor tCsC = smem_thr_copy_C.partition_S(sC);
        Tensor tCrC = make_tensor<ElementC>(layout(tDrD));
        Tensor tCrC_copy_view = smem_thr_copy_C.retile_D(tCrC);
        copy(smem_tiled_copy_C, tCsC, tCrC_copy_view);

        cutlass::NumericConverter<ElementD, typename MMA::ValTypeD> converterD;
        Tensor tDrD_merge = make_tensor<ElementD>(layout(tDrD));
        for(int i = 0; i < size(tDrD_merge); i++)
        {
            // convert accumulator from 'float' to 'bfloat16' and accumulating
            // D <- D + C
            tDrD_merge(i) = converterD(tDrD(i) + tCrC(i));    
        }


        //Epilogue writeback

        // D: R -> S
        auto smem_tiled_copy_D = make_tiled_copy_C(SmemCopyAtomD{}, tiled_mma);
        auto smem_thr_copy_D = smem_tiled_copy_D.get_thread_slice(warp_group_thread_idx);
        Tensor tDrD_copy_view = smem_thr_copy_D.retile_S(tDrD_merge); // replace with tDrD_merge
        Tensor tDsD_r2s = smem_thr_copy_D.partition_D(sD);

        
        // step 1: Copy from register to shared memory
        copy(smem_tiled_copy_D, tDrD_copy_view, tDsD_r2s);

        // async proxy fence: make sure the shared memory writing is visible to TMA
        cutlass::arch::fence_view_async_shared();
        // bar thread sync: make sure the customer thread has arrived
        cutlass::arch::NamedBarrier::sync(size(tiled_mma), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // step 2: Launch TMA Store
        if(warp_group_thread_idx == 0)
        {
            // D: S -> G
            auto block_tma_d = tma_store_d.get_slice(0);
            Tensor tDsD_s2g = block_tma_d.partition_S(sD);
            Tensor tDgD_s2g = block_tma_d.partition_D(gD);
            copy(tma_store_d, tDsD_s2g, tDgD_s2g);
        }
        
        // For debug
        // if (warp_group_thread_idx == 0) {
            // print_latex(smem_tiled_copy_A);
            // print_latex(smem_tiled_copy_B);
            // print_latex(smem_tiled_copy_SFA);
            // print_latex(smem_tiled_copy_SFB);
            // print_latex(smem_tiled_copy_C);
            // print_latex(smem_tiled_copy_D);
        // }
    }

}

#endif //SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_WARPSPECIALIZED_H
