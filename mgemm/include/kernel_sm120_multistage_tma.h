#ifndef SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_H
#define SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/container/array.hpp>
#include <cutlass/cutlass.h>
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

using namespace cute;
template<
        class LayoutA, class SmemLayoutA, class TmaLoadA, class SmemCopyAtomA,
        class LayoutB, class SmemLayoutB, class TmaLoadB, class SmemCopyAtomB,
        class LayoutSFA, class SmemLayoutSFA, class TmaLoadSFA, class SmemCopyAtomSFA,
        class LayoutSFB, class SmemLayoutSFB, class TmaLoadSFB, class SmemCopyAtomSFB,
        class ElementC, class LayoutC,
        class MMA, class ProbShape_MNK, class TileShape_MNK, class SharedStorage,
        class MainloopPipeline, class PipelineState, int N_STAGE, int TmaTransactionBytes>
__launch_bounds__(128, 1)
__global__
void
gemm_device_multistage(__grid_constant__ const TmaLoadA tma_load_a, LayoutA layout_A,
                       __grid_constant__ const TmaLoadB tma_load_b, LayoutB layout_B,
                       __grid_constant__ const TmaLoadSFA tma_load_sfa, LayoutSFA layout_SFA,
                       __grid_constant__ const TmaLoadSFB tma_load_sfb, LayoutSFB layout_SFB,
                       ElementC *ptr_C, LayoutC layout_C) {

    //Smem Alloc
    extern __shared__ uint8_t smem_ptr[];
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_ptr);

    auto const thread_idx = threadIdx.x;
//    int lane_predicate = cute::elect_one_sync();
    auto const block_x = blockIdx.x;
    auto const block_y = blockIdx.y;

    if (thread_idx == 0) {
        cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_load_sfb.get_tma_descriptor());
    }

    Tensor mA = tma_load_a.get_tma_tensor(shape(layout_A));             // (M, K)
    Tensor mB = tma_load_b.get_tma_tensor(shape(layout_B));             // (N, K)
    Tensor mSFA = tma_load_sfa.get_tma_tensor(shape(layout_SFA));
    Tensor mSFB = tma_load_sfb.get_tma_tensor(shape(layout_SFB));

    Tensor mC = make_tensor(make_gmem_ptr(ptr_C), layout_C);

    auto const block_coord = make_coord(block_x, block_y, _);
    Tensor gA = local_tile(mA, TileShape_MNK{}, block_coord, Step<_1, X, _1>{});             // (BLK_M, BLK_K, k_tile)
    Tensor gB = local_tile(mB, TileShape_MNK{}, block_coord, Step<X, _1, _1>{});             // (BLK_N, BLK_K, k_tile)
    Tensor gSFA = local_tile(mSFA, TileShape_MNK{}, block_coord, Step<_1, X, _1>{});         // (BLK_M, BLK_K, k_tile)
    Tensor gSFB = local_tile(mSFB, TileShape_MNK{}, block_coord, Step<X, _1, _1>{});         // (BLK_N, BLK_K, k_tile)
    Tensor gC = local_tile(mC, TileShape_MNK{}, block_coord, Step<_1, _1, X>{});             // (BLK_M, BLK_N)


    auto &shared_tensors = shared_storage.tensors;
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                            SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                            SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
                              SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
                              SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

    auto block_tma_a = tma_load_a.get_slice(0);
    auto block_tma_b = tma_load_b.get_slice(0);

    auto block_tma_sfa = tma_load_sfa.get_slice(0);
    auto block_tma_sfb = tma_load_sfb.get_slice(0);

    // Partition source and destination tensors for tma copies
    Tensor tAgA = block_tma_a.partition_S(gA);                                // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                               // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                               // (TMA,TMA_N,TMA_K,PIPE)

    Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);                        // (TMA,TMA_M,TMA_K,k)
    Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);                        // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);                        // (TMA,TMA_N,TMA_K,k)
    Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                        // (TMA,TMA_N,TMA_K,PIPE)


    static_assert(rank(gSFA) == Int<3>{}, "gSFA rank should be 3 (BLK_M, BLK_K, k_tile)");
    static_assert(rank(sSFA) == Int<3>{}, "sSFA rank should be 3 (BLK_M, BLK_K, PIPE)");
    static_assert(rank(tAgSFA) == Int<4>{}, "tAgSFA rank should be 4 (TMA, TMA_M, TMA_K, k)");
    static_assert(rank(tAsSFA) == Int<4>{}, "tAsSFA rank should be 4 (TMA, TMA_M, TMA_K, PIPE)");

    static_assert(rank(gSFB) == Int<3>{}, "gSFB rank should be 3 (BLK_N, BLK_K, k_tile)");
    static_assert(rank(sSFB) == Int<3>{}, "sSFB rank should be 3 (BLK_N, BLK_K, PIPE)");
    static_assert(rank(tBgSFB) == Int<4>{}, "tBgSFB rank should be 4 (TMA, TMA_N, TMA_K, k)");
    static_assert(rank(tBsSFB) == Int<4>{}, "tBsSFB rank should be 4 (TMA, TMA_N, TMA_K, PIPE)");

    MMA tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));          // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));          // (MMA,MMA_N,MMA_K)
    Tensor tCrC = thread_mma.partition_fragment_C(gC);                        // (MMA,MMA_M,MMA_N)
    Tensor tCgC = thread_mma.partition_C(gC);                                 // (MMA,MMA_M,MMA_N)

    // clear accumulator
    clear(tCrC);

    Tensor tCrSFA = partition_fragment_SFA(sSFA(_, _, Int<0>{}),
                                           thread_mma);       // (MMA,MMA_M,MMA_K)
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_, _, Int<0>{}),
                                           thread_mma);       // (MMA,MMA_N,MMA_K)

    //
    // Copy from smem to registers
    //

    // A
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(
            as_position_independent_swizzle_tensor(
                    sA));                                // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(
            tCrA);                            // (CPY,CPY_M,CPY_K)

    // B
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(
            as_position_independent_swizzle_tensor(
                    sB));                                // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(
            tCrB);                            //(CPY,CPY_M,CPY_K)

    // SFA
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk),
                                                               size<2>(tile_shape_mnk))
    );
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA = smem_thr_copy_SFA.partition_S(
            as_position_independent_swizzle_tensor(
                    sSFA));                                  // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(
            tCrSFA);                          // (CPY,CPY_M,CPY_K)

    // SFB
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk),
                                                               size<2>(tile_shape_mnk))
    );
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB = smem_thr_copy_SFB.partition_S(
            as_position_independent_swizzle_tensor(
                    sSFB));                                 // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(
            tCrSFB);                                        // (CPY,CPY_N,CPY_K)
    if(0 && thread0() && block_x == 0 && block_y == 0)
    {
        print_latex(tma_load_a);
        print_latex(smem_tiled_copy_A);
        print_latex(smem_tiled_copy_B);
        print_latex(smem_tiled_copy_SFA);
        print_latex(smem_tiled_copy_SFB);
    }
    if(0 && thread0() && block_x == 0 && block_y == 0)
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

    }
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                        // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                        // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));                                  // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));                                  // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                  // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                  // PIPE
    CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sA));                                   // PIPE
    CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sB));                                   // PIPE

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));                    // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));                    // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(tCrC));                                // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(tCrC));                                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));                              // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));                              // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));                                    // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));                                    // PIPE

    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
    // TMA Params set
    typename MainloopPipeline::Params mainloop_pipeline_params;
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    mainloop_pipeline_params.transaction_bytes = TmaTransactionBytes;
    mainloop_pipeline_params.is_leader = thread_idx == 0;
    mainloop_pipeline_params.num_consumers = size(MMA{});
    static_assert(size(MMA{}) == 128);

    //SM120 No Cluster
    auto cluster_shape = Shape<_1, _1, _1>{};
    MainloopPipeline pipeline(shared_storage.pipeline_storage, mainloop_pipeline_params,
                              cluster_shape);
    __syncthreads();

    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read;

    int num_k_tile = size<2>(gA);
    int k_tile_load = 0;

    KERNEL_DEBUG(num_k_tile);

    if (thread_idx == 0) {
        // Launch N_STAGE-1 TMA load before start
        for (int i = 0; i < min(int(N_STAGE - 1), num_k_tile); i++) {
            pipeline.producer_acquire(smem_pipe_write);
            BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);

            int write_stage = smem_pipe_write.index();
            copy(tma_load_a.with(*tmaBar), tAgA(_, _, _, k_tile_load), tAsA(_, _, _, write_stage));
            copy(tma_load_b.with(*tmaBar), tBgB(_, _, _, k_tile_load), tBsB(_, _, _, write_stage));
            copy(tma_load_sfa.with(*tmaBar), tAgSFA(_, _, _, k_tile_load),
                 tAsSFA(_, _, _, write_stage));
            copy(tma_load_sfb.with(*tmaBar), tBgSFB(_, _, _, k_tile_load),
                 tBsSFB(_, _, _, write_stage));
            ++smem_pipe_write;
            k_tile_load++;
        }
    }

    __syncthreads();
    // Make sure the first load request has arrived
    pipeline.consumer_wait(smem_pipe_read);

    for (int k_tile = 0; k_tile < num_k_tile; k_tile++) {
        // Launch next TMA load
        if (thread_idx == 0 && k_tile_load < num_k_tile) {
            pipeline.producer_acquire(smem_pipe_write);
            BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);

            int write_stage = smem_pipe_write.index();
            copy(tma_load_a.with(*tmaBar), tAgA(_, _, _, k_tile_load), tAsA(_, _, _, write_stage));
            copy(tma_load_b.with(*tmaBar), tBgB(_, _, _, k_tile_load), tBsB(_, _, _, write_stage));
            copy(tma_load_sfa.with(*tmaBar), tAgSFA(_, _, _, k_tile_load),
                 tAsSFA(_, _, _, write_stage));
            copy(tma_load_sfb.with(*tmaBar), tBgSFB(_, _, _, k_tile_load),
                 tBsSFB(_, _, _, write_stage));
            ++smem_pipe_write;
            k_tile_load++;
        }

        auto read_stage = smem_pipe_read.index();
        auto tCsA_stage = tCsA(_, _, _, read_stage);
        auto tCsB_stage = tCsB(_, _, _, read_stage);
        auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);
        auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

        auto num_k_block = size<2>(tCrA);        // MMA_K

        KERNEL_DEBUG(num_k_block);
        // Load the first block to reg before start
        copy(smem_tiled_copy_A, tCsA_stage(_, _, 0), tCrA_copy_view(_, _, 0));
        copy(smem_tiled_copy_B, tCsB_stage(_, _, 0), tCrB_copy_view(_, _, 0));
        copy(tCsSFA_stage(_, _, 0), tCrSFA_copy_view(_, _, 0));
        copy(tCsSFB_stage(_, _, 0), tCrSFB_copy_view(_, _, 0));

        for (int k_block = 0; k_block < num_k_block; k_block++) {
            auto k_block_next = ((k_block + 1) == num_k_block) ? 0 : (k_block + 1);
            // The last block but not the last tile, preload next tile's first block
            if (k_block == num_k_block - 1 && k_tile < num_k_tile - 1) {
                /*cutlass::arch::NamedBarrier::sync(
                        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);*/
                // release current read buffer
                pipeline.consumer_release(smem_pipe_read);
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

            // copy the next block
            copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block_next),
                 tCrA_copy_view(_, _, k_block_next));
            copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block_next),
                 tCrB_copy_view(_, _, k_block_next));
            copy(tCsSFA_stage(_, _, k_block_next), tCrSFA_copy_view(_, _, k_block_next));
            copy(tCsSFB_stage(_, _, k_block_next), tCrSFB_copy_view(_, _, k_block_next));

/*            if(0 && thread0())
            {
                print("tCrA   "); print_tensor(tCrA(_, _, k_block)); print("\n");
                print("tCrSFA "); print_tensor(tCrSFA(_, _, k_block)); print("\n");
                print("tCrB   "); print_tensor(tCrB(_, _, k_block)); print("\n");
                print("tCrSFB "); print_tensor(tCrSFB(_, _, k_block)); print("\n");
            }*/
            //gemm
            gemm(tiled_mma,
                 make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                 make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
                 tCrC);
        }
        __syncthreads();
    }

//    __syncthreads();
    //Epilogue writeback
    //TODO optimizing with shared memory
    copy(tCrC, tCgC);

}
#endif //SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_TMA_H
