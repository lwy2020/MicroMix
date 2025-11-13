#ifndef SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_CPASYNC_H
#define SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_CPASYNC_H

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
        class ElementA, class TensorA, class SmemLayoutA, class GmemCopyA, class SmemCopyAtomA,
        class ElementB, class TensorB, class SmemLayoutB, class GmemCopyB, class SmemCopyAtomB,
        class ElementSFA, class TensorSFA, class SmemLayoutSFA, class GmemCopySFA, class SmemCopyAtomSFA,
        class ElementSFB, class TensorSFB, class SmemLayoutSFB, class GmemCopySFB, class SmemCopyAtomSFB,
        class ElementC, class TensorC, class SmemLayoutC, class GmemCopyC, class SmemCopyAtomC,
        class ElementD, class TensorD, class SmemLayoutD, class GmemCopyD, class SmemCopyAtomD,
        class TiledMMA, class TileShape_MNK, 
        class MainloopSharedStorage, class EpilogueSharedStorage,
        int N_STAGE>
//__launch_bounds__(256, 1)
__global__
void
gemm_device_multistage_cpasync(TensorA mA, TensorB mB, TensorSFA mSFA, TensorSFB mSFB, TensorC mC, TensorD mD) {

    //Smem Alloc
    extern __shared__ uint8_t smem_ptr[];
    MainloopSharedStorage &mainloop_shared_storage = *reinterpret_cast<MainloopSharedStorage *>(smem_ptr);
    EpilogueSharedStorage &epilogue_shared_storage = *reinterpret_cast<EpilogueSharedStorage *>(smem_ptr);

    int thread_idx = int(threadIdx.x);
                                    
    auto const block_x = blockIdx.x;
    auto const block_y = blockIdx.y;

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
    Tensor sC = make_tensor(make_smem_ptr(epilogue_shared_tensors.smem_C.begin()), SmemLayoutC{});          // (BLK_M, BLK_N)
    Tensor sD = make_tensor(make_smem_ptr(epilogue_shared_tensors.smem_D.begin()), SmemLayoutD{});          // (BLK_M, BLK_N)


    GmemCopyA g2s_copy_A;
    GmemCopyB g2s_copy_B;
    GmemCopySFA g2s_copy_SFA;
    GmemCopySFB g2s_copy_SFB;

    auto g2s_thr_copy_A = g2s_copy_A.get_slice(thread_idx);
    auto g2s_thr_copy_B = g2s_copy_B.get_slice(thread_idx);
    auto g2s_thr_copy_SFA = g2s_copy_SFA.get_slice(thread_idx);
    auto g2s_thr_copy_SFB = g2s_copy_SFB.get_slice(thread_idx);


    Tensor tAgA = g2s_thr_copy_A.partition_S(gA);                   // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = g2s_thr_copy_A.partition_D(sA);                   // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = g2s_thr_copy_B.partition_S(gB);                   // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = g2s_thr_copy_B.partition_D(sB);                   // (TMA,TMA_N,TMA_K,PIPE)

    Tensor tAgSFA = g2s_thr_copy_SFA.partition_S(gSFA);             // (TMA,TMA_M,TMA_K,k)
    Tensor tAsSFA = g2s_thr_copy_SFA.partition_D(sSFA);             // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgSFB = g2s_thr_copy_SFB.partition_S(gSFB);             // (TMA,TMA_N,TMA_K,k)
    Tensor tBsSFB = g2s_thr_copy_SFB.partition_D(sSFB);             // (TMA,TMA_N,TMA_K,PIPE)

    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});
    

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)


    // Repeat the partitioning with identity layouts
    Tensor tAcA = g2s_thr_copy_A.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = g2s_thr_copy_B.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    
    
    auto BM = size<0>(TileShape_MNK{});
    auto BN = size<1>(TileShape_MNK{});

    auto residue_m = size<0>(mA) - block_x * BM;
    auto residue_n = size<0>(mB) - block_y * BN;
    if(residue_m > BM) residue_m = BM;
    if(residue_n > BN) residue_n = BN;


    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < residue_m;  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < residue_n;  // blk_n coord < residue_n
    }


    if(0 && thread0() && block_x==0 && block_y==0)
    {
        print("tApA: "); print_tensor(tApA);
        print("tBpB: "); print_tensor(tBpB);
        print("tAcA: "); print_tensor(tAcA);
        print("tBcB: "); print_tensor(tBcB);
        print("cA: "); print_tensor(cA);
        print("cB: "); print_tensor(cB);

    }

    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);

    static_assert(rank(gSFA) == Int<3>{},   "gSFA rank should be 3 (BLK_M, BLK_K, k_tile)");
    static_assert(rank(sSFA) == Int<3>{},   "sSFA rank should be 3 (BLK_M, BLK_K, PIPE)");
    static_assert(rank(tAgSFA) == Int<4>{}, "tAgSFA rank should be 4 (TMA, TMA_M, TMA_K, k)");
    static_assert(rank(tAsSFA) == Int<4>{}, "tAsSFA rank should be 4 (TMA, TMA_M, TMA_K, PIPE)");

    static_assert(rank(gSFB) == Int<3>{},   "gSFB rank should be 3 (BLK_N, BLK_K, k_tile)");
    static_assert(rank(sSFB) == Int<3>{},   "sSFB rank should be 3 (BLK_N, BLK_K, PIPE)");
    static_assert(rank(tBgSFB) == Int<4>{}, "tBgSFB rank should be 4 (TMA, TMA_N, TMA_K, k)");
    static_assert(rank(tBsSFB) == Int<4>{}, "tBsSFB rank should be 4 (TMA, TMA_N, TMA_K, PIPE)");

    CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sA));                 // PIPE
    CUTE_STATIC_ASSERT_V(Int<N_STAGE>{} == size<2>(sB));                 // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));                  // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));                  // PIPE
   
    TiledMMA tiled_mma;
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
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tDsA = smem_thr_copy_A.partition_S(
            as_position_independent_swizzle_tensor(sA));        // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tDrA_copy_view = smem_thr_copy_A.retile_D(tDrA);     // (CPY,CPY_M,CPY_K)

    // B: S->R
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
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
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tDsSFA = smem_thr_copy_SFA.partition_S(
            as_position_independent_swizzle_tensor(sSFA));           // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tDrSFA_copy_view = smem_thr_copy_SFA.retile_D(tDrSFA);    // (CPY,CPY_M,CPY_K)

    // SFB: S->R
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk),
                                                                size<2>(tile_shape_mnk))
    );
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
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


    int num_k_tile = size<2>(gA);

    int k_tile_load = 0;
    int smem_pipe_read = 0;
    int smem_pipe_write = 0;

    auto load_tile = [&](int k_tile)
    {
        if(k_tile >= num_k_tile)
        {
            k_tile = 0;
            clear(tApA);
            clear(tBpB);
        }

        copy_if(g2s_copy_A, tApA, tAgA(_, _, _, k_tile), tAsA(_, _, _, smem_pipe_write));
        copy_if(g2s_copy_B, tBpB, tBgB(_, _, _, k_tile), tBsB(_, _, _, smem_pipe_write));
        cp_async_fence();
        copy_if(g2s_copy_SFA, tApA, tAgSFA(_, _, _, k_tile), tAsSFA(_, _, _, smem_pipe_write));
        copy_if(g2s_copy_SFB, tBpB, tBgSFB(_, _, _, k_tile), tBsSFB(_, _, _, smem_pipe_write));
        smem_pipe_write++;
        if(smem_pipe_write == N_STAGE) smem_pipe_write = 0;
    };
    

    for(int i = 0; i < N_STAGE - 1; i++)
    {
        // Issue N - 1 cp.async load
        load_tile(k_tile_load++);
    }
    cp_async_wait<N_STAGE - 2>();
    __syncthreads();

    // xxx_stage: point to current read stage tensor
    auto tDsA_stage   = tDsA(_, _, _, smem_pipe_read);
    auto tDsB_stage   = tDsB(_, _, _, smem_pipe_read);
    auto tDsSFA_stage = tDsSFA(_, _, _, smem_pipe_read);
    auto tDsSFB_stage = tDsSFB(_, _, _, smem_pipe_read);

    auto load_block = [&](int k_block)
    {
        copy(smem_tiled_copy_A, tDsA_stage(_, _, k_block), tDrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tDsB_stage(_, _, k_block), tDrB_copy_view(_, _, k_block));

        // using MMAOp = typename TiledMMA::MMA_Op;
        // FP4Shift::fp4_shift_A(MMAOp{}, tDrA_copy_view(_, _, k_block));
        // FP4Shift::fp4_shift_B(MMAOp{}, tDrB_copy_view(_, _, k_block));

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

                cp_async_wait<N_STAGE - 2>();
                __syncthreads();

                smem_pipe_read++;
                if(smem_pipe_read == N_STAGE) smem_pipe_read = 0;
                // Read next tile
                tDsA_stage   = tDsA(_, _, _, smem_pipe_read);
                tDsB_stage   = tDsB(_, _, _, smem_pipe_read);
                tDsSFA_stage = tDsSFA(_, _, _, smem_pipe_read);
                tDsSFB_stage = tDsSFB(_, _, _, smem_pipe_read);
            }
            // The first block, pre load next tile
            if(k_block == 0)
            {
                load_tile(k_tile_load++);
            }

            // copy the next block
            load_block(k_block_next);
            
            //gemm
            gemm(tiled_mma,
                    make_zip_tensor(tDrA(_, _, k_block), tDrSFA(_, _, k_block)),
                    make_zip_tensor(tDrB(_, _, k_block), tDrSFB(_, _, k_block)),
                    tDrD);
            if(0 && thread0())
            {
                print("tAgA: "); print_tensor(tAgA(_, _, _, k_tile));
                print("tAsA: "); print_tensor(tAsA(_, _, _, k_tile));
                print("tDsA: "); print_tensor(tDsA(_, _, k_block, k_tile));
                print("tDrA: "); print_tensor(tDrA_copy_view(_, _, k_block));
                print('\n');

                print("tBgB: "); print_tensor(tBgB(_, _, _, k_tile));
                print("tBsB: "); print_tensor(tBsB(_, _, _, k_tile));
                print("tDsB: "); print_tensor(tDsB(_, _, k_block, k_tile));
                print("tDrB: "); print_tensor(tDrB_copy_view(_, _, k_block));
                print('\n');

                print("tDrSFA: "); print_tensor(tDrSFA(_, _, k_block));
                print("tDrSFB: "); print_tensor(tDrSFB(_, _, k_block));
                print("tDrD: "); print_tensor(tDrD);
            }
        }
        // __syncthreads();
    }
    cp_async_wait<0>();
    
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    // Epilogue

    // C: G -> S
    GmemCopyC g2s_copy_C;
    auto g2s_thr_copy_C = g2s_copy_C.get_slice(thread_idx);
    Tensor tCgC = g2s_thr_copy_C.partition_S(gC);                   // (TMA,TMA_M,TMA_N)
    Tensor tCsC = g2s_thr_copy_C.partition_D(sC);                   // (TMA,TMA_M,TMA_N)

    Tensor tCpC = make_tensor<bool>(make_shape(size<1>(tCsC), size<2>(tCsC)), Stride<_1,_1>{});
    Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC)));    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor tCcC = g2s_thr_copy_C.partition_S(cC);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    CUTLASS_PRAGMA_UNROLL
    for(int m = 0; m < size<0>(tCpC); m++)
    {
        for(int n = 0; n < size<1>(tCpC); n++)
        {
            tCpC(m, n) = get<0>(tCcC(0, m, 0)) < residue_m && 
                        get<1>(tCcC(size<0>(tCcC)-1, 0, n)) < residue_n; 
        }
    }
    copy_if(g2s_copy_C, tCpC, tCgC, tCsC);

    if(0 && thread0() && block_x == 0 and block_y ==0)
    {
        print("tCpC "); print_tensor(tCpC);
        print("cC "); print_tensor(cC);
        print("tCcC "); print_tensor(tCcC);

    }
    __syncthreads();

    // C: S -> R
    auto s2r_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
    auto s2r_thr_copy_C = s2r_copy_C.get_thread_slice(thread_idx);
    Tensor tDsC = s2r_thr_copy_C.partition_S(sC);
    Tensor tDrC = make_tensor<ElementC>(layout(tDrD));
    Tensor tDrC_copy_view = s2r_thr_copy_C.retile_D(tDrC);
    copy(s2r_copy_C, tDsC, tDrC_copy_view);


    cutlass::NumericConverter<ElementD, typename TiledMMA::ValTypeD> converterD;
    Tensor tDrD_merge = make_tensor<ElementD>(layout(tDrD));
    for(int i = 0; i < size(tDrD_merge); i++)
    {
        // convert accumulator from 'float' to 'bfloat16' and accumulating
        // D <- D + C
        tDrD_merge(i) = converterD(tDrD(i) + tDrC(i));    
        // tDrD_merge(i) = converterD(tDrD(i));    

    }


    //Epilogue writeback

    // D: R -> S
    auto r2s_copy_D = make_tiled_copy_C(SmemCopyAtomD{}, tiled_mma);
    auto r2s_thr_copy_D = r2s_copy_D.get_thread_slice(thread_idx);
    Tensor tDrD_copy_view = r2s_thr_copy_D.retile_S(tDrD_merge); // replace with tDrD_merge
    Tensor tDsD_r2s = r2s_thr_copy_D.partition_D(sD);

    // step 1: Copy from register to shared memory
    copy(r2s_copy_D, tDrD_copy_view, tDsD_r2s);

    __syncthreads();
 
    // D: S -> G
    GmemCopyD s2g_copy_D;
    auto s2g_thr_copy_D = s2g_copy_D.get_slice(thread_idx);
    Tensor tDsD_s2g = s2g_thr_copy_D.partition_S(sD);
    Tensor tDgD_s2g = s2g_thr_copy_D.partition_D(gD);
    // step 2: Gmem Store
    copy_if(s2g_copy_D, tCpC, tDsD_s2g, tDgD_s2g);
        

    if(0 && thread0() && block_x == 0 && block_y == 0)
    {
        print_tensor(tDrD);
        print_tensor(tDsD_s2g);
        print_tensor(tDgD_s2g);

        print("LayoutA:      "); print(layout(mA)); print('\n');
        print("LayoutB:      "); print(layout(mB)); print('\n');
        print("LayoutC:      "); print(layout(mC)); print('\n');
        print("LayoutD:      "); print(layout(mD)); print('\n');
        print("LayoutSFA:    "); print(layout(mSFA)); print('\n');
        print("LayoutSFB:    "); print(layout(mSFB)); print('\n');

        print('\n');

        print("SmemLayoutA:      "); print(SmemLayoutA{}); print('\n');
        print("SmemLayoutB:      "); print(SmemLayoutB{}); print('\n');
        print("SmemLayoutC:      "); print(SmemLayoutC{}); print('\n');
        print("SmemLayoutD:      "); print(SmemLayoutD{}); print('\n');
        print("SmemLayoutSFA:    "); print(SmemLayoutSFA{}); print('\n');
        print("SmemLayoutSFB:    "); print(SmemLayoutSFA{}); print('\n');
        print('\n');

        print("mA:      "); print(mA); print('\n');
        print("mB:      "); print(mB); print('\n');
        print("mC:      "); print(mC); print('\n');
        print("mD:      "); print(mD); print('\n');
        print("mSFA:    "); print(mSFA); print('\n');
        print("mSFB:    "); print(mSFB); print('\n');
        print('\n');

        print("gA:      "); print(gA); print('\n');
        print("gB:      "); print(gB); print('\n');
        print("gC:      "); print(gC); print('\n');
        print("gD:      "); print(gD); print('\n');
        print("gSFA:    "); print(gSFA); print('\n');
        print("gSFB:    "); print(gSFB); print('\n');
        print('\n');

        print("sA:      "); print(sA); print('\n');
        print("sB:      "); print(sB); print('\n');
        print("sC:      "); print(sC); print('\n');
        print("sD:      "); print(sD); print('\n');
        print("sSFA:    "); print(sSFA); print('\n');
        print("sSFB:    "); print(sSFB); print('\n');
        print('\n');

        print("tAgA:    "); print(tAgA); print('\n');
        print("tBgB:    "); print(tBgB); print('\n');
        print("tCgC:    "); print(tCgC); print('\n');
        print("tAgSFA:  "); print(tAgSFA); print('\n');
        print("tBgSFB:  "); print(tBgSFB); print('\n');
        print('\n');

        print("tAsA:    "); print(tAsA); print('\n');
        print("tBsB:    "); print(tBsB); print('\n');
        print("tCsC:    "); print(tCsC); print('\n');
        print("tAsSFA:  "); print(tAsSFA); print('\n');
        print("tBsSFB:  "); print(tBsSFB); print('\n');
        print('\n');

        print("tDsA:    "); print(tDsA); print('\n');
        print("tDsB:    "); print(tDsB); print('\n');
        print("tDsSFA:  "); print(tDsSFA); print('\n');
        print("tDsSFB:  "); print(tDsSFB); print('\n');
        print('\n');

        print("tDrA_copy_view:    "); print(tDrA_copy_view); print('\n');
        print("tDrB_copy_view:    "); print(tDrB_copy_view); print('\n');
        print("tDrSFA_copy_view:  "); print(tDrSFA_copy_view); print('\n');
        print("tDrSFB_copy_view:  "); print(tDrSFB_copy_view); print('\n');
        print('\n');

        print("tDrA:    "); print(tDrA); print('\n');
        print("tDrB:    "); print(tDrB); print('\n');
        print("tDrC:    "); print(tDrC); print('\n');
        print("tDrSFA:  "); print(tDrSFA); print('\n');
        print("tDrSFB:  "); print(tDrSFB); print('\n');
        print('\n');

    }
}

#endif //SM120_BLOCKSCALED_GEMM_KERNEL_SM120_MULTISTAGE_CPASYNC_H
