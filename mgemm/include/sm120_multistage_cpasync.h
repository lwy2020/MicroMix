#ifndef SM120_BLOCKSCALED_GEMM_SM120_MULTISTAGE_CPASYNC_H
#define SM120_BLOCKSCALED_GEMM_SM120_MULTISTAGE_CPASYNC_H
#include "cutlass/cutlass.h"
#include "gemm_utils.h"
#include "sm120_sf_layout.h"
#include "kernel_sm120_multistage_cpasync.h"
// #include "kernel_sm120_multistage_tma_warpspecialized.h"
#include "gemm_shared_storage_cpasync.h"

using namespace cute;

template<
        int N_STAGE = 2, int BM = 128, int BN = 128, int BK = 128, 
        class ElementA, class ElementB, class ElementC, class ElementD, class ElementSF>
void
gemm_host_tn_cpasync(ElementA *ptr_A, ElementSF *ptr_SFA,
             ElementB *ptr_B, ElementSF *ptr_SFB,
             ElementC *ptr_C, 
             ElementD *ptr_D, 
             int M, int N, int K,
             cudaStream_t stream = 0) {

    // Input pointer check
    assert(ptr_A != nullptr && ptr_B != nullptr && ptr_C != nullptr && ptr_D != nullptr &&
        ptr_SFA != nullptr && ptr_SFB != nullptr);
    // Input shape check
    assert(BM > 0 && BN > 0 && BK > 0 && M > 0 && N > 0 && K > 0 && K % BK == 0);
    
    // Input type check
    // static_assert(BK == 128);
    static_assert(sizeof_bits_v<ElementA> <= 8);
    static_assert(sizeof_bits_v<ElementB> <= 8);
    static_assert(is_same_v<ElementA, ElementB>);
    static_assert(is_same_v<ElementSF, cutlass::float_ue8m0_t>);

    constexpr bool isF4 = is_same_v<ElementA, ElementB> && is_same_v<ElementA, cutlass::float_e2m1_t>;
    constexpr bool isF8F6F4 = !isF4;
    // Define problem size (dynamic)
    auto const prob_shape = make_shape(M, N, K);
    // Define CTA tile sizes (static), assume (32, 32, 128)
    using TileShape_MNK = decltype(make_tile(Int<BM>{}, Int<BN>{}, Int<BK>{}));     // (BLK_M, BLK_N, BLK_K)
    using EpilogueTile_MN = decltype(make_tile(Int<BM>{}, Int<BN>{}));
    // Only support 32 SF vector size
    constexpr int SFVecSize = 32;
    // float accumulator type
    using ElementAccumulator = float;

    auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
    auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
    auto layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

    auto layout_SFA = make_layout(make_shape(M, make_shape(Int<SFVecSize>{}, K/SFVecSize)),
                                  make_stride(K/SFVecSize, make_stride(_0{}, _1{})));
    auto layout_SFB = make_layout(make_shape(N, make_shape(Int<SFVecSize>{}, K/SFVecSize)),
                                  make_stride(K/SFVecSize, make_stride(_0{}, _1{})));
    // auto [layout_SFA, layout_SFB] = sm120_get_SF_layout<BM, BN, BK, SFVecSize>(M, N, K);

    using MMA_OP = decltype(cute::rr_blockscaled_op_selector_sm120<ElementA,
            ElementB,
            ElementAccumulator,
            ElementSF,
            SFVecSize,
            isF8F6F4
    >());
    using MmaAtom = MMA_Atom<MMA_OP>;
    using MmaAtomShape_MNK = typename MmaAtom::Traits::Shape_MNK;
    // Thread expands, four warps per cta
    using MMA_Layout_M = _2;
    using MMA_Layout_N = _2;
    using MMA_Layout_K = _1; // Always be 1
    auto mma_layout_mnk = make_layout(make_shape(MMA_Layout_M{}, MMA_Layout_N{}, MMA_Layout_K{}));

    // Value expand
    using MMA_Perm_M = _32;
    //对 N 维度进行重排，确保同一个warp负责的部分连续。
    using MMA_Perm_N = _32; //Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>;
    using MMA_Perm_K = Int<size<2>(MmaAtomShape_MNK{})>; // Same as MMAop K
    auto mma_perm_mnk = make_tile(MMA_Perm_M{}, MMA_Perm_N{},  MMA_Perm_K{});


    using TiledMMA = decltype(make_tiled_mma(MmaAtom{}, mma_layout_mnk, mma_perm_mnk)); // (32, 32, 32)


    // SmemAllocTypeA 可能和 ElementA不同
    // 例如当 ElementA为 sub-byte float_e2m3_t， SmemAllocTypeA 需要为 int8_t
    using SmemAllocTypeA = cute::conditional_t<isF8F6F4, uint8_t, typename TiledMMA::ValTypeA>;
    using SmemAllocTypeB = cute::conditional_t<isF8F6F4, uint8_t, typename TiledMMA::ValTypeB>;
    using SmemAllocTypeSF = ElementSF;

    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<SmemAllocTypeA, Int<BK>>());
    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<SmemAllocTypeB, Int<BK>>());
    using SmemLayoutAtomD = decltype(cutlass::epilogue::collective::detail::sm120_get_epilogue_smem_swizzle_layout_atom
            <Stride<_32, _1>, ElementD, EpilogueTile_MN>());
    using SmemLayoutAtomC = SmemLayoutAtomD;

    using SF_NUM_K = Int<BK / SFVecSize>;
    using SFShape_K = Shape<Int<SFVecSize>, SF_NUM_K>;
    using SFStride_K = Stride<_0, _1>;

    using SmemLayoutAtomSFA = decltype(make_layout(append(Int<BM>{}, SFShape_K{}), 
                                                    append(SF_NUM_K{}, SFStride_K{})));
    using SmemLayoutAtomSFB = decltype(make_layout(append(Int<BN>{}, SFShape_K{}), 
                                                    append(SF_NUM_K{}, SFStride_K{})));

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape_MNK{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape_MNK{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape_MNK{}) % size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape_MNK{}) % size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    // Tile along modes in a way that maximizes the TMA box size.
    using SmemLayoutA = decltype(tile_to_shape(
            SmemLayoutAtomA{},
            make_shape(Int<BM>{}, Int<BK>{}, Int<N_STAGE>{})));
    using SmemLayoutB = decltype(tile_to_shape(
            SmemLayoutAtomB{},
            make_shape(Int<BN>{}, Int<BK>{}, Int<N_STAGE>{})));
    using SmemLayoutD = decltype(tile_to_shape(
            SmemLayoutAtomD{},
            make_shape(Int<BM>{}, Int<BN>{}),
            Step<_2, _1>{}));
    using SmemLayoutC = SmemLayoutD;
/*    print("SmemLayoutAtomC:      "); print(SmemLayoutAtomC{}); print('\n');
    print("SmemLayoutC:           "); print(SmemLayoutC{}); print('\n');*/
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout A must be rank 3 (BM,BK,PIPE)");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout B must be rank 3 (BN,BK,PIPE)");
    static_assert(rank(SmemLayoutC{}) == 2, "Smem layout C must be rank 2 (BM,BN)");
    static_assert(rank(SmemLayoutD{}) == 2, "Smem layout D must be rank 2 (BM,BN)");

    using SmemLayoutSFA = decltype(make_layout(
            append(shape(SmemLayoutAtomSFA{}), Int<N_STAGE>{}),
            append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
    ));

    using SmemLayoutSFB = decltype(make_layout(
            append(shape(SmemLayoutAtomSFB{}), Int<N_STAGE>{}),
            append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
    ));



    using MyMainloopSharedStorage = CpAsyncSmem::MainloopSharedStorage<
            SmemLayoutA, SmemLayoutB, SmemLayoutSFA, SmemLayoutSFB,
            SmemAllocTypeA, SmemAllocTypeB, ElementSF
    >;
    using MyEpilogueSharedStorage = CpAsyncSmem::EpilogueSharedStorage<SmemLayoutC, ElementC, SmemLayoutD, ElementD>;

    static_assert(sizeof(MyMainloopSharedStorage) <= cutlass::arch::sm120_smem_capacity_bytes);
    static_assert(sizeof(MyEpilogueSharedStorage) <= cutlass::arch::sm120_smem_capacity_bytes);
    constexpr int shared_storage_size = std::max(sizeof(MyMainloopSharedStorage), sizeof(MyEpilogueSharedStorage));
    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                    cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
            cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));

    static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                    cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
            cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

    using GmemCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, SmemAllocTypeA>;
    using GmemCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, SmemAllocTypeA>;
    using GmemCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, ElementC>;
    using GmemCopyAtomD = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
    using GmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

    using CopyThreadLayout = Layout<Shape<_32, _4>, Stride<_4, _1>>;
    using CopyValueLayoutA = Layout<Shape<_1, Int<128/sizeof_bits_v<SmemAllocTypeA>>>>;
    using CopyValueLayoutB = Layout<Shape<_1, Int<128/sizeof_bits_v<SmemAllocTypeA>>>>;
    using CopyValueLayoutC = Layout<Shape<_1, Int<128/sizeof_bits_v<ElementC>>>>;
    using CopyValueLayoutD = Layout<Shape<_1, Int<128/sizeof_bits_v<ElementD>>>>;
    using CopyValueLayoutSF = Layout<Shape<_1, _1>>;


    using GmemCopyA = decltype(make_tiled_copy(GmemCopyAtomA{}, CopyThreadLayout{}, CopyValueLayoutA{}));
    using GmemCopyB = decltype(make_tiled_copy(GmemCopyAtomB{}, CopyThreadLayout{}, CopyValueLayoutB{}));
    using GmemCopyC = decltype(make_tiled_copy(GmemCopyAtomC{}, CopyThreadLayout{}, CopyValueLayoutC{}));
    using GmemCopyD = decltype(make_tiled_copy(GmemCopyAtomD{}, CopyThreadLayout{}, CopyValueLayoutD{}));
    using GmemCopySFA = decltype(make_tiled_copy(GmemCopyAtomSF{}, CopyThreadLayout{}, CopyValueLayoutSF{}));
    using GmemCopySFB = decltype(make_tiled_copy(GmemCopyAtomSF{}, CopyThreadLayout{}, CopyValueLayoutSF{}));

    // print("GmemCopyA "); print(GmemCopyA{}); print('\n');
    // print("GmemCopyB "); print(GmemCopyB{}); print('\n');


    using SmemCopyAtomA = Copy_Atom<decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_A<
            ElementA, ElementB, isF8F6F4>()), SmemAllocTypeA>;
    using SmemCopyAtomB = Copy_Atom<decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
            ElementA, ElementB, isF8F6F4>()), SmemAllocTypeB>;
    // using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, SmemAllocTypeA>; // 16x32 for 8-bit element
    // using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, SmemAllocTypeB>; // 16x32 for 8-bit element
    using SmemCopyAtomC = Copy_Atom<SM75_U32x4_LDSM_N, ElementC>;
    using SmemCopyAtomD = Copy_Atom<SM90_U32x4_STSM_N, ElementD>;       // 8x16 for 16-bit element?

    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>;
    using SmemCopyAtomSFA = SmemCopyAtomSF;
    using SmemCopyAtomSFB = SmemCopyAtomSF;

    using TmaInternalElementA = cute::conditional_t<not isF8F6F4,
            ElementA,
            cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                    cutlass::detail::float_e2m1_unpacksmem_t,
                    cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                            cutlass::detail::float_e2m3_unpacksmem_t,
                            cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                    cutlass::detail::float_e3m2_unpacksmem_t,
                                    uint_bit_t<sizeof_bits_v<ElementA>>>>>>;

    using TmaInternalElementB = cute::conditional_t<not isF8F6F4,
            ElementB,
            cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                    cutlass::detail::float_e2m1_unpacksmem_t,
                    cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                            cutlass::detail::float_e2m3_unpacksmem_t,
                            cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                    cutlass::detail::float_e3m2_unpacksmem_t,
                                    uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

    Tensor mA = make_tensor(recast_ptr<SmemAllocTypeA>(ptr_A), layout_A);
    Tensor mB = make_tensor(recast_ptr<SmemAllocTypeA>(ptr_B), layout_B);
    Tensor mSFA = make_tensor(make_gmem_ptr(ptr_SFA), layout_SFA);
    Tensor mSFB = make_tensor(make_gmem_ptr(ptr_SFB), layout_SFB);
    Tensor mC = make_tensor(make_gmem_ptr(ptr_C), layout_C);
    Tensor mD = make_tensor(make_gmem_ptr(ptr_D), layout_D);

    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blockDim(size(TiledMMA{}));
    // printf("fp%d M=%d N=%d K=%d\n", sizeof_bits_v<ElementA>, M, N, K);
    // A helper lambda to avoid duplicating the kernel launch code
    auto launch = [&](auto kernel_ptr) {
        setKernelSmemSize(kernel_ptr, shared_storage_size);
        checkCudaLastErrors();

        // Launch Kernel
        kernel_ptr<<<gridDim, blockDim, shared_storage_size, stream>>>(mA, mB, mSFA, mSFB, mC, mD);
        // cudaDeviceSynchronize();
        
        checkCudaLastErrors();
    };

    // debug_print_cp<<<1,1>>>(ptr_A, ptr_B, M, N, K);
    // cudaDeviceSynchronize();
    // checkCudaLastErrors();
    
    auto kernel_ptr = gemm_device_multistage_cpasync<
            TmaInternalElementA, decltype(mA), SmemLayoutA, GmemCopyA, SmemCopyAtomA,
            TmaInternalElementB, decltype(mB), SmemLayoutB, GmemCopyB, SmemCopyAtomB,
            ElementSF, decltype(mSFA), SmemLayoutSFA, GmemCopySFA, SmemCopyAtomSFA,
            ElementSF, decltype(mSFB), SmemLayoutSFB, GmemCopySFB, SmemCopyAtomSFB,
            ElementC, decltype(mC), SmemLayoutC, GmemCopyC, SmemCopyAtomC,
            ElementD, decltype(mD), SmemLayoutD, GmemCopyD, SmemCopyAtomD,
            TiledMMA, TileShape_MNK, 
            MyMainloopSharedStorage, MyEpilogueSharedStorage, N_STAGE>;

    launch(kernel_ptr);

}

#endif //SM120_BLOCKSCALED_GEMM_SM120_MULTISTAGE_CPASYNC_H
