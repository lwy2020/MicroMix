#ifndef SM120_BLOCKSCALED_GEMM_SM120_MXF8_MULTISTAGE_TMA_H
#define SM120_BLOCKSCALED_GEMM_SM120_MXF8_MULTISTAGE_TMA_H
#include "cutlass/cutlass.h"
#include "gemm_utils.h"
#include "sm120_sf_layout.h"
// #include "kernel_sm120_multistage_tma.h"
#include "kernel_sm120_multistage_tma_warpspecialized.h"
#include "gemm_shared_storage.h"

using namespace cute;

template<
        int N_STAGE = 2, int BM = 128, int BN = 128, int BK = 128, 
        class ElementA, class ElementB, class ElementC, class ElementD, class ElementSF>
void
gemm_host_tn(ElementA *ptr_A, ElementSF *ptr_SFA,
             ElementB *ptr_B, ElementSF *ptr_SFB,
             ElementC *ptr_C, 
             ElementD *ptr_D, 
             int M, int N, int K) {

    if (ptr_A == nullptr || ptr_B == nullptr ||
        ptr_C == nullptr || ptr_D == nullptr ||
        ptr_SFA == nullptr || ptr_SFB == nullptr) {
        std::cerr << "Error: Null pointer provided to gemm_host_tn" << std::endl;
        return;
    }

    // Shape check
    assert(BM > 0 && BN > 0 && BK > 0 && M > 0 && N > 0 && K > 0 && K % BK == 0);

    // Input type check
    static_assert(BK == 128);
    static_assert(sizeof_bits_v<ElementA> <= 8);
    static_assert(sizeof_bits_v<ElementB> <= 8);
    static_assert(is_same_v<ElementSF, cutlass::float_ue8m0_t>);

    constexpr bool isF4 = is_same_v<ElementA, ElementB> && is_same_v<ElementA, cutlass::float_e2m1_t>;
    constexpr bool isF8F6F4 = !isF4;
    // Define problem size (dynamic)
    auto const prob_shape = make_shape(M, N, K);
    // Define CTA tile sizes (static), assume (32, 32, 128)
    using TileShape_MNK = decltype(make_tile(Int<BM>{}, Int<BN>{}, Int<BK>{}));     // (BLK_M, BLK_N, BLK_K)
    using EpilogueTile_MN = decltype(make_tile(Int<BM>{}, Int<BN>{}));
    // Only support 32 SF vector size
    const int SFVecSize = 32;
    // float accumulator type
    using ElementAccumulator = float;

    auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
    auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
    auto layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

    auto [layout_SFA, layout_SFB] = sm120_get_SF_layout<BM, BN, BK, SFVecSize>(M, N, K);

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

    // Construct SMEM layout for SF
    // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
    // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col).
    // 32bits corresponds to the TMEM word size
//    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
    using Blk_MN    = Int<BM>;//typename Sm1xxBlkScaledConfig::Blk_MN; // 128
    using Blk_SF    = _4;//typename Sm1xxBlkScaledConfig::Blk_SF; //   4
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    static constexpr int MMA_NSF = size<2>(typename TiledMMA::AtomShape_MNK{}) / SFVecSize;
    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<Int<BM/4>,_4>;
    using mnBasicBlockStride = Stride<_16,_4>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;

    using sSFA_shapeM       = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{},   mnBasicBlockShape{}));
    using sSF_strideMN      = decltype(prepend(                        Blk_Elems{},  mnBasicBlockStride{}));
    using sSFA_strideM      = sSF_strideMN;
    using sSF_shapeK        = decltype(prepend(make_shape( Blk_SF{}/Int<MMA_NSF>{},   size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),  kBasicBlockShape{}));

    using sSFA_strideK      = decltype(prepend(make_stride(         Int<MMA_NSF>{},   size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFA_shape        = decltype(make_shape(  sSFA_shapeM{},   sSF_shapeK{}));
    using sSFA_stride       = decltype(make_stride(sSFA_strideM{}, sSFA_strideK{}));
    using SmemLayoutAtomSFA = decltype(make_layout(  sSFA_shape{},  sSFA_stride{}));

    using sSFB_shapeN       = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{},   mnBasicBlockShape{}));
    using sSFB_strideN      = sSF_strideMN;
    using sSFB_strideK      = decltype(prepend(make_stride(Int<MMA_NSF>{},   size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFB_shape        = decltype(make_shape(  sSFB_shapeN{},   sSF_shapeK{}));
    using sSFB_stride       = decltype(make_stride(sSFB_strideN{}, sSFB_strideK{}));
    using SmemLayoutAtomSFB = decltype(make_layout(  sSFB_shape{},  sSFB_stride{}));

//    using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
//    using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

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


    using MainloopPipeline = cutlass::PipelineTmaAsync<N_STAGE>;
    using EpiloguePipeline = cutlass::PipelineTmaAsync<1>;
//    using PipelineParams = typename MainloopPipeline::Params;
    using MainloopPipelineState = typename cutlass::PipelineState<N_STAGE>;
    using EpiloguePipelineState = typename cutlass::PipelineState<1>;

    using MyMainloopSharedStorage = MainloopSharedStorage<
            SmemLayoutA, SmemLayoutB, SmemLayoutC, SmemLayoutSFA, SmemLayoutSFB,
            SmemAllocTypeA, SmemAllocTypeB, ElementC, ElementSF,
            MainloopPipeline, EpiloguePipeline
    >;
    using MyEpilogueSharedStorage = EpilogueSharedStorage<SmemLayoutD, ElementD>;

    static constexpr int MainloopSharedStorageSize = sizeof(MyMainloopSharedStorage);
    static constexpr int EpilogueSharedStorageSize = sizeof(MyEpilogueSharedStorage);
    constexpr int SharedStorageSize = std::max(MainloopSharedStorageSize, EpilogueSharedStorageSize);
    static_assert(MainloopSharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
    static_assert(EpilogueSharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                    cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
            cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));

    static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                    cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
            cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

    constexpr uint32_t MainloopTmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;
    constexpr uint32_t EpilogueTmaTransactionBytes = static_cast<uint32_t>(
            cutlass::bits_to_bytes(cosize(SmemLayoutC{}) * sizeof_bits_v<ElementC>));

    //sub-byte set as unpackment_t
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
    // used for TMA define
    Tensor mA_raw = make_tensor(recast_ptr<TmaInternalElementA>(ptr_A), layout_A);  // (M, K)
    Tensor mB_raw = make_tensor(recast_ptr<TmaInternalElementB>(ptr_B), layout_B);  // (N, K)
    Tensor mSFA_raw = make_tensor(ptr_SFA, layout_SFA);                             // (M, K/32) ?
    Tensor mSFB_raw = make_tensor(ptr_SFB, layout_SFB);                             // (N, K/32) ?
    Tensor mD_raw = make_tensor(ptr_D, layout_D);                                   // (M, N) ?
    Tensor mC_raw = make_tensor(ptr_C, layout_C);                                   // (M, N) ?

    // TMA Copy Define
    auto tma_load_a = make_tma_copy(
            SM90_TMA_LOAD{},
            mA_raw,
            SmemLayoutA{}(_, _, Int<0>{}),
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    auto tma_load_b = make_tma_copy(
            SM90_TMA_LOAD{},
            mB_raw,
            SmemLayoutB{}(_, _, Int<0>{}),
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    auto tma_load_sfa = make_tma_copy<uint16_t>(
            SM90_TMA_LOAD{},
            mSFA_raw,
            SmemLayoutSFA{}(_, _, Int<0>{}),
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    auto tma_load_sfb = make_tma_copy<uint16_t>(
            SM90_TMA_LOAD{},
            mSFB_raw,
            SmemLayoutSFB{}(_, _, Int<0>{}),
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    auto tma_load_c = make_tma_copy(
            SM90_TMA_LOAD{},
            mC_raw,
            SmemLayoutC{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    auto tma_store_d = make_tma_copy(
            SM90_TMA_STORE{},
            mD_raw,
            SmemLayoutD{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{}); // No programmatic multicast

    using SmemCopyAtomA = Copy_Atom<decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_A<
            ElementA, ElementB, isF8F6F4>()), SmemAllocTypeA>;
    using SmemCopyAtomB = Copy_Atom<decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
            ElementA, ElementB, isF8F6F4>()), SmemAllocTypeB>;
    //using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, SmemAllocTypeA>; // 16x32 for 8-bit element
    //using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, SmemAllocTypeB>; // 16x32 for 8-bit element
    using SmemCopyAtomC = Copy_Atom<SM75_U32x4_LDSM_N, ElementC>;
    using SmemCopyAtomD = Copy_Atom<SM90_U32x4_STSM_N, ElementD>;       // 8x16 for 16-bit element?

    // auto-vectorized LDS
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>;
    using SmemCopyAtomSFA = SmemCopyAtomSF;
    using SmemCopyAtomSFB = SmemCopyAtomSF;

    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blockDim(2 * cutlass::NumThreadsPerWarpGroup);
    static_assert(cutlass::NumThreadsPerWarpGroup == size(TiledMMA{}));

    // A helper lambda to avoid duplicating the kernel launch code
    auto launch = [&](auto kernel_ptr) {
        setKernelSmemSize(kernel_ptr, SharedStorageSize);
        checkCudaLastErrors();

        // Launch Kernel
        kernel_ptr<<<gridDim, blockDim, SharedStorageSize>>>(tma_load_a, layout_A,
                                                             tma_load_b, layout_B,
                                                             tma_load_sfa, layout_SFA,
                                                             tma_load_sfb, layout_SFB,
                                                             tma_load_c, layout_C,
                                                             tma_store_d, layout_D);
        cudaDeviceSynchronize();
        checkCudaLastErrors();
    };


    auto kernel_ptr = gemm_device_multistage_warpspecialized<
            ElementA, decltype(layout_A), SmemLayoutA, decltype(tma_load_a), SmemCopyAtomA,
            ElementB, decltype(layout_B), SmemLayoutB, decltype(tma_load_b), SmemCopyAtomB,
            ElementSF, decltype(layout_SFA), SmemLayoutSFA, decltype(tma_load_sfa), SmemCopyAtomSFA,
            ElementSF, decltype(layout_SFB), SmemLayoutSFB, decltype(tma_load_sfb), SmemCopyAtomSFB,
            ElementC, decltype(layout_C), SmemLayoutC, decltype(tma_load_c), SmemCopyAtomC,
            ElementD, decltype(layout_D), SmemLayoutD, decltype(tma_store_d), SmemCopyAtomD,
            TiledMMA, decltype(prob_shape), TileShape_MNK, 
            MyMainloopSharedStorage, MyEpilogueSharedStorage,
            MainloopPipeline, MainloopPipelineState, 
            EpiloguePipeline, EpiloguePipelineState, 
            N_STAGE, MainloopTmaTransactionBytes, EpilogueTmaTransactionBytes>;

    launch(kernel_ptr);

}

#endif //SM120_BLOCKSCALED_GEMM_SM120_MXF8_MULTISTAGE_TMA_H
