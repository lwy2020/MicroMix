#ifndef SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
#define SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
#include "cutlass/cutlass.h"

template<
        class SmemLayoutA, class SmemLayoutB, class SmemLayoutSFA, class SmemLayoutSFB,
        class SmemAllocTypeA, class SmemAllocTypeB, class ElementSF,
        class MainloopPipeline
>
struct MainloopSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
};

template<
        class SmemLayoutC, class ElementC
>
struct EpilogueSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;
    } tensors;
    using TypeC = ElementC;
};
#endif //SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
