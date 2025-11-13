#ifndef SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
#define SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
#include "cutlass/cutlass.h"

namespace TmaSmem{
template<
        class SmemLayoutA, class SmemLayoutB, class SmemLayoutC, class SmemLayoutSFA, class SmemLayoutSFB,
        class SmemAllocTypeA, class SmemAllocTypeB, class ElementC, class ElementSF,
        class MainloopPipeline, class EpiloguePipeline
>
struct MainloopSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
        alignas(1024) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;

        cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;
    using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) MainloopPipelineStorage mainloop_pipeline_storage;
    using EpiloguePipelineStorage = typename EpiloguePipeline::SharedStorage;
    alignas(16) EpiloguePipelineStorage epilogue_pipeline_storage;
};

template<
        class SmemLayoutD, class ElementD
>
struct EpilogueSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    } tensors;
};
}
#endif //SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_H
