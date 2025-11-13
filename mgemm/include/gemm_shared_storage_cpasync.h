#ifndef SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_CPASYNC_H
#define SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_CPASYNC_H
#include "cutlass/cutlass.h"

namespace CpAsyncSmem{
template<
        class SmemLayoutA, class SmemLayoutB, class SmemLayoutSFA, class SmemLayoutSFB,
        class SmemAllocTypeA, class SmemAllocTypeB, class ElementSF
>
struct MainloopSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;

        alignas(1024) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        alignas(1024) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;
};

template<
    class SmemLayoutC, class ElementC,
    class SmemLayoutD, class ElementD
>
struct EpilogueSharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
        alignas(1024) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;
    } tensors;
};
}
#endif //SM120_BLOCKSCALED_GEMM_GEMM_SHARED_STORAGE_CPASYNC_H
