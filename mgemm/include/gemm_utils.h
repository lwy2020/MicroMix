#ifndef SM120_BLOCKSCALED_GEMM_GEMM_UTILS_H
#define SM120_BLOCKSCALED_GEMM_GEMM_UTILS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// For CuTe debug
//#define DEBUG

#ifdef DEBUG
#define KERNEL_DEBUG(x) \
    do {             \
        if(thread0())\
        {               \
            print(__FILE__);                                                 \
            print(":");                                                      \
            print(__LINE__);                                                 \
            print("\n");                                                     \
            print(#x ": "); print(x); print("\n"); \
        }                \
    } while (0)
#else
#define KERNEL_DEBUG(x) do {} while (0)
#endif

#ifdef DEBUG
#define KERNEL_DEBUG_TENSOR(x) \
    do {             \
        if(thread0())\
        {               \
            print(__FILE__);                                                 \
            print(":");                                                      \
            print(__LINE__);                                                 \
            print("\n");                                                     \
            print(#x ": "); print_tensor(x); print("\n"); \
        }                \
    } while (0)
#else
#define DEBUG_PRINT(x) do {} while (0)
#endif
#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINT_TENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

inline void checkCudaError(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define checkCudaLastErrors() checkCudaError(__FILE__, __LINE__)

template <typename KernelFunc>
inline void setKernelSmemSize(KernelFunc &kernel, size_t smem_size) {
    cudaError_t err = cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set shared memory size: %s\n",
                cudaGetErrorString(err));
        exit(1);
    }
}
#endif //SM120_BLOCKSCALED_GEMM_GEMM_UTILS_H
