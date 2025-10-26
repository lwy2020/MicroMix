#ifndef SM120_BLOCKSCALED_GEMM_GEMM_UTILS_H
#define SM120_BLOCKSCALED_GEMM_GEMM_UTILS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/numeric_conversion.h"

template <typename T>
void initialize_matrix(T* data, uint32_t size, float val)
{
    cutlass::NumericConverter<T, float, cutlass::FloatRoundStyle::round_to_nearest> converter;
    for (int i = 0; i < size; ++i) {
        data[i] = converter(val);
    }
}
template <typename T>
void initialize_matrix_random(T* data, uint32_t size) {

    cutlass::NumericConverter<T, float, cutlass::FloatRoundStyle::round_to_nearest> converter;

    float range = 1.0f;

    // 针对不同精度选择不同动态范围
    if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        // e2m1: [-2, 2]
        range = 1.5f;
    } else if constexpr (std::is_same_v<T, cutlass::float_e3m2_t>) {
        // e3m2: [-16, 16]
        range = 4.0f;
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        // e4m3: [-448, 448]
        range = 40.0f;
    } else {
        // 默认 float32 范围
        range = 1.0f;
    }

    for (int i = 0; i < size; ++i) {
        float rnd = static_cast<float>(std::rand()) / RAND_MAX;  // [0, 1)
        float val = (rnd * 2.0f - 1.0f) * range;                 // [-range, range]
        data[i] = converter(val);
    }
}


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
