#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "helper.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void matmul_host(
        const cutlass::float_e2m1_t *AN,
        const cutlass::float_e2m1_t *BN,
        const cutlass::float_e3m2_t *AS,
        const cutlass::float_e3m2_t *BS,
        const cutlass::float_e4m3_t *AO,
        const cutlass::float_e4m3_t *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        cutlass::bfloat16_t *C,
        cutlass::bfloat16_t *D,
        const cutlass::float_ue8m0_t *SFAN,
        const cutlass::float_ue8m0_t *SFBN,
        const cutlass::float_ue8m0_t *SFAS,
        const cutlass::float_ue8m0_t *SFBS,
        const cutlass::float_ue8m0_t *SFAO,
        const cutlass::float_ue8m0_t *SFBO
);