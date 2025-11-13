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
        cutlass::float_e2m1_t *AN,
        cutlass::float_e2m1_t *BN,
        cutlass::float_e3m2_t *AS,
        cutlass::float_e3m2_t *BS,
        cutlass::float_e4m3_t *AO,
        cutlass::float_e4m3_t *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        cutlass::bfloat16_t *C,
        cutlass::bfloat16_t *D,
        cutlass::float_ue8m0_t *SFAN,
        cutlass::float_ue8m0_t *SFBN,
        cutlass::float_ue8m0_t *SFAS,
        cutlass::float_ue8m0_t *SFBS,
        cutlass::float_ue8m0_t *SFAO,
        cutlass::float_ue8m0_t *SFBO
);

void matmul_host_dev(
        cutlass::float_e2m1_t *AN,
        cutlass::float_e2m1_t *BN,
        cutlass::float_e3m2_t *AS,
        cutlass::float_e3m2_t *BS,
        cutlass::float_e4m3_t *AO,
        cutlass::float_e4m3_t *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        cutlass::bfloat16_t *C,
        cutlass::bfloat16_t *D,
        cutlass::float_ue8m0_t *SFAN,
        cutlass::float_ue8m0_t *SFBN,
        cutlass::float_ue8m0_t *SFAS,
        cutlass::float_ue8m0_t *SFBS,
        cutlass::float_ue8m0_t *SFAO,
        cutlass::float_ue8m0_t *SFBO,
        cudaStream_t stream = 0
);

void matmul_w4_host(
        cutlass::float_e2m1_t *AN,
        cutlass::float_e2m1_t *BN,
        cutlass::float_e3m2_t *AS,
        cutlass::float_e2m1_t *BS,
        cutlass::float_e4m3_t *AO,
        cutlass::float_e2m1_t *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        cutlass::bfloat16_t *C,
        cutlass::bfloat16_t *D,
        cutlass::float_ue8m0_t *SFAN,
        cutlass::float_ue8m0_t *SFBN,
        cutlass::float_ue8m0_t *SFAS,
        cutlass::float_ue8m0_t *SFBS,
        cutlass::float_ue8m0_t *SFAO,
        cutlass::float_ue8m0_t *SFBO
);