#include "w4a4.h"
#include "w6a6.h"
#include "w8a8.h"
#include "w4a6.h"
#include "w4a8.h"
#include "gemm.h"
#include "sm120_multistage_tma.h"
#include "sm120_multistage_cpasync.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementANormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         ElementASensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using         ElementAOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand

// B matrix configuration
using         ElementBNormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         ElementBSensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for B matrix operand
using         ElementBOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for B matrix operand

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand

void matmul_host(
        ElementANormal::DataType *AN,
        ElementBNormal::DataType *BN,
        ElementASensitive::DataType *AS,
        ElementBSensitive::DataType *BS,
        ElementAOutlier::DataType *AO,
        ElementBOutlier::DataType *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        ElementC *C,
        ElementD *D,
        ElementANormal::ScaleFactorType *SFAN,
        ElementBNormal::ScaleFactorType *SFBN,
        ElementASensitive::ScaleFactorType *SFAS,
        ElementBSensitive::ScaleFactorType *SFBS,
        ElementAOutlier::ScaleFactorType *SFAO,
        ElementBOutlier::ScaleFactorType *SFBO
)
{
    
    // constexpr int sm_count_threshold = 150;
    // bool use_cutlass_kernel = (cute::ceil_div(M , 128) * cute::ceil_div(N , 128)) > sm_count_threshold;
    // printf("M = %d, N = %d, KN = %d, KS = %d, KO = %d, CUTLASS_ON: %d\n", M, N, KN, KS, KO, use_cutlass_kernel);
 
    if(KN!=0)matmul_host_w4a4(AN, BN, M, N, KN, C, D, SFAN, SFBN);
    if(KS!=0)matmul_host_w6a6(AS, BS, M, N, KS, D, D, SFAS, SFBS);
    if(KO!=0)matmul_host_w8a8(AO, BO, M, N, KO, D, D, SFAO, SFBO);
    // cudaDeviceSynchronize();

}


void matmul_host_dev(
        ElementANormal::DataType *AN,
        ElementBNormal::DataType *BN,
        ElementASensitive::DataType *AS,
        ElementBSensitive::DataType *BS,
        ElementAOutlier::DataType *AO,
        ElementBOutlier::DataType *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        ElementC *C,
        ElementD *D,
        ElementANormal::ScaleFactorType *SFAN,
        ElementBNormal::ScaleFactorType *SFBN,
        ElementASensitive::ScaleFactorType *SFAS,
        ElementBSensitive::ScaleFactorType *SFBS,
        ElementAOutlier::ScaleFactorType *SFAO,
        ElementBOutlier::ScaleFactorType *SFBO,
        cudaStream_t stream
)
{
    // printf("M = %d, N = %d, KN = %d, KS = %d, KO = %d\n", M, N, KN, KS, KO);
    constexpr int BLK_M = 32, BLK_N = 32, BLK_K = 128;
    // gemm_host_tn<6, BLK_M, BLK_N, BLK_K>(AN, SFAN, BN, SFBN, C, D, M, N, KN, stream); //4
    gemm_host_tn_cpasync<8, BLK_M, BLK_N, BLK_K>(AN, SFAN, BN, SFBN, C, D, M, N, KN, stream); //4
    gemm_host_tn_cpasync<8, BLK_M, BLK_N, BLK_K>(AS, SFAS, BS, SFBS, D, D, M, N, KS, stream); //6
    gemm_host_tn_cpasync<8, BLK_M, BLK_N, BLK_K>(AO, SFAO, BO, SFBO, D, D, M, N, KO, stream); //8

    // cudaDeviceSynchronize();
    // cudaStreamSynchronize(stream);
}

void matmul_w4_host(
        ElementANormal::DataType *AN,
        ElementBNormal::DataType *BN,
        ElementASensitive::DataType *AS,
        ElementBNormal::DataType *BS,
        ElementAOutlier::DataType *AO,
        ElementBNormal::DataType *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        ElementC *C,
        ElementD *D,
        ElementANormal::ScaleFactorType *SFAN,
        ElementBNormal::ScaleFactorType *SFBN,
        ElementASensitive::ScaleFactorType *SFAS,
        ElementBNormal::ScaleFactorType *SFBS,
        ElementAOutlier::ScaleFactorType *SFAO,
        ElementBNormal::ScaleFactorType *SFBO
)
{
    constexpr int BLK_M = 32, BLK_N = 32, BLK_K = 128;
    constexpr int sm_count_threshold = 150;
    bool use_cutlass_kernel = (cute::ceil_div(M , 128) * cute::ceil_div(N , 128)) > sm_count_threshold;
    if(use_cutlass_kernel)
    {
        if(KN!=0)matmul_host_w4a4(AN, BN, M, N, KN, C, D, SFAN, SFBN);
        if(KS!=0)matmul_host_w4a6(AS, BS, M, N, KS, D, D, SFAS, SFBS);
        if(KO!=0)matmul_host_w4a8(AO, BO, M, N, KO, D, D, SFAO, SFBO);
    }
    else
    {
        if(KN!=0) gemm_host_tn<6, BLK_M, BLK_N, BLK_K>(AN, SFAN, BN, SFBN, C, D, M, N, KN); //4x4
        if(KS!=0) gemm_host_tn<5, BLK_M, BLK_N, BLK_K>(AS, SFAS, BS, SFBS, D, D, M, N, KS); //6x4
        if(KO!=0) gemm_host_tn<5, BLK_M, BLK_N, BLK_K>(AO, SFAO, BO, SFBO, D, D, M, N, KO); //8x4
    }
    
}
