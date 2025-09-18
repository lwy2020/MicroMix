#include "fp4.h"
#include "fp6.h"
#include "fp8.h"
#include "gemm.h"

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
        const ElementANormal::DataType *AN,
        const ElementBNormal::DataType *BN,
        const ElementASensitive::DataType *AS,
        const ElementBSensitive::DataType *BS,
        const ElementAOutlier::DataType *AO,
        const ElementBOutlier::DataType *BO,
        int M,
        int N,
        int KN,
        int KS,
        int KO,
        ElementC *C,
        ElementD *D,
        const ElementANormal::ScaleFactorType *SFAN,
        const ElementBNormal::ScaleFactorType *SFBN,
        const ElementASensitive::ScaleFactorType *SFAS,
        const ElementBSensitive::ScaleFactorType *SFBS,
        const ElementAOutlier::ScaleFactorType *SFAO,
        const ElementBOutlier::ScaleFactorType *SFBO
)
{
    if(KN!=0)matmul_host4(AN, BN, M, N, KN, C, D, SFAN, SFBN);
    if(KS!=0)matmul_host6(AS, BS, M, N, KS, D, D, SFAS, SFBS);
    if(KO!=0)matmul_host8(AO, BO, M, N, KO, D, D, SFAO, SFBO);
}
