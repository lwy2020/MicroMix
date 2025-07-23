#include "gemm.h"

#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementANormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         ElementASensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using         ElementAOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand

// B matrix configuration
using         ElementBNormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         ElementBSensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using         ElementBOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand

int main() {
    
    const int M = 4096;
    const int N = 4096;
    const int KN = 2048;
    const int KS = 0;
    const int KO = 2048;
    const int block_size = 32; 
    
    ElementANormal::DataType *AN;
    ElementASensitive::DataType *AS;
    ElementAOutlier::DataType *AO;
    ElementBNormal::DataType *BN;
    ElementBSensitive::DataType *BS;
    ElementBOutlier::DataType *BO;
    ElementC *C;
    ElementD *D;
    AN = new ElementANormal::DataType[M * KN];
    AS = new ElementASensitive::DataType[M * KS];
    AO = new ElementAOutlier::DataType[M * KO];
    BN = new ElementBNormal::DataType[N * KN];
    BS = new ElementBSensitive::DataType[N * KS];
    BO = new ElementBOutlier::DataType[N * KO];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    
    // 创建 scale 数组（每 block_size 个元素对应一个缩放因子）
    int szAN = ((M * KN + block_size - 1) / block_size);
    ElementANormal::ScaleFactorType *scaleAN = new ElementANormal::ScaleFactorType[((M * KN + block_size - 1) / block_size)];
    int szBN = ((N * KN + block_size - 1) / block_size);
    ElementBNormal::ScaleFactorType *scaleBN = new ElementBNormal::ScaleFactorType[((N * KN + block_size - 1) / block_size)];

    int szAS = ((M * KS + block_size - 1) / block_size);
    ElementASensitive::ScaleFactorType *scaleAS = new ElementASensitive::ScaleFactorType[((M * KS + block_size - 1) / block_size)];
    int szBS = ((N * KS + block_size - 1) / block_size);
    ElementBSensitive::ScaleFactorType *scaleBS = new ElementBSensitive::ScaleFactorType[((N * KS + block_size - 1) / block_size)];

    int szAO = ((M * KO + block_size - 1) / block_size);
    ElementAOutlier::ScaleFactorType *scaleAO = new ElementAOutlier::ScaleFactorType[((M * KO + block_size - 1) / block_size)];
    int szBO = ((N * KO + block_size - 1) / block_size);
    ElementBOutlier::ScaleFactorType *scaleBO = new ElementBOutlier::ScaleFactorType[((N * KO + block_size - 1) / block_size)];
    
    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementANormal::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAN;
    cutlass::NumericConverter<ElementASensitive::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAS;
    cutlass::NumericConverter<ElementAOutlier::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAO;
    cutlass::NumericConverter<ElementBNormal::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterBN;
    cutlass::NumericConverter<ElementBSensitive::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterBS;
    cutlass::NumericConverter<ElementBOutlier::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterBO;
    cutlass::NumericConverter<ElementANormal::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFA;
    cutlass::NumericConverter<ElementBNormal::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFB;
    
    for (int i = 0; i < M * KN; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 12.0f - 6.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AN[i] = converterAN(f);
    }

    for (int i = 0; i < M * KS; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 56.0f - 28.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AS[i] = converterAS(f);
    }

    for (int i = 0; i < M * KO; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AO[i] = converterAO(f);
    }

    for (int i = 0; i < M * N; ++i) {
        // 模拟浮点值
        ElementC f = static_cast<ElementC>(12.0 * std::rand() / RAND_MAX - 6.0);
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        C[i] = f;
    }
    for (int i = 0; i < N * KN; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 12.0f - 6.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BN[i] = converterBN(f);
    }
    for (int i = 0; i < N * KS; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 12.0f - 6.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BS[i] = converterBS(f);
    }

    for (int i = 0; i < N * KO; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 12.0f - 6.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BO[i] = converterBO(f);
    }


    // 随机初始化 scale（每 block 一个）
    for (size_t i = 0; i < szAN; ++i) {
        scaleAN[i] = converterSFA(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBN; ++i) {
        scaleBN[i] = converterSFB(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szAS; ++i) {
        scaleAS[i] = converterSFA(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBS; ++i) {
        scaleBS[i] = converterSFB(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szAO; ++i) {
        scaleAO[i] = converterSFA(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBO; ++i) {
        scaleBO[i] = converterSFB(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    ElementANormal::DataType *AN_d;
    ElementASensitive::DataType *AS_d;
    ElementAOutlier::DataType *AO_d;
    ElementBNormal::DataType *BN_d;
    ElementBSensitive::DataType *BS_d;
    ElementBOutlier::DataType *BO_d;
    ElementC *C_d;
    ElementD *D_d;    
    ElementANormal::ScaleFactorType *SFAN_d;
    ElementASensitive::ScaleFactorType *SFAS_d;
    ElementAOutlier::ScaleFactorType *SFAO_d;
    ElementBNormal::ScaleFactorType *SFBN_d;
    ElementBSensitive::ScaleFactorType *SFBS_d;
    ElementBOutlier::ScaleFactorType *SFBO_d;

    cudaMalloc((void**)&AN_d, M * KN * sizeof(ElementANormal::DataType));
    cudaMalloc((void**)&AS_d, M * KS * sizeof(ElementASensitive::DataType));
    cudaMalloc((void**)&AO_d, M * KO * sizeof(ElementAOutlier::DataType));
    cudaMalloc((void**)&BN_d, N * KN * sizeof(ElementBNormal::DataType));
    cudaMalloc((void**)&BS_d, N * KS * sizeof(ElementBSensitive::DataType));
    cudaMalloc((void**)&BO_d, N * KO * sizeof(ElementBOutlier::DataType));
    cudaMalloc((void**)&C_d, M * N * sizeof(ElementC));
    cudaMalloc((void**)&D_d, M * N * sizeof(ElementD));
    cudaMalloc((void**)&SFAN_d, szAN * sizeof(ElementANormal::ScaleFactorType));
    cudaMalloc((void**)&SFAS_d, szAS * sizeof(ElementASensitive::ScaleFactorType));
    cudaMalloc((void**)&SFAO_d, szAO * sizeof(ElementAOutlier::ScaleFactorType));
    cudaMalloc((void**)&SFBN_d, szBN * sizeof(ElementBNormal::ScaleFactorType));
    cudaMalloc((void**)&SFBS_d, szBS * sizeof(ElementBSensitive::ScaleFactorType));
    cudaMalloc((void**)&SFBO_d, szBO * sizeof(ElementBOutlier::ScaleFactorType));
    cudaMemcpy(AN_d, AN, M * KN * sizeof(ElementANormal::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(AS_d, AS, M * KS * sizeof(ElementASensitive::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(AO_d, AO, M * KO * sizeof(ElementAOutlier::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(BN_d, BN, N * KN * sizeof(ElementBNormal::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(BS_d, BS, N * KS * sizeof(ElementBSensitive::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(BO_d, BO, N * KO * sizeof(ElementBOutlier::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);
    cudaMemcpy(SFAN_d, scaleAN, szAN * sizeof(ElementANormal::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFAS_d, scaleAS, szAS * sizeof(ElementASensitive::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFAO_d, scaleAO, szAO * sizeof(ElementAOutlier::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFBN_d, scaleBN, szBN * sizeof(ElementBNormal::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFBS_d, scaleBS, szBS * sizeof(ElementBSensitive::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFBO_d, scaleBO, szBO * sizeof(ElementBOutlier::ScaleFactorType), cudaMemcpyHostToDevice);

    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int it = 0; it < 200; it ++) {
        matmul_host(AN_d, BN_d, AS_d, BS_d, AO_d, BO_d, M, N, KN, KS, KO, C_d, D_d, SFAN_d, SFBN_d, SFAS_d, SFBS_d, SFAO_d, SFBO_d);
    }
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < 400; it ++) {
        matmul_host(AN_d, BN_d, AS_d, BS_d, AO_d, BO_d, M, N, KN, KS, KO, C_d, D_d, SFAN_d, SFBN_d, SFAS_d, SFBS_d, SFAO_d, SFBO_d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    cudaMemcpy(D, D_d, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);

    std::printf("GEMM completed in %.3f ms\n", milliseconds / 400);
    std::cout << "mixed gemm finished." << std::endl;
    cudaFree(AN_d);
    cudaFree(BN_d);
    cudaFree(AS_d);
    cudaFree(BS_d);
    cudaFree(AO_d);
    cudaFree(BO_d);
    cudaFree(C_d);
    cudaFree(D_d);
    cudaFree(SFAN_d);
    cudaFree(SFBN_d);
    cudaFree(SFAS_d);
    cudaFree(SFBS_d);
    cudaFree(SFAO_d);
    cudaFree(SFBO_d);
    return 0; 
}