#include "gemm.h"
#include "reorder.cuh"

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
    
    const int M = 2048;
    const int N = 4096;
    const int KN = 0;
    const int KS = 0;
    const int KO = 4096;
    const int K = 4096;
    const int block_size = 32; 
    
    ElementANormal::DataType *AN;
    ElementASensitive::DataType *AS;
    ElementAOutlier::DataType *AO;
    ElementBNormal::DataType *BN;
    ElementBSensitive::DataType *BS;
    ElementBOutlier::DataType *BO;
    ElementC *X;
    ElementD *W;
    // ElementC *C;
    // ElementD *D;
    AN = new ElementANormal::DataType[M * KN];
    AS = new ElementASensitive::DataType[M * KS];
    AO = new ElementAOutlier::DataType[M * KO];
    BN = new ElementBNormal::DataType[N * KN];
    BS = new ElementBSensitive::DataType[N * KS];
    BO = new ElementBOutlier::DataType[N * KO];
    X = new ElementC[M * K];
    W = new ElementD[N * K];
    // C = new ElementC[M * N];
    // D = new ElementD[M * N];
    
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
    cutlass::NumericConverter<ElementC, float, cutlass::FloatRoundStyle::round_to_nearest> converterX;
    cutlass::NumericConverter<ElementD, float, cutlass::FloatRoundStyle::round_to_nearest> converterW;
    
    for (int i = 0; i < M * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2000000000.0f - 1000000000.0f;
        
        X[i] = converterX(f);
    }
    // for (int i = 0; i < M * N; ++i) {
    //     // 模拟浮点值
    //     ElementC f = static_cast<ElementC>(12.0 * std::rand() / RAND_MAX - 6.0);
        
    //     // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
    //     // 否则使用构造函数转换
    //     C[i] = f;
    // }
    for (int i = 0; i < N * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2000000000.0f - 1000000000.0f;
        
        W[i] = converterW(f);
    }
    int16_t *reorder_index = new int16_t[K];
    for(int i = 0; i < K; i++) {
        reorder_index[i] = i;
    }
    std::random_shuffle(reorder_index, reorder_index + K);
    ElementANormal::DataType *AN_d;
    ElementASensitive::DataType *AS_d;
    ElementAOutlier::DataType *AO_d;
    ElementBNormal::DataType *BN_d;
    ElementBSensitive::DataType *BS_d;
    ElementBOutlier::DataType *BO_d;
    ElementC *X_d;
    ElementD *W_d;    
    // ElementC *C_d;
    // ElementD *D_d;   
    int16_t *reorder_index_d;
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
    cudaMalloc((void**)&X_d, M * K * sizeof(ElementC));
    cudaMalloc((void**)&W_d, N * K * sizeof(ElementD));
    // cudaMalloc((void**)&C_d, M * N * sizeof(ElementC));
    // cudaMalloc((void**)&D_d, M * N * sizeof(ElementD));
    cudaMalloc((void**)&reorder_index_d, K * sizeof(int16_t));
    cudaMalloc((void**)&SFAN_d, szAN * sizeof(ElementANormal::ScaleFactorType));
    cudaMalloc((void**)&SFAS_d, szAS * sizeof(ElementASensitive::ScaleFactorType));
    cudaMalloc((void**)&SFAO_d, szAO * sizeof(ElementAOutlier::ScaleFactorType));
    cudaMalloc((void**)&SFBN_d, szBN * sizeof(ElementBNormal::ScaleFactorType));
    cudaMalloc((void**)&SFBS_d, szBS * sizeof(ElementBSensitive::ScaleFactorType));
    cudaMalloc((void**)&SFBO_d, szBO * sizeof(ElementBOutlier::ScaleFactorType));
    cudaMemcpy(X_d, X, M * K * sizeof(ElementC), cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, N * K * sizeof(ElementD), cudaMemcpyHostToDevice);
    // cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);
    cudaMemcpy(reorder_index_d, reorder_index, K * sizeof(int16_t), cudaMemcpyHostToDevice);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int it = 0; it < 200; it ++) {
        run_reorder_bf16_mixed<32, K>(
            X_d, M, reorder_index_d, 
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d), 
            SFAN_d, SFAS_d, SFAO_d, KN, KS, KO
        );
        run_reorder_bf16_mixed<32, K>(
            W_d, N, reorder_index_d, 
            reinterpret_cast<uint8_t*>(BN_d), reinterpret_cast<uint8_t*>(BS_d), reinterpret_cast<uint8_t*>(BO_d), 
            SFBN_d, SFBS_d, SFBO_d, KN, KS, KO
        );
        // matmul_host(AN_d, BN_d, AS_d, BS_d, AO_d, BO_d, M, N, KN, KS, KO, C_d, D_d, SFAN_d, SFBN_d, SFAS_d, SFBS_d, SFAO_d, SFBO_d);
    }
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < 400; it ++) {
        run_reorder_bf16_mixed<32, K>(
            X_d, M, reorder_index_d, 
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d), 
            SFAN_d, SFAS_d, SFAO_d, KN, KS, KO
        );
        // run_reorder_bf16_mixed<32, K>(
        //     W_d, N, reorder_index_d, 
        //     reinterpret_cast<uint8_t*>(BN_d), reinterpret_cast<uint8_t*>(BS_d), reinterpret_cast<uint8_t*>(BO_d), 
        //     SFBN_d, SFBS_d, SFBO_d, KN, KS, KO
        // );
        // matmul_host(AN_d, BN_d, AS_d, BS_d, AO_d, BO_d, M, N, KN, KS, KO, C_d, D_d, SFAN_d, SFBN_d, SFAS_d, SFBS_d, SFAO_d, SFBO_d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    // CRITICAL: Synchronize and check for errors immediately after kernel launch
    cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    if (kernel_err != cudaSuccess) {
        std::cerr << "CUDA error after launching: "
                << cudaGetErrorString(kernel_err) << std::endl;
        // Optionally, throw an exception to propagate the error to Python
        throw std::runtime_error(std::string("CUDA error in : ") + cudaGetErrorString(kernel_err));
    }

    cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    if (sync_err != cudaSuccess) {
        std::cerr << "CUDA error during/after kernel synchronization: "
                << cudaGetErrorString(sync_err) << std::endl;
        throw std::runtime_error(std::string("CUDA sync error in kernel: ") + cudaGetErrorString(sync_err));
    }
    std::cout << "kernel finished and synced successfully." << std::endl; std::cout.flush();

    std::printf("REORDER kernel completed in %.3f ms\n", milliseconds / 400);
    std::cout << "reorder finished." << std::endl;
    cudaFree(AN_d);
    cudaFree(BN_d);
    cudaFree(AS_d);
    cudaFree(BS_d);
    cudaFree(AO_d);
    cudaFree(BO_d);
    cudaFree(X_d);
    cudaFree(W_d);
    cudaFree(SFAN_d);
    cudaFree(SFBN_d);
    cudaFree(SFAS_d);
    cudaFree(SFBS_d);
    cudaFree(SFAO_d);
    cudaFree(SFBO_d);
    return 0; 
}