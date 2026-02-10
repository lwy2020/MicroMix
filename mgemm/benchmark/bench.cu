#include "gemm.h"
#include "reorder.cuh"
#include "cutlass/numeric_conversion.h"

// 增加必要头文件
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda_runtime_api.h>

// 定义 CHECK_CUDA 宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementANormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using ElementASensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using ElementAOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand

// B matrix configuration
using ElementBNormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using ElementBSensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for B matrix operand
using ElementBOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for B matrix operand

// C/D matrix configuration
using ElementD = cutlass::bfloat16_t;                            // Element type for D matrix operand
using ElementC = cutlass::bfloat16_t;                            // Element type for C matrix operand

// 模拟分离的 reorder 和 quantize 函数（需根据实际实现替换）
// 注：此处为占位，需替换为真实的分离算子实现
template <int block_size, int K>
void run_reorder_only_bf16(
    ElementC* X_d, int M, int16_t* reorder_index_d,
    uint8_t* AN_d, uint8_t* AS_d, uint8_t* AO_d,
    int KN, int KS, int KO
) {
    // 仅执行 reorder 逻辑（示例占位）
    dim3 grid((M * K + 1023) / 1024);
    dim3 block(256);
    // reorder_only_kernel<block_size, K><<<grid, block>>>(X_d, M, reorder_index_d, AN_d, AS_d, AO_d, KN, KS, KO);
    CHECK_CUDA(cudaGetLastError());
}

template <int block_size, int K>
void run_quantize_only_bf16(
    uint8_t* AN_d, uint8_t* AS_d, uint8_t* AO_d,
    typename ElementANormal::ScaleFactorType* SFAN_d,
    typename ElementASensitive::ScaleFactorType* SFAS_d,
    typename ElementAOutlier::ScaleFactorType* SFAO_d,
    int M, int KN, int KS, int KO
) {
    // 仅执行 quantize 逻辑（示例占位）
    dim3 grid((M * (KN+KS+KO) + 1023) / 1024);
    dim3 block(256);
    // quantize_only_kernel<block_size, K><<<grid, block>>>(AN_d, AS_d, AO_d, SFAN_d, SFAS_d, SFAO_d, M, KN, KS, KO);
    CHECK_CUDA(cudaGetLastError());
}

int main() {
    // 矩阵维度配置
    const int M = 2048;
    const int N = 4096;
    const int KN = 0;
    const int KS = 0;
    const int KO = 4096;
    const int K = 4096;
    const int block_size = 32; 
    const int warmup_iter = 200;  // 热身迭代数
    const int test_iter = 400;    // 测试迭代数
    
    // 主机内存分配
    ElementANormal::DataType *AN = new ElementANormal::DataType[M * KN];
    ElementASensitive::DataType *AS = new ElementASensitive::DataType[M * KS];
    ElementAOutlier::DataType *AO = new ElementAOutlier::DataType[M * KO];
    ElementBNormal::DataType *BN = new ElementBNormal::DataType[N * KN];
    ElementBSensitive::DataType *BS = new ElementBSensitive::DataType[N * KS];
    ElementBOutlier::DataType *BO = new ElementBOutlier::DataType[N * KO];
    ElementC *X = new ElementC[M * K];
    ElementD *W = new ElementD[N * K];
    
    // 缩放因子内存分配
    int szAN = ((M * KN + block_size - 1) / block_size);
    ElementANormal::ScaleFactorType *scaleAN = new ElementANormal::ScaleFactorType[szAN];
    int szBN = ((N * KN + block_size - 1) / block_size);
    ElementBNormal::ScaleFactorType *scaleBN = new ElementBNormal::ScaleFactorType[szBN];
    int szAS = ((M * KS + block_size - 1) / block_size);
    ElementASensitive::ScaleFactorType *scaleAS = new ElementASensitive::ScaleFactorType[szAS];
    int szBS = ((N * KS + block_size - 1) / block_size);
    ElementBSensitive::ScaleFactorType *scaleBS = new ElementBSensitive::ScaleFactorType[szBS];
    int szAO = ((M * KO + block_size - 1) / block_size);
    ElementAOutlier::ScaleFactorType *scaleAO = new ElementAOutlier::ScaleFactorType[szAO];
    int szBO = ((N * KO + block_size - 1) / block_size);
    ElementBOutlier::ScaleFactorType *scaleBO = new ElementBOutlier::ScaleFactorType[szBO];
    
    // 随机数初始化
    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementC, float, cutlass::FloatRoundStyle::round_to_nearest> converterX;
    cutlass::NumericConverter<ElementD, float, cutlass::FloatRoundStyle::round_to_nearest> converterW;
    
    for (int i = 0; i < M * K; ++i) {
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2000000000.0f - 1000000000.0f;
        X[i] = converterX(f);
    }
    for (int i = 0; i < N * K; ++i) {
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2000000000.0f - 1000000000.0f;
        W[i] = converterW(f);
    }
    
    // 重排索引初始化
    int16_t *reorder_index = new int16_t[K];
    for(int i = 0; i < K; i++) reorder_index[i] = i;
    std::random_shuffle(reorder_index, reorder_index + K);
    
    // 设备内存分配
    ElementANormal::DataType *AN_d;
    ElementASensitive::DataType *AS_d;
    ElementAOutlier::DataType *AO_d;
    ElementBNormal::DataType *BN_d;
    ElementBSensitive::DataType *BS_d;
    ElementBOutlier::DataType *BO_d;
    ElementC *X_d;
    ElementD *W_d;    
    int16_t *reorder_index_d;
    ElementANormal::ScaleFactorType *SFAN_d;
    ElementASensitive::ScaleFactorType *SFAS_d;
    ElementAOutlier::ScaleFactorType *SFAO_d;
    ElementBNormal::ScaleFactorType *SFBN_d;
    ElementBSensitive::ScaleFactorType *SFBS_d;
    ElementBOutlier::ScaleFactorType *SFBO_d;

    CHECK_CUDA(cudaMalloc((void**)&AN_d, M * KN * sizeof(ElementANormal::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&AS_d, M * KS * sizeof(ElementASensitive::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&AO_d, M * KO * sizeof(ElementAOutlier::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&BN_d, N * KN * sizeof(ElementBNormal::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&BS_d, N * KS * sizeof(ElementBSensitive::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&BO_d, N * KO * sizeof(ElementBOutlier::DataType)));
    CHECK_CUDA(cudaMalloc((void**)&X_d, M * K * sizeof(ElementC)));
    CHECK_CUDA(cudaMalloc((void**)&W_d, N * K * sizeof(ElementD)));
    CHECK_CUDA(cudaMalloc((void**)&reorder_index_d, K * sizeof(int16_t)));
    CHECK_CUDA(cudaMalloc((void**)&SFAN_d, szAN * sizeof(ElementANormal::ScaleFactorType)));
    CHECK_CUDA(cudaMalloc((void**)&SFAS_d, szAS * sizeof(ElementASensitive::ScaleFactorType)));
    CHECK_CUDA(cudaMalloc((void**)&SFAO_d, szAO * sizeof(ElementAOutlier::ScaleFactorType)));
    CHECK_CUDA(cudaMalloc((void**)&SFBN_d, szBN * sizeof(ElementBNormal::ScaleFactorType)));
    CHECK_CUDA(cudaMalloc((void**)&SFBS_d, szBS * sizeof(ElementBSensitive::ScaleFactorType)));
    CHECK_CUDA(cudaMalloc((void**)&SFBO_d, szBO * sizeof(ElementBOutlier::ScaleFactorType)));
    
    // 主机到设备数据拷贝
    CHECK_CUDA(cudaMemcpy(X_d, X, M * K * sizeof(ElementC), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W_d, W, N * K * sizeof(ElementD), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(reorder_index_d, reorder_index, K * sizeof(int16_t), cudaMemcpyHostToDevice));

    // CUDA Event 创建
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float elapsed_separate = 0.0f;  // 分离执行总耗时
    float elapsed_fused = 0.0f;     // 融合执行总耗时

    // -------------------------- 1. 分离执行 (Reorder + Quantize) --------------------------
    // 热身
    for (int it = 0; it < warmup_iter; it ++) {
        // 分离执行：先Reorder，后Quantize
        run_reorder_only_bf16<block_size, K>(
            X_d, M, reorder_index_d,
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d),
            KN, KS, KO
        );
        run_quantize_only_bf16<block_size, K>(
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d),
            SFAN_d, SFAS_d, SFAO_d,
            M, KN, KS, KO
        );
    }

    // 计时分离执行
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < test_iter; it ++) {
        run_reorder_only_bf16<block_size, K>(
            X_d, M, reorder_index_d,
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d),
            KN, KS, KO
        );
        run_quantize_only_bf16<block_size, K>(
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d),
            SFAN_d, SFAS_d, SFAO_d,
            M, KN, KS, KO
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_separate, start, stop));

    // 检查CUDA错误
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        std::cerr << "CUDA error (separate): " << cudaGetErrorString(kernel_err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // -------------------------- 2. 融合执行 (Reorder + Quantize) --------------------------
    // 热身
    for (int it = 0; it < warmup_iter; it ++) {
        run_reorder_bf16_mixed<block_size, K>(
            X_d, M, reorder_index_d, 
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d), 
            SFAN_d, SFAS_d, SFAO_d, KN, KS, KO
        );
    }

    // 计时融合执行
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < test_iter; it ++) {
        run_reorder_bf16_mixed<block_size, K>(
            X_d, M, reorder_index_d, 
            reinterpret_cast<uint8_t*>(AN_d), reinterpret_cast<uint8_t*>(AS_d), reinterpret_cast<uint8_t*>(AO_d), 
            SFAN_d, SFAS_d, SFAO_d, KN, KS, KO
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_fused, start, stop));

    // 检查CUDA错误
    kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        std::cerr << "CUDA error (fused): " << cudaGetErrorString(kernel_err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // -------------------------- 3. 输出性能对比 --------------------------
    float avg_separate = elapsed_separate / test_iter;  // 分离执行单轮耗时(ms)
    float avg_fused = elapsed_fused / test_iter;        // 融合执行单轮耗时(ms)
    float speedup = avg_separate / avg_fused;           // 加速比

    std::cout << "========================================" << std::endl;
    std::cout << "性能对比结果 (单轮执行耗时)：" << std::endl;
    std::cout << "分离执行 (Reorder + Quantize): " << avg_separate << " ms" << std::endl;
    std::cout << "融合执行 (Reorder+Quantize)  : " << avg_fused << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "融合算子加速比: " << speedup << "x" << std::endl;
    std::cout << "性能提升百分比: " << (speedup - 1) * 100 << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    // 内存释放
    CHECK_CUDA(cudaFree(AN_d));
    CHECK_CUDA(cudaFree(AS_d));
    CHECK_CUDA(cudaFree(AO_d));
    CHECK_CUDA(cudaFree(BN_d));
    CHECK_CUDA(cudaFree(BS_d));
    CHECK_CUDA(cudaFree(BO_d));
    CHECK_CUDA(cudaFree(X_d));
    CHECK_CUDA(cudaFree(W_d));
    CHECK_CUDA(cudaFree(reorder_index_d));
    CHECK_CUDA(cudaFree(SFAN_d));
    CHECK_CUDA(cudaFree(SFAS_d));
    CHECK_CUDA(cudaFree(SFAO_d));
    CHECK_CUDA(cudaFree(SFBN_d));
    CHECK_CUDA(cudaFree(SFBS_d));
    CHECK_CUDA(cudaFree(SFBO_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    delete[] AN;
    delete[] AS;
    delete[] AO;
    delete[] BN;
    delete[] BS;
    delete[] BO;
    delete[] X;
    delete[] W;
    delete[] reorder_index;
    delete[] scaleAN;
    delete[] scaleBN;
    delete[] scaleAS;
    delete[] scaleBS;
    delete[] scaleAO;
    delete[] scaleBO;

    return 0; 
}