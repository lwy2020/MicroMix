#include "fp6.h"
#include "gemm.h"

#include "cutlass/numeric_conversion.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand

// B matrix configuration
using         ElementB    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for B matrix operand

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand

int main() {
    const int M = 2048;
    const int N = 4096;
    const int K = 4096;
    const int block_size = 32;
    
    ElementA::DataType *A;
    ElementB::DataType *B;
    ElementC *C;
    ElementD *D;
    A = new ElementA::DataType[M * K];
    B = new ElementB::DataType[N * K];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    
    // 创建 scale 数组（每 block_size 个元素对应一个缩放因子）
    int szA = ((M * K + block_size - 1) / block_size);
    ElementA::ScaleFactorType *scaleA = new ElementA::ScaleFactorType[((M * K + block_size - 1) / block_size)];
    int szB = ((N * K + block_size - 1) / block_size);
    ElementB::ScaleFactorType *scaleB = new ElementB::ScaleFactorType[((N * K + block_size - 1) / block_size)];
    

    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementA::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterA;
    cutlass::NumericConverter<ElementB::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterB;
    cutlass::NumericConverter<ElementA::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFA;
    cutlass::NumericConverter<ElementB::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFB;
    
    for (int i = 0; i < M * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 56.0f - 28.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        A[i] = converterA(f);
    }

    for (int i = 0; i < M * N; ++i) {
        // 模拟浮点值
        ElementC f = static_cast<ElementC>(12.0 * std::rand() / RAND_MAX - 6.0);
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        C[i] = f;
    }
    for (int i = 0; i < N * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 56.0f - 28.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        B[i] = converterB(f);
    }


    // 随机初始化 scale（每 block 一个）
    for (size_t i = 0; i < szA; ++i) {
        scaleA[i] = converterSFA(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szB; ++i) {
        scaleB[i] = converterSFB(static_cast<float>(std::rand()) / RAND_MAX * 255.0f);  // [0.1, 1.0]
    }
    ElementA::DataType *A_d;
    ElementB::DataType *B_d;
    ElementC *C_d;
    ElementD *D_d;    
    ElementA::ScaleFactorType *SFA_d;
    ElementB::ScaleFactorType *SFB_d;

    cudaMalloc((void**)&A_d, M * K * sizeof(ElementA::DataType));
    cudaMalloc((void**)&B_d, N * K * sizeof(ElementB::DataType));
    cudaMalloc((void**)&C_d, M * N * sizeof(ElementC));
    cudaMalloc((void**)&D_d, M * N * sizeof(ElementD));
    cudaMalloc((void**)&SFA_d, szA * sizeof(ElementA::ScaleFactorType));
    cudaMalloc((void**)&SFB_d, szB * sizeof(ElementB::ScaleFactorType));
    cudaMemcpy(A_d, A, M * K * sizeof(ElementA::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * K * sizeof(ElementB::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);
    cudaMemcpy(SFA_d, scaleA, szA * sizeof(ElementA::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFB_d, scaleB, szB * sizeof(ElementB::ScaleFactorType), cudaMemcpyHostToDevice);
    
    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int it = 0; it < 200; it ++) {
        matmul_host6(A_d, B_d, M, N, K, C_d, D_d, SFA_d, SFB_d);
    }
    
    // timed runs
    const int timed_iters = 400;
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < timed_iters; it ++) {
        matmul_host6(A_d, B_d, M, N, K, C_d, D_d, SFA_d, SFB_d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    cudaMemcpy(D, D_d, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);

    double time_sec = static_cast<double>(milliseconds) / 1000.0;
    
    double flops_per_gemm = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double total_flops = flops_per_gemm * static_cast<double>(timed_iters);
    double gflops = total_flops / (time_sec * 1.0e9);
    double tflops = total_flops / (time_sec * 1.0e12);

    std::printf("GEMM average time per run: %.3f ms\n", milliseconds / timed_iters);
    std::printf("Total runs: %d, problem size M=%d N=%d K=%d\n", timed_iters, M, N, K);
    std::printf("FLOPs per run: %.0f\n", flops_per_gemm);
    std::printf("Total FLOPs (all runs): %.0f\n", total_flops);
    std::printf("Achieved performance: %.3f TFLOPS\n", tflops);
    std::cout << "mxfp6 gemm finished." << std::endl;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);
    cudaFree(SFA_d);
    cudaFree(SFB_d);
    return 0;
}