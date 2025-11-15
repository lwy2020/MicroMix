#include "sm120_multistage_tma.h"
#include "cutlass/numeric_conversion.h"
#include "w4a4.h"
#include "w4a6.h"
#include "w4a8.h"
#include "w6a6.h"
#include "w6a8.h"
#include "w8a8.h"

#include <string>
#include <iostream>
#include <functional>

#include "gemm_utils.h"

using namespace cute;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;

using Type4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using Type6 = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using Type8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;


constexpr int block_size = 32;

// 性能测试辅助函数
template <typename KernelCallable>
static float perform_benchmark(
    int timed_iters,
    KernelCallable kernel_to_run) 
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm up
    for (int it = 0; it < 10; ++it) {
        kernel_to_run();
    }

    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < timed_iters; ++it) {
        kernel_to_run();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return milliseconds;
}


template <typename ElementA, typename ElementB,
          typename KernelCallable>
static void run_benchmark(
    int M, int N, int K, int timed_iters, 
    KernelCallable kernel_to_run
) {

    typename ElementA::DataType *A;
    typename ElementB::DataType *B;
    ElementC *C;
    ElementD *D;
    
    A = new typename ElementA::DataType[M * K];
    B = new typename ElementB::DataType[N * K];
    C = new ElementC[M * N];
    D = new ElementD[M * N];

    int szA = M * ((K + block_size - 1) / block_size);
    typename ElementA::ScaleFactorType *scaleA = new typename ElementA::ScaleFactorType[szA];
    int szB = N * ((K + block_size - 1) / block_size);
    typename ElementB::ScaleFactorType *scaleB = new typename ElementB::ScaleFactorType[szB];

    std::srand(static_cast<unsigned int>(123));
    
    initialize_matrix_random(A, M * K);
    initialize_matrix_random(B, N * K);
    initialize_matrix_random(C, M * N);
    initialize_matrix_random(scaleA, szA);
    initialize_matrix_random(scaleB, szB);

    typename ElementA::DataType *A_d;
    typename ElementB::DataType *B_d;
    typename ElementA::ScaleFactorType *SFA_d;
    typename ElementB::ScaleFactorType *SFB_d;
    ElementC *C_d;
    ElementD *D_d;

    cudaMalloc(&A_d, M * K * sizeof(typename ElementA::DataType));
    cudaMalloc(&B_d, N * K * sizeof(typename ElementB::DataType));
    cudaMalloc(&SFA_d, szA * sizeof(typename ElementA::ScaleFactorType));
    cudaMalloc(&SFB_d, szB * sizeof(typename ElementB::ScaleFactorType));
    cudaMalloc(&C_d, M * N * sizeof(ElementC));
    cudaMalloc(&D_d, M * N * sizeof(ElementD));
    checkCudaLastErrors();

    cudaMemcpy(A_d, A, M * K * sizeof(typename ElementA::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * K * sizeof(typename ElementB::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFA_d, scaleA, szA * sizeof(typename ElementA::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFB_d, scaleB, szB * sizeof(typename ElementB::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);

    // Lambda 封装
    auto kernel_bench_lambda = [&]() {
        kernel_to_run(A_d, SFA_d, B_d, SFB_d, C_d, D_d, M, N, K);
    };

    float milliseconds = perform_benchmark(timed_iters, kernel_bench_lambda);

    double time_sec = static_cast<double>(milliseconds) / 1000.0 / timed_iters;
    double flops_per_gemm = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tflops = flops_per_gemm / (time_sec * 1.0e12);

    printf("Iter Runs = %4d\n", timed_iters);
    printf("M =%5d, N =%5d, K =%5d, ", M, N, K);
    printf("Time = %12.8lf ms, ", time_sec * 1000);
    printf("AVG Performance = %10.4lf TFLOPs\n", tflops);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);
    cudaFree(SFA_d);
    cudaFree(SFB_d);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] scaleA;
    delete[] scaleB;
}


template<typename EA, typename EB, typename RefKernel>
static void launch_and_run_benchmark(int M, int N, int K, int timed_iters, RefKernel ref_kernel) {

    auto target_kernel =
            [=](typename EA::DataType* pA, typename EA::ScaleFactorType* pSFA,
                typename EB::DataType* pB, typename EB::ScaleFactorType* pSFB,
                ElementC* pC,
                ElementD* pD, 
                int m, int n, int k) {
        
        ref_kernel(pA, pB, m, n, k, pC, pD, pSFA, pSFB);
    };

    run_benchmark<EA, EB>(M, N, K, timed_iters, target_kernel);
}

template<class EA, class EB>
static void default_matmul_host(
        typename EA::DataType *A,
        typename EB::DataType *B,
        int M,
        int N,
        int K,
        ElementC *C,  
        ElementD *D, 
        typename EA::ScaleFactorType *SFA,
        typename EB::ScaleFactorType *SFB
){
    assert(0);
    // 用于那些没有 Ref 实现的精度组合
    // 实际运行时这将非常快，因为只打印不计算，TFLOPs 会虚高
}

int main(int argc, char** argv) {
    int M = 2048;
    int N = 4096;
    int K = 4096;
    int timed_iters = 100;
    std::string prec_str = "8x8";

    if (argc >= 2) M = std::atoi(argv[1]);
    if (argc >= 3) N = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);
    if (argc >= 5) timed_iters = std::atoi(argv[4]);
    if (argc >= 6) prec_str = argv[5];
    
    std::string prec_a, prec_b;
    size_t x_pos = prec_str.find('x');
    
    if (x_pos == std::string::npos) {
        prec_a = prec_str;
        prec_b = prec_str;
    } else {
        prec_a = prec_str.substr(0, x_pos);
        prec_b = prec_str.substr(x_pos + 1);
    }


    if (false) {
    }
    // A=4
    else if (prec_a == "4" && prec_b == "4") {
        printf("Running MXFP4 (E2M1) x MXFP4 (E2M1)\n");
        auto ref_kernel = matmul_host_w4a4;
        launch_and_run_benchmark<Type4, Type4>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "4" && prec_b == "6") {
        printf("Running MXFP4 (E2M1) x MXFP6 (E3M2)\n");
        auto ref_kernel = default_matmul_host<Type4, Type6>;
        launch_and_run_benchmark<Type4, Type6>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "4" && prec_b == "8") {
        printf("Running MXFP4 (E2M1) x MXFP8 (E4M3)\n");
        auto ref_kernel = default_matmul_host<Type4, Type8>;
        launch_and_run_benchmark<Type4, Type8>(M, N, K, timed_iters, ref_kernel);
    }
    // A=6
    else if (prec_a == "6" && prec_b == "4") {
        printf("Running MXFP6 (E3M2) x MXFP4 (E2M1)\n");
        auto ref_kernel = matmul_host_w4a6;
        launch_and_run_benchmark<Type6, Type4>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "6" && prec_b == "6") {
        printf("Running MXFP6 (E3M2) x MXFP6 (E3M2)\n");
        auto ref_kernel = matmul_host_w6a6;
        launch_and_run_benchmark<Type6, Type6>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "6" && prec_b == "8") {
        printf("Running MXFP6 (E3M2) x MXFP8 (E4M3)\n");
        auto ref_kernel = default_matmul_host<Type6, Type8>;
        launch_and_run_benchmark<Type6, Type8>(M, N, K, timed_iters, ref_kernel);
    }
    // A=8
    else if (prec_a == "8" && prec_b == "4") {
        printf("Running MXFP8 (E4M3) x MXFP4 (E2M1)\n");
        auto ref_kernel = matmul_host_w4a8;
        launch_and_run_benchmark<Type8, Type4>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "8" && prec_b == "6") {
        printf("Running MXFP8 (E4M3) x MXFP6 (E3M2)\n");
        auto ref_kernel = matmul_host_w6a8;
        launch_and_run_benchmark<Type8, Type6>(M, N, K, timed_iters, ref_kernel);
    }
    else if (prec_a == "8" && prec_b == "8") {
        printf("Running MXFP8 (E4M3) x MXFP8 (E4M3)\n");
        auto ref_kernel = matmul_host_w8a8;
        launch_and_run_benchmark<Type8, Type8>(M, N, K, timed_iters, ref_kernel);
    }
    else {
        fprintf(stderr, "Error: Unsupported precision string: %s (parsed as A=%s, B=%s)\n", 
                prec_str.c_str(), prec_a.c_str(), prec_b.c_str());
        fprintf(stderr, "Supported formats are '4', '6', '8', '4x4', '4x6', '6x8', '8x8', etc.\n");
        fprintf(stderr, "Usage: %s [M N K iters prec_str]\n", argv[0]);
        return 1;
    }

    return 0;
}