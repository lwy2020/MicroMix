#include "sm120_multistage_tma.h"
#include "cutlass/numeric_conversion.h"
#include "w4a4.h"
#include "w4a6.h"
#include "w4a8.h"
#include "w6a6.h"
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

constexpr int BM = 32;
constexpr int BN = BM;
constexpr int BK = 128;
constexpr int N_STAGE = (BM == 128) ? 2 : 8;

constexpr int block_size = 32;


template <typename KernelCallable, typename RefKernelCallable>
bool perform_validation(
    int M, int N,
    ElementD* D, ElementD* D_ref,
    ElementD* D_d, ElementD* D_d_ref,
    KernelCallable kernel_to_run,
    RefKernelCallable ref_kernel_to_run
) {

    ref_kernel_to_run();
    kernel_to_run();
    cudaMemcpy(D, D_d, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);
    cudaMemcpy(D_ref, D_d_ref, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);
    cutlass::NumericConverter<float, ElementD, cutlass::FloatRoundStyle::round_to_nearest> converterD;
    bool gemm_pass = true;
    for(int i = 0; i < M && gemm_pass; i++) {
        for(int j = 0; j < N && gemm_pass; j++) {
            int idx = i * N + j;
            float val_D = converterD(D[idx]);
            float val_D_ref = converterD(D_ref[idx]);
            if(std::abs(val_D - val_D_ref) > 1e-5) { 
                printf("Result mismatch at [%4d, %4d]: VAL: %.8f, REF: %.8f with error %.8f\n",
                       i,j,val_D, val_D_ref, std::abs(val_D - val_D_ref));
                gemm_pass = false;
            }
        }
    }
    print(gemm_pass ? "\nGEMM VAL PASS!\n\n" : "\nGEMM VAL FAILED!\n\n");
    return gemm_pass;
}
template <typename KernelCallable>
float perform_benchmark(
    int timed_iters,
    KernelCallable kernel_to_run) 
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // warp up
    // for (int it = 0; it < 20; ++it) {
    //     kernel_to_run();
    // }

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
          typename KernelCallable,
          typename RefKernelCallable>
void run_benchmark(
    int M, int N, int K, int timed_iters, bool do_validation, 
    KernelCallable kernel_to_run,
    RefKernelCallable ref_kernel_to_run
) {

    typename ElementA::DataType *A;
    typename ElementB::DataType *B;
    ElementC *C;
    ElementD *D;
    ElementD *D_ref;
    
    A = new typename ElementA::DataType[M * K];
    B = new typename ElementB::DataType[N * K];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    D_ref = new ElementD[M * N];

    int szA = M * ((K + block_size - 1) / block_size);
    typename ElementA::ScaleFactorType *scaleA = new typename ElementA::ScaleFactorType[szA];
    int szB = N * ((K + block_size - 1) / block_size);
    typename ElementB::ScaleFactorType *scaleB = new typename ElementB::ScaleFactorType[szB];

    std::srand(static_cast<unsigned int>(123));
    cutlass::NumericConverter<typename ElementA::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterA;
    cutlass::NumericConverter<typename ElementB::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterB;
    cutlass::NumericConverter<typename ElementA::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFA;
    cutlass::NumericConverter<typename ElementB::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFB;

    initialize_matrix_random(A, M * K);
    initialize_matrix_random(B, N * K);
    initialize_matrix_random(C, M * N);
    // initialize_matrix(A, M * K, 0.f);
    // initialize_matrix(B, N * K, 0.f);
    // initialize_matrix(C, M * N, 0.5f);

    initialize_matrix_random(scaleA, szA);
    initialize_matrix_random(scaleB, szB);


    // for (int i = 0; i < M * K; ++i) { A[i] = converterA(static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f); }
    // for (int i = 0; i < N * K; ++i) { B[i] = converterB(static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f); }
    // for (int i = 0; i < M * N; ++i) { C[i] = static_cast<ElementC>(static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f); }
    // for (size_t i = 0; i < szA; ++i) { scaleA[i] = converterSFA(static_cast<float>(std::rand()) / RAND_MAX * 255.0f); }
    // for (size_t i = 0; i < szB; ++i) { scaleB[i] = converterSFB(static_cast<float>(std::rand()) / RAND_MAX * 255.0f); }

    typename ElementA::DataType *A_d;
    typename ElementB::DataType *B_d;
    typename ElementA::ScaleFactorType *SFA_d;
    typename ElementB::ScaleFactorType *SFB_d;
    ElementC *C_d;
    ElementD *D_d;
    ElementD *D_d_ref;

    cudaMalloc(&A_d, M * K * sizeof(typename ElementA::DataType));
    cudaMalloc(&B_d, N * K * sizeof(typename ElementB::DataType));
    cudaMalloc(&SFA_d, szA * sizeof(typename ElementA::ScaleFactorType));
    cudaMalloc(&SFB_d, szB * sizeof(typename ElementB::ScaleFactorType));
    cudaMalloc(&C_d, M * N * sizeof(ElementC));
    cudaMalloc(&D_d, M * N * sizeof(ElementD));
    cudaMalloc(&D_d_ref, M * N * sizeof(ElementD));
    checkCudaLastErrors();

    cudaMemcpy(A_d, A, M * K * sizeof(typename ElementA::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * K * sizeof(typename ElementB::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFA_d, scaleA, szA * sizeof(typename ElementA::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(SFB_d, scaleB, szB * sizeof(typename ElementB::ScaleFactorType), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);

    if (do_validation) {
        auto kernel_val_lambda = [&]() {
            kernel_to_run(A_d, SFA_d, B_d, SFB_d, C_d, D_d, M, N, K);
        };
        auto ref_val_lambda = [&]() {
            ref_kernel_to_run(A_d, B_d, M, N, K, C_d, D_d_ref, SFA_d, SFB_d);
        };
        perform_validation(M, N, D, D_ref, D_d, D_d_ref, kernel_val_lambda, ref_val_lambda);
    }

    auto kernel_bench_lambda = [&]() {
        kernel_to_run(A_d, SFA_d, B_d, SFB_d, C_d, D_d, M, N, K);
    };

    float milliseconds = perform_benchmark(timed_iters, kernel_bench_lambda);

    double time_sec = static_cast<double>(milliseconds) / 1000.0 / timed_iters;
    double flops_per_gemm = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tflops = flops_per_gemm / (time_sec * 1.0e12);

    printf("Iter Runs = %4d\n", timed_iters);
    printf("BM=%5d, BN=%5d, BK=%5d, ", BM, BN, BK);
    printf("Pipeline Stage = %d\n", N_STAGE);
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
void launch_and_run_benchmark(int M, int N, int K, int timed_iters, bool do_validation, RefKernel ref_kernel) {

    auto kernel_launcher =
        [=](typename EA::DataType* pA, typename EA::ScaleFactorType* pSFA,
            typename EB::DataType* pB, typename EB::ScaleFactorType* pSFB,
            ElementC* pC,
            ElementD* pD, 
            int m, int n, int k) {
        gemm_host_tn<N_STAGE, BM, BN, BK>(pA, pSFA, pB, pSFB, pC, pD, m, n, k);
    };
    auto ref_kernel_launcher =
            [=](typename EA::DataType* pA, typename EB::DataType* pB,
                int m, int n, int k, ElementC* pC, ElementD* pD_ref,
                typename EA::ScaleFactorType* pSFA, typename EB::ScaleFactorType* pSFB) {
            ref_kernel(pA, pB, m, n, k, pC, pD_ref, pSFA, pSFB);
        };
    run_benchmark<EA, EB>(M, N, K, timed_iters, do_validation, kernel_launcher, ref_kernel_launcher);
}

template<class EA, class EB>
void default_matmul_host(
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
    print("\n\nThis GEMM kernel has no REF implementation, does not support VAL!\n\n");
}
int main(int argc, char** argv) {
    int M = 2048;
    int N = 4096;
    int K = 4096;
    int timed_iters = 100;
    bool do_validation = false;
    std::string prec_str = "8x8";

    if (argc >= 2) M = std::atoi(argv[1]);
    if (argc >= 3) N = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);
    if (argc >= 5) timed_iters = std::atoi(argv[4]);
    if (argc >= 6) prec_str = argv[5];
    if (argc >= 7 && std::string(argv[6]) == "--validate") { 
        do_validation = true;
        printf("Validation Enabled.\n");
    }
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
        launch_and_run_benchmark<Type4, Type4>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "4" && prec_b == "6") {
        printf("Running MXFP4 (E2M1) x MXFP6 (E3M2)\n");
        auto ref_kernel = default_matmul_host<Type4, Type6>;
        launch_and_run_benchmark<Type4, Type6>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "4" && prec_b == "8") {
        printf("Running MXFP4 (E2M1) x MXFP8 (E4M3)\n");
        auto ref_kernel = default_matmul_host<Type4, Type8>;
        launch_and_run_benchmark<Type4, Type8>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    // A=6
    else if (prec_a == "6" && prec_b == "4") {
        printf("Running MXFP6 (E3M2) x MXFP4 (E2M1)\n");
        auto ref_kernel = matmul_host_w4a6;
        launch_and_run_benchmark<Type6, Type4>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "6" && prec_b == "6") {
        printf("Running MXFP6 (E3M2) x MXFP6 (E3M2)\n");
        auto ref_kernel = matmul_host_w6a6;
        launch_and_run_benchmark<Type6, Type6>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "6" && prec_b == "8") {
        printf("Running MXFP6 (E3M2) x MXFP8 (E4M3)\n");
        auto ref_kernel = default_matmul_host<Type6, Type8>;
        launch_and_run_benchmark<Type6, Type8>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    // A=8
    else if (prec_a == "8" && prec_b == "4") {
        printf("Running MXFP8 (E4M3) x MXFP4 (E2M1)\n");
        auto ref_kernel = matmul_host_w4a8;
        launch_and_run_benchmark<Type8, Type4>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "8" && prec_b == "6") {
        printf("Running MXFP8 (E4M3) x MXFP6 (E3M2)\n");
        auto ref_kernel = default_matmul_host<Type8, Type6>;
        launch_and_run_benchmark<Type8, Type6>(M, N, K, timed_iters, do_validation, ref_kernel);
    }
    else if (prec_a == "8" && prec_b == "8") {
        printf("Running MXFP8 (E4M3) x MXFP8 (E4M3)\n");
        auto ref_kernel = matmul_host_w8a8;
        launch_and_run_benchmark<Type8, Type8>(M, N, K, timed_iters, do_validation, ref_kernel);
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
