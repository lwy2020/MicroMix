#include <torch/extension.h>

// Include all files
#include "gemm.h"
#include "reorder.cuh"
#include "mxdtype.h"


torch::Tensor matmul(
        const torch::Tensor &AN,
        const torch::Tensor &BN,
        const torch::Tensor &AS,
        const torch::Tensor &BS,
        const torch::Tensor &AO,
        const torch::Tensor &BO,
        const torch::Tensor &SFAN,
        const torch::Tensor &SFBN,
        const torch::Tensor &SFAS,
        const torch::Tensor &SFBS,
        const torch::Tensor &SFAO,
        const torch::Tensor &SFBO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    uint32_t M = AN.size(0);
    uint32_t N = BN.size(0);
    uint32_t KN = AN.size(1) * 2;  // 4bit packing is on the columns
    uint32_t KS = AS.size(1) * 4 / 3;  // 6bit packing is on the columns
    uint32_t KO = AO.size(1) * 1;  // 8bit packing is on the columns
    // assert(KN == 2560 && KS == 1408 && KO == 128);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(AN.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;

    matmul_host(
        reinterpret_cast<cutlass::float_e2m1_t *>(AN.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_e2m1_t *>(BN.data_ptr<uint8_t>()),
        reinterpret_cast<cutlass::float_e3m2_t *>(AS.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_e3m2_t *>(BS.data_ptr<uint8_t>()), 
        reinterpret_cast<cutlass::float_e4m3_t *>(AO.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_e4m3_t *>(BO.data_ptr<uint8_t>()),
        M, N,
        KN, KS, KO,
        (cutlass::bfloat16_t *)C.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)C.data_ptr<at::BFloat16>(),
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFAN.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_ue8m0_t *>(SFBN.data_ptr<uint8_t>()),
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFAS.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_ue8m0_t *>(SFBS.data_ptr<uint8_t>()),
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFAO.data_ptr<uint8_t>()), reinterpret_cast<cutlass::float_ue8m0_t *>(SFBO.data_ptr<uint8_t>())
    );
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching GEMM: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in GEMM: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after GEMM kernel synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in GEMM kernel: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "GEMM kernel finished and synced successfully." << std::endl; std::cout.flush();
    return C;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reorder_quantize_x(
        const torch::Tensor &X,
        const torch::Tensor &reorder_index,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int M = X.size(0);
    int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto XN = torch::empty({M, KN / 2}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XS = torch::empty({M, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XO = torch::empty({M, KO}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXN = torch::empty({(M / 128 + 1) * 128 * KN / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXS = torch::empty({(M / 128 + 1) * 128 * KS / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXO = torch::empty({(M / 128 + 1) * 128 * KO / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
        run_reorder_bf16_mixed<32, 4096>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
        run_reorder_bf16_mixed<32, 5120>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
        run_reorder_bf16_mixed<32, 3584>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
        run_reorder_bf16_mixed<32, 3072>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 8192) {
        run_reorder_bf16_mixed<32, 8192>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 14336) {
        run_reorder_bf16_mixed<32, 14336>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 18944) {
        run_reorder_bf16_mixed<32, 18944>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 12288) {
        run_reorder_bf16_mixed<32, 12288>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 13824) {
        run_reorder_bf16_mixed<32, 13824>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_bf16_mixed "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_bf16_mixed: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_bf16_mixed: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_bf16_mixed synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_bf16_mixed: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_bf16_mixed kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(XN, XS, XO, SFXN, SFXS, SFXO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reorder_quantize_w(
        const torch::Tensor &W,
        const torch::Tensor &reorder_index,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int N = W.size(0);
    const int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto WN = torch::empty({N, KN / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WS = torch::empty({N, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WO = torch::empty({N, KO}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWN = torch::empty({N * KN / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWS = torch::empty({N * KS / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWO = torch::empty({N * KO / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
         run_reorder_bf16_fp4<32, 4096>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
         run_reorder_bf16_fp4<32, 5120>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
         run_reorder_bf16_fp4<32, 3584>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
         run_reorder_bf16_fp4<32, 3072>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 8192) {
         run_reorder_bf16_fp4<32, 8192>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 14336) {
         run_reorder_bf16_fp4<32, 14336>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 18944) {
         run_reorder_bf16_fp4<32, 18944>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 12288) {
         run_reorder_bf16_fp4<32, 12288>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 13824) {
         run_reorder_bf16_fp4<32, 13824>(
            (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, reorder_index.data_ptr<int16_t>(), 
            WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_reorder_bf16_fp4 "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_bf16_fp4: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_bf16_fp4: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_bf16_fp4 synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_bf16_fp4: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_bf16_fp4 kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(WN, WS, WO, SFWN, SFWS, SFWO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rmsnorm_quantize_x(
        const torch::Tensor &X,
        const torch::Tensor &W,
        const float eps,
        const torch::Tensor &reorder_index,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int M = X.size(0);
    int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto XN = torch::empty({M, KN / 2}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XS = torch::empty({M, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(X.device()));
    auto XO = torch::empty({M, KO}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXN = torch::empty({(M / 128 + 1) * 128 * KN / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXS = torch::empty({(M / 128 + 1) * 128 * KS / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    auto SFXO = torch::empty({(M / 128 + 1) * 128 * KO / 32}, torch::dtype(torch::kUInt8).device(X.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    if (K == 4096) {
        run_rmsnorm_bf16_mixed<32, 4096>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps,
            M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 5120) {
        run_rmsnorm_bf16_mixed<32, 5120>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps,
            M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3072) {
        run_rmsnorm_bf16_mixed<32, 3072>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps,
            M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else if (K == 3584) {
        run_rmsnorm_bf16_mixed<32, 3584>(
            (cutlass::bfloat16_t *)X.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), eps,
            M, reorder_index.data_ptr<int16_t>(), 
            XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
            KN, KS, KO
        );
    }
    else {
        std::cerr << "K value is not valid !" << std::endl;
        throw std::runtime_error(std::string("Value error in run_rmsnorm_bf16_mixed "));
    }
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_bf16_mixed: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_bf16_mixed: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_bf16_mixed synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_bf16_mixed: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_bf16_mixed kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(XN, XS, XO, SFXN, SFXS, SFXO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> activate_quantize_x(
        const torch::Tensor &A,
        const torch::Tensor &B,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int M = A.size(0);
    int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto XN = torch::empty({M, KN / 2}, torch::dtype(torch::kUInt8).device(A.device()));
    auto XS = torch::empty({M, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(A.device()));
    auto XO = torch::empty({M, KO}, torch::dtype(torch::kUInt8).device(A.device()));
    auto SFXN = torch::empty({(M / 128 + 1) * 128 * KN / 32}, torch::dtype(torch::kUInt8).device(A.device()));
    auto SFXS = torch::empty({(M / 128 + 1) * 128 * KS / 32}, torch::dtype(torch::kUInt8).device(A.device()));
    auto SFXO = torch::empty({(M / 128 + 1) * 128 * KO / 32}, torch::dtype(torch::kUInt8).device(A.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    run_activate_bf16_mixed(
        (cutlass::bfloat16_t *)A.data_ptr<at::BFloat16>(), (cutlass::bfloat16_t *)B.data_ptr<at::BFloat16>(), M, K, 
        XN.data_ptr<uint8_t>(), XS.data_ptr<uint8_t>(), XO.data_ptr<uint8_t>(), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFXN.data_ptr<uint8_t>()), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFXS.data_ptr<uint8_t>()), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFXO.data_ptr<uint8_t>()), 
        KN, KS, KO
    );
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_bf16_mixed: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_bf16_mixed: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_bf16_mixed synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_bf16_mixed: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_bf16_mixed kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(XN, XS, XO, SFXN, SFXS, SFXO);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> downproj_quantize_w(
        const torch::Tensor &W,
        const int KN,
        const int KS,
        const int KO
)
{
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    // torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    int N = W.size(0);
    int K = KN + KS + KO;
    // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
    auto WN = torch::empty({N, KN / 2}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WS = torch::empty({N, KS / 4 * 3}, torch::dtype(torch::kUInt8).device(W.device()));
    auto WO = torch::empty({N, KO}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWN = torch::empty({(N / 128 + 1) * 128 * KN / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWS = torch::empty({(N / 128 + 1) * 128 * KS / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    auto SFWO = torch::empty({(N / 128 + 1) * 128 * KO / 32}, torch::dtype(torch::kUInt8).device(W.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    run_downproj_bf16_mixed(
        (cutlass::bfloat16_t *)W.data_ptr<at::BFloat16>(), N, K, 
        WN.data_ptr<uint8_t>(), WS.data_ptr<uint8_t>(), WO.data_ptr<uint8_t>(), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFWN.data_ptr<uint8_t>()), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFWS.data_ptr<uint8_t>()), 
        reinterpret_cast<cutlass::float_ue8m0_t *>(SFWO.data_ptr<uint8_t>()), 
        KN, KS, KO
    );
    // // CRITICAL: Synchronize and check for errors immediately after kernel launch
    // cudaError_t kernel_err = cudaGetLastError(); // Check for asynchronous errors from the kernel
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "CUDA error after launching run_reorder_bf16_mixed: "
    //             << cudaGetErrorString(kernel_err) << std::endl;
    //     // Optionally, throw an exception to propagate the error to Python
    //     throw std::runtime_error(std::string("CUDA error in run_reorder_bf16_mixed: ") + cudaGetErrorString(kernel_err));
    // }

    // cudaError_t sync_err = cudaDeviceSynchronize(); // Wait for the kernel to complete and check for runtime errors
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "CUDA error during/after run_reorder_bf16_mixed synchronization: "
    //             << cudaGetErrorString(sync_err) << std::endl;
    //     throw std::runtime_error(std::string("CUDA sync error in run_reorder_bf16_mixed: ") + cudaGetErrorString(sync_err));
    // }
    // std::cout << "run_reorder_bf16_mixed kernel finished and synced successfully." << std::endl; std::cout.flush();
    return std::make_tuple(WN, WS, WO, SFWN, SFWS, SFWO);
}




//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul", &matmul,
          "input: (AN: torch.Tensor(M x KN, UINT8, CUDA), BN: torch.Tensor(N x KN, "
          "UINT8, CUDA), AS: torch.Tensor(M x KS, UINT8, CUDA), BS: torch.Tensor(N x KS, "
          "UINT8, CUDA), AO: torch.Tensor(M x KO, UINT8, CUDA), BO: torch.Tensor(N x KO, "
          "UINT8, CUDA), SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)\n"
          "output: torch.Tensor(M x N, BFLOAT16, CUDA)\n"
          "output = A @ B^T",
          py::arg("AN"), py::arg("BN"),
          py::arg("AS"), py::arg("BS"),
          py::arg("AO"), py::arg("BO"),
          py::arg("SFAN"), py::arg("SFBN"),
          py::arg("SFAS"), py::arg("SFBS"),
          py::arg("SFAO"), py::arg("SFBO")
        );
    m.def("test_function", []() { return "Hello from test_function!"; });
    m.def("reorder_quantize_x", &reorder_quantize_x,
          "Reorder and quantize activation",
          py::arg("X"), py::arg("reorder_index"),
          py::arg("KN"), py::arg("KS"), py::arg("KO")
        );
    m.def("reorder_quantize_w", &reorder_quantize_w,
          "Reorder and quantize weight",
          py::arg("W"), py::arg("reorder_index"),
          py::arg("KN"), py::arg("KS"), py::arg("KO")
        );

    m.def("rmsnorm_quantize_x", &rmsnorm_quantize_x,
          "Normalize and quantize activation",
          py::arg("X"), py::arg("W"), py::arg("eps"), py::arg("reorder_index"),
          py::arg("KN"), py::arg("KS"), py::arg("KO")
        );
    m.def("activate_quantize_x", &activate_quantize_x,
          "Activate and quantize activation",
          py::arg("A"), py::arg("B"),
          py::arg("KN"), py::arg("KS"), py::arg("KO")
        );
    m.def("downproj_quantize_w", &downproj_quantize_w,
          "Quantize down_proj weight",
          py::arg("W"),
          py::arg("KN"), py::arg("KS"), py::arg("KO")
        );

}