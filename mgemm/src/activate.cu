#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_conversion.h>
#include <cmath>
#include <cstdio>

#include "reorder.cuh"
#include <cute/tensor.hpp>

using namespace cute;

// --- 辅助函数和结构体 (与之前版本相同) ---
#define DEVICE __forceinline__ __device__

#define FP4_MAX 6.0f
#define FP6_MAX 28.0f
#define FP8_MAX 448.0f

typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_e3m2_t fp6_t;
typedef cutlass::float_e4m3_t fp8_t;
typedef cutlass::bfloat16_t bf16_t;
typedef cutlass::float_ue8m0_t sf_t;

namespace cg = cooperative_groups;

struct PackFp4 { int8_t low : 4; int8_t high : 4; };
DEVICE float silu(float x) { return x / (1.0f + expf(-x)); }
DEVICE void pack_4_fp6_to_3_bytes(const uint8_t v0, const uint8_t v1, const uint8_t v2, const uint8_t v3, uint8_t* output_bytes) {
    uint8_t c0 = v0 & 0x3F, c1 = v1 & 0x3F, c2 = v2 & 0x3F, c3 = v3 & 0x3F;
    output_bytes[0] = c0 | (c1 << 6);
    output_bytes[1] = (c1 >> 2) | (c2 << 4);
    output_bytes[2] = (c2 >> 4) | (c3 << 2);
}
enum class QuantType { FP4, FP6, FP8 };



template <
    int GROUP_SIZE, int ELEMENTS_PER_THREAD,
    typename F4ScaleTensor, typename F6ScaleTensor, typename F8ScaleTensor
>
__global__ void activate_quantize_kernel_with_cute_layout(
    const bf16_t *A, const bf16_t *B,
    uint8_t *f4out, uint8_t *f6out, uint8_t *f8out,
    F4ScaleTensor f4scale, F6ScaleTensor f6scale, F8ScaleTensor f8scale, // 接收 CUTE Tensor 对象
    const int KN, const int KS, const int KO,
    const int total_rows
) {
    // --- 1. 初始化和设置 (与之前版本相同) ---
    static_assert(GROUP_SIZE % 32 == 0, "GROUP_SIZE must be a multiple of 32.");
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / ELEMENTS_PER_THREAD;
    static_assert(THREADS_PER_GROUP == 8, "This kernel is optimized for 8 threads per quantization group.");
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<THREADS_PER_GROUP> segment = cg::tiled_partition<THREADS_PER_GROUP>(cta);
    cutlass::NumericConverter<fp4_t, float> converterN;
    cutlass::NumericConverter<fp6_t, float> converterS;
    cutlass::NumericConverter<fp8_t, float> converterO;
    cutlass::NumericConverter<bf16_t, float> converterBF16;
    cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    constexpr int GROUPS_PER_BLOCK = 256 / THREADS_PER_GROUP;
    __shared__ uint8_t smem_packed[GROUPS_PER_BLOCK * 24];

    // --- 2. 网格跨步循环 (与之前版本相同) ---
    const int groups_per_block = blockDim.x / THREADS_PER_GROUP;
    const int start_group_offset = blockIdx.x * groups_per_block + (threadIdx.x / THREADS_PER_GROUP);
    const int total_groups_per_row = (KN + KS + KO) / GROUP_SIZE;

    for (int row_id = blockIdx.y; row_id < total_rows; row_id += gridDim.y) {
        for (int group_idx = start_group_offset; group_idx < total_groups_per_row; group_idx += gridDim.x * groups_per_block) {

            // --- 3. 确定类型和指针 (大部分与之前相同) ---
            QuantType q_type;
            float quant_max_bound;
            uint8_t* data_out_ptr;
            int group_idx_in_type;

            const int KN_GROUPS = KN / GROUP_SIZE;
            const int KS_GROUPS = KS / GROUP_SIZE;

            if (group_idx < KN_GROUPS) {
                q_type = QuantType::FP4; quant_max_bound = FP4_MAX; group_idx_in_type = group_idx;
                data_out_ptr = f4out + row_id * (KN / 2) + group_idx_in_type * (GROUP_SIZE / 2);
            } else if (group_idx < KN_GROUPS + KS_GROUPS) {
                q_type = QuantType::FP6; quant_max_bound = FP6_MAX; group_idx_in_type = group_idx - KN_GROUPS;
                data_out_ptr = f6out + row_id * (KS * 3 / 4) + group_idx_in_type * (GROUP_SIZE * 3 / 4);
            } else {
                q_type = QuantType::FP8; quant_max_bound = FP8_MAX; group_idx_in_type = group_idx - (KN_GROUPS + KS_GROUPS);
                data_out_ptr = f8out + row_id * KO + group_idx_in_type * GROUP_SIZE;
            }

            // --- 4. 加载、激活、归约 (与之前版本相同) ---
            const bf16_t* a_ptr = A + row_id * (KN + KS + KO) + group_idx * GROUP_SIZE;
            const bf16_t* b_ptr = B + row_id * (KN + KS + KO) + group_idx * GROUP_SIZE;
            float input_float[ELEMENTS_PER_THREAD];
            bf16_t local_a[ELEMENTS_PER_THREAD], local_b[ELEMENTS_PER_THREAD];
            const bf16_t* thread_a_ptr = a_ptr + segment.thread_rank() * ELEMENTS_PER_THREAD;
            const bf16_t* thread_b_ptr = b_ptr + segment.thread_rank() * ELEMENTS_PER_THREAD;
            reinterpret_cast<float2*>(local_a)[0] = reinterpret_cast<const float2*>(thread_a_ptr)[0];
            reinterpret_cast<float2*>(local_a)[1] = reinterpret_cast<const float2*>(thread_a_ptr)[1];
            reinterpret_cast<float2*>(local_b)[0] = reinterpret_cast<const float2*>(thread_b_ptr)[0];
            reinterpret_cast<float2*>(local_b)[1] = reinterpret_cast<const float2*>(thread_b_ptr)[1];
            float maxv = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                input_float[i] = silu(static_cast<float>(local_a[i])) * static_cast<float>(local_b[i]);
                maxv = fmaxf(maxv, fabsf(input_float[i]));
            }
            maxv = cg::reduce(segment, maxv, [](float a, float b) {
                return fmaxf(a, b);
            });
            
            // --- 5. 计算 Scale 并使用 CUTE 布局存储 ---
            // (这是核心修改点)
            float scale = 1.0f;
            if (segment.thread_rank() == 0) {
                if (maxv > 1e-6f) {
                    scale = ldexpf(1.0f, static_cast<int>(ceilf(log2f(maxv / quant_max_bound))));
                }

                // 创建逻辑坐标并使用 CUTE Tensor 写入
                auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
                auto logical_coord1 = make_coord(make_coord(0, group_idx_in_type % 4), group_idx_in_type / 4);
                auto logical_coord2 = make_coord(0, 0);
                auto logical_coord = make_coord(logical_coord0, logical_coord1, logical_coord2);

                switch(q_type) {
                    case QuantType::FP4: f4scale(logical_coord) = converterSF(scale); break;
                    case QuantType::FP6: f6scale(logical_coord) = converterSF(scale); break;
                    case QuantType::FP8: f8scale(logical_coord) = converterSF(scale); break;
                }
            }
            scale = segment.shfl(scale, 0);
            const float r_scale = 1.0f / scale;

            // --- 6. 量化、打包、存储 (与之前版本相同) ---
            switch (q_type) {
                case QuantType::FP8: {
                    uint8_t packed_data[ELEMENTS_PER_THREAD];
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                        float val = roundf(input_float[i] * r_scale);
                        val = fminf(fmaxf(val, -FP8_MAX), FP8_MAX);
                        packed_data[i] = converterO(val).storage;
                    }
                    // 合并写入4个字节
                    *reinterpret_cast<uint32_t*>(data_out_ptr + segment.thread_rank() * 4) = *reinterpret_cast<uint32_t*>(packed_data);
                    break;
                }
                case QuantType::FP4: {
                    PackFp4 packed_data[ELEMENTS_PER_THREAD / 2];
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_THREAD / 2; ++i) {
                        float val1 = roundf(input_float[i*2] * r_scale);
                        val1 = fminf(fmaxf(val1, -FP4_MAX), FP4_MAX);
                        float val2 = roundf(input_float[i*2+1] * r_scale);
                        val2 = fminf(fmaxf(val2, -FP4_MAX), FP4_MAX);
                        packed_data[i].low = converterN(val1).storage;
                        packed_data[i].high = converterN(val2).storage;
                    }
                    // 合并写入2个字节
                    *reinterpret_cast<uint16_t*>(data_out_ptr + segment.thread_rank() * 2) = *reinterpret_cast<uint16_t*>(packed_data);
                    break;
                }
                case QuantType::FP6: {
                    uint8_t temp_packed[3];
                    // 量化4个值
                    uint8_t v[4];
                    #pragma unroll
                    for(int i = 0; i < 4; ++i) {
                        float val = roundf(input_float[i] * r_scale);
                        val = fminf(fmaxf(val, -FP6_MAX), FP6_MAX);
                        v[i] = converterS(val).storage;
                    }
                    // 打包到3个字节中
                    pack_4_fp6_to_3_bytes(v[0], v[1], v[2], v[3], temp_packed);

                    // 使用共享内存实现合并写入
                    int smem_group_offset = (threadIdx.x / THREADS_PER_GROUP) * 24; // 24 bytes per group
                    
                    // 非合并写入到共享内存
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 0] = temp_packed[0];
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 1] = temp_packed[1];
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 2] = temp_packed[2];
                    
                    cta.sync();

                    // 从共享内存合并读取，并写入到全局内存
                    // 每个segment需要写入24字节，我们让前6个线程每个写4字节(uint32_t)
                    if (segment.thread_rank() < 6) {
                        uint32_t val_to_write = *reinterpret_cast<uint32_t*>(&smem_packed[smem_group_offset + segment.thread_rank() * 4]);
                        *reinterpret_cast<uint32_t*>(data_out_ptr + segment.thread_rank() * 4) = val_to_write;
                    }
                    
                    cta.sync(); // 确保写完后再进入下一次循环
                    break;
                }
            }
        }
    }
}

template <
    int GROUP_SIZE, int ELEMENTS_PER_THREAD,
    typename F4ScaleTensor, typename F6ScaleTensor, typename F8ScaleTensor
>
__global__ void downproj_quantize_kernel_with_cute_layout(
    const bf16_t *W,
    uint8_t *f4out, uint8_t *f6out, uint8_t *f8out,
    F4ScaleTensor f4scale, F6ScaleTensor f6scale, F8ScaleTensor f8scale, // 接收 CUTE Tensor 对象
    const int KN, const int KS, const int KO,
    const int total_rows
) {
    // --- 1. 初始化和设置 (与之前版本相同) ---
    static_assert(GROUP_SIZE % 32 == 0, "GROUP_SIZE must be a multiple of 32.");
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / ELEMENTS_PER_THREAD;
    static_assert(THREADS_PER_GROUP == 8, "This kernel is optimized for 8 threads per quantization group.");
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<THREADS_PER_GROUP> segment = cg::tiled_partition<THREADS_PER_GROUP>(cta);
    cutlass::NumericConverter<fp4_t, float> converterN;
    cutlass::NumericConverter<fp6_t, float> converterS;
    cutlass::NumericConverter<fp8_t, float> converterO;
    cutlass::NumericConverter<bf16_t, float> converterBF16;
    cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    constexpr int GROUPS_PER_BLOCK = 256 / THREADS_PER_GROUP;
    __shared__ uint8_t smem_packed[GROUPS_PER_BLOCK * 24];

    // --- 2. 网格跨步循环 (与之前版本相同) ---
    const int groups_per_block = blockDim.x / THREADS_PER_GROUP;
    const int start_group_offset = blockIdx.x * groups_per_block + (threadIdx.x / THREADS_PER_GROUP);
    const int total_groups_per_row = (KN + KS + KO) / GROUP_SIZE;

    for (int row_id = blockIdx.y; row_id < total_rows; row_id += gridDim.y) {
        for (int group_idx = start_group_offset; group_idx < total_groups_per_row; group_idx += gridDim.x * groups_per_block) {

            // --- 3. 确定类型和指针 (大部分与之前相同) ---
            QuantType q_type;
            float quant_max_bound;
            uint8_t* data_out_ptr;
            int group_idx_in_type;

            const int KN_GROUPS = KN / GROUP_SIZE;
            const int KS_GROUPS = KS / GROUP_SIZE;

            if (group_idx < KN_GROUPS) {
                q_type = QuantType::FP4; quant_max_bound = FP4_MAX; group_idx_in_type = group_idx;
                data_out_ptr = f4out + row_id * (KN / 2) + group_idx_in_type * (GROUP_SIZE / 2);
            } else if (group_idx < KN_GROUPS + KS_GROUPS) {
                q_type = QuantType::FP6; quant_max_bound = FP6_MAX; group_idx_in_type = group_idx - KN_GROUPS;
                data_out_ptr = f6out + row_id * (KS * 3 / 4) + group_idx_in_type * (GROUP_SIZE * 3 / 4);
            } else {
                q_type = QuantType::FP8; quant_max_bound = FP8_MAX; group_idx_in_type = group_idx - (KN_GROUPS + KS_GROUPS);
                data_out_ptr = f8out + row_id * KO + group_idx_in_type * GROUP_SIZE;
            }

            // --- 4. 加载、激活、归约 (与之前版本相同) ---
            const bf16_t* a_ptr = W + row_id * (KN + KS + KO) + group_idx * GROUP_SIZE;
            float input_float[ELEMENTS_PER_THREAD];
            bf16_t local_a[ELEMENTS_PER_THREAD];
            const bf16_t* thread_a_ptr = a_ptr + segment.thread_rank() * ELEMENTS_PER_THREAD;
            reinterpret_cast<float2*>(local_a)[0] = reinterpret_cast<const float2*>(thread_a_ptr)[0];
            reinterpret_cast<float2*>(local_a)[1] = reinterpret_cast<const float2*>(thread_a_ptr)[1];
            float maxv = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                input_float[i] = static_cast<float>(local_a[i]);
                maxv = fmaxf(maxv, fabsf(input_float[i]));
            }
            maxv = cg::reduce(segment, maxv, [](float a, float b) {
                return fmaxf(a, b);
            });
            
            // --- 5. 计算 Scale 并使用 CUTE 布局存储 ---
            // (这是核心修改点)
            float scale = 1.0f;
            if (segment.thread_rank() == 0) {
                if (maxv > 1e-6f) {
                    scale = ldexpf(1.0f, static_cast<int>(ceilf(log2f(maxv / quant_max_bound))));
                }

                // 创建逻辑坐标并使用 CUTE Tensor 写入
                auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
                auto logical_coord1 = make_coord(make_coord(0, group_idx_in_type % 4), group_idx_in_type / 4);
                auto logical_coord2 = make_coord(0, 0);
                auto logical_coord = make_coord(logical_coord0, logical_coord1, logical_coord2);

                switch(q_type) {
                    case QuantType::FP4: f4scale(logical_coord) = converterSF(scale); break;
                    case QuantType::FP6: f6scale(logical_coord) = converterSF(scale); break;
                    case QuantType::FP8: f8scale(logical_coord) = converterSF(scale); break;
                }
            }
            scale = segment.shfl(scale, 0);
            const float r_scale = 1.0f / scale;

            // --- 6. 量化、打包、存储 (与之前版本相同) ---
            switch (q_type) {
                case QuantType::FP8: {
                    uint8_t packed_data[ELEMENTS_PER_THREAD];
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                        float val = roundf(input_float[i] * r_scale);
                        val = fminf(fmaxf(val, -FP8_MAX), FP8_MAX);
                        packed_data[i] = converterO(val).storage;
                    }
                    // 合并写入4个字节
                    *reinterpret_cast<uint32_t*>(data_out_ptr + segment.thread_rank() * 4) = *reinterpret_cast<uint32_t*>(packed_data);
                    break;
                }
                case QuantType::FP4: {
                    PackFp4 packed_data[ELEMENTS_PER_THREAD / 2];
                    #pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_THREAD / 2; ++i) {
                        float val1 = roundf(input_float[i*2] * r_scale);
                        val1 = fminf(fmaxf(val1, -FP4_MAX), FP4_MAX);
                        float val2 = roundf(input_float[i*2+1] * r_scale);
                        val2 = fminf(fmaxf(val2, -FP4_MAX), FP4_MAX);
                        packed_data[i].low = converterN(val1).storage;
                        packed_data[i].high = converterN(val2).storage;
                    }
                    // 合并写入2个字节
                    *reinterpret_cast<uint16_t*>(data_out_ptr + segment.thread_rank() * 2) = *reinterpret_cast<uint16_t*>(packed_data);
                    break;
                }
                case QuantType::FP6: {
                    uint8_t temp_packed[3];
                    // 量化4个值
                    uint8_t v[4];
                    #pragma unroll
                    for(int i = 0; i < 4; ++i) {
                        float val = roundf(input_float[i] * r_scale);
                        val = fminf(fmaxf(val, -FP6_MAX), FP6_MAX);
                        v[i] = converterS(val).storage;
                    }
                    // 打包到3个字节中
                    pack_4_fp6_to_3_bytes(v[0], v[1], v[2], v[3], temp_packed);

                    // 使用共享内存实现合并写入
                    int smem_group_offset = (threadIdx.x / THREADS_PER_GROUP) * 24; // 24 bytes per group
                    
                    // 非合并写入到共享内存
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 0] = temp_packed[0];
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 1] = temp_packed[1];
                    smem_packed[smem_group_offset + segment.thread_rank() * 3 + 2] = temp_packed[2];
                    
                    cta.sync();

                    // 从共享内存合并读取，并写入到全局内存
                    // 每个segment需要写入24字节，我们让前6个线程每个写4字节(uint32_t)
                    if (segment.thread_rank() < 6) {
                        uint32_t val_to_write = *reinterpret_cast<uint32_t*>(&smem_packed[smem_group_offset + segment.thread_rank() * 4]);
                        *reinterpret_cast<uint32_t*>(data_out_ptr + segment.thread_rank() * 4) = val_to_write;
                    }
                    
                    cta.sync(); // 确保写完后再进入下一次循环
                    break;
                }
            }
        }
    }
}


// --- 主机端启动函数  ---
void run_activate_bf16_mixed(
  bf16_t *d_A, bf16_t *d_B, int seq_len, int hidden_dim,
  uint8_t *d_o_normal, uint8_t *d_o_sensitive, uint8_t *d_o_outlier,
  sf_t *d_normal_scale, sf_t *d_sensitive_scale, sf_t *d_outlier_scale,
  int KN, int KS, int KO
) {
    constexpr int GROUP_SIZE = 32;
    constexpr int ELEMENTS_PER_THREAD = 4;
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / ELEMENTS_PER_THREAD;

    if (hidden_dim != (KN + KS + KO)) {
        fprintf(stderr, "Error: hidden_dim does not match sum of KN, KS, KO.\n");
        return;
    }
    
    // --- 创建 CUTE Tensor 对象 ---

    Tensor f4scale_tensor = cute::make_tensor(d_normal_scale, filter_zeros(normal::get_layoutSFA(seq_len, KN)));
    Tensor f6scale_tensor = cute::make_tensor(d_sensitive_scale, filter_zeros(sensitive::get_layoutSFA(seq_len, KS)));
    Tensor f8scale_tensor = cute::make_tensor(d_outlier_scale, filter_zeros(outlier::get_layoutSFA(seq_len, KO)));

    // --- 设置启动配置 ---
    const int max_grid_y = 65536/4;
    const int threads_per_block = 128; 
    const int groups_per_block = threads_per_block / THREADS_PER_GROUP;
    const int total_groups = hidden_dim / GROUP_SIZE;
    dim3 grids((total_groups + groups_per_block - 1) / groups_per_block, std::min(max_grid_y, seq_len));
    dim3 blocks(threads_per_block);

    // --- 启动内核，传入 Tensor 对象 ---
    activate_quantize_kernel_with_cute_layout<GROUP_SIZE, ELEMENTS_PER_THREAD><<<grids, blocks>>>(
        d_A, d_B,
        d_o_normal, d_o_sensitive, d_o_outlier,
        f4scale_tensor, f6scale_tensor, f8scale_tensor, // 传递 Tensor 对象
        KN, KS, KO, seq_len
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}

void run_downproj_bf16_mixed(
  bf16_t *d_W, int out_features, int hidden_dim,
  uint8_t *d_o_normal, uint8_t *d_o_sensitive, uint8_t *d_o_outlier,
  sf_t *d_normal_scale, sf_t *d_sensitive_scale, sf_t *d_outlier_scale,
  int KN, int KS, int KO
) {
    constexpr int GROUP_SIZE = 32;
    constexpr int ELEMENTS_PER_THREAD = 4;
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / ELEMENTS_PER_THREAD;

    if (hidden_dim != (KN + KS + KO)) {
        fprintf(stderr, "Error: hidden_dim does not match sum of KN, KS, KO.\n");
        return;
    }
    
    // --- 创建 CUTE Tensor 对象 ---

    Tensor f4scale_tensor = cute::make_tensor(d_normal_scale, filter_zeros(normal::get_layoutSFB(out_features, KN)));
    Tensor f6scale_tensor = cute::make_tensor(d_sensitive_scale, filter_zeros(sensitive::get_layoutSFB(out_features, KS)));
    Tensor f8scale_tensor = cute::make_tensor(d_outlier_scale, filter_zeros(outlier::get_layoutSFB(out_features, KO)));

    // --- 设置启动配置 ---
    const int max_grid_y = 65536/4;
    const int threads_per_block = 128; 
    const int groups_per_block = threads_per_block / THREADS_PER_GROUP;
    const int total_groups = hidden_dim / GROUP_SIZE;
    dim3 grids((total_groups + groups_per_block - 1) / groups_per_block, std::min(max_grid_y, out_features));
    dim3 blocks(threads_per_block);

    // --- 启动内核，传入 Tensor 对象 ---
    downproj_quantize_kernel_with_cute_layout<GROUP_SIZE, ELEMENTS_PER_THREAD><<<grids, blocks>>>(
        d_W,
        d_o_normal, d_o_sensitive, d_o_outlier,
        f4scale_tensor, f6scale_tensor, f8scale_tensor, // 传递 Tensor 对象
        KN, KS, KO, out_features
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
}