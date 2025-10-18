#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "reorder.cuh"
#include "cutlass/numeric_conversion.h"

#include <cstdio>


#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

#define FP4_MAX 6
#define FP6_MAX 28
#define FP8_MAX 448

typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_e3m2_t fp6_t;
typedef cutlass::float_e4m3_t fp8_t;
typedef cutlass::float_ue8m0_t sf_t;
typedef cutlass::bfloat16_t bf16_t;

namespace cg = cooperative_groups;
using namespace cute;

struct PackFp4 {
  int8_t low : 4;
  int8_t high : 4;
};

// HOST_DEVICE int cdiv(int a, int b) { return (a + b - 1) / b; }

HOST_DEVICE float fpmax(float a, float b) { return (a) > (b) ? (a) : (b); }

HOST_DEVICE float fpmin(float a, float b) { return (a) < (b) ? (a) : (b); }

HOST_DEVICE float clamp(float x, float a, float b) { return fpmax(a, fpmin(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
HOST_DEVICE Accum local_sum_p2(U *vec, Accum sumv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    sumv += (Accum)view[i] * (Accum)view[i];
  }
  return sumv;
}
HOST_DEVICE void pack_4_fp6_to_3_bytes(
    uint8_t fp6_v0, uint8_t fp6_v1, uint8_t fp6_v2, uint8_t fp6_v3,
    uint8_t* output_bytes // Array of 3 uint8_t
) {
    fp6_v0 &= 0x3F; fp6_v1 &= 0x3F; fp6_v2 &= 0x3F; fp6_v3 &= 0x3F;

    output_bytes[0] = (fp6_v0) | ((fp6_v1 & 0x03) << 6);
    output_bytes[1] = (fp6_v1 >> 2) | ((fp6_v2 & 0x0F) << 4);
    output_bytes[2] = (fp6_v2 >> 4) | (fp6_v3 << 2);
}
/*
 * Given a row index, return the start index of the scale.
*/
// HOST_DEVICE int scale_index(int row_id){
//   int bottomUpper = (row_id / 8) % 2;
//   int group_idx = row_id % 8;
//   int group_nums = row_id / 16;
//   return (group_nums * 64) + (group_idx * 8) + bottomUpper;
// }

/*
 * Given the row numbers, calculate the leading dimension of scales.
 * In unit of half.
*/
#define SCALE_SIZE(x) ((x) / 32)
#define GROUP_NUM(x) ((x) / 32)

#define mymax(a, b) ((a) > (b) ? (a) : (b))

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
DEVICE float local_abs_max(U *vec, float maxv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    maxv = mymax((float)maxv, (float)abs((float)view[i]));
  }
  return maxv;
}


template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void reorder_quantize_mixed_kernel(
  bf16_t *input,
  int16_t *reorder_index,
  uint8_t *f4out,
  uint8_t *f6out,
  uint8_t *f8out,
  // cute::Tensor<cute::ViewEngine<sf_t*>, normal::LayoutSFA> f4scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, sensitive::LayoutSFA> f6scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, outlier::LayoutSFA> f8scale,
  auto f4scale,
  auto f6scale,
  auto f8scale,
  int f4scaleldm,
  int f6scaleldm,
  int f8scaleldm,
  int KN, int KS, int KO
){
  // static_assert(GROUP_SIZE == 32 && HIDDEN_DIM == 4096, "Current only support 32x4096.");
  // static_assert(bdx == 128, "Current 128 threads per block.");
  // static_assert(bdx == HIDDEN_DIM / GROUP_SIZE, "Current only support 4096/32.");
  constexpr int elements_per_thread = GROUP_SIZE;

  cg::thread_block cta = cg::this_thread_block();

  // One block solves one row of hidden states.
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(bf16_t)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem);

  // Local memory stores the reordered hidden states.
  bf16_t input_frag[elements_per_thread];

  // Row are independent
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  f4out = f4out + row_id * (GROUP_SIZE * GROUP_NUM(KN)) / 2;
  f6out = f6out + row_id * (GROUP_SIZE * GROUP_NUM(KS)) / 4 * 3;
  f8out = f8out + row_id * (GROUP_SIZE * GROUP_NUM(KO));

  // Coalesced access global memory
  int tx = threadIdx.x;
  int tid = tx;
  constexpr int bytes_per_iter = bdx * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter;
  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterN;
  cutlass::NumericConverter<fp6_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterS;
  cutlass::NumericConverter<fp8_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterO;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
  cutlass::NumericConverter<bf16_t, int, cutlass::FloatRoundStyle::round_to_nearest> converterBF;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterScale;
  cutlass::NumericConverter<int, fp4_t, cutlass::FloatRoundStyle::round_to_nearest> converter4i;

  #pragma unroll
  for(int i = 0;i < iters;++i){
    // Each thread loads 16 bytes
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  cta.sync();
  // Reorder
  #pragma unroll 4
  for(int i = 0;i < elements_per_thread;++i){
    int offset = tid * GROUP_SIZE + i;
    input_frag[i] = input_smem[reorder_index[offset]];
  }
  // Reduce to get max
  // Each ty should get its max value
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  float *input_frag_float = reinterpret_cast<float *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(bf16_t) / sizeof(float4);
  float maxv = 0,  scale = 1.0, r_scale = 1.0;

  #pragma unroll
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();
  // Calculate scales
  // Specific layout
  // int replicated_row_id = scale_index(row_id);
  float lower_bound, upper_bound;
  if (tid >= bdx - GROUP_NUM(KO)) {
    // fp8 quantize
    lower_bound = -FP8_MAX;
    upper_bound = FP8_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP8_MAX)))));
    int idx = (tid + GROUP_NUM(KO) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f8scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)) {
    // fp6 quant
    lower_bound = -FP6_MAX;
    upper_bound = FP6_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP6_MAX)))));
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f6scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
    auto logical_coord2 = make_coord(0, 0);
    f4scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }

  // Use reverse scale to replace devision by multiplication
   r_scale = 1.0 / scale;

  // Quantize each thread's value
  // int lower_bound = (ty == bdy - 1) ? -128 : -8;
  // int upper_bound = (ty == bdy - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  fp8_t* input_frag_fp8 = reinterpret_cast<fp8_t*>(input_frag);
  uint8_t* input_frag_fp6 = reinterpret_cast<uint8_t*>(input_frag);
  PackFp4* input_frag_fp4 = reinterpret_cast<PackFp4*>(input_frag);
  for(int i = 0; i < elements_per_thread; i += 4){
    bf16_t result_0, result_1, result_2, result_3;
    result_0 = converterScale(clamp(((float)input_frag[i + 0] * r_scale), lower_bound, upper_bound));
    result_1 = converterScale(clamp(((float)input_frag[i + 1] * r_scale), lower_bound, upper_bound));
    result_2 = converterScale(clamp(((float)input_frag[i + 2] * r_scale), lower_bound, upper_bound));
    result_3 = converterScale(clamp(((float)input_frag[i + 3] * r_scale), lower_bound, upper_bound));
    if(tid >= bdx - GROUP_NUM(KO)){
      input_frag_fp8[i + 0] = converterO(result_0);
      input_frag_fp8[i + 1] = converterO(result_1);
      input_frag_fp8[i + 2] = converterO(result_2);
      input_frag_fp8[i + 3] = converterO(result_3);
    }
    else if(tid >= bdx - GROUP_NUM(KO + KS)) {
      pack_4_fp6_to_3_bytes(
        converterS(result_0).storage, // Corrected
        converterS(result_1).storage, // Corrected
        converterS(result_2).storage, // Corrected
        converterS(result_3).storage, // Corrected
        (input_frag_fp6 + (i / 4) * 3)
      );
    }
    else {
      input_frag_fp4[i / 2].low = converterN(result_0).storage;
      input_frag_fp4[i / 2].high = converterN(result_1).storage;
      input_frag_fp4[i / 2 + 1].low = converterN(result_2).storage;
      input_frag_fp4[i / 2 + 1].high = converterN(result_3).storage;
    }
  }
  // Store frag out to global memory
  if(tid >= bdx - GROUP_NUM(KO)){
    // Store fp8_t quantized result
    float4* f8out_float4 = reinterpret_cast<float4*>(f8out);
    f8out_float4[(tid + GROUP_NUM(KO) - bdx) * 2 + 0] = input_frag_float4[0];
    f8out_float4[(tid + GROUP_NUM(KO) - bdx) * 2 + 1] = input_frag_float4[1];
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)){ // FP6 data processing path
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    int64_t* f6out_ll = reinterpret_cast<int64_t*>(f6out);
    int64_t* input_frag_ll = reinterpret_cast<int64_t*>(input_frag);
    f6out_ll[idx * 3 + 0] = input_frag_ll[0];
    f6out_ll[idx * 3 + 1] = input_frag_ll[1];
    f6out_ll[idx * 3 + 2] = input_frag_ll[2];
  }
  else {
    // Store fp4_t quantized result
    float4* f4out_float4 = reinterpret_cast<float4*>(f4out);
    f4out_float4[tid] = input_frag_float4[0];
  }
}

template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void reorder_quantize_mxfp4_kernel(
  bf16_t *input,
  int16_t *reorder_index,
  uint8_t *f4out,
  uint8_t *f6out,
  uint8_t *f8out,
  // cute::Tensor<cute::ViewEngine<sf_t*>, normal::LayoutSFB> f4scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, sensitive::LayoutSFB> f6scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, outlier::LayoutSFB> f8scale,
  auto f4scale,
  auto f6scale,
  auto f8scale,
  int f4scaleldm,
  int f6scaleldm,
  int f8scaleldm, 
  int KN, int KS, int KO
){
  // static_assert(GROUP_SIZE == 32 && HIDDEN_DIM == 4096, "Current only support 32x4096.");
  // static_assert(bdx == 128, "Current 128 threads per block.");
  // static_assert(bdx == HIDDEN_DIM / GROUP_SIZE, "Current only support 4096/32.");
  constexpr int elements_per_thread = GROUP_SIZE;

  cg::thread_block cta = cg::this_thread_block();

  // One block solves one row of hidden states.
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(bf16_t)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem);

  // Local memory stores the reordered hidden states.
  bf16_t input_frag[elements_per_thread];

  // Row are independent
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  f4out = f4out + row_id * (GROUP_SIZE * GROUP_NUM(KN)) / 2;
  f6out = f6out + row_id * (GROUP_SIZE * GROUP_NUM(KS)) / 2;
  f8out = f8out + row_id * (GROUP_SIZE * GROUP_NUM(KO)) / 2;

  // Coalesced access global memory
  int tx = threadIdx.x;
  int tid = tx;
  constexpr int bytes_per_iter = bdx * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter;
  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterN;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
  cutlass::NumericConverter<bf16_t, int, cutlass::FloatRoundStyle::round_to_nearest> converterBF;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterScale;

  #pragma unroll
  for(int i = 0;i < iters;++i){
    // Each thread loads 16 bytes
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  cta.sync();
  // Reorder
  #pragma unroll 4
  for(int i = 0;i < elements_per_thread;++i){
    int offset = tid * GROUP_SIZE + i;
    input_frag[i] = input_smem[reorder_index[offset]];
  }
  // Reduce to get max
  // Each ty should get its max value
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  float *input_frag_float = reinterpret_cast<float *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(bf16_t) / sizeof(float4);
  float maxv = 0, scale = 1.0, r_scale = 1.0;

  #pragma unroll
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();
  // Calculate scales
  // Specific layout
  // int replicated_row_id = scale_index(row_id);
  float lower_bound, upper_bound;
  if (tid >= bdx - GROUP_NUM(KO)) {
    // fp4 quantize
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    int idx = (tid + GROUP_NUM(KO) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f8scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)) {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f6scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
    auto logical_coord2 = make_coord(0, 0);
    f4scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }

  // Use reverse scale to replace devision by multiplication
  r_scale = 1.0 / scale;

  // Quantize each thread's value
  // int lower_bound = (ty == bdy - 1) ? -128 : -8;
  // int upper_bound = (ty == bdy - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  // fp4_t* input_frag_fp8 = reinterpret_cast<fp4_t*>(input_frag);
  // fp4_t* input_frag_fp6 = reinterpret_cast<fp4_t*>(input_frag);
  // fp4_t* input_frag_fp4 = reinterpret_cast<fp4_t*>(input_frag);
  PackFp4* input_frag_fp4 = reinterpret_cast<PackFp4*>(input_frag);
  PackFp4* input_frag_fp6 = reinterpret_cast<PackFp4*>(input_frag);
  PackFp4* input_frag_fp8 = reinterpret_cast<PackFp4*>(input_frag);
  
  for(int i = 0; i < elements_per_thread; i += 2){
    bf16_t result_0, result_1;
    result_0 = converterScale(clamp(input_frag[i] * r_scale, lower_bound, upper_bound));
    result_1 = converterScale(clamp(input_frag[i + 1] * r_scale, lower_bound, upper_bound));
    if(tid >= bdx - GROUP_NUM(KO)){
      input_frag_fp8[i / 2].low = converterN(result_0).storage;
      input_frag_fp8[i / 2].high = converterN(result_1).storage;
    }
    else if(tid >= bdx - GROUP_NUM(KO + KS)) {
      input_frag_fp6[i / 2].low = converterN(result_0).storage;
      input_frag_fp6[i / 2].high = converterN(result_1).storage;
    }
    else {
      input_frag_fp4[i / 2].low = converterN(result_0).storage;
      input_frag_fp4[i / 2].high = converterN(result_1).storage;
    }
  }
  // Store frag out to global memory
  if(tid >= bdx - GROUP_NUM(KO)){
    // Store fp4_t quantized result
    float4* f8out_float4 = reinterpret_cast<float4*>(f8out);
    f8out_float4[(tid + GROUP_NUM(KO) - bdx)] = input_frag_float4[0];
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)){
    // Store fp4_t quantized result
    float4* f6out_float4 = reinterpret_cast<float4*>(f6out);
    f6out_float4[(tid + GROUP_NUM(KO + KS) - bdx)] = input_frag_float4[0];
  }
  else {
    // Store fp4_t quantized result
    float4* f4out_float4 = reinterpret_cast<float4*>(f4out);
    f4out_float4[tid] = input_frag_float4[0];
  }
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_x(
  bf16_t *hidden_states,
  int seq_len,
  // int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  dim3 grids(seq_len);
  dim3 blocks(hidden_dim / 32);
  Tensor sfan_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFA(seq_len, KN)));
  Tensor sfas_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFA(seq_len, KS)));
  Tensor sfao_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFA(seq_len, KO)));
  reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfan_tensor,
    sfas_tensor,
    sfao_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w(
  bf16_t *hidden_states,
  // int seq_len,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  dim3 grids(out_features);
  dim3 blocks(hidden_dim / 32);
  Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFB(out_features, KN)));
  Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFB(out_features, KS)));
  Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFB(out_features, KO)));
  reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfbn_tensor,
    sfbs_tensor,
    sfbo_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w4(
  bf16_t *hidden_states,
  // int seq_len,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  dim3 grids(out_features);
  dim3 blocks(hidden_dim / 32);
  Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFB(out_features, KN)));
  Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFB(out_features, KS)));
  Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFB(out_features, KO)));
  reorder_quantize_mxfp4_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfbn_tensor,
    sfbs_tensor,
    sfbo_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template void run_reorder_quantize_x<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);


template void run_reorder_quantize_x<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

// template void run_reorder_quantize_x<32, 27648>(
//   bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
//   sf_t*, sf_t*, sf_t*, int, int, int
// );

// template void run_reorder_quantize_w<32, 27648>(
//   bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
//   sf_t*, sf_t*, sf_t*, int, int, int
// );
