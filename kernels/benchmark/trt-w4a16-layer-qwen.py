import tensorrt as trt
import torch
import numpy as np
import time
import traceback
from collections import OrderedDict

BATCH_SIZE = 32
SEQ_LEN = 64
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 40
NUM_KV_HEADS = 8  
FFN_HIDDEN_SIZE = 13824
THETA_BASE = 1000000.0
RMS_NORM_EPS = 1e-5
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
GQA_FACTOR = NUM_ATTENTION_HEADS // NUM_KV_HEADS 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# (Profiler and other functions up to build_decoder_layer_engine remain the same)
class SimpleProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer_times = OrderedDict()
        self.total_time_ms = 0
    def report_layer_time(self, layer_name, ms):
        self.layer_times[layer_name] = self.layer_times.get(layer_name, 0) + ms
    def print_report(self):
        if not self.layer_times: print("No layer timing information available."); return
        self.total_time_ms = sum(self.layer_times.values())
        print("\n--- TensorRT Layer-wise Performance Report (Qwen2.5-7B GQA) ---")
        print(f"Total GPU time for one inference: {self.total_time_ms:.6f} ms")
        print("-" * 60)
        print(f"{'Layer Name':<45} | {'Time (ms)':<12} | {'Percentage'}")
        print("-" * 60)
        for name, ms in sorted(self.layer_times.items(), key=lambda item: item[1], reverse=True):
            percentage = (ms / self.total_time_ms) * 100 if self.total_time_ms > 0 else 0
            display_name = (name[:42] + '...') if len(name) > 45 else name
            print(f"{display_name:<45} | {ms:<12.6f} | {percentage:8.2f}%")
        print("-" * 60)
        self.layer_times.clear(); self.total_time_ms = 0

def add_rmsnorm(network, input_tensor, weight_shape):
    epsilon = 1e-5; pow2_tensor = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0); reduce_axes = 1 << (len(input_tensor.shape) - 1); mean_tensor = network.add_reduce(pow2_tensor, trt.ReduceOperation.AVG, axes=reduce_axes, keep_dims=True).get_output(0); epsilon_tensor = network.add_constant(shape=(1,) * len(input_tensor.shape), weights=trt.Weights(np.array([epsilon], dtype=np.float16))).get_output(0); add_eps_tensor = network.add_elementwise(mean_tensor, epsilon_tensor, trt.ElementWiseOperation.SUM).get_output(0); sqrt_tensor = network.add_unary(add_eps_tensor, trt.UnaryOperation.SQRT).get_output(0); reciprocal_sqrt_tensor = network.add_unary(sqrt_tensor, trt.UnaryOperation.RECIP).get_output(0); normalized_tensor = network.add_elementwise(input_tensor, reciprocal_sqrt_tensor, trt.ElementWiseOperation.PROD).get_output(0); weight_const_1d = network.add_constant(weight_shape, create_dummy_weights(weight_shape)).get_output(0); shuffle_layer = network.add_shuffle(weight_const_1d); reshape_dims = [1] * len(input_tensor.shape); reshape_dims[-1] = weight_shape[0]; shuffle_layer.reshape_dims = tuple(reshape_dims); weight_const_reshaped = shuffle_layer.get_output(0); output_tensor = network.add_elementwise(normalized_tensor, weight_const_reshaped, trt.ElementWiseOperation.PROD).get_output(0); return output_tensor

def quantize_and_pack_w4(fp32_weights: torch.Tensor):
    weights = fp32_weights.cpu()
    K, N = weights.shape
    if N % 2 != 0: raise ValueError("N must be even for INT4 packing")
    scales = weights.abs().max(dim=0)[0] / 7.0; scales = torch.clamp(scales, min=1e-5)
    quantized_int8 = torch.round(weights / scales).to(torch.int8)
    quantized_int8 = torch.clamp(quantized_int8, min=-8, max=7)
    low_nibbles = quantized_int8[:, 0::2] & 0x0F
    high_nibbles = quantized_int8[:, 1::2] & 0x0F
    packed_uint8_weights = (high_nibbles << 4) | low_nibbles
    return packed_uint8_weights.to(torch.uint8), scales.to(torch.float16)

def add_w4a16_linear_weight(network, shape, name_prefix, weight_holder: list):
    K, N = shape
    fp32_weights = torch.randn(K, N, dtype=torch.float32)
    packed_weights_uint8, scales_fp16 = quantize_and_pack_w4(fp32_weights)
    weights_np = np.ascontiguousarray(packed_weights_uint8.numpy())
    scales_np = np.ascontiguousarray(scales_fp16.numpy())
    weight_holder.append(weights_np); weight_holder.append(scales_np)
    trt_int4_weights = trt.Weights(trt.DataType.INT4, weights_np.ctypes.data, weights_np.size * 2)
    quant_w_const = network.add_constant(trt.Dims([K, N]), trt_int4_weights); quant_w_const.name = f"{name_prefix}_int4_weights"
    scales_const = network.add_constant(scales_np.shape, trt.Weights(scales_np)); scales_const.name = f"{name_prefix}_fp16_scales"
    dequantize_layer = network.add_dequantize(quant_w_const.get_output(0), scales_const.get_output(0), trt.DataType.HALF)
    dequantize_layer.axis = 1; dequantize_layer.name = f"{name_prefix}_dequantize"
    return dequantize_layer.get_output(0)

def create_dummy_weights(shape, dtype=np.float16): return trt.Weights(np.random.rand(*shape).astype(dtype))

def add_bias(network, input_tensor, bias_shape, name_prefix):
    bias_weights = create_dummy_weights(bias_shape, dtype=np.float16)
    bias_const = network.add_constant(bias_shape, bias_weights); bias_const.name = f"{name_prefix}_bias"
    shuffle_layer = network.add_shuffle(bias_const.get_output(0))
    reshape_dims = [1] * len(input_tensor.shape); reshape_dims[-1] = bias_shape[0]
    shuffle_layer.reshape_dims = tuple(reshape_dims)
    bias_reshaped = shuffle_layer.get_output(0)
    add_bias_layer = network.add_elementwise(input_tensor, bias_reshaped, trt.ElementWiseOperation.SUM)
    add_bias_layer.name = f"{name_prefix}_add_bias"
    return add_bias_layer.get_output(0)

def _create_rope_cache(network, dtype):
    theta = THETA_BASE ** (-2.0 * np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
    position = np.arange(SEQ_LEN, dtype=np.float32)
    freqs = np.outer(position, theta); emb = np.concatenate((freqs, freqs), axis=-1)
    cos_cache = np.cos(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    sin_cache = np.sin(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    cos_cache_reshaped = np.transpose(cos_cache, (0, 2, 1, 3))
    sin_cache_reshaped = np.transpose(sin_cache, (0, 2, 1, 3))
    cos_const = network.add_constant(cos_cache_reshaped.shape, trt.Weights(cos_cache_reshaped)).get_output(0)
    sin_const = network.add_constant(sin_cache_reshaped.shape, trt.Weights(sin_cache_reshaped)).get_output(0)
    return cos_const, sin_const

def add_rope(network, input_tensor, cos_cache, sin_cache, num_heads):
    head_dim_half = HEAD_DIM // 2; slice_shape = list(input_tensor.shape); slice_shape[3] = head_dim_half
    slice_layer1 = network.add_slice(input_tensor, start=[0,0,0,0], shape=slice_shape, stride=[1,1,1,1])
    slice_layer2 = network.add_slice(input_tensor, start=[0,0,0,head_dim_half], shape=slice_shape, stride=[1,1,1,1])
    x1, x2 = slice_layer1.get_output(0), slice_layer2.get_output(0)
    neg_x2 = network.add_unary(x2, trt.UnaryOperation.NEG).get_output(0)
    concat_layer = network.add_concatenation([neg_x2, x1]); concat_layer.axis = 3
    rotated_input = concat_layer.get_output(0)
    cos_term = network.add_elementwise(input_tensor, cos_cache, trt.ElementWiseOperation.PROD).get_output(0)
    sin_term = network.add_elementwise(rotated_input, sin_cache, trt.ElementWiseOperation.PROD).get_output(0)
    output_tensor = network.add_elementwise(cos_term, sin_term, trt.ElementWiseOperation.SUM).get_output(0)
    return output_tensor

def build_decoder_layer_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    if hasattr(trt.BuilderFlag, 'WEIGHT_ONLY_QUANT'):
        config.set_flag(trt.BuilderFlag.WEIGHT_ONLY_QUANT)
    
    weight_holder = []

    # --- 网络构建 ---
    input_tensor = network.add_input(name="input_hidden_state", dtype=trt.float16, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    normed_input = add_rmsnorm(network, input_tensor, (HIDDEN_SIZE,))
    
    q_proj_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, HIDDEN_SIZE), "q_proj", weight_holder)
    k_proj_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), "k_proj", weight_holder)
    v_proj_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), "v_proj", weight_holder)
    o_proj_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, HIDDEN_SIZE), "o_proj", weight_holder)

    q_proj_linear = network.add_einsum([normed_input, q_proj_w], "bsk,kn->bsn").get_output(0)
    q_proj = add_bias(network, q_proj_linear, (HIDDEN_SIZE,), "q_proj")
    k_proj_linear = network.add_einsum([normed_input, k_proj_w], "bsk,kn->bsn").get_output(0)
    k_proj = add_bias(network, k_proj_linear, (NUM_KV_HEADS * HEAD_DIM,), "k_proj")
    v_proj_linear = network.add_einsum([normed_input, v_proj_w], "bsk,kn->bsn").get_output(0)
    v_proj = add_bias(network, v_proj_linear, (NUM_KV_HEADS * HEAD_DIM,), "v_proj")

    def reshape_and_transpose(tensor, num_heads):
        shuffle_layer = network.add_shuffle(tensor)
        shuffle_layer.reshape_dims = (BATCH_SIZE, SEQ_LEN, num_heads, HEAD_DIM)
        shuffle_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
        return shuffle_layer.get_output(0)
        
    q_reshaped = reshape_and_transpose(q_proj, NUM_ATTENTION_HEADS)
    k_reshaped = reshape_and_transpose(k_proj, NUM_KV_HEADS)
    v_reshaped = reshape_and_transpose(v_proj, NUM_KV_HEADS)
    
    cos_cache, sin_cache = _create_rope_cache(network, np.float16)
    q_with_rope = add_rope(network, q_reshaped, cos_cache, sin_cache, NUM_ATTENTION_HEADS)
    k_with_rope = add_rope(network, k_reshaped, cos_cache, sin_cache, NUM_KV_HEADS)

    def repeat_kv(kv_tensor):
        if GQA_FACTOR == 1: 
            return kv_tensor
        concat_layer = network.add_concatenation([kv_tensor] * GQA_FACTOR)
        concat_layer.axis = 1 # Concatenate along the head dimension
        return concat_layer.get_output(0)

    # 使用 repeat_kv 函数来处理 k 和 v
    k_repeated = repeat_kv(k_with_rope)
    v_repeated = repeat_kv(v_reshaped) # V 不应用RoPE, 但需要重复

    qkT = network.add_matrix_multiply(q_with_rope, trt.MatrixOperation.NONE, k_repeated, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale_factor = 1.0 / (HEAD_DIM ** 0.5)
    scale_const = network.add_constant((1,1,1,1), trt.Weights(np.array([scale_factor], dtype=np.float16))).get_output(0)
    qkT_scaled = network.add_elementwise(qkT, scale_const, trt.ElementWiseOperation.PROD).get_output(0)
    attention_probs_layer = network.add_softmax(qkT_scaled); attention_probs_layer.axes = 1 << 3
    attention_probs = attention_probs_layer.get_output(0)
    attn_out_bshd = network.add_matrix_multiply(attention_probs, trt.MatrixOperation.NONE, v_repeated, trt.MatrixOperation.NONE).get_output(0)
    
    shuffle_out = network.add_shuffle(attn_out_bshd)
    shuffle_out.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle_out.reshape_dims = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    attn_out_bsn = shuffle_out.get_output(0)
    attention_output = network.add_einsum([attn_out_bsn, o_proj_w], "bsk,kn->bsn").get_output(0)

    residual1 = network.add_elementwise(input_tensor, attention_output, trt.ElementWiseOperation.SUM).get_output(0)
    normed_residual1 = add_rmsnorm(network, residual1, (HIDDEN_SIZE,))
    
    ffn_gate_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, FFN_HIDDEN_SIZE), "ffn_gate", weight_holder)
    ffn_up_w = add_w4a16_linear_weight(network, (HIDDEN_SIZE, FFN_HIDDEN_SIZE), "ffn_up", weight_holder)
    ffn_down_w = add_w4a16_linear_weight(network, (FFN_HIDDEN_SIZE, HIDDEN_SIZE), "ffn_down", weight_holder)

    gate_proj = network.add_einsum([normed_residual1, ffn_gate_w], "bsk,kn->bsn").get_output(0)
    up_proj = network.add_einsum([normed_residual1, ffn_up_w], "bsk,kn->bsn").get_output(0)
    
    sigmoid_gate = network.add_activation(gate_proj, trt.ActivationType.SIGMOID).get_output(0)
    silu_out = network.add_elementwise(gate_proj, sigmoid_gate, trt.ElementWiseOperation.PROD).get_output(0)
    gated_ffn = network.add_elementwise(silu_out, up_proj, trt.ElementWiseOperation.PROD).get_output(0)
    
    ffn_output = network.add_einsum([gated_ffn, ffn_down_w], "bsk,kn->bsn").get_output(0)
    final_output = network.add_elementwise(residual1, ffn_output, trt.ElementWiseOperation.SUM).get_output(0)

    final_output.name = "output_hidden_state"
    network.mark_output(final_output)
    
    print(f"Building TensorRT engine for Qwen2.5 Layer (BS={BATCH_SIZE}, GQA, W4A16)...")
    start_time = time.time()
    plan = builder.build_serialized_network(network, config)
    end_time = time.time()
    
    if not plan: raise RuntimeError("Engine build failed.")
    
    print(f"Engine build successful! Time taken: {end_time - start_time:.2f} seconds.")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    return engine

def benchmark_and_profile(engine):
    profiler = SimpleProfiler()
    context = engine.create_execution_context()
    context.profiler = profiler
    input_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    output_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    input_tensor = torch.randn(input_shape, dtype=torch.float16).cuda()
    output_tensor = torch.empty(output_shape, dtype=torch.float16).cuda()
    context.set_tensor_address("input_hidden_state", input_tensor.data_ptr())
    context.set_tensor_address("output_hidden_state", output_tensor.data_ptr())
    print("Warming up...")
    warmup_runs = 96 * 2048 * 8 // BATCH_SIZE // SEQ_LEN
    for _ in range(warmup_runs): context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    num_runs = 400 * 2048 * 8 // BATCH_SIZE // SEQ_LEN
    start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
    print(f"Running benchmark for {num_runs} iterations...")
    start_event.record()
    for _ in range(num_runs): context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    end_event.record(); torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event); avg_latency = total_time_ms / num_runs
    throughput = (BATCH_SIZE * SEQ_LEN) / (avg_latency / 1000)
    print("\n--- Overall Benchmark Results (Qwen2.5 W4A16) ---")
    print(f"Parameters: BATCH={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}, GQA_FACTOR={GQA_FACTOR}")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Tokens per Second (Throughput): {throughput:.2f} tokens/sec")
    print("\nRunning a single inference to measure peak active memory...")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    memory_before_b = torch.cuda.memory_allocated()
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
    peak_memory_b = torch.cuda.max_memory_allocated()
    print("\n--- Peak Memory Usage Report ---")
    print(f"Memory allocated before inference (Engine): {memory_before_b / 1024**2:.2f} MB")
    print(f"Peak memory during inference (Total):    {peak_memory_b / 1024**2:.2f} MB")
    print(f"Inference overhead (Activations):      {(peak_memory_b - memory_before_b) / 1024**2:.2f} MB")
    print("-" * 40)
    print("\nRunning one final inference to capture layer-wise timings...")
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
    profiler.print_report()

if __name__ == "__main__":
    if not hasattr(trt.DataType, 'INT4'):
        print("Error: This script requires a TensorRT version that supports trt.DataType.INT4 (e.g., TRT 8.6+).")
    else:
        print(f"TensorRT version: {trt.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        try:
            engine = build_decoder_layer_engine()
            if engine:
                benchmark_and_profile(engine)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            traceback.print_exc()