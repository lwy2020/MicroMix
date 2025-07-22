import tensorrt as trt
import torch
import numpy as np
import time
import traceback

# 1. 定义 Llama 3-8B 结构参数
BATCH_SIZE = 32
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
FFN_HIDDEN_SIZE = 11008
GQA_FACTOR = NUM_ATTENTION_HEADS // NUM_KV_HEADS
NUM_DECODER_LAYERS = 32

# TensorRT 日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_dummy_weights(shape, dtype=np.float16):
    """创建一个包含随机数据的 trt.Weights 对象"""
    return trt.Weights(np.random.rand(*shape).astype(dtype))

def _create_rope_cache(network, dtype):
    """预计算 RoPE 的 sin 和 cos 缓存并作为常量添加到网络中"""
    print("Creating RoPE cache for all layers...")
    theta_base = 500000.0 # Llama 3 使用了更高的 theta base
    
    theta = 1.0 / (theta_base ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    position = np.arange(SEQ_LEN, dtype=np.float32)
    freqs = np.outer(position, theta)
    
    emb = np.concatenate((freqs, freqs), axis=-1)
    
    cos_cache = np.cos(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    sin_cache = np.sin(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    
    cos_cache_reshaped = np.transpose(cos_cache, (0, 2, 1, 3))
    sin_cache_reshaped = np.transpose(sin_cache, (0, 2, 1, 3))

    cos_const = network.add_constant(cos_cache_reshaped.shape, trt.Weights(cos_cache_reshaped)).get_output(0)
    sin_const = network.add_constant(sin_cache_reshaped.shape, trt.Weights(sin_cache_reshaped)).get_output(0)
    
    return cos_const, sin_const

def add_rope(network, input_tensor, cos_cache, sin_cache):
    """在网络中添加 RoPE 层"""
    head_dim_half = HEAD_DIM // 2

    b, h, s, d = input_tensor.shape
    slice_shape = (b, h, s, head_dim_half)
    
    slice_layer1 = network.add_slice(input_tensor, start=[0,0,0,0], shape=slice_shape, stride=[1,1,1,1])
    slice_layer2 = network.add_slice(input_tensor, start=[0,0,0,head_dim_half], shape=slice_shape, stride=[1,1,1,1])
    x1 = slice_layer1.get_output(0)
    x2 = slice_layer2.get_output(0)

    neg_x2 = network.add_unary(x2, trt.UnaryOperation.NEG).get_output(0)

    concat_layer = network.add_concatenation([neg_x2, x1])
    concat_layer.axis = 3
    rotated_input = concat_layer.get_output(0)

    cos_term = network.add_elementwise(input_tensor, cos_cache, trt.ElementWiseOperation.PROD).get_output(0)
    sin_term = network.add_elementwise(rotated_input, sin_cache, trt.ElementWiseOperation.PROD).get_output(0)
    
    output_tensor = network.add_elementwise(cos_term, sin_term, trt.ElementWiseOperation.SUM).get_output(0)
    return output_tensor

def add_rmsnorm(network, input_tensor, weight_shape, op_name=""):
    """在网络中添加 RMSNorm 层"""
    epsilon = 1e-5
    dtype = np.float16

    pow2_tensor = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    
    reduce_axes = 1 << (len(input_tensor.shape) - 1)
    mean_tensor = network.add_reduce(pow2_tensor, trt.ReduceOperation.AVG, axes=reduce_axes, keep_dims=True).get_output(0)
    
    epsilon_tensor = network.add_constant(shape=(1,) * len(mean_tensor.shape), weights=trt.Weights(np.array([epsilon], dtype=dtype))).get_output(0)
    add_eps_tensor = network.add_elementwise(mean_tensor, epsilon_tensor, trt.ElementWiseOperation.SUM).get_output(0)

    rsqrt_tensor = network.add_unary(add_eps_tensor, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal_sqrt_tensor = network.add_unary(rsqrt_tensor, trt.UnaryOperation.RECIP).get_output(0)
    
    normalized_tensor = network.add_elementwise(input_tensor, reciprocal_sqrt_tensor, trt.ElementWiseOperation.PROD).get_output(0)

    weight_const_1d = network.add_constant(weight_shape, create_dummy_weights(weight_shape, dtype=dtype)).get_output(0)
    weight_const_1d.name = f"{op_name}_weight"
    
    shuffle_layer = network.add_shuffle(weight_const_1d)
    reshape_dims = [1] * (len(input_tensor.shape) - 1) + [weight_shape[0]]
    shuffle_layer.reshape_dims = tuple(reshape_dims)
    weight_const_reshaped = shuffle_layer.get_output(0)

    output_tensor = network.add_elementwise(normalized_tensor, weight_const_reshaped, trt.ElementWiseOperation.PROD).get_output(0)
    output_tensor.name = op_name
    return output_tensor


def build_llama_prefill_engine():
    """构建一个使用 FP8 的 Llama 3 完整 32 层 prefill 阶段的 TensorRT 引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1024 ** 3)) # 4 GiB

    scale_val = 0.1 
    scale_tensor = network.add_constant((), trt.Weights(np.array([scale_val], dtype=np.float32))).get_output(0)
    scale_tensor.name = "Global_FP8_Scale"

    def add_fp8_linear_op(input_tensor, weight_tensor, op_type, equation=None, transpose_b=False, op_name=""):
        """通过构建一个 Q-Dq-MatMul 模式来触发 TensorRT 的 FP8 算子融合"""
        input_q = network.add_quantize(input_tensor, scale_tensor, trt.DataType.FP8)
        input_q.name = f"{op_name}_input_quant"
        input_dq = network.add_dequantize(input_q.get_output(0), scale_tensor, trt.float16)
        input_dq.name = f"{op_name}_input_dequant"
        dequantized_input = input_dq.get_output(0)

        weight_q = network.add_quantize(weight_tensor, scale_tensor, trt.DataType.FP8)
        weight_q.name = f"{op_name}_weight_quant"
        weight_dq = network.add_dequantize(weight_q.get_output(0), scale_tensor, trt.float16)
        weight_dq.name = f"{op_name}_weight_dequant"
        dequantized_weight = weight_dq.get_output(0)

        if op_type == 'einsum':
            layer = network.add_einsum([dequantized_input, dequantized_weight], equation)
        elif op_type == 'matmul':
            op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
            layer = network.add_matrix_multiply(dequantized_input, trt.MatrixOperation.NONE, dequantized_weight, op_b)
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")
        
        layer.name = op_name
        return layer.get_output(0)

    input_tensor = network.add_input(name="input_hidden_state", dtype=trt.float16, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    
    cos_cache, sin_cache = _create_rope_cache(network, np.float16)
    
    current_hidden_state = input_tensor
    
    for i in range(NUM_DECODER_LAYERS):
        print(f"--- Building Decoder Layer {i+1}/{NUM_DECODER_LAYERS} ---")
        layer_input = current_hidden_state

        normed_input = add_rmsnorm(network, layer_input, (HIDDEN_SIZE,), op_name=f"L{i}_Attn_RMSNorm")
        
        q_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
        k_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
        v_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
        o_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)

        q_proj = add_fp8_linear_op(normed_input, q_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_Q_Proj")
        k_proj = add_fp8_linear_op(normed_input, k_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_K_Proj")
        v_proj = add_fp8_linear_op(normed_input, v_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_V_Proj")

        def reshape_and_transpose(tensor, num_heads, op_name=""):
            shuffle_layer = network.add_shuffle(tensor)
            shuffle_layer.reshape_dims = (BATCH_SIZE, SEQ_LEN, num_heads, HEAD_DIM)
            shuffle_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
            shuffle_layer.name = op_name
            return shuffle_layer.get_output(0)

        q_reshaped = reshape_and_transpose(q_proj, NUM_ATTENTION_HEADS, op_name=f"L{i}_Q_Reshape")
        k_reshaped = reshape_and_transpose(k_proj, NUM_KV_HEADS, op_name=f"L{i}_K_Reshape")
        v_reshaped = reshape_and_transpose(v_proj, NUM_KV_HEADS, op_name=f"L{i}_V_Reshape")

        q_with_rope = add_rope(network, q_reshaped, cos_cache, sin_cache)
        k_with_rope = add_rope(network, k_reshaped, cos_cache, sin_cache)

        # 将当前层的 K 和 V 缓存标记为网络输出
        # 这就是 KV Cache 的初始化内容
        k_with_rope.name = f"L{i}_present_k_cache"
        v_reshaped.name = f"L{i}_present_v_cache" # 注意：V 通常在 RoPE 之前被缓存
        network.mark_output(k_with_rope)
        network.mark_output(v_reshaped)
        
        if GQA_FACTOR > 1:
            # --- FIX: 使用 add_concatenation 替换 add_tile ---
            # 重复 K 头
            concat_k = network.add_concatenation([k_with_rope] * GQA_FACTOR)
            concat_k.axis = 1 # 沿头部维度(B,H,S,D)拼接
            concat_k.name = f"L{i}_K_Repeat"
            k_repeated = concat_k.get_output(0)
            
            # 重复 V 头
            concat_v = network.add_concatenation([v_reshaped] * GQA_FACTOR)
            concat_v.axis = 1 # 沿头部维度(B,H,S,D)拼接
            concat_v.name = f"L{i}_V_Repeat"
            v_repeated = concat_v.get_output(0)
        else:
            k_repeated = k_with_rope
            v_repeated = v_reshaped

        qkT = add_fp8_linear_op(q_with_rope, k_repeated, 'matmul', transpose_b=True, op_name=f"L{i}_QKT_MatMul")

        scale_factor = 1.0 / (HEAD_DIM ** 0.5)
        scale_const = network.add_constant((1,1,1,1), trt.Weights(np.array([scale_factor], dtype=np.float16))).get_output(0)
        qkT_scaled = network.add_elementwise(qkT, scale_const, trt.ElementWiseOperation.PROD).get_output(0)

        softmax_layer = network.add_softmax(qkT_scaled)
        softmax_layer.axes = 1 << 3
        attention_probs = softmax_layer.get_output(0)
        
        attn_out_bshd = add_fp8_linear_op(attention_probs, v_repeated, 'matmul', op_name=f"L{i}_ProbsV_MatMul")

        shuffle_out = network.add_shuffle(attn_out_bshd)
        shuffle_out.first_transpose = trt.Permutation([0, 2, 1, 3])
        shuffle_out.reshape_dims = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        shuffle_out.name = f"L{i}_Attn_Out_Reshape"
        attn_out_bsn = shuffle_out.get_output(0)

        attention_output = add_fp8_linear_op(attn_out_bsn, o_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_O_Proj")
        attention_output.name = f"L{i}_attention_output"

        residual1 = network.add_elementwise(layer_input, attention_output, trt.ElementWiseOperation.SUM).get_output(0)
        residual1.name = f"L{i}_residual1"
        
        normed_residual1 = add_rmsnorm(network, residual1, (HIDDEN_SIZE,), op_name=f"L{i}_FFN_RMSNorm")

        ffn_gate_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
        ffn_up_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
        ffn_down_w = network.add_constant((FFN_HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((FFN_HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)

        gate_proj = add_fp8_linear_op(normed_residual1, ffn_gate_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_FFN_Gate")
        up_proj = add_fp8_linear_op(normed_residual1, ffn_up_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_FFN_Up")
        
        silu_gate = network.add_activation(gate_proj, trt.ActivationType.SIGMOID).get_output(0)
        activated_gate = network.add_elementwise(gate_proj, silu_gate, trt.ElementWiseOperation.PROD).get_output(0)
        
        gated_ffn = network.add_elementwise(activated_gate, up_proj, trt.ElementWiseOperation.PROD).get_output(0)
        
        ffn_output = add_fp8_linear_op(gated_ffn, ffn_down_w, 'einsum', "bsk,kn->bsn", op_name=f"L{i}_FFN_Down")
        ffn_output.name = f"L{i}_ffn_output"

        final_layer_output = network.add_elementwise(residual1, ffn_output, trt.ElementWiseOperation.SUM).get_output(0)
        final_layer_output.name = f"L{i}_final_output"
        
        current_hidden_state = final_layer_output
    
    final_norm_output = add_rmsnorm(network, current_hidden_state, (HIDDEN_SIZE,), op_name="Final_RMSNorm")
    
    final_norm_output.name = "output_hidden_state"
    network.mark_output(final_norm_output)
    
    print("\nBuilding 32-Layer Llama TensorRT engine... (This may take several minutes)")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("ERROR: Engine build failed.")
        return None
    
    print("Engine build successful!")
    return plan

def benchmark(engine_plan):
    """使用构建的引擎进行性能测试"""
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    context = engine.create_execution_context()

    input_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    input_tensor = torch.randn(input_shape, dtype=torch.float16).cuda()
    context.set_tensor_address("input_hidden_state", input_tensor.data_ptr())

    output_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    output_tensor = torch.empty(output_shape, dtype=torch.float16).cuda()
    context.set_tensor_address("output_hidden_state", output_tensor.data_ptr())


    k_cache_shape = (BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
    v_cache_shape = (BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
    
    present_k_caches = []
    present_v_caches = []

    for i in range(NUM_DECODER_LAYERS):
        # 创建 K cache 张量并绑定
        k_cache_tensor = torch.empty(k_cache_shape, dtype=torch.float16).cuda()
        present_k_caches.append(k_cache_tensor)
        context.set_tensor_address(f"L{i}_present_k_cache", k_cache_tensor.data_ptr())

        # 创建 V cache 张量并绑定
        v_cache_tensor = torch.empty(v_cache_shape, dtype=torch.float16).cuda()
        present_v_caches.append(v_cache_tensor)
        context.set_tensor_address(f"L{i}_present_v_cache", v_cache_tensor.data_ptr())


    print("Warming up...")
    for _ in range(5):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    print("Warm-up finished.")

    num_runs = 40
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Running benchmark for {num_runs} iterations...")
    start_event.record()
    for _ in range(num_runs):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency = total_time_ms / num_runs
    throughput = (BATCH_SIZE * SEQ_LEN) / (avg_latency / 1000) if avg_latency > 0 else 0

    print("\n--- Benchmark Results ---")
    print(f"Model: Llama 3 8B-like ({NUM_DECODER_LAYERS} Layers)")
    print(f"Phase: Prefill")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Precision: FP8 (triggered by fusible Q/Dq pattern)")
    print("Features: Full Model Graph with RoPE, GQA, SwiGLU")
    print("---")
    print(f"Average Latency for full prefill: {avg_latency:.3f} ms")
    print(f"Tokens per Second (Throughput): {throughput:.2f} tokens/sec")
    print("--------------------------")


if __name__ == "__main__":
    print(f"TensorRT version: {trt.__version__}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
    elif torch.cuda.get_device_properties(0).major < 9:
        print("ERROR: FP8 is only supported on GPUs with compute capability 9.0 (Hopper architecture) or newer.")
        print(f"Your GPU's compute capability is {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}. Aborting.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} (Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor})")
        engine_plan = None
        try:
            engine_plan = build_llama_prefill_engine()
            if engine_plan:
                benchmark(engine_plan)
        except Exception as e:
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()