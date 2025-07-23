import tensorrt as trt
import torch
import numpy as np
import time
import traceback

# 1. 定义 Qwen2.5-7B 解码器层的结构参数
BATCH_SIZE = 1
SEQ_LEN = 64
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
FFN_HIDDEN_SIZE = 13824
GQA_FACTOR = NUM_ATTENTION_HEADS // NUM_KV_HEADS 

# TensorRT 日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_dummy_weights(shape, dtype=np.float16):
    """创建一个包含随机数据的 trt.Weights 对象"""
    return trt.Weights(np.random.rand(*shape).astype(dtype))

def _create_rope_cache(network, dtype):
    """预计算 RoPE 的 sin 和 cos 缓存并作为常量添加到网络中"""
    print("Creating RoPE cache...")
    # 修改：将 theta_base 更新为 Qwen2 的值
    theta_base = 1000000.0
    
    theta = theta_base ** (-2.0 * np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
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

    slice_shape = list(input_tensor.shape)
    slice_shape[-1] = head_dim_half
    
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

def add_rmsnorm(network, input_tensor, weight_shape):
    """在网络中添加 RMSNorm 层"""
    epsilon = 1e-5
    dtype = np.float16

    pow2_tensor = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    
    reduce_axes = 1 << (len(input_tensor.shape) - 1)
    mean_tensor = network.add_reduce(pow2_tensor, trt.ReduceOperation.AVG, axes=reduce_axes, keep_dims=True).get_output(0)
    
    epsilon_tensor = network.add_constant(shape=(1,) * len(input_tensor.shape), weights=trt.Weights(np.array([epsilon], dtype=dtype))).get_output(0)
    add_eps_tensor = network.add_elementwise(mean_tensor, epsilon_tensor, trt.ElementWiseOperation.SUM).get_output(0)

    sqrt_tensor = network.add_unary(add_eps_tensor, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal_sqrt_tensor = network.add_unary(sqrt_tensor, trt.UnaryOperation.RECIP).get_output(0)
    normalized_tensor = network.add_elementwise(input_tensor, reciprocal_sqrt_tensor, trt.ElementWiseOperation.PROD).get_output(0)

    weight_const_1d = network.add_constant(weight_shape, create_dummy_weights(weight_shape, dtype=dtype)).get_output(0)
    
    shuffle_layer = network.add_shuffle(weight_const_1d)
    reshape_dims = [1] * len(input_tensor.shape)
    reshape_dims[-1] = weight_shape[0]
    shuffle_layer.reshape_dims = tuple(reshape_dims)
    weight_const_reshaped = shuffle_layer.get_output(0)

    output_tensor = network.add_elementwise(normalized_tensor, weight_const_reshaped, trt.ElementWiseOperation.PROD).get_output(0)
    return output_tensor


def build_decoder_layer_engine():
    """构建使用 FP8 的 Qwen2.5-7B 解码器层 TensorRT 引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    scale_val = 0.1
    scale_tensor = network.add_constant((), trt.Weights(np.array([scale_val], dtype=np.float32))).get_output(0)
    scale_tensor.name = "Global_FP8_Scale"

    # --- 修改：FP8 辅助函数现在可以处理偏置项 ---
    def add_fp8_linear_op(input_tensor, weight_tensor, op_type, bias_tensor=None, equation=None, transpose_b=False, op_name=""):
        """
        这个函数通过构建一个 Q-Dq-MatMul-BiasAdd 模式来触发 TensorRT 的 FP8 算子融合。
        我们定义一个显式的 FP16 计算路径，TensorRT 的优化器会自动将其替换为
        一个单一的、高效的 FP8 内核。
        """
        # 1. 创建 Q-Dq 子图来包裹输入张量
        input_q = network.add_quantize(input_tensor, scale_tensor, trt.DataType.FP8)
        input_q.name = f"{op_name}_input_quant"
        input_dq = network.add_dequantize(input_q.get_output(0), scale_tensor, trt.float16)
        input_dq.name = f"{op_name}_input_dequant"
        dequantized_input = input_dq.get_output(0)

        # 2. 创建 Q-Dq 子图来包裹权重张量
        weight_q = network.add_quantize(weight_tensor, scale_tensor, trt.DataType.FP8)
        weight_q.name = f"{op_name}_weight_quant"
        weight_dq = network.add_dequantize(weight_q.get_output(0), scale_tensor, trt.float16)
        weight_dq.name = f"{op_name}_weight_dequant"
        dequantized_weight = weight_dq.get_output(0)

        # 3. 在 FP16 精度下定义矩阵乘法，等待 TensorRT 进行融合
        if op_type == 'einsum':
            layer = network.add_einsum([dequantized_input, dequantized_weight], equation)
        elif op_type == 'matmul':
            op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
            layer = network.add_matrix_multiply(dequantized_input, trt.MatrixOperation.NONE, dequantized_weight, op_b)
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")
        
        layer.name = op_name
        matmul_output = layer.get_output(0)
        
        # 4. (新增) 如果提供了偏置，则添加 ElementWise SUM 操作
        if bias_tensor is not None:
            bias_add_layer = network.add_elementwise(matmul_output, bias_tensor, trt.ElementWiseOperation.SUM)
            bias_add_layer.name = f"{op_name}_bias_add"
            return bias_add_layer.get_output(0)
            
        return matmul_output

    input_tensor = network.add_input(name="input_hidden_state", dtype=trt.float16, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    
    normed_input = add_rmsnorm(network, input_tensor, (HIDDEN_SIZE,))
    
    print("Building Attention Block by defining a fusible FP8 pattern...")
    
    # 修改：为每个线性层添加权重和偏置
    # Qwen2.5-7B K/V 头的维度和Q头相同，都是 HIDDEN_SIZE
    q_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
    q_proj_b = network.add_constant((1, 1, HIDDEN_SIZE), create_dummy_weights((1, 1, HIDDEN_SIZE))).get_output(0)
    k_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    k_proj_b = network.add_constant((1, 1, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((1, 1, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    v_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    v_proj_b = network.add_constant((1, 1, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((1, 1, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    o_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
    o_proj_b = network.add_constant((1, 1, HIDDEN_SIZE), create_dummy_weights((1, 1, HIDDEN_SIZE))).get_output(0)
    
    # 修改：在线性层调用中传入偏置
    q_proj = add_fp8_linear_op(normed_input, q_proj_w, 'einsum', bias_tensor=q_proj_b, equation="bsk,kn->bsn", op_name="Q_Proj_Einsum")
    k_proj = add_fp8_linear_op(normed_input, k_proj_w, 'einsum', bias_tensor=k_proj_b, equation="bsk,kn->bsn", op_name="K_Proj_Einsum")
    v_proj = add_fp8_linear_op(normed_input, v_proj_w, 'einsum', bias_tensor=v_proj_b, equation="bsk,kn->bsn", op_name="V_Proj_Einsum")

    def reshape_and_transpose(tensor, num_heads):
        shuffle_layer = network.add_shuffle(tensor)
        shuffle_layer.reshape_dims = (BATCH_SIZE, SEQ_LEN, num_heads, HEAD_DIM)
        shuffle_layer.second_transpose = trt.Permutation([0, 2, 1, 3])
        return shuffle_layer.get_output(0)

    q_reshaped = reshape_and_transpose(q_proj, NUM_ATTENTION_HEADS)
    k_reshaped = reshape_and_transpose(k_proj, NUM_KV_HEADS)
    v_reshaped = reshape_and_transpose(v_proj, NUM_KV_HEADS)

    cos_cache, sin_cache = _create_rope_cache(network, np.float16)
    q_with_rope = add_rope(network, q_reshaped, cos_cache, sin_cache)
    k_with_rope = add_rope(network, k_reshaped, cos_cache, sin_cache)

    def repeat_kv(kv_tensor):
        # 对于 Qwen2.5-7B, GQA_FACTOR 为 1, 此函数不起作用，直接返回原张量
        if GQA_FACTOR == 1:
            return kv_tensor
        return network.add_concatenation([kv_tensor] * GQA_FACTOR).get_output(0)

    k_repeated = repeat_kv(k_with_rope)
    v_repeated = repeat_kv(v_reshaped)

    qkT = add_fp8_linear_op(q_with_rope, k_repeated, 'matmul', transpose_b=True, op_name="QKT_MatMul")

    scale_factor = 1.0 / (HEAD_DIM ** 0.5)
    scale_const = network.add_constant((1,1,1,1), trt.Weights(np.array([scale_factor], dtype=np.float16))).get_output(0)
    qkT_scaled = network.add_elementwise(qkT, scale_const, trt.ElementWiseOperation.PROD).get_output(0)

    softmax_layer = network.add_softmax(qkT_scaled)
    softmax_layer.axes = 1 << 3
    attention_probs = softmax_layer.get_output(0)
    
    attn_out_bshd = add_fp8_linear_op(attention_probs, v_repeated, 'matmul', op_name="ProbsV_MatMul")

    shuffle_out = network.add_shuffle(attn_out_bshd)
    shuffle_out.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle_out.reshape_dims = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    attn_out_bsn = shuffle_out.get_output(0)

    # 修改：为 O-Projection 添加偏置
    attention_output = add_fp8_linear_op(attn_out_bsn, o_proj_w, 'einsum', bias_tensor=o_proj_b, equation="bsk,kn->bsn", op_name="O_Proj_Einsum")

    residual1 = network.add_elementwise(input_tensor, attention_output, trt.ElementWiseOperation.SUM).get_output(0)
    normed_residual1 = add_rmsnorm(network, residual1, (HIDDEN_SIZE,))

    # 修改：为 FFN 层添加权重和偏置
    ffn_gate_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_gate_b = network.add_constant((1, 1, FFN_HIDDEN_SIZE), create_dummy_weights((1, 1, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_up_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_up_b = network.add_constant((1, 1, FFN_HIDDEN_SIZE), create_dummy_weights((1, 1, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_down_w = network.add_constant((FFN_HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((FFN_HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
    ffn_down_b = network.add_constant((1, 1, HIDDEN_SIZE), create_dummy_weights((1, 1, HIDDEN_SIZE))).get_output(0)

    # 修改：在 FFN 线性层调用中传入偏置
    gate_proj = add_fp8_linear_op(normed_residual1, ffn_gate_w, 'einsum', bias_tensor=ffn_gate_b, equation="bsk,kn->bsn", op_name="FFN_Gate_Einsum")
    up_proj = add_fp8_linear_op(normed_residual1, ffn_up_w, 'einsum', bias_tensor=ffn_up_b, equation="bsk,kn->bsn", op_name="FFN_Up_Einsum")
    
    # Qwen2 和 Llama3 都使用 SwiGLU, 这部分逻辑保持不变
    # 1. 计算 sigmoid(gate_proj)
    sigmoid_gate = network.add_activation(gate_proj, trt.ActivationType.SIGMOID).get_output(0)

    # 2. 计算 SiLU(gate_proj) = gate_proj * sigmoid(gate_proj)
    silu_out = network.add_elementwise(gate_proj, sigmoid_gate, trt.ElementWiseOperation.PROD).get_output(0)
    silu_out.name = "SiLU_Output"

    # 3. 计算最终的 FFN 门控结果
    gated_ffn = network.add_elementwise(silu_out, up_proj, trt.ElementWiseOperation.PROD).get_output(0)
    
    # 修改：在 FFN Down-Projection 中添加偏置
    ffn_output = add_fp8_linear_op(gated_ffn, ffn_down_w, 'einsum', bias_tensor=ffn_down_b, equation="bsk,kn->bsn", op_name="FFN_Down_Einsum")
    
    final_output = network.add_elementwise(residual1, ffn_output, trt.ElementWiseOperation.SUM).get_output(0)

    final_output.name = "output_hidden_state"
    network.mark_output(final_output)
    
    print("Building TensorRT engine... (This may take a few minutes)")
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
    output_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    
    input_tensor = torch.randn(input_shape, dtype=torch.float16).cuda()
    output_tensor = torch.empty(output_shape, dtype=torch.float16).cuda()
    
    context.set_tensor_address("input_hidden_state", input_tensor.data_ptr())
    context.set_tensor_address("output_hidden_state", output_tensor.data_ptr())

    print("Warming up...")
    for _ in range(96 * 2048 * 8 // BATCH_SIZE // SEQ_LEN):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    print("Warm-up finished.")

    num_runs = 400 * 2048 * 8 // BATCH_SIZE // SEQ_LEN
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
    print(f"Model: Qwen2.5-7B Decoder Layer") # 修改
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Precision: FP8 (triggered by fusible Q/Dq pattern)")
    print("Features: Full Decoder Layer with RoPE, MHA, and Biases") # 修改
    print("---")
    print(f"Average Latency: {avg_latency:.3f} ms")
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
            engine_plan = build_decoder_layer_engine()
            if engine_plan:
                benchmark(engine_plan)
        except Exception as e:
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()