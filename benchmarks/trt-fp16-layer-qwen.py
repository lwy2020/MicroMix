import tensorrt as trt
import torch
import numpy as np
import time

# 1. 定义 Qwen2.5-7B 解码器层的结构参数
BATCH_SIZE = 1
SEQ_LEN = 4096
# --- [修改] Qwen2.5-7B 结构参数 ---
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 40
NUM_KV_HEADS = 8
FFN_HIDDEN_SIZE = 13824
# --- [修改结束] ---
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
GQA_FACTOR = NUM_ATTENTION_HEADS // NUM_KV_HEADS

# TensorRT 日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_dummy_weights(shape, dtype=np.float16):
    """创建一个包含随机数据的 trt.Weights 对象 (用于无需关心数值的权重)"""
    return trt.Weights(np.random.rand(*shape).astype(dtype))

def _create_rope_cache(network, dtype):
    """预计算 RoPE 的 sin 和 cos 缓存并作为常量添加到网络中"""
    print("Creating RoPE cache...")
    # Qwen2.5 同样使用 10000.0 作为基础 theta
    theta_base = 10000.0
    
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

def add_rope(network, input_tensor, cos_cache, sin_cache, num_heads):
    """在网络中添加 RoPE 层"""
    head_dim_half = HEAD_DIM // 2

    slice_shape = [BATCH_SIZE, num_heads, SEQ_LEN, head_dim_half]
    
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

    pow2_tensor = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    
    reduce_axes = 1 << (len(input_tensor.shape) - 1)
    mean_tensor = network.add_reduce(pow2_tensor, trt.ReduceOperation.AVG, axes=reduce_axes, keep_dims=True).get_output(0)
    
    epsilon_tensor = network.add_constant(shape=(1,) * len(input_tensor.shape), weights=trt.Weights(np.array([epsilon], dtype=np.float16))).get_output(0)
    add_eps_tensor = network.add_elementwise(mean_tensor, epsilon_tensor, trt.ElementWiseOperation.SUM).get_output(0)

    sqrt_tensor = network.add_unary(add_eps_tensor, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal_sqrt_tensor = network.add_unary(sqrt_tensor, trt.UnaryOperation.RECIP).get_output(0)
    normalized_tensor = network.add_elementwise(input_tensor, reciprocal_sqrt_tensor, trt.ElementWiseOperation.PROD).get_output(0)

    weight_const_1d = network.add_constant(weight_shape, create_dummy_weights(weight_shape)).get_output(0)
    
    shuffle_layer = network.add_shuffle(weight_const_1d)
    reshape_dims = [1] * len(input_tensor.shape)
    reshape_dims[-1] = weight_shape[0]
    shuffle_layer.reshape_dims = tuple(reshape_dims)
    weight_const_reshaped = shuffle_layer.get_output(0)

    output_tensor = network.add_elementwise(normalized_tensor, weight_const_reshaped, trt.ElementWiseOperation.PROD).get_output(0)
    return output_tensor


def build_decoder_layer_engine():
    """构建使用 FP16 的 Qwen2.5-7B 解码器层 TensorRT 引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    input_tensor = network.add_input(name="input_hidden_state", dtype=trt.float16, shape=(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    
    normed_input = add_rmsnorm(network, input_tensor, (HIDDEN_SIZE,))
    
    print("Building Attention Block with RoPE...")
    
    # --- [修改] Qwen2.5 的权重和偏置 ---
    q_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
    k_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    v_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    o_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)

    # Qwen2.5 在 QKV 和 O 投射中加入了偏置
    q_proj_b = network.add_constant((1, 1, HIDDEN_SIZE), create_dummy_weights((1, 1, HIDDEN_SIZE))).get_output(0)
    k_proj_b = network.add_constant((1, 1, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((1, 1, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    v_proj_b = network.add_constant((1, 1, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((1, 1, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
    o_proj_b = network.add_constant((1, 1, HIDDEN_SIZE), create_dummy_weights((1, 1, HIDDEN_SIZE))).get_output(0)
    # --- [修改结束] ---

    q_einsum = network.add_einsum([normed_input, q_proj_w], "bsk,kn->bsn"); q_proj = q_einsum.get_output(0)
    k_einsum = network.add_einsum([normed_input, k_proj_w], "bsk,kn->bsn"); k_proj = k_einsum.get_output(0)
    v_einsum = network.add_einsum([normed_input, v_proj_w], "bsk,kn->bsn"); v_proj = v_einsum.get_output(0)

    # --- [新增] 添加偏置 ---
    q_proj = network.add_elementwise(q_proj, q_proj_b, trt.ElementWiseOperation.SUM).get_output(0)
    k_proj = network.add_elementwise(k_proj, k_proj_b, trt.ElementWiseOperation.SUM).get_output(0)
    v_proj = network.add_elementwise(v_proj, v_proj_b, trt.ElementWiseOperation.SUM).get_output(0)
    # --- [新增结束] ---

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
        tensors_to_concat = [kv_tensor] * GQA_FACTOR
        concat_layer = network.add_concatenation(tensors_to_concat)
        concat_layer.axis = 1
        return concat_layer.get_output(0)

    k_repeated = repeat_kv(k_with_rope)
    v_repeated = repeat_kv(v_reshaped)

    qkT_layer = network.add_matrix_multiply(q_with_rope, trt.MatrixOperation.NONE, k_repeated, trt.MatrixOperation.TRANSPOSE)
    qkT = qkT_layer.get_output(0)

    scale_factor = 1.0 / (HEAD_DIM ** 0.5)
    scale_const = network.add_constant((1,1,1,1), trt.Weights(np.array([scale_factor], dtype=np.float16))).get_output(0)
    qkT_scaled = network.add_elementwise(qkT, scale_const, trt.ElementWiseOperation.PROD).get_output(0)

    softmax_layer = network.add_softmax(qkT_scaled)
    softmax_layer.axes = 1 << 3
    attention_probs = softmax_layer.get_output(0)
    
    attn_out_layer = network.add_matrix_multiply(attention_probs, trt.MatrixOperation.NONE, v_repeated, trt.MatrixOperation.NONE)
    attn_out_bshd = attn_out_layer.get_output(0)

    shuffle_out = network.add_shuffle(attn_out_bshd)
    shuffle_out.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle_out.reshape_dims = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    attn_out_bsn = shuffle_out.get_output(0)

    o_einsum = network.add_einsum([attn_out_bsn, o_proj_w], "bsk,kn->bsn")
    attention_output = o_einsum.get_output(0)

    # --- [新增] 添加输出偏置 ---
    attention_output = network.add_elementwise(attention_output, o_proj_b, trt.ElementWiseOperation.SUM).get_output(0)
    # --- [新增结束] ---

    residual1 = network.add_elementwise(input_tensor, attention_output, trt.ElementWiseOperation.SUM).get_output(0)

    normed_residual1 = add_rmsnorm(network, residual1, (HIDDEN_SIZE,))

    # --- [修改] Qwen2.5 的 FFN 权重 ---
    ffn_gate_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_up_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
    ffn_down_w = network.add_constant((FFN_HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((FFN_HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
    # --- [修改结束] ---

    gate_einsum = network.add_einsum([normed_residual1, ffn_gate_w], "bsk,kn->bsn"); gate_proj = gate_einsum.get_output(0)
    up_einsum = network.add_einsum([normed_residual1, ffn_up_w], "bsk,kn->bsn"); up_proj = up_einsum.get_output(0)
    
    # Qwen2.5 使用 SwiGLU, 其实现和 Llama 类似
    # 1. 计算 sigmoid(gate_proj)
    sigmoid_gate = network.add_activation(gate_proj, trt.ActivationType.SIGMOID).get_output(0)

    # 2. 计算 SiLU(gate_proj) = gate_proj * sigmoid(gate_proj)
    silu_out = network.add_elementwise(gate_proj, sigmoid_gate, trt.ElementWiseOperation.PROD).get_output(0)
    silu_out.name = "SiLU_Output"
    
    gated_ffn = network.add_elementwise(silu_out, up_proj, trt.ElementWiseOperation.PROD).get_output(0)
    
    down_einsum = network.add_einsum([gated_ffn, ffn_down_w], "bsk,kn->bsn"); ffn_output = down_einsum.get_output(0)
    
    final_output = network.add_elementwise(residual1, ffn_output, trt.ElementWiseOperation.SUM).get_output(0)

    final_output.name = "output_hidden_state"
    network.mark_output(final_output)
    
    print("Building TensorRT engine... (This may take a few minutes)")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("ERROR: Engine build failed.")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            print(f"Layer {i}: {layer.name}, Type: {layer.type}")
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                if tensor:
                    print(f"  Input {j}: {tensor.name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                if tensor:
                    print(f"  Output {j}: {tensor.name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
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
    # 适当减少warmup次数以加快启动速度
    for _ in range(96 * 2048 * 8 // BATCH_SIZE // SEQ_LEN):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    print("Warm-up finished.")

    num_runs = 400 * 2048 * 8 // BATCH_SIZE // SEQ_LEN
    latencies = []
    print(f"Running benchmark for {num_runs} iterations...")
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        latencies.append(latency)

    # 移除前几个可能不稳定的结果
    latencies = latencies[5:]
    avg_latency = np.mean(latencies)
    throughput = (BATCH_SIZE * SEQ_LEN) / (avg_latency / 1000) if avg_latency > 0 else 0

    print("\n--- Benchmark Results (Qwen2.5-7B) ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Precision: FP16")
    print("Features: Full Decoder Layer with RoPE, GQA, and QKV Bias")
    print("---")
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"Tokens per Second (Throughput): {throughput:.2f} tokens/sec")
    print("---------------------------------------")


if __name__ == "__main__":
    print(f"TensorRT version: {trt.__version__}")
    engine_plan = build_decoder_layer_engine()

    if engine_plan:
        benchmark(engine_plan)