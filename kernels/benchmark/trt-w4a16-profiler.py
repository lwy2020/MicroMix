import tensorrt as trt
import torch
import numpy as np
import time
import traceback
from collections import OrderedDict

# --- 配置参数 ---
M, N, K = 64, 5120, 5120
N_ITERATIONS = 6400 * 2048 // M
N_WARMUP = 1600 * 2048 // M

# 确保N是偶数，因为我们将两个INT4值打包成一个INT8
if N % 2 != 0:
    raise ValueError(f"N must be an even number for INT4 packing, but got {N}")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. 自定义的Profiler类 (与原版相同，无需修改)
class SimpleProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer_times = OrderedDict() # 使用有序字典保持层的顺序
        self.total_time_ms = 0

    def report_layer_time(self, layer_name, ms):
        """TensorRT在每次推理后会调用这个方法。"""
        self.layer_times[layer_name] = ms

    def print_report(self):
        """打印格式化的性能报告。"""
        if not self.layer_times:
            print("No layer timing information available. Was a profiled execution run?")
            return
        
        # 重新计算总时间，以处理融合层的情况
        self.total_time_ms = sum(self.layer_times.values())
        
        print("\n--- TensorRT Layer-wise Performance Report ---")
        print(f"Total GPU time for one inference: {self.total_time_ms:.6f} ms")
        print("-" * 60)
        print(f"{'Layer Name':<45} | {'Time (ms)':<12} | {'Percentage'}")
        print("-" * 60)

        for name, ms in self.layer_times.items():
            percentage = (ms / self.total_time_ms) * 100 if self.total_time_ms > 0 else 0
            # 缩短过长的层名以便显示
            display_name = (name[:42] + '...') if len(name) > 45 else name
            print(f"{display_name:<45} | {ms:<12.6f} | {percentage:8.2f}%")
        
        print("-" * 60)
        # 清空记录，以便下次分析
        self.layer_times.clear()
        self.total_time_ms = 0

# --- W4A16 核心改动部分 ---

def quantize_and_pack_w4(fp32_weights: torch.Tensor):
    """
    将FP32权重矩阵量化为INT4，并打包成UINT8。
    这里使用per-channel的对称量化。
    
    Args:
        fp32_weights (torch.Tensor): 形状为 (K, N) 的FP32权重。
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - packed_uint8_weights (torch.Tensor): 形状为 (K, N/2) 的UINT8打包权重。
        - scales (torch.Tensor): 形状为 (N,) 的FP16缩放因子。
    """
    print("Quantizing FP32 weights to INT4 and packing...")
    # 确保权重在CPU上
    weights = fp32_weights.cpu()
    K, N = weights.shape
    
    # 1. 计算缩放因子 (Per-channel)
    # 对于INT4，我们将范围映射到 [-8, 7]。对称量化使用absmax。
    # scale = absmax / 7.0
    scales = weights.abs().max(dim=0)[0] / 7.0
    
    # 2. 量化
    # quantized_val = round(float_val / scale)
    # 使用广播进行量化
    quantized_int8 = torch.round(weights / scales).to(torch.int8)
    
    # 3. 裁剪到INT4范围 [-8, 7]
    quantized_int8 = torch.clamp(quantized_int8, min=-8, max=7)

    # 4. 打包 (Pack)
    # 将两个4-bit整数打包到一个8-bit整数中。
    # 我们将低4位存储第一个值，高4位存储第二个值。
    # PyTorch/Numpy没有int4类型, 所以我们操作int8然后打包。
    # `& 0x0F` 确保我们只取低4位 (等价于 % 16，但更高效)
    low_nibbles = quantized_int8[:, 0::2] & 0x0F
    high_nibbles = quantized_int8[:, 1::2] & 0x0F
    
    # 将高位的半字节左移4位，然后与低位半字节进行或运算
    packed_uint8_weights = (high_nibbles << 4) | low_nibbles
    
    # 将缩放因子转换为FP16，因为激活和计算都将在FP16中进行
    return packed_uint8_weights.to(torch.uint8), scales.to(torch.float16)


def build_w4a16_engine():
    """构建一个使用W4A16 GEMM的TensorRT引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    # config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)
    if hasattr(trt.BuilderFlag, 'WEIGHT_ONLY_QUANT'):
        config.set_flag(trt.BuilderFlag.WEIGHT_ONLY_QUANT)

    input_a = network.add_input(name='A', dtype=trt.float16, shape=(M, K))

    weights_b_cpu_fp32 = torch.randn(K, N, dtype=torch.float32)
    packed_weights_b, scales_b = quantize_and_pack_w4(weights_b_cpu_fp32)
    
    # --- 关键修改在这里 ---
    # 将打包好的权重和缩放因子添加到网络中作为常量
    
    # 错误的方式:
    # weights_b_trt = trt.Weights(packed_weights_b.numpy(), type=trt.DataType.INT4)
    
    # 正确的方式: 使用(type, ptr, count)构造函数
    weights_np = packed_weights_b.numpy()
    # 明确告诉TRT数据类型是INT4，并提供数据指针和INT4元素的总数
    # weights_np.size 是 uint8 元素的数量 (K * N / 2)
    # INT4 元素的总数是它的两倍，即 K * N
    weights_b_trt = trt.Weights(
        trt.DataType.INT4,          # 逻辑数据类型
        weights_np.ctypes.data,     # 指向原始数据的内存指针
        weights_np.size * 2         # INT4 元素的总数量
    )

    # 接下来的代码和原来一样
    quant_b_const = network.add_constant((K, N), weights_b_trt)
    quant_b_const.name = "Packed_INT4_Weights_B"
    
    scales_b_trt = trt.Weights(scales_b.numpy())
    scales_b_const = network.add_constant(scales_b.shape, scales_b_trt)
    scales_b_const.name = "FP16_Scales_B"
    
    dequant_b_layer = network.add_dequantize(
        quant_b_const.get_output(0), 
        scales_b_const.get_output(0), 
        trt.DataType.HALF
    )
    dequant_b_layer.axis = 1 
    dequant_b_layer.name = "Dequantize_B"
    dequant_b_output = dequant_b_layer.get_output(0)

    gemm_layer = network.add_matrix_multiply(
        input_a, trt.MatrixOperation.NONE,
        dequant_b_output, trt.MatrixOperation.NONE
    )
    gemm_layer.name = "MatrixMultiply(FP16)"
    
    final_output = gemm_layer.get_output(0)
    network.mark_output(final_output)
    final_output.name = "output_c"
    final_output.dtype = trt.float16

    print("Building W4A16 engine...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise RuntimeError("Engine build failed.")
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    print("Engine build successful!")
    return engine

def benchmark_and_profile(engine):
    """对引擎进行测速，并使用IProfiler分析单次推理的内部耗时"""
    profiler = SimpleProfiler()
    context = engine.create_execution_context()
    context.profiler = profiler

    # --- W4A16 基准测试改动 ---
    # 输入和输出张量现在是 FP16
    input_a_gpu = torch.randn(M, K, dtype=torch.float16).cuda()
    output_gpu = torch.empty((M, N), dtype=torch.float16).cuda()
    
    input_name = "A"
    output_name = "output_c"
    context.set_tensor_address(input_name, input_a_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())

    # --- 常规基准测试 (与原版逻辑相同) ---
    print(f"Warming up for {N_WARMUP} iterations...")
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"Running benchmark for {N_ITERATIONS} iterations...")
    start_event.record()
    for _ in range(N_ITERATIONS):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms_cuda = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms_cuda / N_ITERATIONS
    # FLOPS计算保持不变，因为我们计算的是等效的浮点运算次数
    throughput_tflops = (2 * M * N * K) / (avg_latency_ms * 1e-3) / 1e12

    print("\n--- Overall Benchmark Results (W4A16) ---")
    print(f"Average GPU Latency (from {N_ITERATIONS} runs): {avg_latency_ms:.6f} ms")
    print(f"Throughput: {throughput_tflops:.4f} TFLOPS")
    print("-------------------------------------------")

    # --- 单次推理以进行性能分析 ---
    print("\nRunning one final inference to capture layer-wise timings...")
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # 打印Profiler报告，我们期望看到Dequantize和MatrixMultiply被融合成一个层
    profiler.print_report()


if __name__ == "__main__":
    # W4A16通常在支持FP16的GPU上表现良好 (Compute Capability 7.0+ / Volta及更新架构)
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 7:
         print("W4A16 benchmark is best suited for GPUs with compute capability 7.0 (Volta) or newer.")
    # 检查TensorRT版本是否支持INT4类型
    elif not hasattr(trt.DataType, 'INT4'):
        print("Your version of TensorRT does not support `trt.DataType.INT4`. Please use a newer version (e.g., 8.6+).")
    else:
        try:
            engine = build_w4a16_engine()
            if engine:
                benchmark_and_profile(engine)
        except Exception as e:
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()