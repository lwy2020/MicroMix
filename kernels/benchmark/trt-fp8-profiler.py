import tensorrt as trt
import torch
import numpy as np
import time
import traceback
from collections import OrderedDict

# --- 配置参数 ---
M, N, K = 64, 5120, 5120
N_ITERATIONS = 6400 * 2048 * 2 // M
N_WARMUP = 1600 * 2048 * 2 // M

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. 创建一个自定义的Profiler类
class SimpleProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer_times = OrderedDict() # 使用有序字典保持层的顺序
        self.total_time_ms = 0

    def report_layer_time(self, layer_name, ms):
        """TensorRT在每次推理后会调用这个方法。"""
        # 注意：对于融合后的层，layer_name可能是 '(Layer1 + Layer2 + ...)' 的形式
        self.layer_times[layer_name] = ms
        # 简单累加所有层的时间，这约等于总的GPU执行时间
        if "(" not in layer_name: # 避免重复计算融合层和其子层的时间
            self.total_time_ms += ms

    def print_report(self):
        """打印格式化的性能报告。"""
        if not self.layer_times:
            print("No layer timing information available. Was a profiled execution run?")
            return

        # 重新计算总时间，以处理融合层的情况
        # 我们只关心最终的、实际执行的层，所以直接加总字典中的值
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

def build_fp8_engine_the_right_way():
    """与原版相同，此处省略详细注释"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    input_a = network.add_input(name='A', dtype=trt.float32, shape=(M, K))
    weights_b_cpu = torch.randn(K, N, dtype=torch.float32)
    weights_b = network.add_constant(weights_b_cpu.shape, weights_b_cpu.numpy()).get_output(0)
    weights_b.name = "Weights_B"

    scale_val = 0.02
    scale_tensor = network.add_constant((), trt.Weights(np.array([scale_val], dtype=np.float32))).get_output(0)
    scale_tensor.name = "Scale"
    
    # 给层命名，以便在Profiler中识别
    quant_a_layer = network.add_quantize(input_a, scale_tensor, trt.DataType.FP8)
    quant_a_layer.name = "Quantize_A"
    fp8_a = quant_a_layer.get_output(0)
    
    dequant_a_layer = network.add_dequantize(fp8_a, scale_tensor, trt.DataType.FLOAT)
    dequant_a_layer.name = "Dequantize_A"
    dequant_a_output = dequant_a_layer.get_output(0)
    
    quant_b_layer = network.add_quantize(weights_b, scale_tensor, trt.DataType.FP8)
    quant_b_layer.name = "Quantize_B"
    fp8_b = quant_b_layer.get_output(0)

    dequant_b_layer = network.add_dequantize(fp8_b, scale_tensor, trt.DataType.FLOAT)
    dequant_b_layer.name = "Dequantize_B"
    dequant_b_output = dequant_b_layer.get_output(0)
    
    gemm_layer = network.add_matrix_multiply(
        dequant_a_output, trt.MatrixOperation.NONE,
        dequant_b_output, trt.MatrixOperation.NONE
    )
    gemm_layer.name = "MatrixMultiply"
    
    final_output = gemm_layer.get_output(0)
    network.mark_output(final_output)
    final_output.name = "output_c"

    print("Building engine with automatic fusion pattern recognition...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise RuntimeError("Engine build failed.")
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    print("Engine build successful!")
    return engine

def benchmark_and_profile(engine):
    """对引擎进行测速，并使用IProfiler分析单次推理的内部耗时"""
    # 2. 实例化我们的Profiler
    profiler = SimpleProfiler()
    
    context = engine.create_execution_context()
    # 3. 将Profiler关联到Execution Context
    context.profiler = profiler

    input_a_gpu = torch.randn(M, K, dtype=torch.float32).cuda()
    output_gpu = torch.empty((M, N), dtype=torch.float32).cuda()
    input_name = "A"
    output_name = "output_c"
    context.set_tensor_address(input_name, input_a_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())

    # --- 常规基准测试 ---
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
    throughput_tflops = (2 * M * N * K) / (avg_latency_ms * 1e-3) / 1e12

    print("\n--- Overall Benchmark Results ---")
    print(f"Average GPU Latency (from {N_ITERATIONS} runs): {avg_latency_ms:.6f} ms")
    print(f"Throughput: {throughput_tflops:.4f} TFLOPS")
    print("---------------------------------")

    # --- 单次推理以进行性能分析 ---
    # 我们在所有基准测试运行后，再执行一次，以捕获一个“热”运行的层耗时
    print("\nRunning one final inference to capture layer-wise timings...")
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # 4. 打印Profiler收集到的报告
    profiler.print_report()


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 9:
         print("FP8 is only supported on GPUs with compute capability 9.0 (Hopper architecture) or newer.")
    else:
        try:
            engine = build_fp8_engine_the_right_way()
            if engine:
                benchmark_and_profile(engine)
        except Exception as e:
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()