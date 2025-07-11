import argparse
import gc
import functools
import pprint
import numpy as np
import torch
import time

import flashinfer
import torch
import transformers
import dataclasses

@dataclasses.dataclass
class ModelConfig:
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int
    _attn_implementation: str
    dtype: str = dataclasses.field(default="bfloat16")
    device: str = dataclasses.field(default="cuda:0")
    
MODEL_CFGS = {
    "8b":
        ModelConfig(
            num_layers=1,
            num_heads=32,
            hidden_size=4096,
            intermediate_size=14336,
            _attn_implementation='flash_attention_2',
            # page_size=16,
            # max_num_pages=128
        ),
    "14b":
        ModelConfig(
            num_layers=1,
            num_heads=40,
            hidden_size=5120,
            intermediate_size=13824,
            _attn_implementation='flash_attention_2',
            # page_size=16,
            # max_num_pages=128
        ),
}


benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 3
num_bench_steps = 10

def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@repeated_run()
def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    torch.cuda.reset_max_memory_allocated()
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


def _build_cache(batch_size, length, layer, disable_quant, num_key_value_heads, hidden_size, device):
    num_heads = num_key_value_heads
    model_dim = hidden_size
    head_dim = model_dim // num_heads
    return quarot.transformers.MultiLayerPagedKVCache4Bit(
        batch_size=batch_size,
        page_size=length, 
        max_seq_len=length, 
        device=device, 
        n_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        disable_quant=disable_quant,
        hadamard_dtype=None if disable_quant else torch.float16
    )


def get_model_quantized(model_cfg):
    from modeling_micromix import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=model_cfg.hidden_size,
            num_heads=model_cfg.num_heads,
            intermediate_size=model_cfg.intermediate_size,
            num_hidden_layers=model_cfg.num_layers,
        )).to(model_cfg.device)
    # return model, None, model.config.hidden_size
    
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
   
    return model, wrapper


# def get_model_hf(config_name):
#     return transformers.LlamaForCausalLM.from_pretrained(
#         config_name, 
#         torch_dtype=torch.float16, 
#         attn_implementation="flash_attention_2"
#     ), None, model.config.hidden_size

def get_model_fp16(model_cfg):
    from modeling_fp16 import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=model_cfg.hidden_size,
            num_heads=model_cfg.num_heads,
            intermediate_size=model_cfg.intermediate_size,
            num_hidden_layers=model_cfg.num_layers,
        )).to(model_cfg.device)
    
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchAttention(kv_layout="NHD")
    return model, wrapper, model.config.hidden_size


def run_prefill(layer, wrapper, bsz, prefill_length, config):
    device = 'cuda'
    test_input = torch.rand((bsz, prefill_length, config.hidden_size), dtype=torch.bfloat16, device=device)
    if wrapper is None:
        def _prefill():
            layer(test_input)
    else:
#         nnz_qo = bsz*prefill_length
#         qo_indptr = torch.tensor(
#     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
# )
#         paged_kv_indices = torch.arange(config.max_num_pages).int().to("cuda:0")
#         paged_kv_indptr = torch.tensor(
#     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
# )
#         paged_kv_last_page_len = torch.tensor(
#     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
# )
#         wrapper.plan(
#                 qo_indptr,
#                 paged_kv_indptr,
#                 paged_kv_indices,
#                 paged_kv_last_page_len,
#                 config.num_heads,
#                 config.num_heads,
#                 config.head_dim,
#                 config.page_size,
#                 causal=True,
#             )
        
        def _prefill():
            layer(test_input)
    return module_benchmark(_prefill)


def run_decode(layer, wrapper, bsz, prefill_length, decode_steps, hidden_size):
    device = layer.self_attn.v_proj.weight.device
    test_input = torch.rand((bsz, prefill_length, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((bsz, 1, hidden_size), dtype=torch.float16, device=device)
    assert wrapper is not None
    past_key_values = wrapper(bsz, prefill_length + decode_steps, layer)
    layer(test_input, past_key_value=past_key_values)
    def _decode_for_multiple_steps():
        past_key_values.length = prefill_length
        for i in range(decode_steps):
            layer(next_input, past_key_value=past_key_values, 
            position_ids=torch.tensor([[prefill_length + i]] * bsz, device=past_key_values.device, dtype=torch.int32))
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(layer, wrapper, bsz, prefill_length, decode_steps, hidden_size):
    device = layer.self_attn.v_proj.weight.device
    test_input = torch.rand((bsz, prefill_length, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((bsz, 1, hidden_size), dtype=torch.float16, device=device)
    assert wrapper is not None
    past_key_values = wrapper(bsz, prefill_length + decode_steps, layer)
    def _prefill_and_decode_for_multiple_steps():
        past_key_values.length = 0
        past_key_values._needs_init[0] = True
        layer(test_input, past_key_value=past_key_values)
        for i in range(decode_steps):
            layer(next_input, past_key_value=past_key_values, 
            position_ids=torch.tensor([[prefill_length + i]] * bsz, device=device, dtype=torch.int32))
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(layer, wrapper, bsz, prefill, decode, config):
    layer = layer.cuda()
    layer.eval()
    time_prefill, memory_prefill = run_prefill(layer, wrapper, bsz, prefill, config)
    
    _cleanup()
    # if decode is not None:
    #     time_decode, memory_decode = run_decode(layer, wrapper, bsz, prefill, decode, hidden_size)
    #     _cleanup()
    #     time_e2e, _ = run_e2e(layer, wrapper, bsz, prefill, decode, hidden_size)
    #     _cleanup()
    # else:
    #     time_decode = time_e2e = None
    #     memory_decode = None
    return time_prefill, memory_prefill

def benchmark(args):

    model, wrapper = get_model_quantized(MODEL_CFGS[args.model])
    layer = model.model.layers[0]
    del model
    _cleanup()
    time_prefill_i4, mem_i4 = run_all_for_model(
        layer, wrapper, args.batch_size, args.prefill_seq_len, args.decode_steps, MODEL_CFGS[args.model])
    del layer
    _cleanup()
    # model, wrapper, hidden_size = get_model_fp16(MODEL_CFGS[args.model])
    # layer = model.model.layers[0]
    # del model
    # _cleanup()
    # time_prefill_f16, mem_f16 = run_all_for_model(
    #     layer, wrapper, args.batch_size, args.prefill_seq_len, args.decode_steps, hidden_size)
    # del layer
    # _cleanup()

    print(f"Prefill Int4 time: {np.mean(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f}ms")
    # print(f"Prefill FP16 time: {np.mean(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f}ms")
    # print(f"Speedup: {np.mean(time_prefill_f16) / np.mean(time_prefill_i4):.3f}x")
    # print(f'Prefill & {args.model} & {args.batch_size} & {args.prefill_seq_len} & {np.mean(time_prefill_f16):.3f} & {np.mean(time_prefill_i4):.3f}\\\\')

#         if args.decode_steps is not None:
#             print(f"Decode Int4 time: {np.mean(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f}ms")
#             print(f"Decode FP16 time: {np.mean(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f}ms")
#             print(f"Speedup: {np.mean(time_decode_f16) / np.mean(time_decode_i4):.3f}x")
#             print(f'Decode & {args.model} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_decode_f16):.3f} & {np.mean(time_decode_i4):.3f}\\\\')

#             print(f"E2E Int4 time: {np.mean(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f}ms")
#             print(f"E2E FP16 time: {np.mean(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f}ms")
#             print(f"Speedup: {np.mean(time_e2e_f16) / np.mean(time_e2e_i4):.3f}x")
#             print(f'E2E & {args.model} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(time_e2e_f16):.3f} & {np.mean(time_e2e_i4):.3f}\\\\')

    # table-style output

    print(f"Int4 memory: {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_i4):.3f}")
    # print(f"FP16 memory: {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem_f16):.3f}")
    # print(f"Memory saving: {np.mean(mem_f16) / np.mean(mem_i4):.3f}x")
    # print(f'Memory saving & {args.model} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & {np.mean(mem_i4) / (1024 * 1024 * 1024):.3f}GB & {np.mean(mem_f16) / (1024 * 1024 * 1024):.3f}GB\\\\')
        
    print('--------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        default='8b'
    )
    
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=64,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=4096,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=None,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    benchmark(args)