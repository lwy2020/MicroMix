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



def get_model_fp16(name):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map='auto',
        torch_dtype=torch.bfloat16
        )
    return model



def run_prefill(model, bsz, prefill_length):
    device = 'cuda'
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    def _prefill():
        model(test_input)
   
    return module_benchmark(_prefill)

def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    out = model(test_input)
    past_key_values = out.past_key_values
    del out
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        for _ in range(decode_steps):
            model(next_input, past_key_values=past_key_values)
    return module_benchmark(_decode_for_multiple_steps)

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    if decode is None:
        time_prefill, memory_prefill = run_prefill(model, bsz, prefill)
        _cleanup()
        return time_prefill, memory_prefill
    else:
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        return time_decode, memory_decode

def benchmark(args):
    model = get_model_fp16(args.model)
    time, mem = run_all_for_model(
        model, args.batch_size, args.prefill_seq_len, args.decode_steps)
    del model
    _cleanup()

    print(f"FP16 time: {np.mean(time):.3f} +- {1.96 * np.std(time):.3f}ms")
    print(f"FP16 memory: {np.mean(mem) / (1024 * 1024 * 1024):.3f}GB +- {1.96 * np.std(mem):.3f}")
    print('--------------')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        default='llama-3.1-8b'
    )
    
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=32,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
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
