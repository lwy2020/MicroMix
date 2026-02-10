from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import torch
import functools
from typing import List
import time
import tqdm
import argparse
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path of the hf model")
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--lamda", type=float, default=1.0)
args = parser.parse_args()

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True,** kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc


@torch.no_grad()
def get_act_stats(model, device_, tokenizer, seqlen=2048, num_samples=32):
    device = next(model.parameters()).device
    act_scales = {}
    total_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).float().detach().cpu().abs()
        comming_scales = torch.mean(tensor, dim=0).float()
        if name in act_scales:
            total_scales[name].append(tensor)
            act_scales[name] = torch.max(act_scales[name], comming_scales)
        
        else:
            total_scales[name] = [tensor]
            act_scales[name] = comming_scales
  

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input", x)
        # stat_tensor(name + ".output", y)

    def reorder_tensor(tensor):
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional"
        sorted_tensor, sorted_index = torch.sort(tensor, descending=False) # For putting outliers at last
        # _, sorted_index = torch.sort(tensor, descending=True) # For putting outliers at first

        return sorted_index

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    dataset = dataset.shuffle(seed=0)
    
    for i in tqdm.tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seqlen, truncation=True
        ).input_ids.to(device)
        if input_ids.shape[-1] == 0:
            continue
        model(input_ids)

    p8_nums = {}
    p6_nums = {}
    total_elements = 0
    total_bits = 0
    act_orders = {}
    average_bits_dict = {}  # 新增：存储每个key的average bits

    for key, value in total_scales.items():
        act_orders[key] = reorder_tensor(act_scales[key])
        value = torch.cat(value, dim=0)
        seqlen, in_features = value.shape 
        
        p4_threshold = value.max(dim=-1, keepdim=True)[0] * 448 / 6 / math.pow(2, 10) * args.lamda
        p6_threshold = value.max(dim=-1, keepdim=True)[0] * 448 / 28 / math.pow(2, 6) * args.lamda
        
        p4_ratio = (value < p4_threshold).sum() / value.numel()
        p6_ratio = (value < p6_threshold).sum() / value.numel() - p4_ratio
        p8_ratio = 1 - p4_ratio - p6_ratio
        p6_num = math.ceil(in_features * p6_ratio / 128) * 128
        p8_num = math.ceil(in_features * p8_ratio / 128) * 128
        p4_num = in_features - p8_num - p6_num
        average_bits = 4 * p4_ratio + 6 * p6_ratio + 8 * p8_ratio
        
        # 保存每个key的average bits
        average_bits_dict[key] = average_bits
        
        total_elements += in_features
        total_bits += 4 * p4_num + 8 * p8_num + 6 * p6_num
        print(key, f'p4_num is {p4_num}, p8_num is {p8_num}, avg:{average_bits:.2f}')
        p6_nums[key] = p6_num
        p8_nums[key] = p8_num

    print(f'average bits is {total_bits / total_elements}')
    

    for h in hooks:
        h.remove()
    del act_scales
    del total_scales
    gc.collect()
    return act_orders, p8_nums, p6_nums
        
def main():
    model, enc = load_model(args.model)
    folder_path = "./saved"
    path = args.model.rstrip('/')
    model_name = path.split('/')[-1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    start_time = time.time()
    print("Getting reorder indices and 4/8 ratios...")
    
    reorder_index, p8_num, p6_num = get_act_stats(
        model, "cuda:0", enc, seqlen=args.seqlen, num_samples=args.samples
    )
    print(f'calibration time: {(time.time()-start_time):.2f}s')

    torch.save(reorder_index, f'./saved/{model_name}_reorder_index_wikitext2.pt')
    torch.save(p8_num, f'./saved/{model_name}_p8_num_wikitext2.pt')
    torch.save(p6_num, f'./saved/{model_name}_p6_num_wikitext2.pt')
    
if __name__ == "__main__":
    main()