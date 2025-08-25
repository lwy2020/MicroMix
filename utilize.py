from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM
from datasets import load_dataset
import torch.nn as nn
import gc
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import sys


@torch.no_grad()
def get_reorder_index(model, act_scales):
    act_orders = {}
    def reorder_tensor(tensor):
        # assert dimension == 1
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional"
        sorted_tensor, sorted_index = torch.sort(tensor, descending=False) # For putting outliers at last
        # _, sorted_index = torch.sort(tensor, descending=True) # For putting outliers at first

        return sorted_index
        
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            m.name = name
            # Reorder Index of each layer's input
            # Used to reorder the weight and previous layer's output
            inputName = name + ".input"
            act_orders[inputName] = reorder_tensor(act_scales[inputName])

            assert act_orders[inputName].dim() == 1, "Return Index must be 1 dimensional"

    return act_orders



def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc



@torch.no_grad()
def get_act_stats(model, dataloader, device_, metric='mean', seqlen=2048):
    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            tensorH = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        else:
            # Here we use abs since symmetric quantization use absmax.
            comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input", x)
        stat_tensor(name + ".output", y)

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.squeeze(0)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    assert cache['i'] == nsamples, "Captured samples should be equal to nsamples"
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    for h in hooks:
        h.remove()

    return act_scales

    

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
  
    import random
    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps 


@torch.no_grad()
def search_p4_p6_proportions(model, dataloader, device_, seqlen, reorder_index, hessian):
    nsamples = len(dataloader)
    device = device_
    
    p8_nums = {}
    p6_nums = {}
    average_bits = {}
   
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        act_scales[name+".input"] = x
        act_scales[name+".output"] = y
     
    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            cache['inps'] = inp
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    dataloader = torch.stack(dataloader, dim=0).squeeze(1)

    try:
        model(torch.tensor(dataloader).to(device))
    except ValueError:
        pass

    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    inps = cache['inps']
  
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
  
    for i in tqdm(range(len(layers))):
        act_scales = {}
        layer = layers[i].to(device)
  
        inps = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
       
        for name, keys in act_scales.items():
            if 'output' in name:
                continue
                
            keys = keys.reshape(-1, keys.shape[-1]).contiguous()
            seqlen, in_features = keys.shape 
       
            p4_threshold = keys.max(dim=-1, keepdim=True)[0] * math.pow(2, 1) / 254 * 8 / 6 
            # p6_threshold = keys.max(dim=-1, keepdim=True)[0] * math.pow(2, 1) / 254 * 32 / 7.5  #E2M3
            p6_threshold = keys.max(dim=-1, keepdim=True)[0] * math.pow(2, 3) / 254 * 32 / 28   #E3M2
     
            p4_ratio = (keys < p4_threshold).sum() / keys.numel()
            p6_ratio = (keys < p6_threshold).sum() / keys.numel() - p4_ratio
            p8_ratio = 1 - p4_ratio - p6_ratio
            p8_num = math.ceil(in_features * p8_ratio / 128) * 128
            p6_num = math.ceil(in_features * p6_ratio / 128) * 128
            p4_num = in_features - p8_num - p6_num
            average_bits[name] = 4 * p4_ratio + 6 * p6_ratio + 8 * p8_ratio
            
            print(f'p4_ratio is {p4_ratio}, p6_ratio is {p6_ratio}, p8_ratio is {p8_ratio}, avg:{average_bits[name]}')
            p6_nums[name] = p6_num
            p8_nums[name] = p8_num

            del keys
            
            gc.collect()
            torch.cuda.empty_cache()
            
        
        
        layer.cpu()
        del act_scales
        del layer
        gc.collect()
        torch.cuda.empty_cache()
            

    for h in hooks:
        h.remove()

    return p6_nums, p8_nums, average_bits
