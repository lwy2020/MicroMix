import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from qLinearLayer import find_qlinear_layers
from qLlamaLayer import QLlamaDecoderLayer
from qQwenLayer import QQwen2DecoderLayer
from qMixtralLayer import QMixtralDecoderLayer

from functools import partial

import math


def reorder_model_llama(model, device, kv_cache, reorder_index, p6_nums, p8_nums):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                kv_cache=kv_cache,
                p8_nums=p8_nums,
                p6_nums=p6_nums,
                reorder_index=reorder_index,
                layer_idx=i
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]
            
        nameTemplate = 'layers.{}.{}.{}.{}'
        m.mlp.register_buffer('up_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')].to(torch.int16))
        m.mlp.register_buffer('down_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('q_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('o_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')].to(torch.int16))
        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def reorder_model_qwen(model, device, kv_cache, reorder_index, p6_nums, p8_nums):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], Qwen2DecoderLayer):
            m = QQwen2DecoderLayer(
                originalLayer=layers[i],
                kv_cache=kv_cache,
                p8_nums=p8_nums,
                p6_nums=p6_nums,
                reorder_index=reorder_index,
                layer_idx=i
            )
            
        nameTemplate = 'layers.{}.{}.{}.{}'
        m.mlp.register_buffer('up_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')].to(torch.int16))
        m.mlp.register_buffer('down_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('q_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('o_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')].to(torch.int16))
        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def reorder_model_mixtral(model, device, kv_cache, reorder_index, p6_nums, p8_nums):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], MixtralDecoderLayer):
            m = QMixtralDecoderLayer(
                originalLayer=layers[i],
                kv_cache=kv_cache,
                p8_nums=p8_nums,
                p6_nums=p6_nums,
                reorder_index=reorder_index,
                layer_idx=i
            )
        elif isinstance(layers[i], QMixtralDecoderLayer):
            m = layers[i]
            

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model



import torch
import typing
import transformers
import utils
import os
import logging

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


