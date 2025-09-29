import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from qLinearLayer import find_qlinear_layers
from qQwenLayer import QQwen2DecoderLayer


from functools import partial



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