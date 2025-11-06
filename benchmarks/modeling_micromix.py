import math
from typing import Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (ACT2FN,
    LlamaConfig,
    LlamaMLP,
    PreTrainedModel,
    rotate_half,
    LlamaRotaryEmbedding
)

# from transformers.integrations.flash_attention import flash_attention_forward
import flashinfer

import sys
sys.path.append('./mgemm/build/')
import mixedgemm
from kv_cache import *

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    
class QLinearLayer(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        p8_num, 
        p6_num,
        reorder_index=None
    ) -> None:
        factory_kwargs = {"device": 'cuda', "dtype": torch.bfloat16}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.p6_num = p6_num
        self.p8_num = p8_num
        self.p4_num = self.in_features - p8_num - p6_num
    
        self.BN = torch.zeros(out_features, self.p4_num//2, dtype=torch.uint8, device='cuda')
        self.BS = torch.zeros(out_features, self.p6_num*6//8, dtype=torch.uint8, device='cuda')
        self.BO = torch.zeros(out_features, self.p8_num, dtype=torch.uint8, device='cuda')
        self.SFBN = torch.ones(out_features*self.p4_num//32, dtype=torch.uint8, device='cuda') * 127 
        self.SFBS = torch.ones(out_features*self.p6_num//32, dtype=torch.uint8, device='cuda') * 127
        self.SFBO = torch.ones(out_features*self.p8_num//32, dtype=torch.uint8, device='cuda') * 127
        self.reorder_index = torch.arange(self.in_features, dtype=torch.int16, device='cuda') 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AN, AS, AO, SFAN, SFAS, SFAO = x[:6]
        y = mixedgemm.matmul(AN, self.BN, AS, self.BS, AO, self.BO, SFAN, self.SFBN, SFAS, self.SFBS, SFAO, self.SFBO)
        if self.bias is not None:
            y = y + self.bias
        
        return y


def rotary_pos_emb(q, k, beg):
    device = q.device
    dtype = q.dtype
    bsz, nhead, seqlen, dim = q.shape
    end = beg + seqlen

    base = 10000
    inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(beg, end, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)
    cos = emb.cos()
    sin = emb.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

class QLlamaMLP(nn.Module):
    def __init__(
        self,
        config,
        p8_nums,
        p6_nums,
        i,
        reorder_index=None,
    ):
        super().__init__()
        nameTemplate = 'layers.{}.{}.{}.{}'
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias=config.mlp_bias,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
        )
        self.down_proj = QLinearLayer(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias=config.mlp_bias,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
        )
        self.up_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias=config.mlp_bias,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            
        )
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):   
        bsz, q_len = x[-2], x[-1]
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(mixedgemm.activate_quantize_x(self.gate_proj(x), self.up_proj(x), self.down_proj.p4_num, self.down_proj.p6_num, self.down_proj.p8_num)).reshape(bsz, q_len, -1)


    
class QLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        config,
        p8_nums,
        p6_nums,
        i,
        reorder_index=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        
        self.layer_idx = i
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.q_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=config.attention_bias,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
        )
        self.k_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=config.attention_bias,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
        )
        self.v_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=config.attention_bias,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
        )
        self.o_proj = QLinearLayer(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=config.attention_bias,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
        )
    
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states[-2], hidden_states[-1]
    
    
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        
        kv_seq_len = key_states.shape[1]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        cache_kwargs = {"sin": sin, "cos": cos}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p= 0.0,
                    is_causal=True
                ).contiguous()
        
            # attn_output, attn_weights = flash_attention_forward(
            #     self,
            #     query_states,
            #     key_states,
            #     value_states,
            #     attention_mask,
            #     dropout=0.0,
            #     scaling=self.scaling,
            # ).contiguous()
        else:
            attn_output = cache_out(query_states).contiguous()
        
        attn_output = attn_output.reshape(bsz*q_len, -1).contiguous()

        # output projection
        torch.cuda.nvtx.range_push("qkvo")
        # (AN, AS, AO, SFAN, SFAS, SFAO)
        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(attn_output, self.o_proj.reorder_index, self.o_proj.p4_num, self.o_proj.p6_num, self.o_proj.p8_num)
        attn_output = self.o_proj((AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)).reshape(bsz, q_len, -1)
        torch.cuda.nvtx.range_pop()
    
        return attn_output, past_key_value


class LlamaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size, eps, p8_num, p6_num, reorder_index=None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps
        self.p6_num = p6_num
        self.p8_num = p8_num
        
        self.p4_num = len(self.weight) - p8_num - p6_num
        self.reorder_index = torch.arange(len(self.weight), dtype=torch.int16, device='cuda') 

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz*q_len, -1).contiguous()
        # print(self.p4_num, self.p6_num, self.p8_num)
        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.rmsnorm_quantize_x(hidden_states, self.weight, self.variance_epsilon, self.reorder_index, self.p4_num, self.p6_num, self.p8_num)
        return (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
        # return rms_norm(hidden_states, self.weight, self.variance_epsilon)
    
class FP16LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        p8_nums,
        p6_nums,
        layer_idx,
        reorder_index=None,
    ):
        super().__init__()
        
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.hidden_size = config.hidden_size
        self.self_attn = QLlamaAttention(
            config,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        self.mlp = QLlamaMLP(
            config,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, p8_num=p8_nums[nameTemplate.format(layer_idx, 'self_attn', 'q_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(layer_idx, 'self_attn', 'q_proj', 'input')], 
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, p8_num=p8_nums[nameTemplate.format(layer_idx, 'mlp', 'gate_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(layer_idx, 'mlp', 'gate_proj', 'input')],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value=None,
        position_ids=None
    ) -> torch.Tensor:
        residual = hidden_states

        torch.cuda.nvtx.range_push("input_norm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        torch.cuda.nvtx.range_push("LlamaAttention")
        hidden_states, past_key_value = self.self_attn(hidden_states, past_key_value, position_ids)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        # Fully Connected
        residual = hidden_states
        torch.cuda.nvtx.range_push("norm")
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("mlp")
        hidden_states = self.mlp(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        return hidden_states, past_key_value


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [
      r"decoder\.version",
      r"self_attn\.rotary_emb\.inv_freq",
    ]
    
class FP16LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        
class LlamaModel(LlamaPreTrainedModel):

    def __init__(self, name: str, config: LlamaConfig, layer_idx=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
    
        p6_num_filename = f'./saved/{name}_p6_num_wikitext2_mean.pt'
        p8_num_filename = f'./saved/{name}_p8_num_wikitext2_mean.pt'
        p6_nums = torch.load(p6_num_filename, weights_only=False)
        p8_nums = torch.load(p8_num_filename, weights_only=False)
        if layer_idx is not None:
            self.layers = nn.ModuleList(
        [LlamaDecoderLayer(config, p8_nums, p6_nums, layer_idx)])
        else:
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, p8_nums, p6_nums, i) for i in range(config.num_hidden_layers)],)

        self.norm = FP16LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.cache_dtype = "bfloat16"
        self.config = config
        self.page_len = 128
        self.head_dim = config.hidden_size // config.num_heads
        
    def build_cache(self, batch_size, page_size, max_length):
        device = 'cuda'
        dtype = self.cache_dtype
        
        num_heads = self.config.num_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16"
        return MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16
        )
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_value=None,
        position_ids=None
      ) -> torch.Tensor:
        if past_key_value is None:
            max_length = input_ids.shape[1] * 2
            past_key_value = self.build_cache(
                input_ids.shape[0], 
                page_size=max_length,
                max_length=max_length
            )
        torch.cuda.nvtx.range_push(f"embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()
        hidden_states = hidden_states.to(torch.bfloat16)
        bsz, q_len, _ = hidden_states.shape
        if position_ids is None:
            # 获取 input_ids 的设备和形状
            device = input_ids.device
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)
        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            # print(hidden_states.dtype)
            hidden_states, past_key_value = decoder_layer(hidden_states, past_key_value, position_ids)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states, past_key_value


class LlamaForCausalLM(LlamaModel):

    def __init__(self, name, config, layer_idx=None):
        super().__init__(name, config, layer_idx)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.attention_bias, dtype=torch.bfloat16)
        self.post_init()
        self.config = config
    def forward(
      self,
      input_ids: torch.Tensor,
    past_key_value=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states, past_key_value = super().forward(
            input_ids, 
            past_key_value,
        )        
        torch.cuda.nvtx.range_push("lm_head")
        
        logits = self.lm_head(hidden_states.to(torch.bfloat16))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return past_key_value
