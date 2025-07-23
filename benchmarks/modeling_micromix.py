# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math
from typing import Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (ACT2FN,
    LlamaConfig,
    LlamaMLP,
    PreTrainedModel,
    rotate_half,
)

import flashinfer

import sys
sys.path.append('./mgemm/build/')
import mixedgemm

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
    
        self.layer_idx = i
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
        self.page_len = 128
        
    def forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value=None
) -> torch.Tensor:
        bsz, q_len = hidden_states[-2], hidden_states[-1]
    
    
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).contiguous()

        if past_key_value is not None:
            seqlens = [q_len] * bsz
            x_indptr = torch.tensor([0] + [q_len * (i + 1) for i in range(bsz)], dtype=torch.int32, device=query_states.device)
            seq_lens = torch.tensor(seqlens, dtype=torch.int32, device=query_states.device)
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                x_indptr, seq_lens, bsz * q_len
            )
            flashinfer.append_paged_kv_cache(
                key_states.view(-1, self.num_heads, self.head_dim),
                value_states.view(-1, self.num_heads, self.head_dim),
                batch_indices,
                positions,
                past_key_value[self.layer_idx]["layer_buf"],
                past_key_value[self.layer_idx]["kv_indices"],
                past_key_value[self.layer_idx]["kv_indptr"],
                past_key_value[self.layer_idx]["kv_last_page_len"],
                "NHD"
            )

  
        stack_attn_output = []
        for i in range(bsz):
            # print
            o = flashinfer.single_prefill_with_kv_cache(query_states[i], key_states[i], value_states[i], causal=True, pos_encoding_mode="ROPE_LLAMA") # append attention with LLaMA style RoPE on-the-fly, apply causal mask
            stack_attn_output.append(o)
        
        if len(stack_attn_output) == 1:
            attn_output = stack_attn_output[0]
        else:
            attn_output = torch.cat(stack_attn_output, dim=0)
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
        # self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
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
      past_key_value=None
  ) -> torch.Tensor:
        residual = hidden_states

        torch.cuda.nvtx.range_push("input_norm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        torch.cuda.nvtx.range_push("LlamaAttention")
        hidden_states, past_key_value = self.self_attn(hidden_states, past_key_value)
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
    def forward(
      self,
      input_ids: torch.Tensor,
      past_key_value=None
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push(f"embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()
        hidden_states = hidden_states.to(torch.bfloat16)
        bsz, q_len, _ = hidden_states.shape
        
        if past_key_value is None:
            # 初始化 KV Cache 张量（仅在第一次调用时）
            device = hidden_states.device
            seqlens = [q_len] * bsz
            # total_pages = int(256000 / self.page_len)
            total_pages = math.ceil(bsz * q_len / self.page_len) 
            layer_buf = torch.empty(
            (total_pages, 2, self.page_len, self.config.num_heads, self.head_dim),
            dtype=torch.bfloat16,
            device="cuda"
            )
            kv_indices_host = []
            kv_indptr_host = [0]
            next_page_id = 0
            for seqlen in seqlens:
                npages = (seqlen + self.page_len - 1) // self.page_len
                kv_indices_host.extend(range(next_page_id, next_page_id + npages))
                next_page_id += npages
                kv_indptr_host.append(len(kv_indices_host))
            kv_indices = torch.tensor(kv_indices_host, device=device, dtype=torch.int32)
            kv_indptr = torch.tensor(kv_indptr_host, device=device, dtype=torch.int32)
            # kv_indices = torch.arange(total_pages, dtype=torch.int32, device=hidden_states.device)
            # kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)
            kv_last_page_len = torch.tensor([(seqlen - 1) % self.page_len + 1 for seqlen in seqlens], dtype=torch.int32, device=hidden_states.device)
            past_key_value = [
                {
                "layer_buf": layer_buf,
                "kv_indices": kv_indices,
                "kv_indptr": kv_indptr,
                "kv_last_page_len": kv_last_page_len,
                # "next_page_id": i,
            } for i in range(self.config.num_hidden_layers)]
    
        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            # print(hidden_states.dtype)
            hidden_states, past_key_value = decoder_layer(hidden_states, past_key_value)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states, past_key_value


class LlamaForCausalLM(LlamaModel):

    def __init__(self, name, config, layer_idx=None):
        super().__init__(name, config, layer_idx)

        self.model = LlamaModel(name, config, layer_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.attention_bias, dtype=torch.bfloat16)
        self.post_init()
        self.config = config
    def forward(
      self,
      input_ids: torch.Tensor,
        past_key_value=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states, past_key_value = self.model(input_ids, past_key_value)
        torch.cuda.nvtx.range_push("lm_head")
        
        logits = self.lm_head(hidden_states.to(torch.bfloat16))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        # return CausalLMOutputWithPast(
        #     past_key_value=past_key_value,
        # )