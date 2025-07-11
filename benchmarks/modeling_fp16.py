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
import quarot
import quarot.transformers


from transformers.utils import is_flash_attn_greater_or_equal_2_10
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)


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

class LlamaMLP(nn.Module):
    def __init__(
        self,
        config,
        i
    ):
        super().__init__()
   
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias=False
        )
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):   
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    


    
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        config,
        i
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
    
        self.layer_idx = i
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
       
        self.q_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False
        )
        self.k_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False
        )
        self.v_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False
        )
        self.o_proj = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False
        )
        
     
    def forward(
      self,
      hidden_states: torch.Tensor,
      past_key_value
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        
        torch.cuda.nvtx.range_push("qkvo")
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        torch.cuda.nvtx.range_pop()

      
        
        torch.cuda.nvtx.range_push("pos_emb")
        query_states, key_states = rotary_pos_emb(query_states, key_states, 0)
        torch.cuda.nvtx.range_pop()
        cache_kwargs = {}
        # torch.cuda.nvtx.range_push("append_kv")
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("flash_attn")
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True)
        torch.cuda.nvtx.range_pop()
    
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        # output projection
        torch.cuda.nvtx.range_push("qkvo")

        attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()
        
        
        return attn_output, past_key_value


    
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
        layer_idx
    ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            config,
            i=layer_idx
        )
        self.mlp = LlamaMLP(
            config,
            i=layer_idx
        )
        self.input_layernorm = FP16LlamaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FP16LlamaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps)

    def forward(
      self,
      hidden_states: torch.Tensor,
      past_key_value
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

class LlamaModel(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList(
            # [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        [LlamaDecoderLayer(config, 0)]) # Hack for memory
        self.norm = FP16LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.cache_dtype = "float16"
        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16" 
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None 
        )

    def forward(
      self,
      input_ids: torch.Tensor,
      past_key_values=None
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push(f"embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()
        hidden_states = hidden_states.to(torch.bfloat16)
        # print(hidden_states.dtype)
        if past_key_values is None:
            max_length = input_ids.shape[1]
        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            # print(hidden_states.dtype)
            hidden_states, past_key_values = decoder_layer(hidden_states, past_key_values)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states, past_key_values


class LlamaForCausalLM(LlamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
      self,
      input_ids: torch.Tensor,
        past_key_values=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states, past_key_values = self.model(input_ids, past_key_values)
        torch.cuda.nvtx.range_push("lm_head")
        logits = self.lm_head(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return CausalLMOutputWithPast(
            past_key_values=past_key_values,
        )