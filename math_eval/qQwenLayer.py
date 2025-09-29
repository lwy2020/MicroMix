import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2Attention, Qwen2MLP

from qLinearLayer import QLinearLayer
import mixedgemm

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_hadamard(n): 
    if n == 1:
        return torch.tensor([[1.]], dtype=torch.float32)
    else:
        assert n % 1 == 0, "The size should be divided by 2."
        H_n_minus_1 = get_hadamard(n//2)
        return torch.cat([torch.cat([H_n_minus_1, H_n_minus_1], dim=1),
                          torch.cat([H_n_minus_1, -H_n_minus_1], dim=1)], dim=0) / math.sqrt(2)

        
class QQwen2RMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: Qwen2RMSNorm,
    ):
        super().__init__()
        self.originalNorm = originalNorm

    @torch.no_grad()
    def forward(self, hidden_states):
        result = self.originalNorm(hidden_states)


        return result
    
    def to(self, *args, **kwargs):
        super(QQwen2RMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
        return self
    
class QQwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: Qwen2DecoderLayer,
        kv_cache,
        p8_nums,
        p6_nums,
        reorder_index,
        layer_idx
    ):
        super().__init__()
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QQwen2Attention(
            originalLayer.self_attn,
            kv_cache,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        # self.self_attn = originalLayer.self_attn
        self.mlp = QQwen2MLP(
            originalLayer.mlp,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        self.input_layernorm = QQwen2RMSNorm(
            originalLayer.input_layernorm, 
        )
        self.post_attention_layernorm = QQwen2RMSNorm(
            originalLayer.post_attention_layernorm, 
        )
   
        
    def to(self, *args, **kwargs):
        super(QQwen2DecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            # position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class QQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        originalAttn: Qwen2Attention,
        kv_cache,
        p8_nums,
        p6_nums,
        reorder_index,
        i
    ):
        super().__init__()
        self.layer_idx = i
        self.q_kv_cache = kv_cache
        self.config = originalAttn.config
        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta
        self.attention_dropout = originalAttn.attention_dropout
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.q_proj = QLinearLayer(
            originalAttn.q_proj,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')]
        )
        self.k_proj = QLinearLayer(
            originalAttn.k_proj,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')]
        )
        self.v_proj = QLinearLayer(
            originalAttn.v_proj,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')]
        )
        self.o_proj = QLinearLayer(
            originalAttn.o_proj,
            p8_num=p8_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            p6_num=p6_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')]
        )
        self.rotary_emb = originalAttn.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QQwen2Attention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs)
        
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        hidden_states = hidden_states.reshape(bsz*q_len, -1).contiguous().detach()
        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(hidden_states, self.q_reorder_index, self.q_proj.p4_num, self.q_proj.p6_num, self.q_proj.p8_num)
        torch.cuda.synchronize()
        
        hidden_states = (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # kv_seq_len += past_key_value[0].shape[-2]
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # [bsz, nh, t, hd]
    
        if past_key_value is not None:
            # reuse k, v, self_attention
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            
        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=128)
            
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Reorder the BMM output to feed into o.proj
        # if self.act_mean is not None:
        #     # self.act_mean = torch.zeros_like(attn_output[-1])
        #     attn_output -= self.act_mean
          
       
        attn_output = attn_output.reshape(bsz*q_len, -1).contiguous().detach()

        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(attn_output, self.o_reorder_index, self.o_proj.p4_num, self.o_proj.p6_num, self.o_proj.p8_num)
        torch.cuda.synchronize()
        attn_output = (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class QQwen2MLP(nn.Module):
    def __init__(
        self,
        originalMLP: Qwen2MLP,
        p8_nums,
        p6_nums,
        reorder_index,
        i
    ):
        super().__init__()
        
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.gate_proj = QLinearLayer(
            originalMLP.gate_proj,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')]
        )
        self.down_proj = QLinearLayer(
            originalMLP.down_proj,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')]
        )
        self.up_proj = QLinearLayer(
            originalMLP.up_proj,
            p6_num=p6_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            p8_num=p8_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')]
        )
        self.act_fn = originalMLP.act_fn

    def to(self, *args, **kwargs):
        super(QQwen2MLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
      
        return self

   

    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized
#         if self.quant_type == 'fp':
        bsz, q_len, _ = x.shape
        x = x.reshape(bsz*q_len, -1).contiguous().detach()

        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.up_reorder_index, self.up_proj.p4_num, self.up_proj.p6_num, self.up_proj.p8_num)
        torch.cuda.synchronize()
        x = (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Quantize the activations and feed into down_proj

        bsz, q_len, _ = tmpResult.shape
        tmpResult = tmpResult.reshape(bsz*q_len, -1).contiguous().detach()
        if _ > 25000:
            AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.downproj_quantize_w(torch.index_select(tmpResult, 1, self.down_reorder_index.to(torch.int32)), self.down_proj.p4_num, self.down_proj.p6_num, self.down_proj.p8_num)
        else: 
            AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(tmpResult, self.down_reorder_index, self.down_proj.p4_num, self.down_proj.p6_num, self.down_proj.p8_num)
        torch.cuda.synchronize()
        tmpResult = (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
       
        return self.down_proj(tmpResult)
