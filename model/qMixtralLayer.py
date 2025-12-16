import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from tqdm import tqdm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm, MixtralAttention, MixtralSparseMoeBlock, MixtralBlockSparseTop2MLP
from qLinearLayer import QLinearLayer

import sys
sys.path.append('./mgemm/build/')
import mixedgemm

@torch.no_grad()
def quantize_int_group(w, nbits, group_size):
    savedShape = w.shape
    w = w.reshape(-1, group_size)
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    q_max = (2**(nbits)-1)
    q_min = (0)
    scales = (w_max-w_min).clamp(min=1e-5) / q_max
    base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    return w.reshape(savedShape)

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

class QMixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: MixtralDecoderLayer,
        kv_cache,
        p8_nums,
        p6_nums,
        reorder_index,
        layer_idx
    ):
        super().__init__()
       
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QMixtralAttention(
            originalLayer.self_attn,
            kv_cache,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        # self.self_attn = originalLayer.self_attn
        self.block_sparse_moe = QMixtralSparseMoeBlock(
            originalLayer.block_sparse_moe,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx
        )
        # self.mlp = originalLayer.mlp
        self.input_layernorm = QMixtralRMSNorm(
            originalLayer.input_layernorm, 
            
        )
        self.post_attention_layernorm = QMixtralRMSNorm(
            originalLayer.post_attention_layernorm, 
            
        )

    def to(self, *args, **kwargs):
        super(QMixtralDecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.block_sparse_moe = self.block_sparse_moe.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings = None
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
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
            
        return outputs
    
   
        
class QMixtralRMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: MixtralRMSNorm,
    ):
        super().__init__()
        self.originalNorm = originalNorm
    

       
    @torch.no_grad()
    def forward(self, hidden_states):
        result = self.originalNorm(hidden_states)
            
#         if self.args.abits < 16:
#             result = self.act_quant(result)
        
        
        return result
    
    def to(self, *args, **kwargs):
        super(QMixtralRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
       
        return self

class QMixtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        originalAttn: MixtralAttention,
        kv_cache,
        p8_nums,
        p6_nums,
        reorder_index,
        i
    ):
        super().__init__()
        
        self.q_kv_cache = kv_cache
        self.config = originalAttn.config
        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta
        self.layer_idx = i
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
        self.register_buffer('q_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')].to(torch.int16))
        self.register_buffer('o_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')].to(torch.int16))
        
        self.rotary_emb = originalAttn.rotary_emb
        self.attention_dropout=originalAttn.attention_dropout

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QMixtralAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs)
        self.q_reorder_index = self.q_reorder_index.to(*args, **kwargs)
        self.o_reorder_index = self.o_reorder_index.to(*args, **kwargs)
      
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
         
        else:
            cos, sin = position_embeddings

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Quantize the attention output
      
        attn_output = attn_output.reshape(bsz*q_len, -1).contiguous().detach()

        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(attn_output, self.o_reorder_index, self.o_proj.p4_num, self.o_proj.p6_num, self.o_proj.p8_num)
        torch.cuda.synchronize()
        attn_output = (AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

class QMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self, 
        originalSparseMoeBlock: MixtralSparseMoeBlock,
        p8_nums,
        p6_nums,
        reorder_index,
        i
    ):
        super().__init__()
        self.hidden_dim = originalSparseMoeBlock.hidden_dim
        self.ffn_dim = originalSparseMoeBlock.ffn_dim
        self.num_experts = originalSparseMoeBlock.num_experts
        self.top_k = originalSparseMoeBlock.top_k



        nameTemplate = 'layers.{}.{}.{}.{}'
        self.gate = originalSparseMoeBlock.gate

        self.experts = originalSparseMoeBlock.experts

        for j in range(self.num_experts):
            self.experts[j] = QMixtralBlockSparseTop2MLP(originalSparseMoeBlock.experts[j], p8_nums, p6_nums, reorder_index, i, j)

        # Jitter parameters
        # self.jitter_noise = originalSparseMoeBlock.router_jitter_noise

    def to(self, *args, **kwargs):
        super(QMixtralSparseMoeBlock, self).to(*args, **kwargs)
        self.gate = self.gate.to(*args, **kwargs)
        self.experts = self.experts.to(*args, **kwargs)
    
        
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:  # numel() 返回元素总数，0表示空
                continue  
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
class QMixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, 
                originalBlock: MixtralBlockSparseTop2MLP,
                p8_nums,
                p6_nums,
                reorder_index,
                layer_idx,
                moe_idx
            ):
        super().__init__()
        self.ffn_dim = originalBlock.ffn_dim
        self.hidden_dim = originalBlock.hidden_dim

        nameTemplate = 'layers.{}.{}.{}.{}.{}.{}'
        self.w1 = QLinearLayer(
            originalBlock.w1,
            p8_num=p8_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w1', 'input')],
            p6_num=p6_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w1', 'input')],
            reorder_index=reorder_index[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w1', 'input')]
        )
        self.w3 = QLinearLayer(
            originalBlock.w3,
            p8_num=p8_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w3', 'input')],
            p6_num=p6_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w3', 'input')],
            reorder_index=reorder_index[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w3', 'input')]
        )
        self.w2 = QLinearLayer(
            originalBlock.w2,
            p8_num=p8_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w2', 'input')],
            p6_num=p6_nums[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w2', 'input')],
            reorder_index=reorder_index[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w2', 'input')]
        )
        self.register_buffer('w1_reorder_index', reorder_index[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w1', 'input')].to(torch.int16))
        self.register_buffer('w2_reorder_index', reorder_index[nameTemplate.format(layer_idx, 'block_sparse_moe', 'experts', moe_idx, 'w2', 'input')].to(torch.int16))

        self.act_fn = originalBlock.act_fn
        
    def to(self, *args, **kwargs):
        super(QMixtralBlockSparseTop2MLP, self).to(*args, **kwargs)
        self.w1 = self.w1.to(*args, **kwargs)
        self.w2 = self.w2.to(*args, **kwargs)
        self.w3 = self.w3.to(*args, **kwargs)
        self.w2_reorder_index = self.w2_reorder_index.to(*args, **kwargs)
        self.w1_reorder_index = self.w1_reorder_index.to(*args, **kwargs)
        
        return self

    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized
        q_len, _ = x.shape
        # x = x.reshape(bsz*q_len, -1).contiguous().detach()

        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.w1_reorder_index, self.w1.p4_num, self.w1.p6_num, self.w1.p8_num)
        torch.cuda.synchronize()
        x = (AN, AS, AO, SFAN, SFAS, SFAO, None, q_len)
        tmpResult = self.act_fn(self.w1(x)) * self.w3(x)
        # Quantize the activations and feed into down_proj
        # bsz, q_len, _ = tmpResult.shape
        # tmpResult = tmpResult.reshape(bsz*q_len, -1).contiguous().detach()

        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(tmpResult, self.w2_reorder_index, self.w2.p4_num, self.w2.p6_num, self.w2.p8_num)
        torch.cuda.synchronize()
        tmpResult = (AN, AS, AO, SFAN, SFAS, SFAO, None, q_len)
       
        return self.w2(tmpResult)