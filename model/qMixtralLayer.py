import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from tqdm import tqdm
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm, MixtralAttention, MixtralSparseMoeBlock, MixtralBlockSparseTop2MLP
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import sys
sys.path.append('./mgemm/build/')
import mixedgemm

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        p8_num, 
        p6_num,
        reorder_index,
        out_reorder_index=None,
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.p6_num = p6_num  #p4_num, p6_num, p8_num需要整除128
        self.p8_num = p8_num
        self.p4_num = self.in_features - p6_num -p8_num
       
       
       
        out_features, in_features = originalLayer.weight.data.shape

        # micromix
        self.reorder_index = reorder_index.to(torch.int16).cuda()

        
        self.BN, self.BS, self.BO, self.SFBN, self.SFBS, self.SFBO = mixedgemm.reorder_quantize_w4(originalLayer.weight.data, self.reorder_index, self.p4_num, self.p6_num, self.p8_num)

        # self.BN_d, self.BS_d, self.BO_d, self.SFBN_d, self.SFBS_d, self.SFBO_d = mixedgemm.reorder_quantize_w4(originalLayer.weight.data, self.reorder_index, 0, 0, self.in_features)
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 3:
            bsz, q_len, _ = x.shape
            x = x.reshape(bsz*q_len, -1).contiguous() 
        else: 
            bsz = None
            q_len, _ = x.shape
        if q_len == 1:
            AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.reorder_index, 0, 0, self.in_features)
        else:
            AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.reorder_index, self.p4_num, self.p6_num, self.p8_num)
        y = mixedgemm.matmul(AN, self.BN, AS, self.BS, AO, self.BO, SFAN, self.SFBN, SFAS, self.SFBS, SFAO, self.SFBO)
        if self.bias is not None:
            y = y + self.bias
        if bsz is not None:
            y = y.reshape(bsz, q_len, -1)
        return y



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

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
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
        layer_idx,
        **kwargs
    ):
        super().__init__()
       
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QMixtralAttention(
            originalLayer.self_attn,
            kv_cache,
            p8_nums=p8_nums,
            p6_nums=p6_nums,
            reorder_index=reorder_index,
            i=layer_idx,
            **kwargs
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_embeddings = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
            
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

        self.head_dim = originalAttn.head_dim
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.layer_idx = i
        self.scaling = self.head_dim**-0.5
        
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
     
        self.attention_dropout=originalAttn.attention_dropout

    def to(self, *args, **kwargs):
        super(QMixtralAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
      
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
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
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
        self.act_fn = originalBlock.act_fn
        
    def to(self, *args, **kwargs):
        super(QMixtralBlockSparseTop2MLP, self).to(*args, **kwargs)
        self.w1 = self.w1.to(*args, **kwargs)
        self.w2 = self.w2.to(*args, **kwargs)
        self.w3 = self.w3.to(*args, **kwargs)
        
        return self

    @torch.no_grad()
    def forward(self, x):
        tmpResult = self.act_fn(self.w1(x)) * self.w3(x)
       
        return self.w2(tmpResult)