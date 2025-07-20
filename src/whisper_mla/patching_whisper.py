import inspect
from typing import Callable, Optional, Tuple

import torch
from transformers import Cache
from transformers.utils import logging
from transformers.processing_utils import Unpack
import modeling_whisper
from modeling_whisper import WhisperAttention
from torch import nn


logger = logging.get_logger(__name__)


def custom_WhisperAttention_mla_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    key_value_states: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    kv_shape = (*input_shape, 1, -1)
    num_q_heads = 12
    q_shape = (*input_shape, num_q_heads, -1)
    query_states = self.q_proj(hidden_states) * self.scaling
    if (
        is_cross_attention
        and past_key_value is not None
        and past_key_value[0].shape[1] == key_value_states.shape[1]
    ):
        # reuse k,v, cross_attentions
        kv_states = past_key_value[0]
        key_r_states = past_key_value[1]
        key_c_states, value_states = self.kv_proj.kv_up(kv_states)
        key_states = torch.cat([key_r_states, key_c_states], dim=-1)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
    # NOTE: value_states = self.v_proj(hidden_states)
    elif is_cross_attention:
        kv_states = self.kv_proj.kv_forward(key_value_states)
        key_c_states, value_states = self.kv_proj.kv_up(kv_states)
        key_r_states = self.k_r_proj(key_value_states)
        key_states = torch.cat([key_r_states, key_c_states], dim=-1)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
    elif past_key_value is not None:
        kv_states = self.kv_proj.kv_forward(hidden_states)
        kv_states = torch.cat([past_key_value[0], kv_states], dim=1)
        key_c_states, value_states = self.kv_proj.kv_up(kv_states)
        key_r_states = self.k_r_proj(hidden_states)
        key_r_states = torch.cat([past_key_value[1], key_r_states], dim=1)
        key_states = torch.cat([key_r_states, key_c_states], dim=-1)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
    else:
        kv_states = self.kv_proj.kv_forward(hidden_states)
        key_c_states, value_states = self.kv_proj.kv_up(kv_states)
        key_r_states = self.k_r_proj(hidden_states)
        key_states = torch.cat([key_r_states, key_c_states], dim=-1)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
        
    # NOTE: the code below has not been modified.
    if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (kv_states, key_r_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)
   
    return attn_output, attn_weights_reshaped, past_key_value


def mha2mla_mla_whisper():
    WhisperAttention.forward = custom_WhisperAttention_mla_forward