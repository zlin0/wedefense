"""
JIT-compatible MultiheadAttention implementation.

This module provides a JIT-compatible version of MultiheadAttention
that removes internal function definitions for torch.jit.script compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FairseqDropout(nn.Module):

    def __init__(self, p: float, module_name: Optional[str] = None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        return x


def softmax_supporting_onnx_trace(x: torch.Tensor,
                                  dim: int,
                                  onnx_trace: bool = False) -> torch.Tensor:
    """JIT-compatible softmax function (moved outside class to avoid function definition issues).

    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        onnx_trace: Whether to use ONNX-compatible softmax

    Returns:
        Softmax output tensor
    """
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class JITCompatibleMultiheadAttention(nn.Module):
    """JIT-compatible MultiheadAttention matching fairseq behavior."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.onnx_trace = False

    def _pad_masks(
        self,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat(
                [attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat(
                [key_padding_mask,
                 key_padding_mask.new_zeros(shape)], dim=-1)
        return key_padding_mask, attn_mask

    def _add_bias(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        bsz: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        assert self.bias_k is not None and self.bias_v is not None
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def _append_zero_attn(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        k = torch.cat(
            [k,
             torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)],
            dim=-2)
        v = torch.cat(
            [v,
             torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)],
            dim=-2)
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int,
                          src_len: int, bsz: int) -> torch.Tensor:
        return attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ):
        tgt_len, bsz, embed_dim = query.size()

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            assert key is not None
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q * self.scaling

        if self.bias_k is not None:
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k,
                v=v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len,
                                              bsz)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len)
            attn_weights = attn_weights.view(bsz, -1, self.num_heads, tgt_len,
                                             src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(
                    torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        attn_weights_float = softmax_supporting_onnx_trace(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0,
                                  1).contiguous().view(tgt_len, bsz,
                                                       self.embed_dim)
        attn = self.out_proj(attn)

        attn_weights_out: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out
