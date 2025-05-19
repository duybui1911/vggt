# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import time
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention
XFORMERS_AVAILABLE = False

def chunked_scaled_dot_product_attention(q, k, v, chunk_size, dropout_p=0.0):
    batch_size, n_heads, seq_len, dim = q.shape
    output = torch.zeros_like(q)
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        q_chunk = q[:, :, start_idx:end_idx, :]
        chunk_output = F.scaled_dot_product_attention(
            q_chunk, k, v, dropout_p=dropout_p
        )
        output[:, :, start_idx:end_idx, :] = chunk_output
    return output

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        start_attn_event = torch.cuda.Event(enable_timing=True)
        start_ffw_event = torch.cuda.Event(enable_timing=True)
        end_ffw_event = torch.cuda.Event(enable_timing=True)
        start_qkv_event = torch.cuda.Event(enable_timing=True)
        start_scaleattn_event = torch.cuda.Event(enable_timing=True)
        end_scaleattn_event = torch.cuda.Event(enable_timing=True)
        end_prj_event = torch.cuda.Event(enable_timing=True)

        start_attn_event.record()

        B, N, C = x.shape
        chunking = True
        chunk_size = 768
        if chunking:
            start_ffw_event.record()
            x = x.view(1, B * N, C)
            chunks = x.split(chunk_size, dim=1)
            qkv_chunks = []
            for chunk in chunks:
                qkv_chunk = self.qkv(chunk)
                qkv_chunks.append(qkv_chunk)
            end_ffw_event.record()
            qkv = torch.cat(qkv_chunks, dim=1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        start_qkv_event.record()
        if self.rope is not None:
            q = self.rope(q, pos).to(v.dtype)
            k = self.rope(k, pos).to(v.dtype)

        start_scaleattn_event.record()
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        end_scaleattn_event.record()
        x = x.transpose(1, 2).reshape(B, N, C)
        if chunking:
            x = x.view(1, B * N, C)
            chunks = x.split(chunk_size, dim=1)
            state_chunks = []
            for chunk in chunks:
                state_chunk = self.proj(chunk)
                state_chunks.append(state_chunk)
            x = torch.cat(state_chunks, dim=1)
            x = x.reshape(B, N, C)
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        end_prj_event.record()

        torch.cuda.synchronize()

        time_attn = start_attn_event.elapsed_time(end_prj_event) / 1000.0
        time_ffw = start_ffw_event.elapsed_time(end_ffw_event) / 1000.0 if chunking else 0.0
        time_qkv = start_qkv_event.elapsed_time(start_scaleattn_event) / 1000.0
        time_scaleattn = start_scaleattn_event.elapsed_time(end_scaleattn_event) / 1000.0
        time_prj = end_scaleattn_event.elapsed_time(end_prj_event) / 1000.0

        if chunking:
            print(f"Time ffw: {time_ffw}")
        print(f"Time attn: {time_attn}, Time scaleattn: {time_scaleattn}, Time prj: {time_prj}, Time qkv: {time_qkv}")

        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = torch.unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
