"""Attention mechanisms."""

import numpy as np 
from .module import Module, Parameter
from ..tensor import Tensor 
import math 
import nn

class MultiheadAttention(Module):
    """Multi-head attention mechanism"""

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None 

    def forward(self, query, key, value, mask=None):
        batch_size, tgt_len, embed_dim = query.shape 
        src_len = key.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2, -1)) * self.scaling

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = nn.Softmax(dim=-1)(scores)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ V 

        attn_output = attn_output.transpose(1,2).reshape(batch_size, tgt_len, embed_dim)

        output = self.out_proj(attn_output)

        return output 