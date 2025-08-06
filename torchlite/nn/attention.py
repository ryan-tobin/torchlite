"""Attention mechanisms."""

import numpy as np
from .module import Module, Parameter
from ..tensor import Tensor
import math


class MultiheadAttention(Module):
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        # Import here to avoid circular imports
        from .layers import Linear, Dropout

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, query, key, value, mask=None):
        batch_size, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q_data = Q.data.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_data = K.data.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_data = V.data.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = Tensor(Q_data, requires_grad=Q.requires_grad)
        K = Tensor(K_data, requires_grad=K.requires_grad)
        V = Tensor(V_data, requires_grad=V.requires_grad)

        # Compute attention scores
        scores = (Q @ Tensor(K.data.transpose(-2, -1))) * self.scaling

        if mask is not None:
            scores_data = scores.data
            scores_data[mask == 0] = -1e9
            scores = Tensor(scores_data, requires_grad=scores.requires_grad)

        # Apply softmax
        from .activations import Softmax

        attn_weights = Softmax(dim=-1)(scores)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ V

        # Reshape back
        attn_output_data = attn_output.data.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
        attn_output = Tensor(attn_output_data, requires_grad=attn_output.requires_grad)

        # Final projection
        output = self.out_proj(attn_output)

        return output
