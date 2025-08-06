"""Embedding Layer"""

import numpy as np 
from .module import Module, Parameter
from ..tensor import Tensor 

class Embedding(Module):
    """Embedding layer for discrete inputs."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)

    def forward(self, input):
        return Tensor(self.weight.datap[input.data], requires_grad = self.requires_grad)
    
