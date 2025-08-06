"""Recurrent neural network layers"""
import numpy as np 
from .module import Module, Parameter 
from ..tensor import Tensor 
import nn

class RNNCell(Module):
    """Basic RNN cell."""

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        self.weight_ih = Parameter(np.random.randn(input_size, hidden_size) * 0.01)
        self.weight_hh = Parameter(np.random.randn(hidden_size, hidden_size) * 0.01)

        if bias:
            self.bias_ih = Parameter(np.zeros(hidden_size))
            self.bias_hh = Parameter(np.zeros(hidden_size))
        else:
            self.bias_hh = None 
            self.bias_ih = None 
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = Tensor(np.zeros((input.shape[0], self.hidden_size)))

        gi = input @ self.weight_ih 
        gh = hidden @ self.weight_hh 

        if self.bias is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh

        h_new = gi + gh 

        if self.nonlinearity == 'tanh':
            h_new = h_new.tanh()
        elif self.nonlinearity == 'relu':
            h_new = h_new.relu()

        return h_new 

class RNN(Module):
    """Multi-layer RNN"""

    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True,
                 batch_first = False, dropout=0., nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity)
            for i in range(num_layers)
        ])

        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None 

    def forard(self, input, h_0 = None):
        if self.batch_first:
            input = input.transpose(0,1)

        seq_len, batch_size, _ = input.shape 

        if h_0 is None:
            h_0 = [Tensor(np.zeros((batch_size, self.hidden_size)))
                   for _ in range(self.num_layers)]
        
        outputs = []
        h_n = []

        for t in range(seq_len):
            x_t = input[t]
            h_t = list(h_0)

            for layer_idx, cell in enumerate(self.cells):
                h_t[layer_idx] = cell(x_t, h_t[layer_idx])
                x_t = h_t[layer_idx]

                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(x_t)
            h_0 = h_t 

        h_n = h_t 
        output = Tensor(np.stack([o.data for o in outputs]))

        if self.batch_first:
            output = output.transpose(0,1)

        return output, h_n 
    
class LSTMCell(Module):
    """LSTM cell."""
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices for input, forget, cell, and output gates
        self.weight_ih = Parameter(np.random.randn(input_size, 4 * hidden_size) * 0.01)
        self.weight_hh = Parameter(np.random.randn(hidden_size, 4 * hidden_size) * 0.01)
        
        if bias:
            self.bias_ih = Parameter(np.zeros(4 * hidden_size))
            self.bias_hh = Parameter(np.zeros(4 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input, states=None):
        if states is None:
            h = Tensor(np.zeros((input.shape[0], self.hidden_size)))
            c = Tensor(np.zeros((input.shape[0], self.hidden_size)))
        else:
            h, c = states 

        gates = input @ self.weight_ih + h @ self.weight_hh 
        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh 

        i, f, g, o = np.split(gates.data, 4, axis=1)

        i = Tensor(1 / (1 + np.exp(-i)))
        f = Tensor(1 / (1 + np.exp(-f)))
        g = Tensor(np.tanh(g))
        o = Tensor(1 / (1 + np.exp(-o)))

        c_new = f * c + i * g
        h_new = o * Tensor(np.tanh(c_new.data))

        return h_new, c_new 
    
class LSTM(Module):
    """Multi-layer LSTM"""

    def __init__(self, input_size, hidden_size, num_layers=1,bias=True,
                 batch_first=False, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
            for i in range(num_layers)
        ])

        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None 

    def forward(self, input, states=None):
        if self.batch_first:
            input = input.transpose(0,1)

        seq_len, batch_size, _ = input.shape 

        if states is None:
            h_0 = [Tensor(np.zeros((batch_size, self.hidden_size)))
                   for _ in range(self.num_layers)]
            c_0 = [Tensor(np.zeros((batch_size, self.hidden_size)))
                   for _ in range(self.num_layers)]
        else:
            h_0, c_0 = states 

        outputs = []

        for t in range(seq_len):
            x_t = input[t]
            h_t = list(h_0)
            c_t = list(c_0)

            for layer_idx, cell in enumerate(self.cells):
                h_t[layer_idx], c_t[layer_idx] = cell(x_t, (h_t[layer_idx], c_t[layer_idx]))
                x_t = h_t[layer_idx]

                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(x_t)
            h_0 = h_t 
            c_0 = c_t 
        
        output = Tensor(np.stack([o.data for o in outputs]))

        if self.batch_first:
            output = output.transpose(0,1)

        return output, (h_t, c_t)
    
class GRU(Module):
    """Gated Recurrent Unit."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
            