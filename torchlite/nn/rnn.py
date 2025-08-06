"""Recurrent neural network layers"""

import numpy as np

from ..tensor import Tensor
from .module import Module, Parameter


class RNNCell(Module):
    """Basic RNN cell."""

    def __init__(
            self,
            input_size,
            hidden_size,
            bias=True,
            nonlinearity="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        # Weight matrices
        self.weight_ih = Parameter(
            np.random.randn(
                input_size,
                hidden_size) * 0.01)
        self.weight_hh = Parameter(
            np.random.randn(
                hidden_size,
                hidden_size) * 0.01)

        if bias:
            self.bias_ih = Parameter(np.zeros(hidden_size))
            self.bias_hh = Parameter(np.zeros(hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = Tensor(np.zeros((input.shape[0], self.hidden_size)))

        # Compute new hidden state
        gi = input @ self.weight_ih
        gh = hidden @ self.weight_hh

        if self.bias_ih is not None:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh

        h_new = gi + gh

        if self.nonlinearity == "tanh":
            h_new = Tensor(
                np.tanh(
                    h_new.data),
                requires_grad=h_new.requires_grad)
        elif self.nonlinearity == "relu":
            h_new = h_new.relu()

        return h_new


class RNN(Module):
    """Multi-layer RNN."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        nonlinearity="tanh",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Import here to avoid circular import
        from .container import ModuleList
        from .layers import Dropout

        self.cells = ModuleList(
            [
                RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity)
                for i in range(num_layers)
            ]
        )

        if dropout > 0 and num_layers > 1:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input, h_0=None):
        if self.batch_first:
            # Transpose batch and sequence dimensions
            input = Tensor(np.transpose(input.data, (1, 0, 2)))

        seq_len, batch_size, _ = input.shape

        if h_0 is None:
            h_0 = [Tensor(np.zeros((batch_size, self.hidden_size)))
                   for _ in range(self.num_layers)]

        outputs = []
        h_n = []

        for t in range(seq_len):
            x_t = Tensor(input.data[t])
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
            output = Tensor(np.transpose(output.data, (1, 0, 2)))

        return output, h_n


class LSTMCell(Module):
    """LSTM cell."""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for input, forget, cell, and output gates
        self.weight_ih = Parameter(
            np.random.randn(
                input_size,
                4 * hidden_size) * 0.01)
        self.weight_hh = Parameter(
            np.random.randn(
                hidden_size,
                4 * hidden_size) * 0.01)

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

        # Compute gates
        gates = input @ self.weight_ih + h @ self.weight_hh
        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh

        # Split gates
        i, f, g, o = np.split(gates.data, 4, axis=1)

        # Apply activations
        i = Tensor(1 / (1 + np.exp(-i)))  # sigmoid
        f = Tensor(1 / (1 + np.exp(-f)))  # sigmoid
        g = Tensor(np.tanh(g))  # tanh
        o = Tensor(1 / (1 + np.exp(-o)))  # sigmoid

        # Update cell and hidden states
        c_new = f * c + i * g
        h_new = o * Tensor(np.tanh(c_new.data))

        return h_new, c_new


class LSTM(Module):
    """Multi-layer LSTM."""

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Import here to avoid circular import
        from .container import ModuleList
        from .layers import Dropout

        self.cells = ModuleList(
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                for i in range(num_layers)
            ]
        )

        if dropout > 0 and num_layers > 1:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input, states=None):
        if self.batch_first:
            input = Tensor(np.transpose(input.data, (1, 0, 2)))

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
            x_t = Tensor(input.data[t])
            h_t = list(h_0)
            c_t = list(c_0)

            for layer_idx, cell in enumerate(self.cells):
                h_t[layer_idx], c_t[layer_idx] = cell(
                    x_t, (h_t[layer_idx], c_t[layer_idx]))
                x_t = h_t[layer_idx]

                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(x_t)
            h_0 = h_t
            c_0 = c_t

        output = Tensor(np.stack([o.data for o in outputs]))

        if self.batch_first:
            output = Tensor(np.transpose(output.data, (1, 0, 2)))

        return output, (h_t, c_t)


class GRU(Module):
    """Gated Recurrent Unit - simplified implementation."""

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # For now, just use LSTM as placeholder
        # In a full implementation, would create GRUCell
        self._lstm = LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout)

    def forward(self, input, h_0=None):
        # Simplified - use LSTM and ignore cell state
        output, (h_n, _) = self._lstm(
            input, (h_0, h_0) if h_0 is not None else None)
        return output, h_n
