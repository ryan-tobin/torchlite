"""
Gradient computation utilities
"""

from typing import List, Optional, Tuple 
import numpy as np 

def grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    """
    Compute gradients of outputs with respect to inputs.
    
    Args:
        outputs: Output tensors
        inputs: Input tensors to compute gradients for
        grad_outputs: Gradients w.r.t. outputs (default: ones)
        retain_graph: Keep computation graph after backward
        create_graph: Create graph of gradient computation
        
    Returns:
        Tuple of gradients for each input
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if grad_outputs is None:
        grad_outputs = [np.ones_like(out.data) for out in outputs]

    for output, grad_output in zip(outputs, grad_outputs):
        output.backward(grad_output)

    grads = tuple(inp.grad for inp in inputs)

    if not retain_graph:
        for inp in inputs:
            inp.grad = None 

    return grads 

def backward(tensors, grad_tensors=None, retain_graph=False):
    """
    Compute gradients by backpropagation.
    
    Args:
        tensors: Tensors to start backprop from
        grad_tensors: Gradients w.r.t. tensors
        retain_graph: Keep computation graph after backward
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

        if grad_tensors is None:
            grad_tensors = [np.ones_like(t.data) for t in tensors]

        for tensor, grad_tensor in zip(tensors, grad_tensors):
            tensor.backward(grad_tensor)

def check_gradients(func, inputs, eps=1e-6):
    """
    Numerical gradient checking for debugging.
    
    Args:
        func: Function to check gradients for
        inputs: Input tensors
        eps: Finite difference epsilon
        
    Returns:
        Maximum relative error between analytical and numerical gradients
    """
    outputs = func(*inputs)
    outputs.backward()
    analytical_grads = [inp.grad.copy() for inp in inputs]

    numerical_grads = []
    for i, inp in enumerate(inputs):
        grad = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index

            old_value = inp.data[idx]
            inp.data[idx] = old_value + eps 
            pos = func(*inputs).data.copy()

            inp.data[idx] = old_value - eps 
            neg = func(*inputs).data.copy()

            grad[idx] = (pos - neg) / (2 * eps)

            inp.data[idx] = old_value
            it.iternext()

        numerical_grads.append(grad)

    max_error = 0 
    for analytical, numerical in zip(analytical_grads, numerical_grads):
        rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        max_error = max(max_error, np.max(rel_error))

    return max_error