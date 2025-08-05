from .optimizer import Optimizer
import numpy as np 

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0, wieght_decay: float = 0):
        defaults = dict(lr=lr, momentum=momentum, wieght_decay=wieght_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue 

                grad = param.grad 
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * param.data

                if group['momentum'] != 0:
                    if param not in self.state:
                        self.state[param] = {}

                    if 'momentum_buffer' not in self.state[param]:
                        buf = self.state[param]['momentum_buffer'] = np.zeros_like(grad)
                    else:
                        buf = self.state[param]['momentum_buffer']

                    buf = group['momentum'] * buf + grad 
                    grad = buf 

                param.data -= group['lr'] * grad 

