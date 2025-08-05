"""
Learning rate schedulers
"""

from abc import ABC, abstractmethod
import math 
import numpy as np 

class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.step()

    @abstractmethod
    def get_lr(self):
        """Compute learning rate."""
        pass 

    def step(self, epoch=None):
        """Update learning rates."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr 

class StepLR(LRScheduler):
    """Decay learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma 
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
    
class ExponentialLR(LRScheduler):
    """Decay learning rate by gamme every epoch"""

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma 
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
    
class CosineAnnealingLR(LRScheduler):
    """Cosine annealing schedule"""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) * 
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
               (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
               (group['lr'] - self.eta_min) + self.eta_min
               for group in self.optimizer.param_groups]
    
class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric has stopped improving"""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0):
        self.mode = mode 
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best = None 
        self.num_bad_epochs = 0
        super().__init__(optimizer, last_epoch=0)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, metric):
        """Update based on metric."""
        if self.best is None:
            self.best = metric
        elif self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
            self.num_bad_epochs = 0
    
    def _is_better(self, a, b):
        if self.mode == 'min':
            return a < b - self.threshold
        else:
            return a > b + self.threshold
        
class OneCycleLR(LRScheduler):
    """One cycle learning rate schedule"""

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 anneal_strategy='cos', div_factor=25., final_div_factor=1e4):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = self.max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

        super().__init__(optimizer, last_epoch=1)

    def get_lr(self):
        step_num = self.last_epoch

        if step_num < self.total_steps * self.pct_start:
            pct = step_num / (self.total_steps * self.pct_start)
            return [self.initial_lr + pct * (self.max_lr - self.initial_lr)
                    for _ in self.optimizer.param_groups]
        else:
            pct = (step_num - self.total_steps * self.pct_start) / \
                    (self.total_steps * (1 - self.pct_start))
            
            if self.anneal_strategy == 'cos':
                return [self.min_lr + (self.max_lr - self.min_lr) *
                        (1 * math.cos(math.pi * pct)) / 2
                        for _ in self.optimizer.param_groups]
            else:
                return [self.max_lr - pct * (self.max_lr - self.min_lr)
                        for _ in self.optimizer.param_groups]