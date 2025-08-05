"""
Dataset classes for data loading.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union 
import numpy as np 
from ..tensor import Tensor 

class Dataset(ABC):
    """Abstract base class for datasets"""

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset"""
        pass 

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get a single item from the dataset."""
        pass 

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        """Concatenate two datasets"""
        return ConcatDataset([self, other])
    
class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.
    Each sample will be retreived by indexing tensors along the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors 

    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        return tuple(tensor.data[index] for tensor in self.tensors)
    
class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets 
        self.cumulative_sizes = []
        cumsum = 0
        for d in datasets:
            cumsum += len(d)
            self.cumulative_sizes.append(cumsum)

    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Index out of range")
            idx = len(self) + idx 

        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                if i == 0:
                    dataset_idx = idx 
                else:
                    dataset_idx = idx - self.cumulative_sizes[i - 1]
                return self.datasets[i][dataset_idx]
            
        raise IndexError("Index out of range")
    
class Subset(Dataset):
    """
    Subset of a dataset at a specified indicies.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
class Sampler(ABC):
    """Base class for all samplers"""

    @abstractmethod
    def __iter__(self):
        pass 

    @abstractmethod
    def __len__(self):
        pass 

class SequentialSampler(Sampler):
    """Samples elements sequentially"""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source)

class RandomSampler(Sampler):
    """Samples elements randomly"""

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

    @property        
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    
    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(np.random.randint(0, n, size=self.num_samples))
        else:
            return iter(np.random.permutation(n)[:self.num_samples])
        
    def __len__(self):
        return self.num_samples
    
class BatchSampler(Sampler):
    """Wraps another sampler to yield mini-batches"""

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch 
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch 
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        