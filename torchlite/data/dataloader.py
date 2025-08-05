from typing import Iterator, Optional
import numpy as np 
from .dataset import Dataset 

class DataLoader:
    """Data loader for batching and shuffling."""

    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)

            if self.drop_last and end_idx - start_idx < self.batch_size:
                break 

            batch_indices = indices[start_idx:end_idx]
            batch = [self.dataset[i] for i in batch_indices]

            yield self._collate_batch(batch)
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def _collate_batch(self, batch):
        """Collate a list of samples into a batch"""
        if isinstance(batch[0], tuple):
            return tuple(np.stack([item[i] for item in batch]) for i in range(len(batch[0])))
        else:
            return np.stack(batch)
        
# Example usage demonstration
if __name__ == "__main__":
    import torchlite as tl 

    class SimpleNet(tl.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tl.nn.Linear(784, 128)
            self.relu = tl.nn.ReLU()
            self.fc2 = tl.nn.Linear(128, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x 
        
    # Create model and optimizer
    model = SimpleNet()
    optimizer = tl.optim.Adam(model.parameters(), lr=0.001)
    criterion = tl.nn.CrossEntropyLoss()

    # Training loop example
    for epoch in range(10):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.data}")
