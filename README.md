# TorchLite 

A lightweight, educational deep learning framework with a PyTorch-like API. Built for learning and experimentation.

## Features

- **Automatic Differentiation**: Full autograd support with dynamic computation graphs
- **Neural Network Modules**: Familiar API similar to PyTorch
- **Optimizers**: SGD, Adam, and more
- **GPU Support**: Optional CUDA acceleration (via CuPy)
- **Data Pipeline**: Dataset and DataLoader abstractions
- **Model Utilities**: Summary, visualization, and serialization

## Installation
```bash 
pip install torchlite
```

For GPU support:
```bash 
pip install torchlite[cuda]
```

## Quick Start

```python
import torchlite as tl 
import torchlite.nn as nn 
import torchlite.optim as optim 

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model, loss, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.data}')
```

## Documentation 
Full documentation available at [https://github.com/ryan-tobin/torchlite/docs](https://github.com/ryan-tobin/torchlite/docs)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.