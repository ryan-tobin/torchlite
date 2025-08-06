# Replace the content of examples/mnist_cnn.py with this fixed version

"""
MNIST CNN example using TorchLite.
Demonstrates basic CNN training on MNIST dataset.
"""
import numpy as np
import torchlite as tl
import torchlite.nn as nn
import torchlite.optim as optim
from torchlite.data import DataLoader, TensorDataset
from torchlite.data.transforms import Compose, ToTensor, Normalize
import time

# For demonstration, we'll create synthetic MNIST-like data
def create_synthetic_mnist(n_samples=1000):
    """Create synthetic MNIST-like data for demonstration."""
    X = np.random.randn(n_samples, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, n_samples)
    return X, y

class ConvNet(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # Separate ReLU instance
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()  # Separate ReLU instance
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()  # Separate ReLU instance
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_epoch(model, dataloader, criterion, optimizer, device=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Convert to tensors
        data = tl.Tensor(data, requires_grad=True)
        target = tl.Tensor(target)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.data
        pred = np.argmax(output.data, axis=1)
        correct += (pred == target.data).sum()
        total += target.shape[0]
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device=None):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in dataloader:
        data = tl.Tensor(data)
        target = tl.Tensor(target)
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.data
        pred = np.argmax(output.data, axis=1)
        correct += (pred == target.data).sum()
        total += target.shape[0]
    
    return total_loss / len(dataloader), correct / total

def main():
    """Main training script."""
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    print("Creating synthetic MNIST data...")
    # Create synthetic data
    X_train, y_train = create_synthetic_mnist(6000)
    X_test, y_test = create_synthetic_mnist(1000)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(tl.Tensor(X_train), tl.Tensor(y_train))
    test_dataset = TensorDataset(tl.Tensor(X_test), tl.Tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    print("Initializing model...")
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    print("Training completed!")

if __name__ == "__main__":
    main()