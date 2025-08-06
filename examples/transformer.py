"""
Transformer implementation using TorchLite.
Demonstrates how to build complex architectures.
"""
import numpy as np
import torchlite as tl
import torchlite.nn as nn
import torchlite.optim as optim
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tl.Tensor(pe[np.newaxis, :, :])
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        return x + tl.Tensor(self.pe.data[:, :seq_len, :])

class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerClassifier(nn.Module):
    """Transformer for sequence classification."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_len, n_classes, dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.encoder(x, mask)
        
        # Global average pooling
        x = x.mean(axis=1)
        
        # Classification
        return self.classifier(x)

def create_padding_mask(sequences, pad_idx=0):
    """Create padding mask for sequences."""
    return (sequences == pad_idx).astype(np.float32)

def generate_synthetic_data(n_samples, seq_len, vocab_size):
    """Generate synthetic sequence data."""
    # Random sequences
    sequences = np.random.randint(1, vocab_size, (n_samples, seq_len))
    
    # Random labels (binary classification)
    labels = np.random.randint(0, 2, n_samples)
    
    # Add some padding
    for i in range(n_samples):
        pad_len = np.random.randint(0, seq_len // 2)
        if pad_len > 0:
            sequences[i, -pad_len:] = 0
    
    return sequences, labels

def main():
    """Train a transformer classifier."""
    # Hyperparameters
    vocab_size = 1000
    d_model = 128
    n_heads = 8
    n_layers = 6
    d_ff = 512
    max_len = 100
    n_classes = 2
    dropout = 0.1
    
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 5
    
    print("Creating synthetic sequence data...")
    # Generate data
    X_train, y_train = generate_synthetic_data(1000, max_len, vocab_size)
    X_test, y_test = generate_synthetic_data(200, max_len, vocab_size)
    
    # Create model
    print("Initializing transformer model...")
    model = TransformerClassifier(
        vocab_size, d_model, n_heads, n_layers, 
        d_ff, max_len, n_classes, dropout
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print("Starting training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        n_batches = len(X_train) // batch_size
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Create tensors
            x = tl.Tensor(batch_x)
            y = tl.Tensor(batch_y)
            
            # Create padding mask
            mask = create_padding_mask(batch_x)
            
            # Forward pass
            output = model(x, mask)
            loss = criterion(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
        
        avg_loss = total_loss / n_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()