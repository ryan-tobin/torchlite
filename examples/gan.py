"""
GAN (Generative Adversarial Network) example using TorchLite.
Demonstrates how to train competing networks.
"""
import numpy as np
import torchlite as tl
import torchlite.nn as nn
import torchlite.optim as optim
import matplotlib.pyplot as plt

class Generator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.reshape(img.shape[0], *self.img_shape)
        return img

class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.reshape(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

def train_gan(generator, discriminator, data, n_epochs, latent_dim, batch_size=64):
    """Train GAN."""
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function - using BCELoss
    def bce_loss(pred, target):
        # Simple binary cross entropy
        eps = 1e-8
        loss = -(target.data * np.log(pred.data + eps) + 
                (1 - target.data) * np.log(1 - pred.data + eps))
        return tl.Tensor(np.mean(loss), requires_grad=pred.requires_grad)
    
    n_samples = len(data)
    n_batches = n_samples // batch_size
    
    for epoch in range(n_epochs):
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_size_actual = end_idx - start_idx
            
            real_imgs = tl.Tensor(data[start_idx:end_idx])
            
            # Adversarial ground truths
            valid = tl.Tensor(np.ones((batch_size_actual, 1)), requires_grad=True)
            fake = tl.Tensor(np.zeros((batch_size_actual, 1)), requires_grad=True)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Sample noise as generator input
            z = tl.Tensor(np.random.normal(0, 1, (batch_size_actual, latent_dim)), requires_grad=True)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Loss measures discriminator's ability to classify real from generated samples
            real_loss = bce_loss(discriminator(real_imgs), valid)
            fake_loss = bce_loss(discriminator(gen_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = tl.Tensor(np.random.normal(0, 1, (batch_size_actual, latent_dim)), requires_grad=True)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = bce_loss(discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            if i % 50 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{n_batches}] "
                      f"[D loss: {d_loss.data:.4f}] [G loss: {g_loss.data:.4f}]")

def generate_samples(generator, n_samples, latent_dim):
    """Generate samples from trained generator."""
    generator.eval()
    z = tl.Tensor(np.random.normal(0, 1, (n_samples, latent_dim)))
    gen_imgs = generator(z)
    return gen_imgs.data

def main():
    """Main GAN training script."""
    # Configuration
    img_shape = (1, 28, 28)  # MNIST-like images
    latent_dim = 100
    n_epochs = 50
    batch_size = 64
    
    # Create synthetic image data (normally you'd load real data)
    print("Creating synthetic image data...")
    n_samples = 6000
    # Generate simple patterns as "real" images
    real_images = []
    for _ in range(n_samples):
        # Create random patterns
        img = np.random.randn(*img_shape) * 0.1
        # Add some structure
        if np.random.rand() > 0.5:
            # Horizontal line
            img[0, 14, :] = 1
        else:
            # Vertical line
            img[0, :, 14] = 1
        real_images.append(img)
    
    real_images = np.array(real_images).astype(np.float32)
    
    # Initialize networks
    print("Initializing GAN networks...")
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape)
    
    # Train GAN
    print("Starting GAN training...")
    train_gan(generator, discriminator, real_images, n_epochs, latent_dim, batch_size)
    
    # Generate samples
    print("Generating samples...")
    samples = generate_samples(generator, 16, latent_dim)
    
    # Visualize generated samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Samples')
    plt.savefig('gan_samples.png')
    print("Samples saved to gan_samples.png")

if __name__ == "__main__":
    main()