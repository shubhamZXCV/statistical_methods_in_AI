from model import Linear, Model, MSELoss, ReLU, Identity
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

SEED = 42  
np.random.seed(SEED)
USERNAME = "shubham.goel"

class Encoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, learning_rate=0.01):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space (bottleneck)
            learning_rate: Learning rate for all layers
        """
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim, ReLU(), learning_rate))
            prev_dim = dim
            
        # Add bottleneck layer
        layers.append(Linear(prev_dim, latent_dim, Identity(), learning_rate))
        
        self.model = Model(layers, MSELoss())

    def forward(self, x):
        return self.model.forward(x)

class Decoder:
    def __init__(self, latent_dim, hidden_dims, output_dim, learning_rate=0.01):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (in reverse order)
            output_dim: Dimension of output (same as input_dim)
            learning_rate: Learning rate for all layers
        """
        layers = []
        prev_dim = latent_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.append(Linear(prev_dim, dim, ReLU(), learning_rate))
            prev_dim = dim
            
        # Add output layer
        layers.append(Linear(prev_dim, output_dim, Identity(), learning_rate))
        
        self.model = Model(layers, MSELoss())

    def forward(self, x):
        return self.model.forward(x)

class Autoencoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, learning_rate=0.01):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Dimension of input/output data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for all layers
        """
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, learning_rate)
        # Reverse hidden_dims for decoder
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim, learning_rate)
        self.model = Model(
            self.encoder.model.layers + self.decoder.model.layers,
            MSELoss()
        )

    def encode(self, x):
        """Convert input to latent representation."""
        return self.encoder.forward(x)

    def decode(self, z):
        """Reconstruct input from latent representation."""
        return self.decoder.forward(z)

    def forward(self, x):
        """Full forward pass: encode then decode."""
        return self.model.forward(x)

    def train(self, x, y=None):
        """Train the autoencoder using input data as targets."""
        if y is None:
            y = x
        return self.model.train(x, y)

    def save(self, path):
        """Save the autoencoder model."""
        self.model.save_to(path)

    def load(self, path):
        """Load the autoencoder model."""
        self.model.load_from(path)

def train_autoencoder(
    autoencoder,
    dataset,
    batch_size=32,
    grad_accumulation_steps=1,
    max_epochs=10000,
    patience=10,
    relative_loss_threshold=1e-4,
    run_name=None,
    save_model=False,
    use_wandb=False
):
    """
    Train an autoencoder model with early stopping and progress tracking.
    
    Args:
        autoencoder: The autoencoder model to train
        dataset: Training data (samples to reconstruct)
        batch_size: Size of training batches
        grad_accumulation_steps: Number of batches to accumulate gradients over
        max_epochs: Maximum number of epochs to train
        learning_rate: Learning rate for optimization
        patience: Early stopping patience
        relative_loss_threshold: Relative improvement threshold for early stopping
        run_name: Name for this training run
        save_model: Whether to save the model and plots
        use_wandb: Whether to use Weights & Biases for logging
    """
    if run_name is None:
        run_name = f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create run directory if saving
    run_dir = f"runs/{run_name}"
    if save_model:
        os.makedirs(run_dir, exist_ok=True)
    
    losses = []
    samples_seen = []
    epoch_losses = []
    total_samples = 0
    
    best_loss = float('inf')
    print(f"Starting autoencoder training: {run_name}")
    
    num_samples = len(dataset)
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        num_batches = 0

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{max_epochs}")
            
        # Shuffle dataset
        indices = np.random.permutation(num_samples)
        
        # Mini-batch training
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]
            batch_data = dataset[batch_indices]
            
            # Train on batch (reconstruction task)
            batch_loss = autoencoder.train(batch_data)
            epoch_loss += batch_loss
            num_batches += 1
            total_samples += len(batch_data)
            
            # Update parameters every grad_accumulation_steps
            if num_batches % grad_accumulation_steps == 0:
                autoencoder.model.update(grad_accumulation_steps)
            
            losses.append(batch_loss)
            samples_seen.append(total_samples)
            
            # if use_wandb:
            #     wandb.log({"batch_loss": batch_loss, "samples_seen": total_samples})
        
        # Final update if needed
        if num_batches % grad_accumulation_steps != 0:
            autoencoder.model.update(grad_accumulation_steps)
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        
        # if use_wandb:
        #     wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
        
        # Early stopping check
        if len(epoch_losses) > patience:
            recent_loss = avg_epoch_loss
            past_loss = epoch_losses[-(patience + 1)]
            
            if recent_loss >= (1 - relative_loss_threshold) * past_loss:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save best model
        if avg_epoch_loss < best_loss and save_model:
            best_loss = avg_epoch_loss
            autoencoder.save(f"{run_dir}/best_model.npz")
    
    results = {
        'run_name': run_name,
        'final_loss': avg_epoch_loss,
        'best_loss': best_loss,
        'total_epochs': epoch + 1,
        'total_samples': total_samples,
        'losses': losses,
        'samples_seen': samples_seen,
        'epoch_losses': epoch_losses
    }
    
    if save_model:
        # Save final model and training plot
        autoencoder.save(f"{run_dir}/final_model.npz")
        
        plt.figure(figsize=(10, 6))
        plt.plot(samples_seen, losses, alpha=0.7)
        plt.xlabel('Number of Samples Seen')
        plt.ylabel('Reconstruction Loss')
        plt.title('Autoencoder Training Loss vs Samples')
        plt.grid(True)
        plt.text(0.95, 0.95, USERNAME, ha='right', va='top', 
                transform=plt.gca().transAxes, fontsize=10, 
                color='gray', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{run_dir}/training_loss.png", dpi=150)
        plt.close()
        
        print(f"Results saved to {run_dir}/")
    
    print(f"Training completed: {results['total_epochs']} epochs, {total_samples} samples")
    print(f"Final loss: {results['final_loss']:.6f}")
    
    return results
