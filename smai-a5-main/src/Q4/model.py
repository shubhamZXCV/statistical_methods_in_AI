# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, HIDDEN_DIM, LATENT_DIM

class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        # --- 1. Encoder ---
        # Maps input image -> Hidden Layer -> Mean & LogVar
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Output mean (mu)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Output log-variance

        # --- 3. Decoder ---
        # Maps sampled latent vector z -> Hidden Layer -> Reconstructed Image
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encodes the input into distribution parameters.
        """
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """
        Task 4.3 (Part 2): The Reparameterization Trick.
        z = mu + epsilon * sigma
        """
        if self.training:
            std = torch.exp(0.5 * logvar) # Convert log_var to standard deviation
            eps = torch.randn_like(std)   # Sample epsilon from N(0, I)
            return mu + eps * std
        else:
            # During testing, we usually just use the mean
            return mu

    def decode(self, z):
        """
        Reconstructs the image from the latent vector.
        """
        h3 = F.relu(self.fc3(z))
        # Sigmoid ensures output is between 0 and 1 (matching input data)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # Flatten the image (Batch, 1, 28, 28) -> (Batch, 784)
        x = x.view(-1, INPUT_DIM)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar