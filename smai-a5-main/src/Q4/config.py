# config.py
import torch

# Device configuration (use GPU if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
INPUT_DIM = 784  # Fashion-MNIST is 28x28 = 784
HIDDEN_DIM = 400 # Size of the hidden layer
LATENT_DIM = 20  # Size of the latent vector z
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 30      # Keep small for demonstration, increase for better results

# List of Beta values for Task 4.6
BETA_VALUES = [0.1, 0.5, 1.0 , 2.0]

# Classes to visualize for the latent space (Task 4.6)
# 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
# 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
VISUALIZE_CLASSES = [5, 7, 9] # Sandal, Sneaker, Ankle boot