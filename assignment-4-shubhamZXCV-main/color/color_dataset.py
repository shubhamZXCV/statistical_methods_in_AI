import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
import wandb

CONFIG = {
    'wandb_project': "CIFAR10-Colorization-Q2",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    
    # Model Architecture Params
    'NIC': 1,               # Number of Input Channels (Grayscale)
    'NUM_CLASSES': 24,      # NC: Number of Output Classes (Color Centroids)
    'NF': 64,               # Base Number of Filters in the first layer (Feature Maps)
    'kernel_size_conv': 3,  # Standard kernel size for Conv2d
    'kernel_size_tconv': 2, # Kernel size for ConvTranspose2d (to double size)

    # Training Hyperparameters
    'learning_rate': 1e-3,
    'batch_size': 128,
    'epochs': 25,
    'optimizer': 'Adam',    # Options: 'Adam', 'SGD'
    'loss_fn': 'CrossEntropyLoss',
    'num_workers': 4,       # Dataloader workers
    'centroids_path': 'color_centroids.npy' # Path to your k-means result file
}

# --- 3. Custom Dataset ---

class ColorizationDataset(Dataset):
    def __init__(self, is_train=True, config=CONFIG):
        # The self.device attribute is now only for the *model's* device (GPU/CUDA).
        # Data loading will be STRICTLY CPU to avoid the fork error.
        self.device = config['device'] 
        self.centroids_path = config['centroids_path']
        self.num_classes = config['NUM_CLASSES']

        # Load CIFAR-10, transforming to Tensor (0-1 range)
        transform = transforms.ToTensor()
        self.cifar_dataset = CIFAR10(root='./data', train=is_train, download=True, transform=transform)
        
        # Load and prepare centroids: CRITICAL FIX
        # Centroids must be on CPU to be used in the worker processes
        centroids_np = np.load(self.centroids_path).astype(np.float32)
        self.centroids_rgb = torch.from_numpy(centroids_np / 255.0).cpu() # <--- FORCE CENTROIDS TO CPU

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        rgb_image_cpu, _ = self.cifar_dataset[idx] 
        # rgb_image_cpu is already a torch.Tensor on the CPU

        # --- 1. Grayscale Input Generation ---
        gray_np = rgb2gray(rgb_image_cpu.permute(1, 2, 0).numpy())
        grayscale_input = torch.from_numpy(gray_np).float().unsqueeze(0) # [1, 32, 32] (on CPU)

        # --- 2. Class Map Target Generation (Classification Target) ---
        # CRITICAL FIX: Ensure pixels_rgb is a CPU tensor for calculation
        pixels_rgb = rgb_image_cpu.permute(1, 2, 0).reshape(-1, 3) 
        
        # Efficient Euclidean Distance Calculation (NOW ENTIRELY ON CPU)
        pixels_sq = torch.sum(pixels_rgb ** 2, dim=1, keepdim=True)
        # self.centroids_rgb is already on CPU
        centroids_sq = torch.sum(self.centroids_rgb ** 2, dim=1, keepdim=True).T
        dot_product = -2.0 * torch.matmul(pixels_rgb, self.centroids_rgb.T)
        distances = pixels_sq + centroids_sq + dot_product
        
        # Find the nearest centroid (class label)
        class_map_flat = torch.argmin(distances, dim=1)
        
        # Reshape back to [32, 32] and cast to LongTensor 
        class_map = class_map_flat.reshape(32, 32).long()

        # Return all tensors as CPU tensors
        return grayscale_input, class_map, rgb_image_cpu