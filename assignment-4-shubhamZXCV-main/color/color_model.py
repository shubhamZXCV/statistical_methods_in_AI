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


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        # CRITICAL FIX: Dynamically calculate padding to ensure H_out = H_in
        # For a stride of 1, padding = (kernel_size - 1) / 2
        padding = (kernel_size - 1) // 2 
        
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=2, padding=0):
        super().__init__()
        # stride=2, kernel=2, padding=0 ensures perfect doubling of H, W
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ColorizerCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        NF, NC = config['NF'], config['NUM_CLASSES']
        NIC = config['NIC']
        K_CONV = config['kernel_size_conv']
        K_TCONV = config['kernel_size_tconv']
        
        # Encoder (32 -> 16 -> 8 -> 4)
        # Uses the corrected ConvBlock which calculates padding automatically.
        self.enc1 = ConvBlock(NIC, NF, kernel_size=K_CONV) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.enc2 = ConvBlock(NF, 2 * NF, kernel_size=K_CONV)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc3 = ConvBlock(2 * NF, 4 * NF, kernel_size=K_CONV) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Bottleneck: [4NF, 4, 4]
        
        # Decoder (4 -> 8 -> 16 -> 32)
        # Note: kernel_size_tconv is assumed to be 2
        self.dec1 = ConvTransposeBlock(4 * NF, 2 * NF, kernel_size=K_TCONV) # [8, 8]
        self.dec2 = ConvTransposeBlock(2 * NF, NF, kernel_size=K_TCONV)      # [16, 16]
        self.dec3 = ConvTransposeBlock(NF, NC, kernel_size=K_TCONV)          # [NC, 32, 32]

        # Classifier (1x1 Conv to output logits)
        self.classifier = nn.Conv2d(NC, NC, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        
        # Final Logits (Must be [B, 24, 32, 32])
        logits = self.classifier(x) 
        return logits