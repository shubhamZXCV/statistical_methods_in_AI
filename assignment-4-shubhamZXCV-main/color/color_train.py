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

from color_dataset import ColorizationDataset
from color_model import ColorizerCNN

config = {
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

def train_model(config):
    # Initialize wandb and log config
    wandb.init(project=config['wandb_project'], config=config)
    cfg = wandb.config
    
    # Device setup
    device = torch.device(cfg.device)

    # DataLoaders
    # Note: Dataloaders rely on ColorizationDataset returning CPU tensors now.
    train_dataset = ColorizationDataset(is_train=True, config=cfg)
    val_dataset = ColorizationDataset(is_train=False, config=cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # Reload centroids for CPU visualization (matplotlib)
    centroids_rgb_np = np.load(cfg.centroids_path) / 255.0
    
    # Model, Loss, Optimizer
    model = ColorizerCNN(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    
    wandb.watch(model, criterion, log="all", log_freq=100)

    best_val_loss = float('inf')
    checkpoint_path = f'best_model_checkpoint_{cfg.wandb_project}.pth'

    print(f"Starting training on {device}...")
    
    # Training Loop
    for epoch in range(cfg.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for inputs_gray, targets_class, _ in train_loader:
            # FIX: Move TENSORS TO DEVICE HERE
            inputs_gray = inputs_gray.to(device)
            targets_class = targets_class.to(device)

            optimizer.zero_grad()
            outputs_logits = model(inputs_gray)
            
            loss = criterion(outputs_logits, targets_class)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs_gray.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_gray, targets_class, _ in val_loader:
                # FIX: Move TENSORS TO DEVICE HERE
                inputs_gray = inputs_gray.to(device)
                targets_class = targets_class.to(device)
                
                outputs_logits = model(inputs_gray)
                loss = criterion(outputs_logits, targets_class)
                val_loss += loss.item() * inputs_gray.size(0)
    
        val_loss /= len(val_loader.dataset)
        
        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f'Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"-> Saved new best model with Val Loss: {best_val_loss:.4f}")
    print(f"--- Run {wandb.run.name} finished. Best Val Loss: {best_val_loss:.4f} ---")
    # --- 5. Final Evaluation and Visualization ---
    
    # Load the best model
    best_model = ColorizerCNN(cfg).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path))
    best_model.eval()
    
    # Sample 10 images from the validation set
    sample_indices = np.random.choice(len(val_dataset), 10, replace=False)
    
    fig, axes = plt.subplots(10, 3, figsize=(12, 40))
    wandb_images = []

    for i, idx in enumerate(sample_indices):
        # Get data from dataset (all tensors are on CPU)
        inputs_gray_cpu, _, rgb_image_gt_cpu = val_dataset[idx] 
        
        # Move ONLY the input image to the GPU/device for model inference
        inputs_gray = inputs_gray_cpu.unsqueeze(0).to(device) # Add batch dim AND move to device
        
        # Inference
        with torch.no_grad():
            outputs_logits = best_model(inputs_gray) 
        
        # Post-process: Get results back to CPU for numpy/matplotlib handling
        pred_class_map = torch.argmax(outputs_logits.squeeze(0), dim=0).cpu().numpy()
        
        # Post-process: Class Map -> Predicted RGB Image (using centroids)
        pred_rgb_flat = centroids_rgb_np[pred_class_map.flatten()]
        pred_rgb_image = pred_rgb_flat.reshape(32, 32, 3)
        
        # Plotting (requires CPU tensors)
        axes[i, 0].imshow(inputs_gray_cpu.squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title("Input Gray")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_rgb_image)
        axes[i, 1].set_title("Predicted Color")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rgb_image_gt_cpu.permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')

        wandb_images.append(wandb.Image(fig, caption=f"Example {i+1}"))
    
    # Log the required 10 images
    wandb.log({"Colourization Examples (Input, Predicted, GT)": wandb_images})
    plt.tight_layout()
    plt.show() 
    
    wandb.finish()

    return train_loss , best_val_loss