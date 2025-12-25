# loss_utils.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from torchvision.utils import save_image
from config import INPUT_DIM, DEVICE
from torchmetrics.image.fid import FrechetInceptionDistance

# --- Task 4.4: Loss Function ---
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    L = Reconstruction_Loss + Beta * KL_Divergence
    """
    # 1. Reconstruction Loss (Binary Cross Entropy)
    # reduction='sum' aggregates loss over the whole batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')

    # 2. KL Divergence
    # Analytical formula for KL(N(mu, sigma) || N(0, 1))
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + (beta * KLD), BCE, KLD

# --- Task 4.6: Visualization Helper (GIF) ---
def save_latent_plot(model, loader, epoch, beta_val, classes_to_plot):
    """
    Saves a scatter plot of the latent space for specific classes.
    """
    model.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            mu, _ = model.encode(data.view(-1, INPUT_DIM))
            all_z.append(mu.cpu())
            all_labels.append(target)
            
    all_z = torch.cat(all_z).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    plt.figure(figsize=(8, 6))
    
    # Filter for specific classes
    for c in classes_to_plot:
        idx = all_labels == c
        # We plot the first 2 dimensions of z (if latent_dim > 2, this is a projection)
        plt.scatter(all_z[idx, 0], all_z[idx, 1], label=f'Class {c}', alpha=0.5)
        
    plt.legend()
    plt.title(f'Latent Space (Epoch {epoch}, Beta={beta_val})')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    
    filename = f'temp_plot_beta_{beta_val}_epoch_{epoch}.png'
    plt.savefig(filename)
    plt.close()
    return filename

def create_gif(filenames, output_name):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename) # Clean up temp files
    imageio.mimsave(output_name, images, duration=0.5)

# --- Task 4.7: Evaluation Helpers ---
def plot_reconstructions(model, loader, filename='reconstruction.png'):
    model.eval()
    data, _ = next(iter(loader))
    data = data.to(DEVICE)
    
    with torch.no_grad():
        recon, _, _ = model(data)
        
    # Reshape for plotting
    data = data.view(-1, 28, 28).cpu().numpy()
    recon = recon.view(-1, 28, 28).cpu().numpy()
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        # Original
        axes[0, i].imshow(data[i], cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed
        axes[1, i].imshow(recon[i], cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.savefig(filename)
    plt.close()

def calculate_fid(model, loader , beta):
    """
    Task 4.7: FID Calculation. 
    Uses torchmetrics.image.fid.FrechetInceptionDistance.
    Also saves a visualization of the generated images from the first batch.
    """
    
    # Initialize FID metric (requires 3 channels)
    # feature=64 uses a smaller inception layer for speed
    fid = FrechetInceptionDistance(feature=64).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i > 5: break # Limit batches for speed in this example
            
            real_imgs = data.to(DEVICE)
            
            # 1. Generate fake images
            z = torch.randn(data.size(0), model.fc_mu.out_features).to(DEVICE)
            fake_imgs = model.decode(z).view(-1, 1, 28, 28)
            
            # --- NEW: Visualization ---
            # Save the first batch of generated images to disk
            if i == 0:
                # Clamp ensures values are strictly [0, 1] before saving
                save_image(fake_imgs.clamp(0, 1)[:16], f"fid_generated_samples_beta_{beta}.png", nrow=4)
                print(f"Saved visualization to 'fid_generated_samples_beta_{beta}.png'")
            # --------------------------
            
            # 2. Preprocessing for FID (Inception expects 3-channel RGB, uint8 [0-255])
            
            # Convert 1-channel (Grayscale) to 3-channel (RGB)
            real_rgb = real_imgs.repeat(1, 3, 1, 1)
            fake_rgb = fake_imgs.repeat(1, 3, 1, 1)
            
            # Convert float [0, 1] to uint8 [0, 255]
            # We use .clamp(0, 1) to prevent overflow errors if the model outputs slightly <0 or >1
            real_rgb = (real_rgb.clamp(0, 1) * 255).byte()
            fake_rgb = (fake_rgb.clamp(0, 1) * 255).byte()
            
            # 3. Update Metric Stats
            fid.update(real_rgb, real=True)
            fid.update(fake_rgb, real=False)
            
    # Compute final scalar score
    return fid.compute().item()