# main.py
import torch
import matplotlib.pyplot as plt
import numpy as np

from config import *
from dataset import get_dataloaders
from model import VAE
from train import train_vae
from loss_utils import save_latent_plot, create_gif, plot_reconstructions, calculate_fid , loss_function

def run_experiments():
    # 1. Data Preparation
    train_loader, test_loader = get_dataloaders()
    
    # print dataset info
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Dictionary to store results
    results = {}

    # --- Task 4.6: Beta-VAE Experiments ---
    print("Starting Task 4.6: Beta-VAE Experiments...")
    
    for beta in BETA_VALUES:
        print(f"\n--- Training with Beta = {beta} ---")
        model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
        
        # We need to hook into the training loop for the GIF, 
        # so we'll do a custom loop here slightly modifying the train_vae logic
        # or just call snapshotting inside a loop here.
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        gif_filenames = []
        
        epoch_losses = []

        for epoch in range(1, EPOCHS + 1):
            # Train one epoch
            model.train()
            total_loss = 0
            total_bce = 0
            total_kld = 0
            for data, _ in train_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss, bce, kld = loss_function(recon, data, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_bce += bce.item()
                total_kld += kld.item()
            
            print(f"Beta {beta}, Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}",flush=True)
            print(f"  BCE: {total_bce / len(train_loader.dataset):.4f}, KLD: {total_kld / len(train_loader.dataset):.4f}",flush=True)
            # Save snapshot for GIF (Task 4.6)
            fname = save_latent_plot(model, test_loader, epoch, beta, VISUALIZE_CLASSES)
            gif_filenames.append(fname)
        
        # Create GIF
        create_gif(gif_filenames, f'latent_evolution_beta_{beta}.gif')
        
        # --- Task 4.7: Evaluation ---
        print(f"Evaluating Beta={beta}...")
        plot_reconstructions(model, test_loader, f'recon_beta_{beta}.png')
        print(f"Reconstruction plot saved to 'recon_beta_{beta}.png'")
        try:
            fid_score = calculate_fid(model, test_loader , beta)
            print(f"FID Score (Beta={beta}): {fid_score:.4f}")
        except ImportError:
            print("Torchmetrics not installed, skipping FID.")
            
        results[beta] = model # Save model for later

    # --- Task 4.8: Effect of Frozen Latent Parameters ---
    print("\nStarting Task 4.8: Frozen Latent Parameters...")
    
    # Use the model trained with Beta=1.0 for this experiment
    model = results[1.0]
    model.eval()
    
    sigmas = [0.1, 0.5, 1.0]
    fixed_mu = torch.zeros(64, LATENT_DIM).to(DEVICE) # Batch of 64, Mean = 0
    
    fig, axes = plt.subplots(len(sigmas), 8, figsize=(12, 6))
    
    with torch.no_grad():
        for i, sigma_val in enumerate(sigmas):
            # Create fixed distribution
            # z = 0 + epsilon * sigma_val
            epsilon = torch.randn(64, LATENT_DIM).to(DEVICE)
            z = fixed_mu + epsilon * sigma_val
            
            generated = model.decode(z).view(-1, 28, 28).cpu().numpy()
            
            # Plot 8 samples
            for j in range(8):
                axes[i, j].imshow(generated[j], cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f"Sigma={sigma_val}")

    plt.suptitle("Generated Samples with Frozen Mean=0 and Varying Sigma")
    plt.savefig("frozen_parameters_experiment.png")
    print("Frozen parameter experiment saved to 'frozen_parameters_experiment.png'")

if __name__ == "__main__":
    run_experiments()