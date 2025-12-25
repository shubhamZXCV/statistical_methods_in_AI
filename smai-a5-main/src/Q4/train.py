# train.py
import torch
import torch.optim as optim
from loss_utils import loss_function
from config import DEVICE, INPUT_DIM

def train_vae(model, train_loader, epochs, beta, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # It should be available here in newer versions
    # optimizer = torch.optim.Muon(model.parameters(), lr=0.02, momentum=0.95)
    
    # To store loss history
    history = {'total': [], 'bce': [], 'kld': []}
    
    # For GIF generation
    plot_filenames = []

    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        total_bce = 0
        total_kld = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            
            optimizer.step()
        
        # Average losses
        avg_loss = train_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        
        history['total'].append(avg_loss)
        history['bce'].append(avg_bce)
        history['kld'].append(avg_kld)
        
        print(f'Epoch {epoch}, Beta: {beta} | Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')

    return history