# train_tune.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import config
from utils import log_message, clear_log
from data_prep import get_data_loaders
from model import RNNPredictor

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_tuning():
    clear_log()
    log_message("Starting Hyperparameter Tuning (Parsimony Analysis)...")
    
    tuning_results = []
    
    for p in config.P_VALUES_TO_TUNE:
        log_message(f"\n--- Training with History Length p={p} ---")
        
        # 1. Get Data
        train_loader, val_loader, _, _, _ = get_data_loaders(p)
        
        # 2. Setup Model
        # Note: Changing 'p' changes the input dimension of the LSTM if we were treating 
        # history as features, but here input_dim is 1 (sequence) and p is seq_len.
        # However, to strictly measure complexity vs p, we can use a Linear AR model 
        # or note that RNN complexity is constant regarding sequence length, 
        # BUT the difficulty of optimization changes.
        # *If* you want to test parameter count explicitly, we can increase hidden size with p.
        # For this assignment, we keep hidden size constant, but we log the count anyway.
        model = RNNPredictor(hidden_dim=config.HIDDEN_DIM, num_layers=config.LAYERS).to(config.DEVICE)
        
        param_count = count_parameters(model)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        # 3. Training Loop
        for epoch in range(config.EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion)
            v_loss = validate(model, val_loader, criterion)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), f"model_p{p}.pth")
        
        log_message(f"Finished p={p}. Params: {param_count} | Best Val MSE: {best_val_loss:.6f}")
        
        tuning_results.append({
            'p': p,
            'params': param_count, # Logged for Section 2.5
            'best_val_mse': best_val_loss
        })

    # Save results
    results_df = pd.DataFrame(tuning_results)
    results_df.to_csv(config.TUNING_RESULTS_FILE, index=False)
    log_message(f"\nTuning complete. Results saved to {config.TUNING_RESULTS_FILE}")
    
    # Parsimony Selection:
    # We want the smallest 'p' that gives a "good enough" error.
    # Simple heuristic: Pick the p with min error.
    best_run = results_df.loc[results_df['best_val_mse'].idxmin()]
    log_message(f"RECOMMENDATION: Best history length is p={int(best_run['p'])}")

if __name__ == "__main__":
    run_tuning()