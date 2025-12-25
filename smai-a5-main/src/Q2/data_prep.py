# data_prep.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import config

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, p):
        self.sequences = sequences
        self.p = p
    
    def __len__(self):
        return len(self.sequences) - self.p
    
    def __getitem__(self, idx):
        # Input: History [x_{k-p}, ..., x_{k-1}]
        # Target: Next value x_k
        x = self.sequences[idx : idx + self.p]
        y = self.sequences[idx + self.p]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def get_data_loaders(p):
    """
    Loads data, normalizes (fit on train only), and returns DataLoaders for a specific p.
    """
    # Load CSV
    # We assume the CSV has a header. If not, change header=None
    try:
        df = pd.read_csv(config.DATA_FILE)
        data = df.iloc[:, 0].values.astype(float) # Take first column
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {config.DATA_FILE}. Please ensure file exists.")

    total_len = len(data)
    
    # Chronological Splits (70% Train, 15% Val, 15% Test)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    train_raw = data[:train_size]
    val_raw = data[train_size : train_size + val_size]
    test_raw = data[train_size + val_size :]
    
    # Normalization
    # CRITICAL: Fit scaler ONLY on training data
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    val_norm = scaler.transform(val_raw.reshape(-1, 1)).flatten()
    test_norm = scaler.transform(test_raw.reshape(-1, 1)).flatten()
    
    # Create Datasets
    train_ds = TimeSeriesDataset(train_norm, p)
    val_ds = TimeSeriesDataset(val_norm, p)
    test_ds = TimeSeriesDataset(test_norm, p)
    
    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch size 1 for testing
    
    return train_loader, val_loader, test_loader, scaler, test_raw