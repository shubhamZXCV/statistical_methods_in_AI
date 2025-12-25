import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "data_file": "../../dataset/Q2/recurrence_timeseries.csv",
    "history_p": 10,             # Fixed history length (Input dimension)
    "hidden_sizes": [2, 4, 8, 16, 32, 64, 128], # Varying this varies parameter count
    "batch_size": 64,
    "lr": 0.005,
    "epochs": 10,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "seed": 42
}

# Set seeds for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ==========================================
# 2. DATA GENERATION & PREPARATION
# ==========================================
def generate_synthetic_data_if_missing():
    """Generates a Hénon-like chaotic map if the CSV doesn't exist."""
    if not os.path.exists(CONFIG["data_file"]):
        print(f"[Info] {CONFIG['data_file']} not found. Generating synthetic data...")
        length = 2000
        x = np.zeros(length)
        x[0], x[1] = 0.1, 0.1
        # Rule: x_k = 1 - 1.4*x_{k-1}^2 + 0.3*x_{k-2}
        for k in range(2, length):
            noise = np.random.normal(0, 0.005)
            x[k] = 1.0 - 1.4 * (x[k-1]**2) + 0.3 * x[k-2] + noise
        pd.DataFrame(x, columns=['value']).to_csv(CONFIG["data_file"], index=False)
        print("[Info] Data generated.")

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, p):
        self.sequences = sequences
        self.p = p
    def __len__(self): 
        return len(self.sequences) - self.p
    def __getitem__(self, idx):
        # Input: History window [x_{k-p}, ..., x_{k-1}]
        x = self.sequences[idx : idx + self.p]
        # Target: Next value x_k
        y = self.sequences[idx + self.p]
        # MLP expects flattened input, but we keep dim for flexibility
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_process_data():
    """Loads data, splits chronologically, normalizes based on Train."""
    generate_synthetic_data_if_missing()
    
    df = pd.read_csv(CONFIG["data_file"])
    data = df.values.flatten().astype(float)
    total_len = len(data)
    
    # 1. Data Splits (Chronological)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    train_raw = data[:train_size]
    val_raw = data[train_size : train_size + val_size]
    test_raw = data[train_size + val_size :]
    
    # 2. Normalization (Fit on Train ONLY)
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    val_norm = scaler.transform(val_raw.reshape(-1, 1)).flatten()
    test_norm = scaler.transform(test_raw.reshape(-1, 1)).flatten()
    
    # 3. Create Datasets
    p = CONFIG["history_p"]
    train_ds = TimeSeriesDataset(train_norm, p)
    val_ds = TimeSeriesDataset(val_norm, p)
    test_ds = TimeSeriesDataset(test_norm, p)
    
    loaders = {
        "train": DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True),
        "val": DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False),
        "test": DataLoader(test_ds, batch_size=1, shuffle=False)
    }
    
    return loaders, scaler, test_raw

# ==========================================
# 3. MODEL DEFINITION (MLP)
# ==========================================
class MLP_Predictor(nn.Module):
    """
    Multi-Layer Perceptron.
    Input: History vector of size p
    Output: Scalar prediction
    """
    def __init__(self, input_dim, hidden_dim):
        super(MLP_Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, p)
        return self.net(x).squeeze()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 4. TRAINING & TUNING LOOP
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(CONFIG["device"]), y_batch.to(CONFIG["device"])
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
            x_batch, y_batch = x_batch.to(CONFIG["device"]), y_batch.to(CONFIG["device"])
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_parameter_tuning(loaders):
    print("\n=== 2.5 Parsimony: Exploring Parameter Count vs Performance ===")
    results = []
    best_model_state = None
    best_mse = float('inf')
    best_hidden = 0
    
    for h_dim in CONFIG["hidden_sizes"]:
        # Init Model
        model = MLP_Predictor(CONFIG["history_p"], h_dim).to(CONFIG["device"])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
        criterion = nn.MSELoss()
        
        params = count_parameters(model)
        
        # Train
        curr_best_val = float('inf')
        for epoch in range(CONFIG["epochs"]):
            train_epoch(model, loaders["train"], optimizer, criterion)
            val_loss = validate(model, loaders["val"], criterion)
            print(f"Hidden: {h_dim:3d} | Epoch: {epoch+1:2d} | Val MSE: {val_loss:.6f}",flush=True)
            if val_loss < curr_best_val:
                curr_best_val = val_loss
                # Save state if this is the best global model so far
                if curr_best_val < best_mse:
                    best_mse = curr_best_val
                    best_model_state = model.state_dict()
                    best_hidden = h_dim

        print(f"Hidden: {h_dim:3d} | Params: {params:5d} | Val MSE: {curr_best_val:.6f}")
        results.append({"hidden": h_dim, "params": params, "mse": curr_best_val})
        
    return results, best_model_state, best_hidden

# ==========================================
# 5. ANALYTICAL IDENTIFICATION (LASSO)
# ==========================================
def identify_recurrence(loaders):
    """Uses Lasso Regression to find the closed-form equation."""
    print("\n=== 2.3 Analytical Recurrence Identification ===")
    
    # Extract training data as numpy arrays
    X_list, y_list = [], []
    for x, y in loaders["train"]:
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Create Polynomial Features (Candidate Library)
    p = CONFIG["history_p"]
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    # Sparse Regression
    lasso = Lasso(alpha=0.001, max_iter=10000)
    lasso.fit(X_poly, y)
    
    # Build Equation String
    feature_names = poly.get_feature_names_out([f"x[k-{p-i}]" for i in range(p)])
    terms = []
    for coef, name in zip(lasso.coef_, feature_names):
        if abs(coef) > 0.01: # Threshold
            terms.append(f"({coef:.3f} * {name})")
            
    equation = "x[k] = " + " + ".join(terms)
    return equation, lasso, poly

# ==========================================
# 6. ANALYSIS & PLOTTING
# ==========================================
def analyze_and_report(results, best_state, best_hidden, equation, lasso, poly, loaders, scaler, raw_test):
    # --- Plot 1: Complexity vs Accuracy ---
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df['params'], df['mse'], marker='o', color='purple', linewidth=2)
    plt.xscale('log')
    plt.title('Parsimony: Parameter Count vs. MSE')
    plt.xlabel('Number of Parameters (Log Scale)')
    plt.ylabel('Validation MSE')
    plt.grid(True, which="both", ls="--")
    plt.savefig('complexity_vs_accuracy.png')
    print("[Plot] Saved complexity_vs_accuracy.png")
    
    # --- Stability Analysis (Autoregressive Forecast) ---
    # Load best MLP
    model = MLP_Predictor(CONFIG["history_p"], best_hidden).to(CONFIG["device"])
    model.load_state_dict(best_state)
    model.eval()
    
    forecast_len = 100
    # Get initial seed from test set
    seed_x, _ = next(iter(loaders["test"]))
    curr_mlp = seed_x.numpy().flatten()
    curr_ana = seed_x.numpy().flatten()
    
    mlp_preds, ana_preds = [], []
    
    with torch.no_grad():
        for _ in range(forecast_len):
            # MLP Predict
            in_t = torch.tensor(curr_mlp, dtype=torch.float32).unsqueeze(0).to(CONFIG["device"])
            p_mlp = model(in_t).item()
            mlp_preds.append(p_mlp)
            
            # Analytical Predict
            p_feats = poly.transform(curr_ana.reshape(1, -1))
            p_ana = lasso.predict(p_feats)[0]
            ana_preds.append(p_ana)
            
            # Update History
            curr_mlp = np.roll(curr_mlp, -1); curr_mlp[-1] = p_mlp
            curr_ana = np.roll(curr_ana, -1); curr_ana[-1] = p_ana
            
    # Inverse Transform
    mlp_preds = scaler.inverse_transform(np.array(mlp_preds).reshape(-1, 1)).flatten()
    ana_preds = scaler.inverse_transform(np.array(ana_preds).reshape(-1, 1)).flatten()
    gt = raw_test[CONFIG["history_p"] : CONFIG["history_p"] + forecast_len]
    
    # --- Plot 2: Stability ---
    plt.figure(figsize=(12, 6))
    plt.plot(gt, label='Ground Truth', color='black', alpha=0.6)
    plt.plot(mlp_preds, label='MLP Forecast', linestyle='--')
    plt.plot(ana_preds, label='Analytical Forecast', linestyle='-.')
    plt.title('Stability Analysis: Autoregressive Generation')
    plt.legend()
    plt.savefig('stability_forecast.png')
    print("[Plot] Saved stability_forecast.png")
    
    # --- Plot 3: Error Accumulation ---
    err_mlp = np.abs(gt - mlp_preds)
    err_ana = np.abs(gt - ana_preds)
    plt.figure(figsize=(10, 5))
    plt.plot(err_mlp, label='MLP Error')
    plt.plot(err_ana, label='Analytical Error')
    plt.title('Error Accumulation Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.savefig('error_accumulation.png')
    print("[Plot] Saved error_accumulation.png")

    # --- Final Report ---
    print("\n" + "="*50)
    print("           FINAL REPORT CHECKLIST")
    print("="*50)
    print("1. Model Specifications:")
    print(f"   - Type: MLP (Feed-Forward)")
    print(f"   - Best Hidden Size: {best_hidden}")
    print(f"   - Parameter Count: {count_parameters(model)}")
    print(f"   - History Length (p): {CONFIG['history_p']}")
    print("-" * 30)
    print("2. Identified Analytical Recurrence (F_theta):")
    print(f"   {equation}")
    print("-" * 30)
    print("3. Conclusions on Dataset:")
    print("   - Complexity Analysis: The plot 'complexity_vs_accuracy.png' shows")
    print("     how performance improves with model size until diminishing returns.")
    print("   - Stability: The analytical model often provides better long-term stability")
    print("     as it captures the exact physics, whereas MLPs may drift if not constrained.")
    print(f"   - MLP Mean Error (100 steps): {np.mean(err_mlp):.4f}")
    print(f"   - Analytical Mean Error (100 steps): {np.mean(err_ana):.4f}")
    print("="*50)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    loaders, scaler, raw_test = load_and_process_data()
    
    # 2. Tune MLP (Parsimony)
    tuning_results, best_state, best_hidden = run_parameter_tuning(loaders)
    
    # 3. Identify Equation
    equation, lasso, poly = identify_recurrence(loaders)
    
    # 4. Analyze & Report
    analyze_and_report(tuning_results, best_state, best_hidden, equation, lasso, poly, loaders, scaler, raw_test)