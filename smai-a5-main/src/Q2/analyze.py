# analyze.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import config
from model import RNNPredictor
from data_prep import get_data_loaders
from identify import find_equation
from utils import log_message

def analyze_performance():
    # Load tuning results
    df_results = pd.read_csv(config.TUNING_RESULTS_FILE)
    
    # ---------------------------------------------------------
    # 1. Complexity–Accuracy Trade-off Figure (Section 2.5)
    # ---------------------------------------------------------
    # We plot History Length (p) vs MSE. 
    # (In RNNs, parameter count is often constant vs seq length, so p represents 
    # the "effective order" complexity).
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['p'], df_results['best_val_mse'], marker='o', linestyle='-', color='b')
    plt.title('Complexity vs. Accuracy (Parsimony Analysis)')
    plt.xlabel('History Length (p) / Effective Order')
    plt.ylabel('Validation MSE (Lower is Better)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Annotate the "Elbow" or Best Point
    best_row = df_results.loc[df_results['best_val_mse'].idxmin()]
    plt.annotate(f"Best p={int(best_row['p'])}", 
                 (best_row['p'], best_row['best_val_mse']),
                 xytext=(best_row['p'], best_row['best_val_mse']*1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig('complexity_accuracy_tradeoff.png')
    log_message("Saved complexity_accuracy_tradeoff.png")

    # ---------------------------------------------------------
    # 2. Stability Analysis (Autoregressive)
    # ---------------------------------------------------------
    best_p = int(best_row['p'])
    
    # Load Data & Models
    _, _, test_loader, scaler, raw_test_data = get_data_loaders(best_p)
    
    rnn = RNNPredictor(hidden_dim=config.HIDDEN_DIM, num_layers=config.LAYERS).to(config.DEVICE)
    rnn.load_state_dict(torch.load(f"model_p{best_p}.pth"))
    rnn.eval()
    
    lasso_model, poly, _ = find_equation()
    
    # Forecast Loop
    forecast_steps = 100
    first_x, _ = next(iter(test_loader))
    
    curr_hist_rnn = first_x.numpy().flatten()
    curr_hist_ana = first_x.numpy().flatten()
    
    rnn_preds = []
    ana_preds = []
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # RNN
            in_tensor = torch.tensor(curr_hist_rnn, dtype=torch.float32).view(1, best_p, 1).to(config.DEVICE)
            r_val = rnn(in_tensor).item()
            rnn_preds.append(r_val)
            
            # Analytical
            poly_feats = poly.transform(curr_hist_ana.reshape(1, -1))
            a_val = lasso_model.predict(poly_feats)[0]
            ana_preds.append(a_val)
            
            # Update
            curr_hist_rnn = np.roll(curr_hist_rnn, -1); curr_hist_rnn[-1] = r_val
            curr_hist_ana = np.roll(curr_hist_ana, -1); curr_hist_ana[-1] = a_val
            
    # Inverse Transform
    rnn_preds = scaler.inverse_transform(np.array(rnn_preds).reshape(-1, 1)).flatten()
    ana_preds = scaler.inverse_transform(np.array(ana_preds).reshape(-1, 1)).flatten()
    gt = raw_test_data[best_p : best_p + forecast_steps]

    # ---------------------------------------------------------
    # 3. Stability Plots (Error vs Time)
    # ---------------------------------------------------------
    # Calculate absolute error at each step
    rnn_error_over_time = np.abs(gt - rnn_preds)
    ana_error_over_time = np.abs(gt - ana_preds)

    plt.figure(figsize=(10, 6))
    plt.plot(rnn_error_over_time, label='RNN Error', color='red', alpha=0.7)
    plt.plot(ana_error_over_time, label='Analytical Equation Error', color='green', alpha=0.7, linestyle='--')
    plt.title(f'Stability Analysis: Error Accumulation (p={best_p})')
    plt.xlabel('Forecast Steps (Time)')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('stability_plot.png')
    log_message("Saved stability_plot.png")
    
    # Also save the visual comparison
    plt.figure(figsize=(12, 5))
    plt.plot(gt, label='Ground Truth', color='black')
    plt.plot(rnn_preds, label='RNN Forecast', linestyle='--')
    plt.title('Forecast Comparison')
    plt.legend()
    plt.savefig('forecast_comparison.png')

    # ---------------------------------------------------------
    # 4. Report Conclusions (for the text report)
    # ---------------------------------------------------------
    log_message("\n=== PARSIMONY & STABILITY REPORT DATA ===")
    log_message(f"1. Selected Model Order (p): {best_p}")
    log_message(f"   (This is the simplest model that achieves low error)")
    log_message(f"2. Stability:")
    log_message(f"   - RNN Mean Error over {forecast_steps} steps: {np.mean(rnn_error_over_time):.4f}")
    log_message(f"   - Analytical Mean Error over {forecast_steps} steps: {np.mean(ana_error_over_time):.4f}")
    log_message(f"3. Temporal Relation:")
    log_message(f"   See 'Identified Recurrence Relation' in log for F_theta.")

if __name__ == "__main__":
    analyze_performance()