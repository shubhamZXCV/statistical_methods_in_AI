# identify.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from data_prep import get_data_loaders
import config
from utils import log_message

def find_equation():
    # 1. Load Tuning Results to find best p
    try:
        df_results = pd.read_csv(config.TUNING_RESULTS_FILE)
        best_p = int(df_results.loc[df_results['best_val_mse'].idxmin()]['p'])
        log_message(f"\n=== Analytical Identification using Best p={best_p} ===")
    except:
        best_p = 3 # Fallback
        log_message("Warning: Tuning results not found. Using default p=3")

    # 2. Load Data (Train set only for fitting)
    train_loader, _, _, scaler, _ = get_data_loaders(best_p)
    
    # Extract X and y from loader
    X_list = []
    y_list = []
    for x_batch, y_batch in train_loader:
        X_list.append(x_batch.squeeze(-1).numpy())
        y_list.append(y_batch.numpy())
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # 3. Symbolic Regression (Lasso + Polynomials)
    # We look for terms up to degree 2 (e.g., x^2, x*y)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    # Lasso with small alpha enforces sparsity (sets useless coefficients to 0)
    lasso = Lasso(alpha=0.001, max_iter=10000)
    lasso.fit(X_poly, y)
    
    # 4. Interpret Results
    feature_names = poly.get_feature_names_out([f"x[k-{best_p-i}]" for i in range(best_p)])
    
    equation_parts = []
    log_message("Identified Recurrence Relation:")
    
    for coef, name in zip(lasso.coef_, feature_names):
        if abs(coef) > 0.001: # Threshold
            equation_parts.append(f"({coef:.4f} * {name})")
            
    equation_str = "x[k] = " + " + ".join(equation_parts)
    log_message(equation_str)
    
    return lasso, poly, best_p

if __name__ == "__main__":
    find_equation()