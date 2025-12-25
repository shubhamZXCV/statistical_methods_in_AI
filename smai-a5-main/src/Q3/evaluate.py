# evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import os

# --- Import your specific project functions ---
from prep_stars import load_data, clean_and_align_time_series, transform_data_domain, scale_data
from split_repos import create_train_test_split, create_sequences
from classical import fit_arma_model, forecast_arma
from train_models import train_dl_model, predict_dl_model
from dl_models import RNNForecaster, CNN1DForecaster

def calculate_metrics(actual, predictions, prefix=""):
    """Calculates MAE and RMSE and returns them."""
    # Ensure inputs are numpy arrays/flattened
    act_val = actual.values if isinstance(actual, (pd.Series, pd.DataFrame)) else actual
    pred_val = predictions.values if isinstance(predictions, (pd.Series, pd.DataFrame)) else predictions
    
    mae = mean_absolute_error(act_val, pred_val)
    rmse = np.sqrt(mean_squared_error(act_val, pred_val))
    print(f"{prefix} MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return {'MAE': mae, 'RMSE': rmse}

def perform_multi_step_forecasting_dl(model, initial_sequence, steps, scaler, sequence_length, device):
    """
    Performs recursive (autoregressive) forecasting for DL models.
    """
    model.eval()
    # initial_sequence shape: (seq_len, 1) -> convert to (1, seq_len, 1)
    current_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    forecasts_scaled = []
    
    with torch.no_grad():
        for _ in range(steps):
            prediction_scaled_tensor = model(current_sequence) # Output shape (1, 1)
            prediction_scaled = prediction_scaled_tensor.cpu().numpy().flatten()[0]
            forecasts_scaled.append(prediction_scaled)
            
            # Update sequence: drop oldest, add new prediction
            new_val_tensor = torch.tensor([[[prediction_scaled]]], dtype=torch.float32).to(device)
            current_sequence = torch.cat((current_sequence[:, 1:, :], new_val_tensor), dim=1)
            
    # Inverse transform
    dummy_for_inverse = np.zeros((len(forecasts_scaled), scaler.n_features_in_))
    dummy_for_inverse[:, 0] = np.array(forecasts_scaled)
    forecasts_original_scale = scaler.inverse_transform(dummy_for_inverse)[:, 0]
    
    return forecasts_original_scale

def plot_forecast(actual_test_data, forecasts, model_name, repo_id, forecast_dates=None):
    """
    Plots Actual Test Data vs Forecast (Test Set Only) and saves the image.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot only the actual test data provided
    plt.plot(actual_test_data.index, actual_test_data.values, label='Actual Test Data', color='green', linewidth=2)
    
    # Plot the forecast
    if forecast_dates is not None:
        plt.plot(forecast_dates, forecasts, label=f'{model_name} Forecast', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'{model_name} Forecast for {repo_id} (Test Set Only)')
    plt.xlabel('Date')
    plt.ylabel('Stars')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Construct filename (replace spaces with underscores for safety)
    safe_model_name = model_name.replace(" ", "_")
    filename = f'forecast_{repo_id}_{safe_model_name}.png'
    
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    
    plt.show()
    plt.close() # Close plot to free memory

def plot_error_vs_horizon(repo_results, repo_id):
    """
    Plots RMSE vs Horizon for all models and saves the image.
    """
    plt.figure(figsize=(10, 6))
    
    # Iterate through models (ARMA, RNN, CNN)
    for model_name, metrics in repo_results.items():
        if 'Multi-step' in metrics and metrics['Multi-step']:
            # Extract horizons and errors, sort by horizon
            multi_step_data = metrics['Multi-step']
            horizons = sorted(multi_step_data.keys())
            errors = [multi_step_data[h] for h in horizons]
            
            plt.plot(horizons, errors, marker='o', label=model_name, linewidth=2)
    
    plt.title(f'RMSE Degradation vs. Forecast Horizon ({repo_id})')
    plt.xlabel('Forecast Horizon (Days)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f'error_vs_horizon_{repo_id}.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    
    plt.show()
    plt.close()

if __name__ == '__main__':
    # --- Configuration ---
    repo_ids_to_evaluate = ['facebook_react', 'pallets_flask'] 
    sequence_length = 30 
    test_size = 0.2
    val_size_of_train_val = 0.25
    
    # Hyperparameters
    best_rnn_params = {'lr': 0.001, 'bs': 32, 'hidden_size': 64, 'num_layers': 2}
    best_cnn_params = {'lr': 0.001, 'bs': 32, 'num_filters': 64, 'kernel_size': 3}
    arma_order = (2, 0, 1) 
    
    # Horizons to evaluate for RMSE table
    horizons_to_eval = [1, 3, 7, 14, 30]
    max_horizon = max(horizons_to_eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dictionary to store final results for DataFrame
    all_results = {} 

    for repo_id in repo_ids_to_evaluate:
        print(f"\n{'='*40}\nEvaluating Repository: {repo_id}\n{'='*40}")
        all_results[repo_id] = {}

        # 1. Data Prep
        stars_df = load_data()
        repo_cleaned = clean_and_align_time_series(stars_df, repo_id)
        repo_transformed = transform_data_domain(repo_cleaned.copy(), domain='incremental')
        repo_scaled, scaler = scale_data(repo_transformed.copy(), scaler_type='minmax', feature_col='stars_transformed')

        # Split
        train_val_df, test_df = create_train_test_split(repo_scaled[['stars_transformed_scaled']], test_size=test_size)
        train_df, val_df = create_train_test_split(train_val_df, test_size=val_size_of_train_val)

        # Reference Data (Original Scale)
        full_data_orig = repo_transformed['stars_transformed']
        train_data_orig = full_data_orig.loc[train_df.index]
        test_data_orig = full_data_orig.loc[test_df.index]

        # DL Sequences
        X_train, y_train = create_sequences(train_df['stars_transformed_scaled'].values, sequence_length)
        X_val, y_val = create_sequences(val_df['stars_transformed_scaled'].values, sequence_length)
        X_test, y_test = create_sequences(test_df['stars_transformed_scaled'].values, sequence_length)
        
        # Reshape for PyTorch
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Last sequence of training data
        last_train_seq_scaled = train_df['stars_transformed_scaled'].values[-sequence_length:].reshape(sequence_length, 1)
        
        # Prepare dates/values for 30-day comparison plots
        multi_step_dates = pd.date_range(start=train_data_orig.index[-1] + pd.Timedelta(days=1), 
                                         periods=max_horizon, freq='D')
        actual_multistep_values = full_data_orig.loc[multi_step_dates].values
        actual_series_30d = pd.Series(actual_multistep_values, index=multi_step_dates)

        # ==========================================
        # MODEL 1: ARMA
        # ==========================================
        print("\n--- Processing ARMA ---")
        all_results[repo_id]['ARMA'] = {'Multi-step': {}}

        # Fit
        arma_model = fit_arma_model(train_data_orig, order=arma_order)
        
        # A. "Single-step" (Proxy: Evaluate over full test set)
        full_test_horizon = len(test_data_orig)
        arma_full_pred_series, _ = forecast_arma(arma_model, full_test_horizon, train_data_orig)
        
        arma_pred_aligned = arma_full_pred_series.values
        actual_test_aligned = test_data_orig.values[:len(arma_pred_aligned)]
        
        metrics_arma_single = calculate_metrics(actual_test_aligned, arma_pred_aligned, "ARMA Full Set")
        all_results[repo_id]['ARMA']['Single-step'] = metrics_arma_single

        # B. Multi-step Horizons
        for h in horizons_to_eval:
            if len(arma_pred_aligned) >= h:
                pred_slice = arma_pred_aligned[:h]
                act_slice = actual_test_aligned[:h]
                rmse_h = np.sqrt(mean_squared_error(act_slice, pred_slice))
                all_results[repo_id]['ARMA']['Multi-step'][h] = rmse_h
                print(f"  ARMA RMSE (H={h}): {rmse_h:.4f}")

        # --- PLOT ARMA ---
        if len(arma_pred_aligned) >= max_horizon:
             plot_forecast(actual_series_30d, 
                           arma_pred_aligned[:max_horizon], 
                           'ARMA 30-Day', repo_id, multi_step_dates)

        # ==========================================
        # MODEL 2: RNN
        # ==========================================
        print("\n--- Processing RNN ---")
        all_results[repo_id]['RNN'] = {'Multi-step': {}}
        
        # Train
        rnn_model = RNNForecaster(input_size=1, hidden_size=best_rnn_params['hidden_size'], 
                                  num_layers=best_rnn_params['num_layers'], output_size=1).to(device)
        rnn_model, _ = train_dl_model(rnn_model, X_train, y_train, X_val, y_val, 
                                      epochs=30, batch_size=best_rnn_params['bs'], 
                                      learning_rate=best_rnn_params['lr'], loss_fn_name='MSE', device=device)
        
        # A. Single-step
        rnn_pred_scaled = predict_dl_model(rnn_model, X_test, device)
        dummy_inv = np.zeros((len(rnn_pred_scaled), scaler.n_features_in_))
        dummy_inv[:, 0] = rnn_pred_scaled
        rnn_pred_orig = scaler.inverse_transform(dummy_inv)[:, 0]
        
        dummy_y = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_y[:, 0] = y_test
        y_test_orig = scaler.inverse_transform(dummy_y)[:, 0]
        
        metrics_rnn_single = calculate_metrics(y_test_orig, rnn_pred_orig, "RNN Single-step")
        all_results[repo_id]['RNN']['Single-step'] = metrics_rnn_single

        # B. Multi-step
        rnn_forecast_long = perform_multi_step_forecasting_dl(
            rnn_model, last_train_seq_scaled, max_horizon, scaler, sequence_length, device
        )
        
        for h in horizons_to_eval:
            if len(actual_multistep_values) >= h:
                rmse_h = np.sqrt(mean_squared_error(actual_multistep_values[:h], rnn_forecast_long[:h]))
                all_results[repo_id]['RNN']['Multi-step'][h] = rmse_h
                print(f"  RNN RMSE (H={h}): {rmse_h:.4f}")
        
        # --- PLOT RNN ---
        plot_forecast(actual_series_30d, 
                      rnn_forecast_long, 
                      'RNN 30-Day', repo_id, multi_step_dates)

        # ==========================================
        # MODEL 3: CNN
        # ==========================================
        print("\n--- Processing CNN ---")
        all_results[repo_id]['CNN'] = {'Multi-step': {}}

        # Train
        cnn_model = CNN1DForecaster(input_channels=1, output_size=1, sequence_length=sequence_length,
                                    kernel_size=best_cnn_params['kernel_size'], 
                                    num_filters=best_cnn_params['num_filters']).to(device)
        cnn_model, _ = train_dl_model(cnn_model, X_train, y_train, X_val, y_val, 
                                      epochs=30, batch_size=best_cnn_params['bs'], 
                                      learning_rate=best_cnn_params['lr'], loss_fn_name='MSE', device=device)

        # A. Single-step
        cnn_pred_scaled = predict_dl_model(cnn_model, X_test, device)
        dummy_inv[:, 0] = cnn_pred_scaled
        cnn_pred_orig = scaler.inverse_transform(dummy_inv)[:, 0]
        
        metrics_cnn_single = calculate_metrics(y_test_orig, cnn_pred_orig, "CNN Single-step")
        all_results[repo_id]['CNN']['Single-step'] = metrics_cnn_single

        # B. Multi-step
        cnn_forecast_long = perform_multi_step_forecasting_dl(
            cnn_model, last_train_seq_scaled, max_horizon, scaler, sequence_length, device
        )
        
        for h in horizons_to_eval:
            if len(actual_multistep_values) >= h:
                rmse_h = np.sqrt(mean_squared_error(actual_multistep_values[:h], cnn_forecast_long[:h]))
                all_results[repo_id]['CNN']['Multi-step'][h] = rmse_h
                print(f"  CNN RMSE (H={h}): {rmse_h:.4f}")

        # --- PLOT CNN ---
        plot_forecast(actual_series_30d, 
                      cnn_forecast_long, 
                      'CNN 30-Day', repo_id, multi_step_dates)
        
        # ==========================================
        # PLOT ERROR VS HORIZON
        # ==========================================
        print(f"Generating Error vs Horizon plot for {repo_id}...")
        plot_error_vs_horizon(all_results[repo_id], repo_id)

    # ==========================================
    # SUMMARY TABLE GENERATION
    # ==========================================
    print("\n\n--- Final Summary ---")
    summary_rows = []
    
    for repo_name, models_dict in all_results.items():
        for model_name, metrics in models_dict.items():
            row = {
                'Repository': repo_name,
                'Model': model_name,
                'MAE (Single-step)': metrics['Single-step']['MAE'],
                'RMSE (Single-step)': metrics['Single-step']['RMSE']
            }
            
            # Add Multi-step columns
            ms_metrics = metrics.get('Multi-step', {})
            for h in horizons_to_eval:
                row[f'RMSE (H={h})'] = ms_metrics.get(h, np.nan)
            
            summary_rows.append(row)

    cols_order = ['Repository', 'Model', 'MAE (Single-step)', 'RMSE (Single-step)'] + \
                 [f'RMSE (H={h})' for h in horizons_to_eval]
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[cols_order] 
    
    print(summary_df.to_string(index=False))
    summary_df.to_csv('forecasting_summary_metrics.csv', index=False)
    print("\nResults saved to 'forecasting_summary_metrics.csv'")