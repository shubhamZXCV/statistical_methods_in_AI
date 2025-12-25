# train_models.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import models and data preparation functions
from dl_models import RNNForecaster, CNN1DForecaster # Use the robust CNN
from prep_stars import load_data, clean_and_align_time_series, transform_data_domain, scale_data
from split_repos import create_train_test_split, create_sequences

def train_dl_model(model, X_train, y_train, X_val=None, y_val=None, 
                   epochs=50, batch_size=32, learning_rate=0.001, 
                   loss_fn_name='MSE', device='cpu'):
    """
    Trains a deep learning model.
    Includes basic validation if X_val, y_val are provided.
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device) # Add feature dim for target

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    # Define loss function
    if loss_fn_name == 'MSE':
        criterion = nn.MSELoss()
    elif loss_fn_name == 'MAE':
        criterion = nn.L1Loss() # MAE in PyTorch
    else:
        raise ValueError("Loss function must be 'MSE' or 'MAE'")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    history = {'train_loss': [], 'val_loss': []}

    print(f"Training model on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        if val_loader:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
            epoch_val_loss = val_running_loss / len(val_dataset)
            history['val_loss'].append(epoch_val_loss)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}')
            
    print("Training complete.")
    return model, history

def predict_dl_model(model, X_test, device='cpu'):
    """
    Generates predictions using a trained deep learning model.
    """
    model.eval() # Set model to evaluation mode
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    predictions = []
    with torch.no_grad():
        # Predict in batches if X_test is large
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        for inputs in test_loader:
            outputs = model(inputs[0]) # inputs is a tuple (input_tensor,)
            predictions.extend(outputs.cpu().numpy().flatten())
            
    return np.array(predictions)

def perform_hyperparameter_search(X_train, y_train, X_val, y_val, model_type, sequence_length, device):
    """
    Placeholder for hyperparameter search.
    You would typically use libraries like Optuna, Keras Tuner, or GridSearchCV.
    For this example, we'll just try a few manual combinations.
    """
    best_model = None
    best_val_loss = float('inf')
    best_params = {}

    # Example hyperparameters to search
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]
    
    if model_type == 'RNN':
        hidden_sizes = [32, 64]
        num_layers_options = [1, 2]
        
        for lr in learning_rates:
            for bs in batch_sizes:
                for hs in hidden_sizes:
                    for nl in num_layers_options:
                        print(f"\nTrying RNN: LR={lr}, BS={bs}, HS={hs}, NL={nl}")
                        model = RNNForecaster(input_size=1, hidden_size=hs, num_layers=nl, output_size=1).to(device)
                        trained_model, history = train_dl_model(model, X_train, y_train, X_val, y_val, 
                                                                epochs=20, batch_size=bs, learning_rate=lr, 
                                                                loss_fn_name='MSE', device=device)
                        
                        current_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            best_model = trained_model
                            best_params = {'lr': lr, 'bs': bs, 'hidden_size': hs, 'num_layers': nl}
                            print(f"New best RNN model found with Val Loss: {best_val_loss:.4f}")

    elif model_type == 'CNN':
        num_filters_options = [32, 64]
        kernel_sizes = [3, 5]

        for lr in learning_rates:
            for bs in batch_sizes:
                for nf in num_filters_options:
                    for ks in kernel_sizes:
                        print(f"\nTrying CNN: LR={lr}, BS={bs}, NF={nf}, KS={ks}")
                        model = CNN1DForecaster(input_channels=1, output_size=1, 
                                                      sequence_length=sequence_length, 
                                                      kernel_size=ks, num_filters=nf).to(device)
                        trained_model, history = train_dl_model(model, X_train, y_train, X_val, y_val, 
                                                                epochs=20, batch_size=bs, learning_rate=lr, 
                                                                loss_fn_name='MSE', device=device)
                        
                        current_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            best_model = trained_model
                            best_params = {'lr': lr, 'bs': bs, 'num_filters': nf, 'kernel_size': ks}
                            print(f"New best CNN model found with Val Loss: {best_val_loss:.4f}")
    
    return best_model, best_params, best_val_loss


if __name__ == '__main__':
    # --- Setup for a single repository ---
    stars_df = load_data()
    unique_repo_ids = stars_df['repository_id'].unique()
    repo_id_1 = unique_repo_ids[0] # Example ID
    
    repo1_cleaned = clean_and_align_time_series(stars_df, repo_id_1)
    repo1_transformed = transform_data_domain(repo1_cleaned.copy(), domain='incremental')
    repo1_scaled, scaler1 = scale_data(repo1_transformed.copy(), scaler_type='minmax', feature_col='stars_transformed')
    
    # Split into train, validation, and test sets for DL models
    # First, split into train+val and test
    train_val_df, test_df = create_train_test_split(repo1_scaled[['stars_transformed_scaled']], test_size=0.2)
    # Then, split train+val into train and val
    train_df, val_df = create_train_test_split(train_val_df, test_size=0.25) # 0.25 of 80% is 20% of total

    sequence_length = 30 # Example sequence length
    
    X_train, y_train = create_sequences(train_df['stars_transformed_scaled'].values, sequence_length)
    X_val, y_val = create_sequences(val_df['stars_transformed_scaled'].values, sequence_length)
    X_test, y_test = create_sequences(test_df['stars_transformed_scaled'].values, sequence_length)

    # Reshape for DL models: (num_samples, sequence_length, input_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameter Search and Training for RNN ---
    print("\n--- Performing Hyperparameter Search for RNN ---")
    best_rnn_model, best_rnn_params, best_rnn_val_loss = perform_hyperparameter_search(
        X_train, y_train, X_val, y_val, 'RNN', sequence_length, device
    )
    print(f"\nBest RNN Params: {best_rnn_params}, Best Val Loss: {best_rnn_val_loss:.4f}")

    # --- Hyperparameter Search and Training for CNN ---
    print("\n--- Performing Hyperparameter Search for CNN ---")
    best_cnn_model, best_cnn_params, best_cnn_val_loss = perform_hyperparameter_search(
        X_train, y_train, X_val, y_val, 'CNN', sequence_length, device
    )
    print(f"\nBest CNN Params: {best_cnn_params}, Best Val Loss: {best_cnn_val_loss:.4f}")

    # --- Make predictions with the best models on the test set ---
    print("\n--- Making predictions on test set ---")
    rnn_predictions_scaled = predict_dl_model(best_rnn_model, X_test, device)
    cnn_predictions_scaled = predict_dl_model(best_cnn_model, X_test, device)
    
    # Inverse transform predictions to original scale
    # Need to create a dummy array with the correct shape for inverse_transform
    # The scaler was fitted on 'stars_transformed', so we need to inverse transform that.
    # The scaler expects a 2D array, so we reshape.
    
    # For RNN predictions
    dummy_rnn_for_inverse = np.zeros((len(rnn_predictions_scaled), scaler1.n_features_in_))
    dummy_rnn_for_inverse[:, 0] = rnn_predictions_scaled # Place predictions in the first feature column
    rnn_predictions_original_scale = scaler1.inverse_transform(dummy_rnn_for_inverse)[:, 0]

    # For CNN predictions
    dummy_cnn_for_inverse = np.zeros((len(cnn_predictions_scaled), scaler1.n_features_in_))
    dummy_cnn_for_inverse[:, 0] = cnn_predictions_scaled
    cnn_predictions_original_scale = scaler1.inverse_transform(dummy_cnn_for_inverse)[:, 0]

    # Inverse transform actual test values for comparison
    dummy_y_test_for_inverse = np.zeros((len(y_test), scaler1.n_features_in_))
    dummy_y_test_for_inverse[:, 0] = y_test
    y_test_original_scale = scaler1.inverse_transform(dummy_y_test_for_inverse)[:, 0]

    # Evaluate DL models (on original scale)
    rnn_mae = mean_absolute_error(y_test_original_scale, rnn_predictions_original_scale)
    rnn_rmse = np.sqrt(mean_squared_error(y_test_original_scale, rnn_predictions_original_scale))
    print(f"\nRNN Test MAE (original scale): {rnn_mae:.4f}")
    print(f"RNN Test RMSE (original scale): {rnn_rmse:.4f}")

    cnn_mae = mean_absolute_error(y_test_original_scale, cnn_predictions_original_scale)
    cnn_rmse = np.sqrt(mean_squared_error(y_test_original_scale, cnn_predictions_original_scale))
    print(f"\nCNN Test MAE (original scale): {cnn_mae:.4f}")
    print(f"CNN Test RMSE (original scale): {cnn_rmse:.4f}")
    
    # Save models (optional)
    # torch.save(best_rnn_model.state_dict(), f'best_rnn_model_repo_{repo_id_1}.pth')
    # torch.save(best_cnn_model.state_dict(), f'best_cnn_model_repo_{repo_id_1}.pth')