# split_repos.py

import pandas as pd
import numpy as np
# Assuming prep_stars.py is in the same directory or accessible
from prep_stars import load_data, clean_and_align_time_series, transform_data_domain, scale_data

def create_train_test_split(df, test_size=0.2):
    """
    Creates a chronological train-test split for a single time series.
    No future data leakage.
    """
    # Ensure the DataFrame is sorted by index (timestamp)
    df = df.sort_index()
    
    # Calculate split point
    split_point = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, test_df

def create_sequences(data, sequence_length):
    """
    Creates input-output sequences for RNN/CNN models.
    data: A 1D numpy array or pandas Series of the time series values.
    sequence_length: The number of past time steps to use as input (look-back window).
    
    Returns: X (input sequences), y (target values)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # Example Usage:
    stars_df = load_data()
    
    # Choose 2 repositories (same as in prep_stars.py example)
    unique_repo_ids = stars_df['repository_id'].unique()
    repo_id_1 = unique_repo_ids[2] # Example ID
    
    print(f"\nProcessing Repository: {repo_id_1}")
    repo1_cleaned = clean_and_align_time_series(stars_df, repo_id_1)
    repo1_transformed = transform_data_domain(repo1_cleaned.copy(), domain='incremental')
    repo1_scaled, scaler1 = scale_data(repo1_transformed.copy(), scaler_type='minmax', feature_col='stars_transformed')
    
    # Get the scaled time series values
    time_series_values = repo1_scaled['stars_transformed_scaled'].values
    
    # Create train/test split on the scaled data
    train_data_scaled, test_data_scaled = create_train_test_split(
        repo1_scaled[['stars_transformed_scaled']], test_size=0.2
    )
    
    # For deep learning models, you'll need to create sequences
    sequence_length = 30 # Example: Use last 30 days to predict next day
    
    X_train, y_train = create_sequences(train_data_scaled['stars_transformed_scaled'].values, sequence_length)
    X_test, y_test = create_sequences(test_data_scaled['stars_transformed_scaled'].values, sequence_length)
    
    print(f"\nShape of X_train: {X_train.shape}") # (num_samples, sequence_length)
    print(f"Shape of y_train: {y_train.shape}")   # (num_samples,)
    
    # For RNNs, you often need an extra dimension for features (even if it's 1)
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # print(f"Shape of X_train (for RNN): {X_train.shape}")
    
    # You would pass these X_train, y_train, X_test, y_test to your deep learning models.
    # For classical models, you'd use the raw train_data_scaled and test_data_scaled directly.