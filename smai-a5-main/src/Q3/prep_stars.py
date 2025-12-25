# prep_stars.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(stars_path='../../dataset/Q3/stars_data.csv', metadata_path='../../dataset/Q3/repo_metadata.json'):
    """Loads the stars data and (optionally) metadata."""
    stars_df = pd.read_csv(stars_path)
    # Convert 'timestamp' to datetime objects
    stars_df['timestamp'] = pd.to_datetime(stars_df['timestamp'])
    
    return stars_df

def clean_and_align_time_series(df, repo_id):
    """
    Cleans and aligns the time series for a single repository.
    Handles missing timestamps by reindexing and filling.
    Ensures cumulative stars are monotonically increasing.
    Removes trailing data points where the star count plateaus (entries remain the same).
    """
    repo_df = df[df['repository_id'] == repo_id].sort_values('timestamp').copy()
    
    # Set timestamp as index for easier time series operations
    repo_df = repo_df.set_index('timestamp')
    
    # Reindex to a regular frequency (daily)
    full_time_range = pd.date_range(start=repo_df.index.min(), end=repo_df.index.max(), freq='D')
    repo_df = repo_df.reindex(full_time_range)
    
    # Fill missing 'stars' values.
    repo_df['stars'] = repo_df['stars'].ffill()
    repo_df['stars'] = repo_df['stars'].bfill() 
    
    # Ensure stars are non-decreasing (cumulative property)
    repo_df['stars'] = repo_df['stars'].cummax()
    
    # --- UPDATED SECTION STARTS HERE ---
    # Remove trailing identical entries. 
    # Since data is cumulative, the last value is the max value. 
    # We want to keep data only up to the *first* time the final value was achieved.
    
    if not repo_df.empty:
        final_value = repo_df['stars'].iloc[-1]
        
        # Find the first index where the star count reached the final value
        # This determines where the 'plateau' begins
        cutoff_date = repo_df[repo_df['stars'] == final_value].index[0]
        
        # Slice the dataframe to end at that date
        repo_df = repo_df.loc[:cutoff_date]
    # --- UPDATED SECTION ENDS HERE ---

    # Drop the 'repo_id' column as it's constant for this single repo
    repo_df = repo_df[['stars']]
    
    return repo_df

def transform_data_domain(df, domain='cumulative'):
    """
    Transforms the data to cumulative or incremental domain.
    'cumulative': y_t
    'incremental': y_t - y_{t-1}
    """
    if domain == 'incremental':
        # Calculate incremental stars. The first value will be NaN, fill with 0 or the first cumulative value.
        df['stars_transformed'] = df['stars'].diff().fillna(df['stars'].iloc[0])
        # Ensure incremental values are non-negative
        df['stars_transformed'] = df['stars_transformed'].clip(lower=0)
    elif domain == 'cumulative':
        df['stars_transformed'] = df['stars']
    else:
        raise ValueError("Domain must be 'cumulative' or 'incremental'")
    return df

def scale_data(df, scaler_type='minmax', feature_col='stars_transformed'):
    """
    Scales the specified feature column.
    Returns the scaled data and the fitted scaler object.
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Scaler type must be 'minmax' or 'standard'")
    
    # Fit and transform the data
    df[f'{feature_col}_scaled'] = scaler.fit_transform(df[[feature_col]])
    
    return df, scaler

if __name__ == '__main__':
    # Example Usage:
    try:
        stars_df = load_data()
        
        unique_repo_ids = stars_df['repository_id'].unique()
        print(f"Total unique repositories: {len(unique_repo_ids)}")
        
        # Pick the first repository
        repo_id_1 = unique_repo_ids[0]
        
        print(f"\nProcessing Repository: {repo_id_1}")
        repo1_cleaned = clean_and_align_time_series(stars_df, repo_id_1)
        
        print("Cleaned Data Head:")
        print(repo1_cleaned.head())
        print("\nCleaned Data Tail (Should not show a long list of identical values):")
        print(repo1_cleaned.tail())
        
        # Verify the update
        print(f"\nFinal Star Count: {repo1_cleaned['stars'].iloc[-1]}")
        print(f"Last Date Included: {repo1_cleaned.index[-1]}")

        repo1_incremental = transform_data_domain(repo1_cleaned.copy(), domain='incremental')
        repo1_scaled, scaler1 = scale_data(repo1_incremental.copy(), scaler_type='standard', feature_col='stars_transformed')
        print("\nScaled Data Head:")
        print(repo1_scaled.head())
        
    except FileNotFoundError:
        print("Error: Dataset files not found. Please check the paths in load_data().")
    except Exception as e:
        print(f"An error occurred: {e}")