# classical.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fit_arma_model(train_series, order=(2, 0, 1)):
    """
    Fits an ARMA model to the training series.
    For ARMA, the 'd' (differencing) component is 0.
    If you find your series needs differencing, you'd use ARIMA.
    """
    # ARIMA(p,d,q) -> ARMA(p,q) when d=0
    # You might need to determine p and q using ACF/PACF plots or auto_arima
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    print(f"ARMA({order[0]}, {order[1]}, {order[2]}) Model Summary:")
    print(model_fit.summary())
    return model_fit

def forecast_arma(model_fit, steps, train_series):
    """
    Generates multi-step forecasts using the fitted ARMA model.
    Uses the model's predict method for out-of-sample forecasting.
    """
    # For multi-step forecasting, we predict from the end of the training data
    # to 'steps' into the future.
    # The 'start' parameter should be the index after the last training point.
    # The 'end' parameter should be 'start + steps - 1'.
    
    # Get the index of the last training data point
    last_train_index = train_series.index[-1]
    
    # Create a future index for the predictions
    # Assuming daily frequency, adjust if your data has a different frequency
    future_index = pd.date_range(start=last_train_index + pd.Timedelta(days=1),
                                 periods=steps,
                                 freq='D')
    
    # Predict using the fitted model
    # 'dynamic=False' means using actual past values for in-sample predictions,
    # but for out-of-sample, it's always dynamic.
    # 'start' and 'end' specify the range of the *original series index* to predict.
    # We want to predict from the first test point up to 'steps' into the future.
    
    # A more robust way for multi-step out-of-sample forecasting with ARIMA is often:
    forecast_results = model_fit.get_forecast(steps=steps)
    predictions = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int() # Get confidence intervals
    
    # Ensure predictions have the correct index
    predictions.index = future_index
    
    return predictions, conf_int

if __name__ == '__main__':
    # Example Usage:
    # Load and preprocess data (similar to prep_stars.py and split_repos.py)
    from prep_stars import load_data, clean_and_align_time_series, transform_data_domain, scale_data
    from split_repos import create_train_test_split

    stars_df = load_data()
    unique_repo_ids = stars_df['repository_id'].unique()
    repo_id_1 = unique_repo_ids[0] # Example ID
    
    repo1_cleaned = clean_and_align_time_series(stars_df, repo_id_1)
    
    # IMPORTANT: For ARMA/ARIMA, it's often better to work with differenced (incremental)
    # and unscaled data, or scale after differencing.
    # Let's use incremental data for ARMA, but ensure it's stationary.
    # If the incremental data is still not stationary, you might need ARIMA (d > 0).
    repo1_transformed = transform_data_domain(repo1_cleaned.copy(), domain='incremental')
    
    # For classical models, we might not always scale the target directly,
    # or we scale, predict, and then inverse transform.
    # Let's work with the raw incremental values for ARMA for simplicity here.
    # If using cumulative, you'd likely need ARIMA(p,1,q) for differencing.
    
    train_df_classical, test_df_classical = create_train_test_split(
        repo1_transformed[['stars_transformed']], test_size=0.2
    )
    
    train_series = train_df_classical['stars_transformed']
    test_series = test_df_classical['stars_transformed']
    
    # Determine ARMA order (p, d, q) - this often requires ACF/PACF plots or auto_arima
    # For now, let's use an example order (e.g., AR(1))
    arma_order = (1, 0, 0) # p=1, d=0, q=0 (AR(1) model)
    
    print(f"\nFitting ARMA model for Repository {repo_id_1} with order {arma_order}")
    arma_model = fit_arma_model(train_series, order=arma_order)
    
    # Multi-step forecasting
    forecast_horizon = len(test_series) # Forecast for the entire test period
    arma_predictions, arma_conf_int = forecast_arma(arma_model, forecast_horizon, train_series)
    
    print("\nARMA Predictions Head:")
    print(arma_predictions.head())
    
    # Align predictions with actual test data for evaluation
    # Ensure both have the same index for direct comparison
    actual_test_values = test_series.reindex(arma_predictions.index)
    
    # Evaluate performance
    mae = mean_absolute_error(actual_test_values, arma_predictions)
    rmse = np.sqrt(mean_squared_error(actual_test_values, arma_predictions))
    
    print(f"\nARMA MAE: {mae:.4f}")
    print(f"ARMA RMSE: {rmse:.4f}")
    
    # Plotting (optional, but good for understanding)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    plt.plot(train_series.index, train_series, label='Train Data')
    plt.plot(test_series.index, test_series, label='Actual Test Data')
    plt.plot(arma_predictions.index, arma_predictions, label='ARMA Forecast')
    plt.fill_between(arma_conf_int.index, arma_conf_int.iloc[:, 0], arma_conf_int.iloc[:, 1], color='k', alpha=.1, label='95% Confidence Interval')
    plt.title(f'ARMA Forecast for Repository {repo_id_1}')
    plt.xlabel('Date')
    plt.ylabel('Incremental Stars')
    plt.legend()
    plt.show()