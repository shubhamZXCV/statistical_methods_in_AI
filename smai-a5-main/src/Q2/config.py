# config.py
import torch

# File Paths
DATA_FILE = '../../dataset/Q2/recurrence_timeseries.csv'
LOG_FILE = 'experiment_log.txt'
TUNING_RESULTS_FILE = 'tuning_results.csv'

# Tuning Parameters
# We will test these history lengths to find the "Effective Order"
P_VALUES_TO_TUNE = [1, 2, 3, 5, 8, 10, 15, 20] 

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 9
HIDDEN_DIM = 32
LAYERS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')