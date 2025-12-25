# model.py
import torch
import torch.nn as nn

class RNNPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super(RNNPredictor, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear Layer to map hidden state to single output
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_time_step = out[:, -1, :]
        
        prediction = self.fc(last_time_step)
        return prediction.squeeze()