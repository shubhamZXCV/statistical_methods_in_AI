# dl_models.py

import torch
import torch.nn as nn

class RNNForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM is a type of RNN, often performs better than simple RNNs
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to output the forecast
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM layer
        # out: (batch_size, sequence_length, hidden_size)
        # hn, cn: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We only care about the last output of the sequence for prediction
        # out[:, -1, :] gives the output of the last time step for all batches
        out = self.fc(out[:, -1, :])
        return out

class CNN1DForecaster(nn.Module): # Renamed to the robust version
    def __init__(self, input_channels, output_size, sequence_length, kernel_size=3, num_filters=64):
        super(CNN1DForecaster, self).__init__() # Changed to CNN1DForecaster
        self.sequence_length = sequence_length
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Calculate the size for the first FC layer dynamically
        self.fc_input_size = self._get_conv_output_size(input_channels, kernel_size, num_filters, sequence_length)
        self.fc = nn.Linear(self.fc_input_size, output_size)

    def _get_conv_output_size(self, input_channels, kernel_size, num_filters, sequence_length):
        dummy_input = torch.randn(1, input_channels, sequence_length)
        x = self.conv1(dummy_input)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x.numel()

    def forward(self, x):
        x = x.permute(0, 2, 1) # Reshape from (batch, seq_len, features) to (batch, features, seq_len)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Example Usage:
    # Define parameters
    input_size = 1 # Number of features (e.g., scaled stars)
    output_size = 1 # Predicting the next single star count
    sequence_length = 30 # From split_repos.py

    # --- RNN Example ---
    hidden_size_rnn = 50
    num_layers_rnn = 2
    rnn_model = RNNForecaster(input_size, hidden_size_rnn, num_layers_rnn, output_size)
    
    # Create a dummy input tensor: (batch_size, sequence_length, input_size)
    dummy_rnn_input = torch.randn(64, sequence_length, input_size)
    rnn_output = rnn_model(dummy_rnn_input)
    print(f"RNN output shape: {rnn_output.shape}") # Should be (batch_size, output_size)

    # --- 1D CNN Example (using the robust version) ---
    # For CNN, input_channels is usually 1 for a single time series
    input_channels_cnn = 1 
    kernel_size_cnn = 3
    num_filters_cnn = 64
    
    # Now create the actual CNN model with the correct FC layer input size
    # using the CNN1DForecaster (which is now the robust one)
    cnn_model = CNN1DForecaster(input_channels_cnn, output_size, sequence_length, kernel_size_cnn, num_filters_cnn)
    dummy_cnn_input = torch.randn(64, sequence_length, input_channels_cnn)
    cnn_output = cnn_model(dummy_cnn_input)
    print(f"CNN output shape: {cnn_output.shape}") # Should be (batch_size, output_size)