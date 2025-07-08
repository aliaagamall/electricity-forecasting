import torch
import torch.nn as nn
from typing import Literal

class TimeSeriesRNN(nn.Module):
    """RNN model for time series forecasting"""
    def __init__(self, rnn_type: Literal["RNN", "GRU", "LSTM"] = "GRU"):
        super().__init__()
        self.rnn_type = rnn_type
        
        # RNN layer selection
        rnn_constructors = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM
        }
        self.rnn = rnn_constructors[rnn_type](
            input_size=1, 
            hidden_size=32, 
            num_layers=2, 
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state
        h0 = torch.zeros(2, batch_size, 32).to(device)
        if self.rnn_type == "LSTM":
            c0 = torch.zeros(2, batch_size, 32).to(device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
            
        return self.fc(out[:, -1, :])
