import os
from dataclasses import dataclass

@dataclass
class Config:
    """Main configuration class"""

    # Model settings
    seq_length: int = 96

    # File paths
    model_path: str = "../models/TimeSeriesRNN.pth"
    scaler_path: str = "../models/scaler.pkl"

    # Data settings
    timestamp_column: str = "timestamp"
    target_column: str = "consumption"

    # Data Paths
    train_data_path: str = "../data/raw/electricity_train.csv"
    test_data_path: str = "../data/raw/electricity_test.csv"

    # SHAP settings
    max_display: int = 20
    save_plots: bool = True
    plots_dir: str = "plots"

    def __post_init__(self):
        """Create plots directory if it doesn't exist"""
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)

# Global config instance
config = Config()