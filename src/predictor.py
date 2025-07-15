import pandas as pd
import numpy as np
import torch 
from sklearn.preprocessing import PowerTransformer
import pickle
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from model import TimeSeriesRNN

class ElectricityPredictor:
    """Simple electricity consumption predictor"""
    
    def __init__(self, model_path=Config.model_path, scaler_path=Config.scaler_path):
        self.model_path = model_path 
        self.scaler_path = scaler_path 
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.data = None
        
        # Load model and scaler
        self._load_model()
        self._load_scaler()
    
    def _load_model(self):
        """Load trained model"""
        self.model = TimeSeriesRNN()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def _load_scaler(self):
        """Load fitted scaler"""
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {self.scaler_path}")
    
    def load_and_prepare_data(self, csv_path):
        """Load and prepare data for prediction"""
        # Load data
        df = pd.read_csv(csv_path)
        df[Config.timestamp_column] = pd.to_datetime(df[Config.timestamp_column])
        df.sort_values(Config.timestamp_column, inplace=True)
        df.set_index(Config.timestamp_column, inplace=True)
        
        # Apply transformations (same as training)
        df[Config.target_column] = self.power_transformer.fit_transform(df[[Config.target_column]])
        df[Config.target_column] = self.scaler.transform(df[[Config.target_column]])
        
        self.data = df
        print(f"Data prepared with shape: {df.shape}")
        return df
    
    def get_latest_sequence(self):
        """Get latest sequence for prediction"""
        if len(self.data) < Config.seq_length:
            raise ValueError("Not enough data to form a full sequence.")
        
        last_seq = self.data[Config.target_column].values[-Config.seq_length:]
        return torch.tensor(last_seq, dtype=torch.float32).view(1, Config.seq_length, 1)
    
    def predict_next(self, inverse_transform=True):
        """Predict next value"""
        seq = self.get_latest_sequence()
        
        with torch.no_grad():
            pred = self.model(seq)
            pred_val = pred.item() if pred.numel() == 1 else pred[0, 0].item()
        
        if inverse_transform:
            # Inverse transform to original scale
            scaled_back = self.scaler.inverse_transform([[pred_val]])[0][0]
            original_val = self.power_transformer.inverse_transform([[scaled_back]])[0][0]
            return original_val
        
        return pred_val
    
    def predict_multiple(self, n_steps, inverse_transform=True):
        """Predict multiple steps ahead"""
        predictions = []
        current_data = self.data.copy()
        
        for _ in range(n_steps):
            # Get latest sequence
            seq = self._get_sequence_from_data(current_data)
            
            # Predict
            with torch.no_grad():
                pred = self.model(seq)
                pred_val = pred.item() if pred.numel() == 1 else pred[0, 0].item()
            
            predictions.append(pred_val)
            
            # Add prediction to data for next step
            next_timestamp = current_data.index[-1] + pd.Timedelta(minutes=15)
            current_data.loc[next_timestamp] = {Config.target_column: pred_val}
        
        predictions = np.array(predictions)
        
        if inverse_transform:
            predictions = np.array([
                self.power_transformer.inverse_transform(
                    [[self.scaler.inverse_transform([[p]])[0][0]]]
                )[0][0] for p in predictions
            ])
        
        return predictions
    
    def _get_sequence_from_data(self, data):
        """Helper to get sequence from data"""
        last_seq = data[Config.target_column].values[-Config.seq_length:]
        return torch.tensor(last_seq, dtype=torch.float32).view(1, Config.seq_length, 1)
    
    def explain_prediction(self, save_plot=None):
        """Explain prediction using SHAP"""
        seq = self.get_latest_sequence()
        
        # Create background (baseline)
        background = torch.zeros_like(seq)
        
        # Create SHAP explainer
        explainer = shap.GradientExplainer(self.model, background)
        
        # Get SHAP values
        shap_values = explainer.shap_values(seq)
        
        # Handle list output
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        # Get base value
        with torch.no_grad():
            base_output = self.model(background)
            base_value = base_output.item() if base_output.numel() == 1 else base_output[0, 0].item()
        
        # Create explanation
        explanation = shap.Explanation(
            values=shap_vals[0].flatten(),
            base_values=base_value,
            data=seq[0].squeeze().numpy(),
            feature_names=[f"t-{Config.seq_length - i}" for i in range(Config.seq_length)]
        )
        
        # Plot waterfall
        shap.plots.waterfall(explanation, max_display=Config.max_display)
        
        # Save plot if requested
        if save_plot or (save_plot is None and Config.save_plots):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f"{Config.plots_dir}/shap_explanation_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        return explanation
    
    def get_prediction_summary(self):
        """Get summary of current prediction"""
        if self.data is None:
            return "No data loaded"
        
        # Get prediction
        pred = self.predict_next()
        
        # Get current stats
        latest_values = self.data[Config.target_column].tail(10)
        
        # Transform back for comparison
        latest_original = [
            self.power_transformer.inverse_transform(
                [[self.scaler.inverse_transform([[val]])[0][0]]]
            )[0][0] for val in latest_values
        ]
        
        return {
            "prediction": pred,
            "latest_10_values": latest_original,
            "data_range": f"{self.data.index[0]} to {self.data.index[-1]}",
            "total_points": len(self.data)
        }
