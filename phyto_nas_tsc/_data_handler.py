import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional
from importlib.resources import files

__all__ = ['DataHandler', 'validate_inputs']

class DataHandler:
    def __init__(self, data_dir='classification_ozone'):
        self.data_dir = Path(data_dir)
        self.X_analysis = None
        self.y_analysis = None
        self.X_test = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        
    def load_and_preprocess(self):
        """Load and preprocess all data"""
        self.X_analysis = pd.read_csv(self.data_dir/'X_train.csv')
        self.y_analysis = pd.read_csv(self.data_dir/'y_train.csv')
        self.X_test = pd.read_csv(self.data_dir/'X_test.csv')
        self.y_test = pd.read_csv(self.data_dir/'y_test.csv')

        # Preprocessing (same as your working version)
        self.X_analysis.fillna(self.X_analysis.mean(), inplace=True)
        self.X_test.fillna(self.X_test.mean(), inplace=True)
        
        self.X_analysis = self.scaler.fit_transform(self.X_analysis)
        self.X_test = self.scaler.transform(self.X_test)

        self.X_analysis = self.X_analysis.reshape(-1, 1, self.X_analysis.shape[1])
        self.X_test = self.X_test.reshape(-1, 1, self.X_test.shape[1])
        
        # One-hot encoding (same logic)
        self.y_analysis = self.encoder.fit_transform(
            self.y_analysis.to_numpy().reshape(-1, 1))
        self.y_test = self.encoder.transform(
            self.y_test.to_numpy().reshape(-1, 1))

    def get_data_splits(self):
        """Generate data splits for cross-validation"""
        if self.X_analysis is None:
            self.load_and_preprocess()
            
        rkf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in rkf.split(self.X_analysis):
            X_train, X_val = self.X_analysis[train_idx], self.X_analysis[val_idx]
            y_train, y_val = self.y_analysis[train_idx], self.y_analysis[val_idx]

            print("Train classes:", np.unique(np.argmax(y_train, axis=1), return_counts=True))
            print("Val classes:", np.unique(np.argmax(y_val, axis=1), return_counts=True))
            
            # Reshape and convert to tensors
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            yield (
                DataLoader(train_dataset, batch_size=32, shuffle=True, 
                         num_workers=4, persistent_workers=True),
                DataLoader(val_dataset, batch_size=32, shuffle=False,
                         num_workers=4, persistent_workers=True)
            )

# Default instance for backward compatibility
default_handler = DataHandler()
get_data_splits = default_handler.get_data_splits

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input shapes and types
    
    Args:
        X: Input features (n_samples, n_timesteps, n_features)
        y: One-hot encoded labels (n_samples, n_classes)
    
    Raises:
        ValueError: If inputs have invalid shapes or types
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    if len(X.shape) != 3:
        raise ValueError(
            f"X must be 3D array (samples, timesteps, features), got {X.shape}"
        )
    
    if len(y.shape) != 2:
        raise ValueError(
            f"y must be 2D one-hot encoded array (samples, classes), got {y.shape}"
        )
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}"
        )