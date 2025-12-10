# data/data_loader.py
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and manage time series sensor data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config['data']['local_path'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> None:
        """Download NASA bearing dataset if not present."""
        if not list(self.data_dir.glob("*.csv")):
            logger.info("Downloading NASA bearing dataset...")
            # For demo purposes, we'll create synthetic data
            # In real scenario, download from Kaggle
            self._create_synthetic_data()
        else:
            logger.info("Dataset already exists locally.")
    
    def _create_synthetic_data(self) -> None:
        """Create synthetic time series data with anomalies for demonstration."""
        np.random.seed(self.config['data']['random_state'])
        
        # Generate normal time series data
        n_samples = 10000
        time = np.arange(n_samples)
        
        # Create 4 sensor readings with different patterns
        sensors = {}
        
        # Sensor 1: Sine wave with noise
        sensors['sensor_1'] = 10 * np.sin(2 * np.pi * time / 1000) + np.random.normal(0, 0.5, n_samples)
        
        # Sensor 2: Linear trend with seasonal component
        sensors['sensor_2'] = 0.001 * time + 5 * np.sin(2 * np.pi * time / 500) + np.random.normal(0, 1, n_samples)
        
        # Sensor 3: Random walk
        sensors['sensor_3'] = np.cumsum(np.random.normal(0, 1, n_samples))
        
        # Sensor 4: Stationary noise
        sensors['sensor_4'] = np.random.normal(20, 2, n_samples)
        
        # Create anomalies at specific intervals
        anomalies = np.zeros(n_samples, dtype=bool)
        
        # Spike anomalies
        spike_indices = [1500, 3200, 5500, 7800]
        for idx in spike_indices:
            sensors['sensor_1'][idx:idx+5] += np.random.uniform(15, 25, 5)
            sensors['sensor_2'][idx:idx+5] += np.random.uniform(-20, -10, 5)
            anomalies[idx:idx+5] = True
        
        # Drift anomalies
        drift_indices = [2500, 4500, 6500]
        for idx in drift_indices:
            drift_length = np.random.randint(20, 50)
            drift = np.linspace(0, 30, drift_length)
            sensors['sensor_3'][idx:idx+drift_length] += drift
            anomalies[idx:idx+drift_length] = True
        
        # Level shift
        shift_idx = 6000
        shift_length = 100
        sensors['sensor_4'][shift_idx:shift_idx+shift_length] += 15
        anomalies[shift_idx:shift_idx+shift_length] = True
        
        # Save to CSV
        df = pd.DataFrame(sensors)
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=n_samples, freq='T')
        df['is_anomaly'] = anomalies.astype(int)
        
        # Add missing values (5% randomly)
        mask = np.random.random(df.shape) < 0.05
        df.mask(mask, inplace=True)
        
        # Save files
        df.to_csv(self.data_dir / "bearing_sensor_data.csv", index=False)
        logger.info(f"Synthetic data created with shape: {df.shape}")
        
        # Also create a test file
        test_df = df.iloc[-2000:].copy()
        test_df.to_csv(self.data_dir / "test_data.csv", index=False)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load and return sensor data with labels."""
        data_path = self.data_dir / "bearing_sensor_data.csv"
        
        if not data_path.exists():
            self.download_dataset()
        
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Separate features and labels
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        X = df[sensor_cols]
        y = df['is_anomaly'] if 'is_anomaly' in df.columns else pd.Series(np.zeros(len(df)))
        
        logger.info(f"Loaded data shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Anomaly percentage: {y.mean():.2%}")
        
        return X, y, df['timestamp']
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                         timestamps: pd.Series) -> Tuple:
        """Split data into train and test sets preserving temporal order."""
        split_idx = int(len(X) * (1 - self.config['data']['test_size']))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        timestamps_train = timestamps.iloc[:split_idx]
        timestamps_test = timestamps.iloc[split_idx:]
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, timestamps_train, timestamps_test