import numpy as np
import pandas as pd
from scipy import stats, fft
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create features from raw time series data."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_names = []
        
    def create_rolling_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        features_list = []
        
        for window in self.config['features']['rolling_windows']:
            for col in X.columns:
                # Rolling statistics
                rolling_mean = X[col].rolling(window=window, center=True, min_periods=1).mean()
                rolling_std = X[col].rolling(window=window, center=True, min_periods=1).std()
                rolling_min = X[col].rolling(window=window, center=True, min_periods=1).min()
                rolling_max = X[col].rolling(window=window, center=True, min_periods=1).max()
                rolling_median = X[col].rolling(window=window, center=True, min_periods=1).median()
                
                # Rate of change
                rolling_change = X[col].diff(window).abs()
                
                # Create feature names
                features = pd.DataFrame({
                    f'{col}_rolling_mean_{window}': rolling_mean,
                    f'{col}_rolling_std_{window}': rolling_std,
                    f'{col}_rolling_range_{window}': rolling_max - rolling_min,
                    f'{col}_rolling_iqr_{window}': rolling_max - rolling_min,
                    f'{col}_change_rate_{window}': rolling_change / window if window > 0 else 0
                })
                
                features_list.append(features)
        
        rolling_features = pd.concat(features_list, axis=1)
        self.feature_names.extend(rolling_features.columns.tolist())
        
        logger.info(f"Created {rolling_features.shape[1]} rolling features")
        return rolling_features
    
    def create_difference_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create difference features."""
        features_list = []
        
        for lag in self.config['features']['diff_lags']:
            for col in X.columns:
                # First and second differences
                diff_1 = X[col].diff(lag)
                diff_2 = diff_1.diff(lag)
                
                features = pd.DataFrame({
                    f'{col}_diff_{lag}': diff_1,
                    f'{col}_diff2_{lag}': diff_2,
                    f'{col}_abs_diff_{lag}': diff_1.abs(),
                    f'{col}_sign_change_{lag}': (diff_1 * diff_1.shift(1) < 0).astype(float)
                })
                
                features_list.append(features)
        
        diff_features = pd.concat(features_list, axis=1)
        self.feature_names.extend(diff_features.columns.tolist())
        
        logger.info(f"Created {diff_features.shape[1]} difference features")
        return diff_features
    
    def create_spectral_features(self, X: pd.DataFrame, 
                                window_size: int = 100) -> pd.DataFrame:
        """Create frequency domain features."""
        features_list = []
        
        for col in X.columns:
            # Initialize arrays for spectral features
            dominant_freq = np.zeros(len(X))
            spectral_energy = np.zeros(len(X))
            spectral_entropy = np.zeros(len(X))
            
            # Calculate spectral features in rolling windows
            for i in range(len(X)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(X), i + window_size // 2)
                
                if end_idx - start_idx < 10:  # Minimum window size
                    continue
                
                segment = X[col].iloc[start_idx:end_idx].values
                
                if len(segment) > 1:
                    # FFT
                    fft_vals = np.abs(fft.fft(segment - np.mean(segment)))
                    fft_freqs = fft.fftfreq(len(segment))
                    
                    # Get positive frequencies only
                    pos_mask = fft_freqs > 0
                    fft_vals_pos = fft_vals[pos_mask]
                    fft_freqs_pos = fft_freqs[pos_mask]
                    
                    if len(fft_vals_pos) > 0:
                        # Dominant frequency
                        dominant_idx = np.argmax(fft_vals_pos)
                        dominant_freq[i] = fft_freqs_pos[dominant_idx]
                        
                        # Spectral energy
                        spectral_energy[i] = np.sum(fft_vals_pos ** 2)
                        
                        # Spectral entropy
                        normalized_fft = fft_vals_pos / np.sum(fft_vals_pos)
                        spectral_entropy[i] = stats.entropy(normalized_fft)
            
            features = pd.DataFrame({
                f'{col}_dominant_freq': dominant_freq,
                f'{col}_spectral_energy': spectral_energy,
                f'{col}_spectral_entropy': spectral_entropy
            }, index=X.index)
            
            features_list.append(features)
        
        spectral_features = pd.concat(features_list, axis=1)
        self.feature_names.extend(spectral_features.columns.tolist())
        
        logger.info(f"Created {spectral_features.shape[1]} spectral features")
        return spectral_features
    
    def create_statistical_features(self, X: pd.DataFrame, 
                                   window: int = 50) -> pd.DataFrame:
        """Create statistical features."""
        features_list = []
        
        for col in X.columns:
            # Rolling statistical moments
            rolling_skew = X[col].rolling(window=window, center=True, min_periods=1).skew()
            rolling_kurtosis = X[col].rolling(window=window, center=True, min_periods=1).kurt()
            
            # Z-score
            rolling_mean = X[col].rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = X[col].rolling(window=window, center=True, min_periods=1).std()
            z_score = (X[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Autocorrelation
            autocorr_1 = X[col].autocorr(lag=1)
            autocorr_5 = X[col].autocorr(lag=5)
            
            features = pd.DataFrame({
                f'{col}_rolling_skew_{window}': rolling_skew,
                f'{col}_rolling_kurtosis_{window}': rolling_kurtosis,
                f'{col}_z_score_{window}': z_score,
                f'{col}_autocorr_1': autocorr_1,
                f'{col}_autocorr_5': autocorr_5
            })
            
            features_list.append(features)
        
        stat_features = pd.concat(features_list, axis=1)
        self.feature_names.extend(stat_features.columns.tolist())
        
        logger.info(f"Created {stat_features.shape[1]} statistical features")
        return stat_features
    
    def create_cross_sensor_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features based on relationships between sensors."""
        features_list = []
        sensor_cols = [col for col in X.columns if col.startswith('sensor_')]
        
        if len(sensor_cols) >= 2:
            # Ratios between sensors
            for i in range(len(sensor_cols)):
                for j in range(i+1, len(sensor_cols)):
                    col1, col2 = sensor_cols[i], sensor_cols[j]
                    
                    ratio = X[col1] / (X[col2] + 1e-8)
                    diff = X[col1] - X[col2]
                    product = X[col1] * X[col2]
                    
                    features = pd.DataFrame({
                        f'ratio_{col1}_{col2}': ratio,
                        f'diff_{col1}_{col2}': diff,
                        f'product_{col1}_{col2}': product
                    })
                    
                    features_list.append(features)
            
            # Aggregate statistics across all sensors
            sensor_data = X[sensor_cols]
            features_list.extend([
                pd.DataFrame({
                    'sensor_mean': sensor_data.mean(axis=1),
                    'sensor_std': sensor_data.std(axis=1),
                    'sensor_max': sensor_data.max(axis=1),
                    'sensor_min': sensor_data.min(axis=1),
                    'sensor_range': sensor_data.max(axis=1) - sensor_data.min(axis=1)
                })
            ])
        
        if features_list:
            cross_features = pd.concat(features_list, axis=1)
            self.feature_names.extend(cross_features.columns.tolist())
            logger.info(f"Created {cross_features.shape[1]} cross-sensor features")
            return cross_features
        else:
            return pd.DataFrame(index=X.index)
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")
        
        # Reset feature names
        self.feature_names = []
        
        # Create different types of features
        features_list = [X.copy()]  # Start with original features
        
        # Rolling features
        rolling_features = self.create_rolling_features(X)
        features_list.append(rolling_features)
        
        # Difference features
        diff_features = self.create_difference_features(X)
        features_list.append(diff_features)
        
        # Statistical features
        stat_features = self.create_statistical_features(X)
        features_list.append(stat_features)
        
        # Spectral features (if enabled)
        if self.config['features']['spectral_features']:
            spectral_features = self.create_spectral_features(X)
            features_list.append(spectral_features)
        
        # Cross-sensor features (if enabled)
        if self.config['features']['cross_sensor_features']:
            cross_features = self.create_cross_sensor_features(X)
            features_list.append(cross_features)
        
        # Combine all features
        all_features = pd.concat(features_list, axis=1)
        
        # Handle any NaN values from feature creation
        all_features = all_features.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Total features created: {all_features.shape[1]}")
        logger.info(f"Feature engineering completed. Final shape: {all_features.shape}")
        
        return all_features