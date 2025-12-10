import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle missing values, outliers, and data cleaning."""
    
    def __init__(self, config: dict):
        self.config = config
        self.imputer = None
        self.scaler = None
        
    def handle_missing_values(self, X: pd.DataFrame, 
                             strategy: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing values in time series data."""
        X_clean = X.copy()
        
        if strategy == 'forward_fill':
            X_clean = X_clean.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'interpolate':
            X_clean = X_clean.interpolate(method='linear', limit_direction='both')
        elif strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
            X_clean = pd.DataFrame(self.imputer.fit_transform(X_clean), 
                                  columns=X_clean.columns, index=X_clean.index)
        
        missing_percentage = (X.isna().sum().sum() / X.size) * 100
        if missing_percentage > 0:
            logger.info(f"Handled {missing_percentage:.2f}% missing values using {strategy}")
        
        return X_clean
    
    def remove_outliers_iqr(self, X: pd.DataFrame, 
                           threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Replace outliers with bounds instead of removing (to preserve time continuity)
            X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        logger.info("Applied IQR-based outlier handling")
        return X_clean
    
    def scale_features(self, X: pd.DataFrame, 
                       method: str = 'standard', 
                       fit: bool = True) -> Tuple[pd.DataFrame, Optional[object]]:
        """Scale features using specified method."""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            self.scaler = scaler
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        logger.info(f"Scaled features using {method} scaling")
        return X_scaled
    
    def preprocess_pipeline(self, X: pd.DataFrame, 
                           y: Optional[pd.Series] = None,
                           fit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        X_clean = self.handle_missing_values(X, strategy='forward_fill')
        
        # Handle outliers
        X_clean = self.remove_outliers_iqr(X_clean)
        
        # Scale features
        X_scaled = self.scale_features(X_clean, 
                                      method=self.config['features']['scale_method'],
                                      fit=fit)
        
        # Align y with X (ensure same length and no NaN)
        if y is not None:
            # Use positional alignment to avoid index mismatches
            y_aligned = y.iloc[:len(X_scaled)].copy()
            # Ensure no NaN values in labels
            if y_aligned.isna().any():
                logger.warning("Found NaN values in labels. Filling with 0 (normal).")
                y_aligned = y_aligned.fillna(0)
        else:
            y_aligned = None
        
        logger.info("Preprocessing pipeline completed")
        return X_scaled, y_aligned