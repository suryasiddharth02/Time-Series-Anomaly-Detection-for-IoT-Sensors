import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import logging
import joblib

logger = logging.getLogger(__name__)

class IsolationForestModel:
    """Isolation Forest anomaly detection model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.threshold = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """Train Isolation Forest model."""
        logger.info("Training Isolation Forest model...")
        
        # Initialize model with config parameters
        model_config = self.config['models']['isolation_forest']
        
        self.model = IsolationForest(
            n_estimators=model_config['n_estimators'],
            contamination=model_config['contamination'],
            max_samples=model_config['max_samples'],
            random_state=model_config['random_state'],
            n_jobs=-1,
            verbose=0
        )
        
        # Fit model
        self.model.fit(X_train)
        
        # Calculate anomaly scores
        scores = -self.model.decision_function(X_train)  # Negative for anomaly score
        
        # Set threshold (using contamination parameter or percentile)
        self.threshold = np.percentile(scores, 100 * model_config['contamination'])
        
        logger.info(f"Isolation Forest training completed. Threshold: {self.threshold:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        # Convert to binary: 1 for anomaly, 0 for normal
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = -self.model.decision_function(X)
        # Normalize scores to [0, 1] probability-like values
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            probabilities = (scores - min_score) / (max_score - min_score)
        else:
            probabilities = np.zeros_like(scores)
        
        return probabilities
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tune hyperparameters using grid search."""
        logger.info("Tuning Isolation Forest hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': [0.5, 0.8, 1.0],
            'contamination': [0.01, 0.05, 0.1]
        }
        
        base_model = IsolationForest(random_state=self.config['models']['isolation_forest']['random_state'])
        
        # Use GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        # Note: IsolationForest expects normal data only for training
        # In practice, we might use only normal samples or use the entire dataset
        grid_search.fit(X_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self