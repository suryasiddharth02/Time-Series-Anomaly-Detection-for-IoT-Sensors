import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers  # pyright: ignore[reportMissingImports]
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class LSTMAutoencoder:
    """LSTM-based autoencoder for anomaly detection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.encoder = None
        self.threshold = None
        self.scaler = None
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequence_length = self.config['models']['lstm_autoencoder']['sequence_length']
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def _build_model(self, input_shape: Tuple) -> models.Model:
        """Build LSTM autoencoder architecture."""
        model_config = self.config['models']['lstm_autoencoder']
        
        # Encoder
        encoder_input = layers.Input(shape=input_shape)
        
        # First LSTM layer
        x = layers.LSTM(
            model_config['lstm_units'][0],
            activation='tanh',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001)
        )(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Second LSTM layer
        x = layers.LSTM(
            model_config['lstm_units'][1],
            activation='tanh',
            return_sequences=False,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Bottleneck (latent space)
        latent = layers.Dense(model_config['lstm_units'][1] // 2, 
                             activation='tanh',
                             name='latent')(x)
        
        # Decoder
        x = layers.RepeatVector(input_shape[0])(latent)
        
        # First LSTM layer (decoder)
        x = layers.LSTM(
            model_config['lstm_units'][1],
            activation='tanh',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Second LSTM layer (decoder)
        x = layers.LSTM(
            model_config['lstm_units'][0],
            activation='tanh',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        decoder_output = layers.TimeDistributed(
            layers.Dense(input_shape[1], activation='linear')
        )(x)
        
        # Create autoencoder model
        autoencoder = models.Model(encoder_input, decoder_output, name='autoencoder')
        
        # Create encoder model
        self.encoder = models.Model(encoder_input, latent, name='encoder')
        
        return autoencoder
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """Train LSTM autoencoder."""
        logger.info("Training LSTM Autoencoder model...")
        
        model_config = self.config['models']['lstm_autoencoder']
        
        # Create sequences
        X_sequences = self._create_sequences(X_train)
        logger.info(f"Created sequences of shape: {X_sequences.shape}")
        
        # Build model
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        self.model = self._build_model(input_shape)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=model_config['learning_rate']
        )
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_sequences,
            X_sequences,  # Autoencoder reconstructs input
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_split=model_config['validation_split'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Calculate reconstruction errors on training data
        train_reconstructions = self.model.predict(X_sequences, verbose=0)
        train_errors = np.mean(np.square(X_sequences - train_reconstructions), axis=(1, 2))
        
        # Set threshold based on training errors
        self.threshold = np.mean(train_errors) + self.config['training']['anomaly_threshold_std'] * np.std(train_errors)
        
        logger.info(f"LSTM Autoencoder training completed. Threshold: {self.threshold:.4f}")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create sequences
        X_sequences = self._create_sequences(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        
        # Convert to binary predictions (1 for anomaly, 0 for normal)
        binary_predictions = np.zeros(len(X))
        
        # Align predictions with original data (first sequence_length-1 samples have no prediction)
        start_idx = self.config['models']['lstm_autoencoder']['sequence_length'] - 1
        
        # Create rolling window for error aggregation
        error_window = np.zeros(len(X))
        for i in range(len(reconstruction_errors)):
            error_window[start_idx + i] = reconstruction_errors[i]
        
        # Smooth errors using rolling window
        window_size = self.config['training']['reconstruction_error_window']
        smoothed_errors = np.convolve(error_window, 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
        
        # Threshold based on smoothed errors
        binary_predictions = (smoothed_errors > self.threshold).astype(int)
        
        return binary_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly probabilities based on reconstruction error."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create sequences
        X_sequences = self._create_sequences(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        
        # Convert errors to probabilities (0-1 range)
        min_error = reconstruction_errors.min()
        max_error = reconstruction_errors.max()
        
        if max_error > min_error:
            probabilities = (reconstruction_errors - min_error) / (max_error - min_error)
        else:
            probabilities = np.zeros_like(reconstruction_errors)
        
        # Align with original data
        aligned_proba = np.zeros(len(X))
        start_idx = self.config['models']['lstm_autoencoder']['sequence_length'] - 1
        
        for i in range(len(probabilities)):
            if start_idx + i < len(aligned_proba):
                aligned_proba[start_idx + i] = probabilities[i]
        
        return aligned_proba
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = models.load_model(path)
        logger.info(f"Model loaded from {path}")
        return self