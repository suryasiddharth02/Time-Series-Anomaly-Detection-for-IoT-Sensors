# Time Series Anomaly Detection for IoT Sensors

A comprehensive machine learning pipeline for detecting anomalies in time series sensor data using multiple detection algorithms. This project implements both traditional machine learning (Isolation Forest) and deep learning (LSTM Autoencoder) approaches for anomaly detection in IoT sensor data.

## Features

- **Multiple Detection Models**: Implements Isolation Forest and LSTM Autoencoder for anomaly detection
- **Comprehensive Feature Engineering**: Creates 167+ features including rolling statistics, differences, spectral features, and cross-sensor correlations
- **Robust Data Preprocessing**: Handles missing values, outliers, and feature scaling
- **Extensive Evaluation Metrics**: Calculates precision, recall, F1-score, ROC-AUC, and more
- **Rich Visualizations**: Generates ROC curves, confusion matrices, anomaly timelines, and feature distributions
- **Automated Pipeline**: End-to-end pipeline from data loading to model evaluation
- **Configurable**: YAML-based configuration for easy parameter tuning

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the project structure is set up correctly (directories will be created automatically on first run)

## Usage

### Basic Usage

Run the complete pipeline with default configuration:

```bash
python run_pipeline.py
```

### Advanced Usage

Run with custom configuration file:

```bash
python run_pipeline.py --config path/to/config.yaml
```

Run specific model only:

```bash
# Run only Isolation Forest
python run_pipeline.py --model isolation_forest

# Run only LSTM Autoencoder
python run_pipeline.py --model lstm_autoencoder
```

Override data path:

```bash
python run_pipeline.py --data-path ./custom/data/path
```

### Command Line Arguments

- `--config`: Path to configuration file (default: `config/config.yaml`)
- `--data-path`: Override data path specified in config
- `--skip-training`: Skip training and use existing models (not implemented)
- `--model`: Choose which model(s) to run (`both`, `isolation_forest`, `lstm_autoencoder`)

## Project Structure

```
Time series anomaly detection/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── load_data.py            # Data loading utilities
│   ├── preprocessor.py         # Data preprocessing
│   └── bearing_dataset/         # Dataset directory
├── evaluation/
│   └── evaluator.py            # Model evaluation and metrics
├── features/
│   └── feature_engineering.py  # Feature creation
├── models/
│   ├── isolation_forest_model.py  # Isolation Forest implementation
│   └── lstm_autoencoder.py        # LSTM Autoencoder implementation
├── utils/
│   ├── logger.py               # Logging utilities
│   └── visualizer.py           # Visualization functions
├── main.py                     # Main pipeline class
├── run_pipeline.py             # Entry point script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

The pipeline is configured via `config/config.yaml`. Key configuration sections:

### Data Configuration
- `local_path`: Path to dataset directory
- `test_size`: Proportion of data for testing (default: 0.2)
- `random_state`: Random seed for reproducibility

### Feature Engineering
- `rolling_windows`: Window sizes for rolling statistics (e.g., [10, 30, 60])
- `diff_lags`: Lags for difference features
- `spectral_features`: Enable/disable spectral features
- `cross_sensor_features`: Enable/disable cross-sensor correlations
- `scale_method`: Scaling method (`standard` or `robust`)

### Model Configuration

#### Isolation Forest
- `n_estimators`: Number of trees (default: 100)
- `contamination`: Expected proportion of anomalies (default: 0.01)
- `max_samples`: Sample size for each tree (default: 0.8)

#### LSTM Autoencoder
- `sequence_length`: Length of input sequences (default: 10)
- `lstm_units`: List of LSTM layer sizes (default: [64, 32])
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate (default: 0.001)

### Training Configuration
- `anomaly_threshold_std`: Standard deviations above mean for anomaly threshold
- `reconstruction_error_window`: Window size for smoothing reconstruction errors

## Models

### 1. Isolation Forest

An ensemble-based anomaly detection algorithm that isolates anomalies by randomly selecting features and split values. Works well for high-dimensional data and doesn't require labeled training data.

**Key Features:**
- Unsupervised learning
- Fast training and prediction
- Handles high-dimensional data well

### 2. LSTM Autoencoder

A deep learning approach using Long Short-Term Memory (LSTM) networks in an autoencoder architecture. The model learns to reconstruct normal patterns and identifies anomalies based on reconstruction error.

**Key Features:**
- Captures temporal dependencies
- Learns complex patterns in time series
- Uses reconstruction error as anomaly score

## Outputs

The pipeline generates several outputs:

### 1. Visualizations (`visualizations/`)
- `test_time_series.png`: Time series plot of test data
- `iso_forest_anomalies.png`: Anomaly timeline for Isolation Forest
- `lstm_ae_anomalies.png`: Anomaly timeline for LSTM Autoencoder
- `iso_forest_cm.png`: Confusion matrix for Isolation Forest
- `lstm_ae_cm.png`: Confusion matrix for LSTM Autoencoder
- `iso_forest_roc.png`: ROC curve for Isolation Forest
- `lstm_ae_roc.png`: ROC curve for LSTM Autoencoder
- `feature_distributions.png`: Distribution of engineered features
- `correlation_matrix.png`: Correlation matrix of features

### 2. Reports (`reports/`)
- `evaluation_report.txt`: Comprehensive evaluation report with model comparisons

### 3. Results (`results/`)
- `metrics_*.json`: Model metrics saved as JSON
- `predictions_*.npy`: Model predictions saved as NumPy arrays

### 4. Logs (`logs/`)
- `anomaly_detection.log`: Detailed execution logs

## Evaluation Metrics

The pipeline calculates the following metrics:

- **Precision**: Proportion of predicted anomalies that are actually anomalies
- **Recall**: Proportion of actual anomalies that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Average Precision**: Area under the Precision-Recall curve
- **False Positive Rate**: Rate of normal samples incorrectly flagged as anomalies
- **False Negative Rate**: Rate of anomalies missed by the model

## Data Format

The pipeline expects CSV files with the following structure:

- **Sensor columns**: Columns starting with `sensor_` (e.g., `sensor_1`, `sensor_2`)
- **Timestamp column**: A `timestamp` column (optional, for visualization)
- **Label column**: An `is_anomaly` column (0 for normal, 1 for anomaly)

Example:
```csv
timestamp,sensor_1,sensor_2,sensor_3,sensor_4,is_anomaly
2024-01-01 00:00:00,10.5,20.3,15.2,18.7,0
2024-01-01 00:01:00,10.6,20.4,15.3,18.8,0
...
```

## Feature Engineering

The pipeline creates 167+ features from raw sensor data:

1. **Rolling Features** (60 features): Mean, std, min, max, median over multiple windows
2. **Difference Features** (48 features): First and second differences at various lags
3. **Statistical Features** (20 features): Skewness, kurtosis, percentiles
4. **Spectral Features** (12 features): FFT-based frequency domain features
5. **Cross-Sensor Features** (23 features): Correlations and interactions between sensors

## Troubleshooting

### Common Issues

1. **ROC AUC is 0.0 or NaN**
   - This occurs when the test set has no anomalies
   - The pipeline automatically calculates ROC AUC on training data as a fallback
   - Check the logs for warnings about class distribution

2. **Memory Issues**
   - Reduce `rolling_windows` or `diff_lags` in config
   - Reduce `n_estimators` for Isolation Forest
   - Reduce `batch_size` for LSTM Autoencoder

3. **Training Takes Too Long**
   - Reduce `epochs` for LSTM Autoencoder
   - Reduce `n_estimators` for Isolation Forest
   - Use smaller dataset or sample data

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning framework
- **matplotlib/seaborn**: Visualization
- **pyyaml**: Configuration file parsing

## License

This project is provided as-is for educational and research purposes.

## Author

Time Series Anomaly Detection Project

## Acknowledgments

- Uses synthetic data generation for demonstration
- Based on standard anomaly detection algorithms
- Designed for IoT sensor data analysis

