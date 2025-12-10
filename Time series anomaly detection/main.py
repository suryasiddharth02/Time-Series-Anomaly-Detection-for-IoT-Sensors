import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.load_data import DataLoader  # pyright: ignore[reportMissingImports]
from data.preprocessor import DataPreprocessor
from features.feature_engineering import FeatureEngineer  # pyright: ignore[reportMissingImports]
from models.isolation_forest_model import IsolationForestModel
from models.lstm_autoencoder import LSTMAutoencoder
from evaluation.evaluator import ModelEvaluator
from utils.logger import setup_logging
from utils.visualizer import DataVisualizer

import logging

class AnomalyDetectionPipeline:
    """Main pipeline for anomaly detection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration."""
        self.logger = setup_logging(config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = DataVisualizer(self.config)
        
        # Results storage
        self.results = {}
        
        self.logger.info("Anomaly Detection Pipeline initialized")
    
    def run(self) -> None:
        """Run complete anomaly detection pipeline."""
        self.logger.info("Starting anomaly detection pipeline...")
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data...")
            X_raw, y_raw, timestamps = self.data_loader.load_data()
            
            # Step 2: Preprocess data
            self.logger.info("Step 2: Preprocessing data...")
            X_processed, y_processed = self.preprocessor.preprocess_pipeline(
                X_raw, y_raw, fit=True
            )
            
            # Step 3: Split data
            self.logger.info("Step 3: Splitting data...")
            X_train, X_test, y_train, y_test, ts_train, ts_test = self.data_loader.train_test_split(
                X_processed, y_processed, timestamps
            )
            
            # Step 4: Feature engineering
            self.logger.info("Step 4: Engineering features...")
            X_train_features = self.feature_engineer.engineer_features(X_train)
            X_test_features = self.feature_engineer.engineer_features(X_test)
            
            # Step 5: Train and evaluate models
            self.logger.info("Step 5: Training and evaluating models...")
            
            # Model 1: Isolation Forest
            self.logger.info("Training Isolation Forest...")
            iso_forest = IsolationForestModel(self.config)
            iso_forest.train(X_train_features.values)
            
            # Predict with Isolation Forest
            iso_predictions = iso_forest.predict(X_test_features.values)
            iso_probabilities = iso_forest.predict_proba(X_test_features.values)
            
            # Store results
            self.results['isolation_forest'] = {
                'model': iso_forest,
                'predictions': iso_predictions,
                'probabilities': iso_probabilities
            }
            
            # Model 2: LSTM Autoencoder
            self.logger.info("Training LSTM Autoencoder...")
            lstm_ae = LSTMAutoencoder(self.config)
            lstm_ae.train(X_train.values)  # Use raw values for LSTM
            
            # Predict with LSTM Autoencoder
            lstm_predictions = lstm_ae.predict(X_test.values)
            lstm_probabilities = lstm_ae.predict_proba(X_test.values)
            
            # Store results
            self.results['lstm_autoencoder'] = {
                'model': lstm_ae,
                'predictions': lstm_predictions,
                'probabilities': lstm_probabilities
            }
            
            # Step 6: Evaluate models
            self.logger.info("Step 6: Evaluating models...")
            
            # Check if test set has anomalies, if not, calculate ROC AUC on training set
            test_has_anomalies = y_test.mean() > 0
            
            # Evaluate Isolation Forest
            iso_metrics = self.evaluator.calculate_metrics(
                y_test.values, 
                iso_predictions, 
                iso_probabilities,
                model_name='Isolation Forest'
            )
            
            # If test set has no anomalies, calculate ROC AUC on training set
            iso_train_probabilities = None
            if not test_has_anomalies and y_train.mean() > 0:
                self.logger.info("Test set has no anomalies. Calculating ROC AUC on training set for Isolation Forest...")
                iso_train_predictions = iso_forest.predict(X_train_features.values)
                iso_train_probabilities = iso_forest.predict_proba(X_train_features.values)
                train_roc_auc = self.evaluator._calculate_roc_auc_only(y_train.values, iso_train_probabilities)
                if train_roc_auc is not None:
                    iso_metrics['roc_auc'] = train_roc_auc
                    iso_metrics['roc_auc_source'] = 'training_set'
                    self.logger.info(f"Isolation Forest ROC AUC (training set): {train_roc_auc:.4f}")
            
            self.results['isolation_forest']['metrics'] = iso_metrics
            self.results['isolation_forest']['train_probabilities'] = iso_train_probabilities
            
            # Evaluate LSTM Autoencoder
            lstm_metrics = self.evaluator.calculate_metrics(
                y_test.values, 
                lstm_predictions, 
                lstm_probabilities,
                model_name='LSTM Autoencoder'
            )
            
            # If test set has no anomalies, calculate ROC AUC on training set
            lstm_train_probabilities = None
            if not test_has_anomalies and y_train.mean() > 0:
                self.logger.info("Test set has no anomalies. Calculating ROC AUC on training set for LSTM Autoencoder...")
                lstm_train_predictions = lstm_ae.predict(X_train.values)
                lstm_train_probabilities = lstm_ae.predict_proba(X_train.values)
                train_roc_auc = self.evaluator._calculate_roc_auc_only(y_train.values, lstm_train_probabilities)
                if train_roc_auc is not None:
                    lstm_metrics['roc_auc'] = train_roc_auc
                    lstm_metrics['roc_auc_source'] = 'training_set'
                    self.logger.info(f"LSTM Autoencoder ROC AUC (training set): {train_roc_auc:.4f}")
            
            self.results['lstm_autoencoder']['metrics'] = lstm_metrics
            self.results['lstm_autoencoder']['train_probabilities'] = lstm_train_probabilities
            
            # Step 7: Generate visualizations
            self.logger.info("Step 7: Generating visualizations...")
            
            # Create combined dataframe for visualization
            test_df = pd.DataFrame({
                'timestamp': ts_test.values,
                'sensor_1': X_test['sensor_1'].values if 'sensor_1' in X_test.columns else X_test.iloc[:, 0].values,
                'true_anomaly': y_test.values,
                'iso_forest_pred': iso_predictions,
                'lstm_ae_pred': lstm_predictions
            })
            
            # Generate visualizations
            viz_paths = []
            
            # Time series plot
            viz_paths.append(self.visualizer.plot_time_series(
                test_df, timestamp_col='timestamp',
                save_name="test_time_series.png"
            ))
            
            # Anomaly timeline for Isolation Forest
            self.evaluator.plot_anomalies_timeline(
                pd.Series(test_df['timestamp']),
                test_df['sensor_1'].values,
                test_df['true_anomaly'].values,
                test_df['iso_forest_pred'].values,
                model_name='Isolation Forest',
                save_path='visualizations/iso_forest_anomalies.png'
            )
            
            # Anomaly timeline for LSTM Autoencoder
            self.evaluator.plot_anomalies_timeline(
                pd.Series(test_df['timestamp']),
                test_df['sensor_1'].values,
                test_df['true_anomaly'].values,
                test_df['lstm_ae_pred'].values,
                model_name='LSTM Autoencoder',
                save_path='visualizations/lstm_ae_anomalies.png'
            )
            
            # Confusion matrices
            self.evaluator.plot_confusion_matrix(
                test_df['true_anomaly'].values,
                test_df['iso_forest_pred'].values,
                model_name='Isolation Forest',
                save_path='visualizations/iso_forest_cm.png'
            )
            
            self.evaluator.plot_confusion_matrix(
                test_df['true_anomaly'].values,
                test_df['lstm_ae_pred'].values,
                model_name='LSTM Autoencoder',
                save_path='visualizations/lstm_ae_cm.png'
            )
            
            # ROC curves - use training data if test has no anomalies
            if test_has_anomalies:
                # Use test data for ROC curves
                iso_roc_y = test_df['true_anomaly'].values
                iso_roc_proba = iso_probabilities
                lstm_roc_y = test_df['true_anomaly'].values
                lstm_roc_proba = lstm_probabilities
            else:
                # Use training data for ROC curves when test has no anomalies
                iso_roc_y = y_train.values
                iso_roc_proba = self.results['isolation_forest'].get('train_probabilities', iso_probabilities)
                lstm_roc_y = y_train.values
                lstm_roc_proba = self.results['lstm_autoencoder'].get('train_probabilities', lstm_probabilities)
            
            self.evaluator.plot_roc_curve(
                iso_roc_y,
                iso_roc_proba,
                model_name='Isolation Forest',
                save_path='visualizations/iso_forest_roc.png'
            )
            
            self.evaluator.plot_roc_curve(
                lstm_roc_y,
                lstm_roc_proba,
                model_name='LSTM Autoencoder',
                save_path='visualizations/lstm_ae_roc.png'
            )
            
            # Feature distributions
            viz_paths.append(self.visualizer.plot_feature_distributions(
                X_train_features,
                save_name="feature_distributions.png"
            ))
            
            # Correlation matrix
            viz_paths.append(self.visualizer.plot_correlation_matrix(
                X_train_features.iloc[:, :20],  # First 20 features
                save_name="correlation_matrix.png"
            ))
            
            # Step 8: Generate final report
            self.logger.info("Step 8: Generating final report...")
            
            report = self.evaluator.generate_report(
                self.results,
                save_path='reports/evaluation_report.txt'
            )
            
            # Step 9: Save results
            self.logger.info("Step 9: Saving results...")
            self._save_results()
            
            self.logger.info("Pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*80)
            print("PIPELINE SUMMARY")
            print("="*80)
            print(f"Total samples processed: {len(X_raw)}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Anomaly rate in test: {y_test.mean():.2%}")
            print(f"Anomaly rate in train: {y_train.mean():.2%}")
            print(f"Features created: {X_train_features.shape[1]}")
            
            # Warn if test set has no anomalies (ROC AUC can't be calculated)
            if y_test.mean() == 0:
                self.logger.warning("Test set has 0% anomalies. ROC AUC cannot be calculated on test set.")
                self.logger.info("Consider using stratified split or evaluating on training set for ROC AUC.")
            
            print("\nModel Performance:")
            
            comparison_df = self.evaluator.compare_models(self.results)
            
            # Find best model by F1-score (handle ties)
            best_f1_models = comparison_df.loc[comparison_df['f1_rank'] == comparison_df['f1_rank'].min(), 'Model']
            if len(best_f1_models) > 0:
                best_model = best_f1_models.values[0]
                print(f"\nBest model by F1-score: {best_model}")
            else:
                print("\nBest model by F1-score: Unable to determine (all models tied)")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise
    
    def _save_results(self) -> None:
        """Save pipeline results to disk."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_dict = {}
        for model_name, model_data in self.results.items():
            if 'metrics' in model_data:
                metrics_dict[model_name] = model_data['metrics']
        
        metrics_path = results_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Save predictions
        predictions_dict = {}
        for model_name, model_data in self.results.items():
            if 'predictions' in model_data:
                predictions_dict[model_name] = model_data['predictions'].tolist()
        
        predictions_path = results_dir / f"predictions_{timestamp}.npy"
        np.save(predictions_path, predictions_dict)
        
        self.logger.info(f"Results saved to {results_dir}")

def main():
    """Main entry point."""
    pipeline = AnomalyDetectionPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()