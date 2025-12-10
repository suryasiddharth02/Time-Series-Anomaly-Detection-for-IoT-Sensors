import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report,
                            precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate anomaly detection models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = {}
        
    def _calculate_roc_auc_only(self, y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:
        """Calculate ROC AUC only, without other metrics."""
        try:
            # Handle NaN values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_proba))
            if not np.all(valid_mask):
                y_true = y_true[valid_mask]
                y_proba = y_proba[valid_mask]
            
            # Check if we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                return None
            
            # Check for variation in probabilities
            proba_std = np.std(y_proba)
            proba_range = np.max(y_proba) - np.min(y_proba)
            
            if proba_std < 1e-10 or proba_range < 1e-10:
                return None
            
            roc_auc = roc_auc_score(y_true, y_proba)
            return roc_auc if not np.isnan(roc_auc) else None
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {e}")
            return None
    
    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray] = None,
                         model_name: str = "model") -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Handle NaN values - filter out samples with NaN in y_true
        valid_mask = ~np.isnan(y_true)
        if not np.all(valid_mask):
            n_nan = np.sum(~valid_mask)
            logger.warning(f"Found {n_nan} NaN values in y_true. Filtering them out.")
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if y_proba is not None:
                y_proba = y_proba[valid_mask]
        
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_flat = cm.ravel()
        
        # Handle different confusion matrix sizes
        if len(cm_flat) == 1:
            # Only one class present (all predictions and labels are the same)
            if y_pred[0] == 0:
                tn, fp, fn, tp = cm_flat[0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm_flat[0]
        elif len(cm_flat) == 4:
            # Standard 2x2 confusion matrix
            tn, fp, fn, tp = cm_flat
        else:
            # Handle edge cases
            unique_pred = np.unique(y_pred)
            unique_true = np.unique(y_true)
            if len(unique_pred) == 1 and len(unique_true) == 1:
                if unique_pred[0] == 0:
                    tn, fp, fn, tp = len(y_pred), 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, len(y_pred)
            else:
                # Fallback: try to extract from 2D matrix
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                else:
                    # Default to zeros if we can't determine
                    tn, fp, fn, tp = len(y_pred), 0, 0, 0
        
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                # Check if we have both classes in y_true
                unique_classes = np.unique(y_true)
                class_counts = {cls: np.sum(y_true == cls) for cls in unique_classes}
                
                if len(unique_classes) < 2:
                    logger.warning(f"Only one class present in y_true (classes: {unique_classes}, counts: {class_counts}). Cannot calculate ROC AUC.")
                    metrics['roc_auc'] = 0.0
                    metrics['average_precision'] = 0.0
                else:
                    # Check for variation in probabilities (use small epsilon to handle floating point issues)
                    proba_std = np.std(y_proba)
                    proba_range = np.max(y_proba) - np.min(y_proba)
                    proba_mean = np.mean(y_proba)
                    
                    logger.debug(f"Probability stats - mean: {proba_mean:.4f}, std: {proba_std:.4f}, range: {proba_range:.4f}, min: {np.min(y_proba):.4f}, max: {np.max(y_proba):.4f}")
                    
                    if proba_std < 1e-10 or proba_range < 1e-10:
                        logger.warning(f"Probabilities have no variation (std={proba_std:.2e}, range={proba_range:.2e}). Cannot calculate ROC AUC.")
                        metrics['roc_auc'] = 0.0
                        metrics['average_precision'] = 0.0
                    else:
                        # Try to calculate ROC AUC - it should work if we have both classes and varying probabilities
                        roc_auc = roc_auc_score(y_true, y_proba)
                        avg_prec = average_precision_score(y_true, y_proba)
                        
                        # Check for NaN values
                        if np.isnan(roc_auc):
                            logger.warning(f"ROC AUC calculation returned NaN. Classes: {unique_classes}, Class counts: {class_counts}, Proba range: [{np.min(y_proba):.4f}, {np.max(y_proba):.4f}]")
                            metrics['roc_auc'] = 0.0
                        else:
                            metrics['roc_auc'] = roc_auc
                            logger.debug(f"ROC AUC calculated successfully: {roc_auc:.4f}")
                        
                        if np.isnan(avg_prec):
                            logger.warning(f"Average precision calculation returned NaN. Setting to 0.0.")
                            metrics['average_precision'] = 0.0
                        else:
                            metrics['average_precision'] = avg_prec
            except ValueError as e:
                logger.warning(f"Could not calculate probability metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
            except Exception as e:
                logger.warning(f"Unexpected error calculating probability metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        logger.info(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            if metric == 'roc_auc_source':
                logger.info(f"  {metric}: {value}")
            else:
                logger.info(f"  {metric}: {value:.4f}")
        
        # Store metrics
        self.metrics[model_name] = metrics
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models and return results dataframe."""
        comparison_data = []
        
        for model_name, model_results in results.items():
            metrics = model_results.get('metrics', {})
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate differences
        if len(comparison_df) > 1:
            # Fill NaN values with 0.0 before ranking to avoid NaN ranks
            f1_values = comparison_df['f1'].fillna(0.0)
            roc_auc_values = comparison_df['roc_auc'].fillna(0.0)
            
            comparison_df['f1_rank'] = f1_values.rank(ascending=False)
            comparison_df['roc_auc_rank'] = roc_auc_values.rank(ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            model_name: str = "Model",
                            save_path: Optional[str] = None):
        """Plot confusion matrix."""
        # Handle NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.all(valid_mask):
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            # Create directory if it doesn't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_proba: np.ndarray,
                      model_name: str = "Model",
                      save_path: Optional[str] = None):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve
        
        # Handle NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_proba))
        if not np.all(valid_mask):
            y_true = y_true[valid_mask]
            y_proba = y_proba[valid_mask]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            # Create directory if it doesn't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_anomalies_timeline(self, timestamps: pd.Series,
                              values: np.ndarray,
                              y_true: Optional[np.ndarray],
                              y_pred: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None):
        """Plot timeline with detected anomalies."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Time series with true anomalies
        axes[0].plot(timestamps, values, 'b-', alpha=0.7, label='Sensor Value')
        if y_true is not None:
            anomaly_indices = np.where(y_true == 1)[0]
            if len(anomaly_indices) > 0:
                axes[0].scatter(timestamps.iloc[anomaly_indices],
                              values[anomaly_indices],
                              color='red', s=50, label='True Anomalies', zorder=5)
        axes[0].set_ylabel('Sensor Value')
        axes[0].set_title(f'{model_name} - Time Series with True Anomalies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Time series with predicted anomalies
        axes[1].plot(timestamps, values, 'b-', alpha=0.7, label='Sensor Value')
        predicted_indices = np.where(y_pred == 1)[0]
        if len(predicted_indices) > 0:
            axes[1].scatter(timestamps.iloc[predicted_indices],
                          values[predicted_indices],
                          color='orange', s=50, label='Predicted Anomalies', zorder=5)
        axes[1].set_xlabel('Timestamp')
        axes[1].set_ylabel('Sensor Value')
        axes[1].set_title(f'{model_name} - Time Series with Predicted Anomalies')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Anomaly timeline saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Dict[str, Any]], 
                       save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ANOMALY DETECTION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("\n")
        
        # Model comparison
        comparison_df = self.compare_models(results)
        report_lines.append("MODEL COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append(comparison_df.to_string())
        report_lines.append("\n")
        
        # Detailed results for each model
        for model_name, model_results in results.items():
            report_lines.append(f"\n{model_name.upper()} DETAILED RESULTS")
            report_lines.append("-" * 40)
            
            metrics = model_results.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, str):
                    report_lines.append(f"{metric:25}: {value}")
                elif isinstance(value, (int, float)):
                    report_lines.append(f"{metric:25}: {value:.4f}")
                else:
                    report_lines.append(f"{metric:25}: {value}")
            
            # Confusion matrix info
            if 'confusion_matrix' in model_results:
                cm = model_results['confusion_matrix']
                report_lines.append(f"\nConfusion Matrix:")
                report_lines.append(f"True Negatives:  {cm[0, 0]}")
                report_lines.append(f"False Positives: {cm[0, 1]}")
                report_lines.append(f"False Negatives: {cm[1, 0]}")
                report_lines.append(f"True Positives:  {cm[1, 1]}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            # Create directory if it doesn't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report