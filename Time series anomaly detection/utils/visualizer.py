import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DataVisualizer:
    """Create visualizations for EDA and results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.save_dir = Path("visualizations")
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_time_series(self, df: pd.DataFrame, 
                        timestamp_col: str = 'timestamp',
                        save_name: str = "time_series.png"):
        """Plot time series of all sensors."""
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(15, 3*len(sensor_cols)), sharex=True)
        
        if len(sensor_cols) == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(sensor_cols):
            axes[idx].plot(df[timestamp_col], df[sensor], 'b-', alpha=0.7, linewidth=1)
            axes[idx].set_ylabel(sensor)
            axes[idx].grid(True, alpha=0.3)
            
            # Mark anomalies if present
            if 'is_anomaly' in df.columns:
                anomaly_indices = df[df['is_anomaly'] == 1].index
                axes[idx].scatter(df.loc[anomaly_indices, timestamp_col],
                                df.loc[anomaly_indices, sensor],
                                color='red', s=20, alpha=0.6, label='Anomalies')
                axes[idx].legend()
        
        axes[-1].set_xlabel('Timestamp')
        plt.suptitle('Time Series Sensor Data', y=1.02, fontsize=14)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        return save_path
    
    def plot_feature_distributions(self, features: pd.DataFrame, 
                                  save_name: str = "feature_distributions.png"):
        """Plot distributions of engineered features."""
        # Select a subset of features to visualize
        n_features = min(12, len(features.columns))
        selected_features = features.columns[:n_features]
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(selected_features):
            if idx < len(axes):
                axes[idx].hist(features[feature].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(feature[:20] + '...' if len(feature) > 20 else feature, fontsize=10)
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(selected_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', y=1.02, fontsize=14)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        return save_path
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                               save_name: str = "correlation_matrix.png"):
        """Plot correlation matrix of features."""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=False, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        return save_path
    
    def plot_rolling_statistics(self, df: pd.DataFrame, 
                               column: str,
                               timestamp_col: str = 'timestamp',
                               window: int = 50,
                               save_name: str = "rolling_stats.png"):
        """Plot rolling statistics for a specific column."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Original series
        axes[0].plot(df[timestamp_col], df[column], 'b-', alpha=0.5, label='Original')
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'Rolling Statistics - {column}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling mean and std
        rolling_mean = df[column].rolling(window=window, center=True).mean()
        rolling_std = df[column].rolling(window=window, center=True).std()
        
        axes[1].plot(df[timestamp_col], df[column], 'b-', alpha=0.3, label='Original')
        axes[1].plot(df[timestamp_col], rolling_mean, 'r-', linewidth=2, label='Rolling Mean')
        axes[1].fill_between(df[timestamp_col],
                           rolling_mean - 2*rolling_std,
                           rolling_mean + 2*rolling_std,
                           color='red', alpha=0.2, label='±2σ')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Rolling statistics as separate lines
        rolling_min = df[column].rolling(window=window, center=True).min()
        rolling_max = df[column].rolling(window=window, center=True).max()
        
        axes[2].plot(df[timestamp_col], rolling_mean, 'r-', label='Mean')
        axes[2].plot(df[timestamp_col], rolling_min, 'g-', alpha=0.7, label='Min')
        axes[2].plot(df[timestamp_col], rolling_max, 'b-', alpha=0.7, label='Max')
        axes[2].set_xlabel('Timestamp')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        return save_path
    
    def create_interactive_plot(self, df: pd.DataFrame, 
                               timestamp_col: str = 'timestamp',
                               save_name: str = "interactive_plot.html"):
        """Create interactive plot with Plotly."""
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        fig = make_subplots(
            rows=len(sensor_cols), cols=1,
            subplot_titles=sensor_cols,
            vertical_spacing=0.05
        )
        
        for idx, sensor in enumerate(sensor_cols):
            row = idx + 1
            
            # Add trace for sensor data
            fig.add_trace(
                go.Scatter(
                    x=df[timestamp_col],
                    y=df[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Add anomalies if present
            if 'is_anomaly' in df.columns:
                anomaly_df = df[df['is_anomaly'] == 1]
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_df[timestamp_col],
                        y=anomaly_df[sensor],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        fig.update_layout(
            height=300 * len(sensor_cols),
            title_text="Interactive Sensor Data Visualization",
            hovermode='x unified'
        )
        
        save_path = self.save_dir / save_name
        fig.write_html(str(save_path))
        
        return save_path