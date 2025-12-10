# run_pipeline.py
#!/usr/bin/env python3
"""
Run script for the anomaly detection pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import AnomalyDetectionPipeline

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Time Series Anomaly Detection Pipeline'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override data path in config'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and use existing models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['both', 'isolation_forest', 'lstm_autoencoder'],
        default='both',
        help='Which model(s) to run'
    )
    
    return parser.parse_args()

def main():
    """Main execution."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("TIME SERIES ANOMALY DETECTION FOR IOT SENSORS")
    print("="*80)
    
    try:
        # Initialize and run pipeline
        pipeline = AnomalyDetectionPipeline(args.config)
        
        # Override config if needed
        if args.data_path:
            pipeline.config['data']['local_path'] = args.data_path
        
        pipeline.run()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()