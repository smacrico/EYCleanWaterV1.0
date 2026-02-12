"""
CLI Training Tool for EY Water Quality Challenge
Command-line interface for model training
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging, load_config, create_submission_file,
    ensure_directory_structure, get_timestamp
)
from src.data_loading import DataLoader
from src.feature_engineering import FeatureEngineer
from src.geospatial_processing import GeospatialProcessor, create_mock_geospatial_features
from src.model_training import MultiTargetTrainer, WaterQualityModel
from src.snowflake_integration import get_snowflake_client


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='EY Water Quality Challenge - Model Training CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all targets with default config
  python -m src.cli_train
  
  # Train specific targets
  python -m src.cli_train --targets "Alkalinity as CaCO3 (mg/L)" "Electrical Conductivity (µS/cm)"
  
  # Use custom config
  python -m src.cli_train --config configs/custom_config.yaml
  
  # Generate submission after training
  python -m src.cli_train --generate-submission
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file (default: configs/train_config.yaml)'
    )
    
    parser.add_argument(
        '--targets',
        nargs='+',
        default=None,
        help='Specific targets to train (default: all targets)'
    )
    
    parser.add_argument(
        '--no-feature-engineering',
        action='store_true',
        help='Skip feature engineering step'
    )
    
    parser.add_argument(
        '--no-geospatial',
        action='store_true',
        help='Skip geospatial feature extraction'
    )
    
    parser.add_argument(
        '--generate-submission',
        action='store_true',
        help='Generate submission file after training'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models (default: from config)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--use-snowflake',
        action='store_true',
        help='Load data from Snowflake instead of local files'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Ensure directory structure
    ensure_directory_structure()
    
    # Setup logging
    logger = setup_logging(
        log_dir=config['paths']['logs_dir'],
        log_level=args.log_level,
        log_to_file=True
    )
    
    logger.info("="*80)
    logger.info("EY WATER QUALITY CHALLENGE - MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir or config['paths']['models_dir']}")
    
    # Determine targets to train
    if args.targets:
        targets = args.targets
    else:
        targets = config['targets']
    
    logger.info(f"Training targets: {targets}")
    
    # Step 1: Load Data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    
    try:
        if args.use_snowflake:
            from src.snowflake_integration import load_competition_data_from_snowflake
            train_df, val_df = load_competition_data_from_snowflake(config)
        else:
            loader = DataLoader(config['paths']['raw_data_dir'])
            train_df = loader.load_training_data()
            val_df = loader.load_validation_data()
            
            # Handle missing data
            train_df = loader.handle_missing_data(train_df, strategy='median')
            val_df = loader.handle_missing_data(val_df, strategy='median')
        
        logger.info(f"Training data: {train_df.shape}")
        logger.info(f"Validation data: {val_df.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Step 2: Feature Engineering
    if not args.no_feature_engineering:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        fe = FeatureEngineer()
        
        logger.info("Engineering features for training data...")
        train_df = fe.create_all_features(train_df, config)
        
        logger.info("Engineering features for validation data...")
        val_df = fe.create_all_features(val_df, config)
        
        logger.info(f"Training data after FE: {train_df.shape}")
        logger.info(f"Validation data after FE: {val_df.shape}")
    
    # Step 3: Geospatial Features
    if not args.no_geospatial:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: GEOSPATIAL FEATURE EXTRACTION")
        logger.info("="*80)
        
        geo_config = config.get('geospatial', {})
        
        geo_processor = GeospatialProcessor(
            dem_path=config['geospatial_files'].get('dem_elevation'),
            slope_path=config['geospatial_files'].get('slope_map'),
            landcover_path=config['geospatial_files'].get('worldcover')
        )
        
        try:
            logger.info("Extracting geospatial features for training data...")
            train_df = geo_processor.extract_all_geospatial_features(train_df)
            
            logger.info("Extracting geospatial features for validation data...")
            val_df = geo_processor.extract_all_geospatial_features(val_df)
            
        except Exception as e:
            logger.warning(f"Geospatial extraction failed, using mock features: {e}")
            train_df = create_mock_geospatial_features(train_df)
            val_df = create_mock_geospatial_features(val_df)
        
        logger.info(f"Training data after geo: {train_df.shape}")
        logger.info(f"Validation data after geo: {val_df.shape}")
    
    # Save processed data
    processed_dir = Path(config['paths']['processed_data_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(processed_dir / 'train_engineered.parquet', index=False)
    val_df.to_parquet(processed_dir / 'val_engineered.parquet', index=False)
    logger.info(f"Saved processed data to {processed_dir}")
    
    # Step 4: Prepare Features and Targets
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PREPARING FEATURES AND TARGETS")
    logger.info("="*80)
    
    # Exclude non-feature columns
    exclude_cols = ['SamplePointID', 'Date'] + config['targets']
    
    # Handle Date column
    for df in [train_df, val_df]:
        if 'Date' in df.columns and df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if train_df[col].dtype in [np.number, np.float64, np.int64]]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[targets].copy()
    
    X_val = val_df[feature_cols].copy()
    
    # Align columns (ensure validation has same features as training)
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for col in missing_cols:
        X_val[col] = 0
    
    X_val = X_val[X_train.columns]
    
    logger.info(f"Feature matrix: {X_train.shape}")
    logger.info(f"Target matrix: {y_train.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Step 5: Train Models
    logger.info("\n" + "="*80)
    logger.info("STEP 5: TRAINING MODELS")
    logger.info("="*80)
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    logger.info(f"Train set: {X_tr.shape}")
    logger.info(f"Test set: {X_te.shape}")
    
    # Initialize trainer
    trainer = MultiTargetTrainer(
        targets=targets,
        params=config['xgboost_params']
    )
    
    # Train all models
    models = trainer.train_all(
        X_tr, y_tr,
        X_te, y_te,
        early_stopping_rounds=config['training'].get('early_stopping_rounds', 50)
    )
    
    # Print metrics summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING METRICS SUMMARY")
    logger.info("="*80)
    
    metrics_df = trainer.get_metrics_summary()
    print(metrics_df.to_string(index=False))
    
    # Save models
    output_dir = args.output_dir or config['paths']['models_dir']
    trainer.save_all(output_dir)
    
    logger.info(f"\nModels saved to: {output_dir}")
    
    # Save feature importance
    if config['evaluation'].get('save_feature_importance', True):
        for target, model in models.items():
            safe_target = target.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            importance_file = Path(output_dir) / f"feature_importance_{safe_target}.csv"
            model.get_top_features(30).to_csv(importance_file, index=False)
            logger.info(f"Saved feature importance for {target}")
    
    # Step 6: Generate Submission (if requested)
    if args.generate_submission:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: GENERATING SUBMISSION")
        logger.info("="*80)
        
        # Make predictions on validation set
        predictions_df = trainer.predict_all(X_val)
        
        # Create submission file
        timestamp = get_timestamp()
        submission_file = Path(config['paths']['submissions_dir']) / f"submission_{timestamp}.csv"
        
        create_submission_file(
            predictions_df=predictions_df,
            sample_ids=val_df['SamplePointID'],
            output_path=submission_file,
            target_columns=targets
        )
        
        logger.info(f"Submission file created: {submission_file}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
