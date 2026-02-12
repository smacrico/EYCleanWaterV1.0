"""
Data Loading Module for EY Water Quality Challenge
Handles loading, merging, and preprocessing of training and validation data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge water quality datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_training_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data with water quality measurements
        
        Args:
            file_path: Optional path to training data file
            
        Returns:
            DataFrame with training data
        """
        if file_path is None:
            file_path = self.data_dir / "training_data.parquet"
        else:
            file_path = Path(file_path)
            
        logger.info(f"Loading training data from {file_path}")
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} training samples with {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Training data file not found: {file_path}")
            raise
            
    def load_validation_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load validation data (for submission)
        
        Args:
            file_path: Optional path to validation data file
            
        Returns:
            DataFrame with validation data
        """
        if file_path is None:
            file_path = self.data_dir / "validation_data.parquet"
        else:
            file_path = Path(file_path)
            
        logger.info(f"Loading validation data from {file_path}")
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} validation samples with {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Validation data file not found: {file_path}")
            raise
            
    def merge_datasets(self, 
                      train_df: pd.DataFrame,
                      landsat_df: Optional[pd.DataFrame] = None,
                      terraclimate_df: Optional[pd.DataFrame] = None,
                      geo_features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge training data with additional feature datasets
        
        Args:
            train_df: Base training DataFrame
            landsat_df: Landsat satellite imagery features
            terraclimate_df: TerraClimate data
            geo_features_df: Geospatial features (elevation, slope, landcover)
            
        Returns:
            Merged DataFrame
        """
        merged_df = train_df.copy()
        
        # Merge Landsat features
        if landsat_df is not None:
            logger.info("Merging Landsat features")
            merge_keys = ['SamplePointID', 'Date'] if 'Date' in landsat_df.columns else ['SamplePointID']
            merged_df = merged_df.merge(landsat_df, on=merge_keys, how='left')
            logger.info(f"After Landsat merge: {len(merged_df.columns)} columns")
        
        # Merge TerraClimate features
        if terraclimate_df is not None:
            logger.info("Merging TerraClimate features")
            merge_keys = ['SamplePointID', 'Date'] if 'Date' in terraclimate_df.columns else ['SamplePointID']
            merged_df = merged_df.merge(terraclimate_df, on=merge_keys, how='left')
            logger.info(f"After TerraClimate merge: {len(merged_df.columns)} columns")
        
        # Merge geospatial features
        if geo_features_df is not None:
            logger.info("Merging geospatial features")
            merged_df = merged_df.merge(geo_features_df, on='SamplePointID', how='left')
            logger.info(f"After geospatial merge: {len(merged_df.columns)} columns")
        
        return merged_df
    
    def handle_missing_data(self, 
                           df: pd.DataFrame,
                           strategy: str = 'median',
                           fill_value: Optional[float] = None) -> pd.DataFrame:
        """
        Handle missing data in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for imputation ('median', 'mean', 'forward_fill', 'constant')
            fill_value: Value to use for constant strategy
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Identify numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target columns from imputation
        target_cols = [
            "Alkalinity as CaCO3 (mg/L)",
            "Electrical Conductivity (µS/cm)",
            "Dissolved Reactive Phosphorus (mg/L)"
        ]
        numeric_cols = [col for col in numeric_cols if col not in target_cols]
        
        logger.info(f"Handling missing data in {len(numeric_cols)} numeric columns")
        
        if strategy == 'median':
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mean':
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'forward_fill':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'constant':
            if fill_value is not None:
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(fill_value)
            else:
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        missing_count = df_clean[numeric_cols].isna().sum().sum()
        logger.info(f"Remaining missing values: {missing_count}")
        
        return df_clean
    
    def get_feature_target_split(self, 
                                df: pd.DataFrame,
                                target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Exclude non-feature columns
        exclude_cols = [
            'SamplePointID',
            'Date',
            "Alkalinity as CaCO3 (mg/L)",
            "Electrical Conductivity (µS/cm)",
            "Dissolved Reactive Phosphorus (mg/L)"
        ]
        
        # Handle Date column if it exists
        if 'Date' in df.columns and df['Date'].dtype == 'object':
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save processed DataFrame to parquet
        
        Args:
            df: DataFrame to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


def load_and_prepare_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to load and prepare training and validation data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (training_df, validation_df)
    """
    loader = DataLoader(config['paths']['raw_data_dir'])
    
    # Load raw data
    train_df = loader.load_training_data()
    val_df = loader.load_validation_data()
    
    # Handle missing data
    train_df = loader.handle_missing_data(train_df, strategy='median')
    val_df = loader.handle_missing_data(val_df, strategy='median')
    
    return train_df, val_df
