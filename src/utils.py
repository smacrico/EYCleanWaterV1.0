"""
Utility Functions for EY Water Quality Challenge
Logging, submission creation, and common helpers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys
from typing import Optional, Dict, List


def setup_logging(log_dir: Optional[str] = None, 
                 log_level: str = 'INFO',
                 log_to_file: bool = True) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_to_file: Whether to save logs to file
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def create_submission_file(predictions_df: pd.DataFrame,
                          sample_ids: pd.Series,
                          output_path: str,
                          target_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create submission file in the required format
    
    Args:
        predictions_df: DataFrame with predictions for all targets
        sample_ids: Series with SamplePointID values
        output_path: Path to save submission file
        target_columns: List of target column names
        
    Returns:
        Submission DataFrame
    """
    if target_columns is None:
        target_columns = [
            "Alkalinity as CaCO3 (mg/L)",
            "Electrical Conductivity (µS/cm)",
            "Dissolved Reactive Phosphorus (mg/L)"
        ]
    
    # Create submission DataFrame
    submission = pd.DataFrame()
    submission['SamplePointID'] = sample_ids
    
    # Add predictions for each target
    for col in target_columns:
        if col in predictions_df.columns:
            submission[col] = predictions_df[col].values
        else:
            logging.warning(f"Target column {col} not found in predictions")
            submission[col] = 0.0
    
    # Ensure no missing values
    submission.fillna(0, inplace=True)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    logging.info(f"Submission file saved to {output_path}")
    logging.info(f"Submission shape: {submission.shape}")
    
    return submission


def validate_submission_file(submission_path: str,
                            template_path: Optional[str] = None) -> bool:
    """
    Validate submission file format
    
    Args:
        submission_path: Path to submission file
        template_path: Path to submission template (optional)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        submission = pd.read_csv(submission_path)
        
        # Required columns
        required_cols = [
            'SamplePointID',
            "Alkalinity as CaCO3 (mg/L)",
            "Electrical Conductivity (µS/cm)",
            "Dissolved Reactive Phosphorus (mg/L)"
        ]
        
        # Check columns
        missing_cols = [col for col in required_cols if col not in submission.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for missing values
        if submission[required_cols].isna().any().any():
            logging.warning("Submission contains missing values")
            return False
        
        # Check for negative values (water quality metrics should be non-negative)
        for col in required_cols[1:]:  # Skip SamplePointID
            if (submission[col] < 0).any():
                logging.warning(f"Column {col} contains negative values")
        
        # If template provided, check row count
        if template_path:
            template = pd.read_csv(template_path)
            if len(submission) != len(template):
                logging.warning(f"Row count mismatch: submission={len(submission)}, "
                              f"template={len(template)}")
        
        logging.info("Submission file validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False


def get_timestamp() -> str:
    """
    Get current timestamp string
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_directory_structure(base_dir: str = '.'):
    """
    Ensure all required directories exist
    
    Args:
        base_dir: Base directory of the project
    """
    base_dir = Path(base_dir)
    
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'data/external/hydrology_layers',
        'models',
        'outputs/logs',
        'outputs/submissions',
        'notebooks',
        'src',
        'configs',
        'business_plan'
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Directory structure created")


def calculate_competition_metric(y_true: pd.DataFrame, 
                                 y_pred: pd.DataFrame,
                                 targets: Optional[List[str]] = None) -> float:
    """
    Calculate competition metric (average R² across targets)
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        targets: List of target columns
        
    Returns:
        Average R² score
    """
    from sklearn.metrics import r2_score
    
    if targets is None:
        targets = [
            "Alkalinity as CaCO3 (mg/L)",
            "Electrical Conductivity (µS/cm)",
            "Dissolved Reactive Phosphorus (mg/L)"
        ]
    
    r2_scores = []
    for target in targets:
        if target in y_true.columns and target in y_pred.columns:
            r2 = r2_score(y_true[target], y_pred[target])
            r2_scores.append(r2)
            logging.info(f"{target}: R² = {r2:.4f}")
    
    avg_r2 = np.mean(r2_scores)
    logging.info(f"Average R²: {avg_r2:.4f}")
    
    return avg_r2


def print_feature_statistics(df: pd.DataFrame, top_n: int = 20):
    """
    Print statistics about features
    
    Args:
        df: DataFrame with features
        top_n: Number of features to display
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\n{'='*60}")
    print(f"FEATURE STATISTICS")
    print(f"{'='*60}")
    print(f"Total features: {len(numeric_cols)}")
    print(f"Total samples: {len(df)}")
    print(f"\nMissing values per feature (top {top_n}):")
    
    missing = df[numeric_cols].isna().sum().sort_values(ascending=False).head(top_n)
    for feature, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {feature}: {count} ({pct:.2f}%)")
    
    print(f"\nFeature value ranges (sample of {min(top_n, len(numeric_cols))}):")
    for col in numeric_cols[:top_n]:
        print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")


def save_metrics_to_file(metrics: Dict, output_path: str):
    """
    Save metrics dictionary to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save file
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return obj
    
    serializable_metrics = {
        k: convert_to_serializable(v) for k, v in metrics.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {output_path}")


class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: Optional[int] = None, message: str = ""):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        pct = (self.current_step / self.total_steps) * 100
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "N/A"
        
        status = f"{self.description}: {self.current_step}/{self.total_steps} "
        status += f"({pct:.1f}%) - ETA: {eta_str}"
        
        if message:
            status += f" - {message}"
        
        print(f"\r{status}", end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
    
    def finish(self):
        """Mark as finished"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{self.description} complete in {elapsed:.2f}s")


def load_config(config_path: str = "configs/train_config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Configuration loaded from {config_path}")
    return config
