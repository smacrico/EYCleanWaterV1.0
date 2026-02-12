"""
Model Training Module for EY Water Quality Challenge
Train XGBoost models for water quality prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

logger = logging.getLogger(__name__)


class WaterQualityModel:
    """XGBoost model for water quality prediction"""
    
    def __init__(self, params: Optional[Dict] = None, target_name: str = "Target"):
        """
        Initialize model
        
        Args:
            params: XGBoost parameters
            target_name: Name of the target variable
        """
        self.target_name = target_name
        self.params = params or self._get_default_params()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def _get_default_params(self) -> Dict:
        """Get default XGBoost parameters"""
        return {
            "max_depth": 9,
            "learning_rate": 0.035,
            "subsample": 0.82,
            "colsample_bytree": 0.78,
            "n_estimators": 900,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.1,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1
        }
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             early_stopping_rounds: int = 50) -> 'WaterQualityModel':
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Self for chaining
        """
        logger.info(f"Training {self.target_name} model")
        logger.info(f"Training samples: {len(X_train)}, Features: {len(X_train.columns)}")
        
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with or without validation set
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model training complete for {self.target_name}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }
        
        logger.info(f"{self.target_name} - R²: {metrics['r2']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv_folds: int = 5,
                      scoring: str = 'r2') -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Running {cv_folds}-fold cross-validation for {self.target_name}")
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            self.model if self.model else xgb.XGBRegressor(**self.params),
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        results = {
            f'cv_{scoring}_mean': scores.mean(),
            f'cv_{scoring}_std': scores.std(),
            f'cv_{scoring}_scores': scores
        }
        
        logger.info(f"CV {scoring}: {results[f'cv_{scoring}_mean']:.4f} "
                   f"(±{results[f'cv_{scoring}_std']:.4f})")
        
        return results
    
    def get_top_features(self, n: int = 30) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with top features
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance.head(n)
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'target_name': self.target_name
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'WaterQualityModel':
        """
        Load model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        data = joblib.load(filepath)
        
        model_instance = cls(params=data['params'], target_name=data['target_name'])
        model_instance.model = data['model']
        model_instance.feature_names = data['feature_names']
        model_instance.feature_importance = data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model_instance


class MultiTargetTrainer:
    """Train models for multiple target variables"""
    
    def __init__(self, targets: List[str], params: Optional[Dict] = None):
        """
        Initialize multi-target trainer
        
        Args:
            targets: List of target variable names
            params: XGBoost parameters (same for all targets)
        """
        self.targets = targets
        self.params = params
        self.models = {}
        self.metrics = {}
    
    def train_all(self,
                 X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 X_val: Optional[pd.DataFrame] = None,
                 y_val: Optional[pd.DataFrame] = None,
                 early_stopping_rounds: int = 50) -> Dict[str, WaterQualityModel]:
        """
        Train models for all targets
        
        Args:
            X_train: Training features
            y_train: Training targets (DataFrame with all target columns)
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training models for {len(self.targets)} targets")
        
        for target in self.targets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for: {target}")
            logger.info(f"{'='*60}")
            
            # Initialize model
            model = WaterQualityModel(params=self.params, target_name=target)
            
            # Train
            if X_val is not None and y_val is not None:
                model.train(
                    X_train, y_train[target],
                    X_val, y_val[target],
                    early_stopping_rounds=early_stopping_rounds
                )
            else:
                model.train(X_train, y_train[target])
            
            # Evaluate
            train_metrics = model.evaluate(X_train, y_train[target])
            
            if X_val is not None and y_val is not None:
                val_metrics = model.evaluate(X_val, y_val[target])
                self.metrics[target] = {
                    'train': train_metrics,
                    'val': val_metrics
                }
            else:
                self.metrics[target] = {'train': train_metrics}
            
            # Store model
            self.models[target] = model
        
        return self.models
    
    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all targets
        
        Args:
            X: Features
            
        Returns:
            DataFrame with predictions for all targets
        """
        predictions = {}
        
        for target, model in self.models.items():
            predictions[target] = model.predict(X)
        
        return pd.DataFrame(predictions)
    
    def save_all(self, output_dir: str):
        """
        Save all models
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            # Create safe filename
            safe_target = target.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            filepath = output_dir / f"model_{safe_target}.joblib"
            model.save(filepath)
        
        logger.info(f"All models saved to {output_dir}")
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of all model metrics
        
        Returns:
            DataFrame with metrics for all targets
        """
        summary = []
        
        for target, metrics in self.metrics.items():
            row = {'target': target}
            for split, split_metrics in metrics.items():
                for metric_name, metric_value in split_metrics.items():
                    row[f'{split}_{metric_name}'] = metric_value
            summary.append(row)
        
        return pd.DataFrame(summary)
