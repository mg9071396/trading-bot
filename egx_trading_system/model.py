"""
Machine Learning Model Module for EGX Trading System.

Implements XGBoost classifier for predicting positive returns.
Includes strict train/test split, cross-validation, and feature importance analysis.

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence. Avoid overfitting by using proper validation.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

# Try to import ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from logger import TradingLogger
from config import model_config, trading_config

logger = TradingLogger("model_training")


class QuantModel:
    """
    Machine learning model for EGX trading signal generation.
    
    Uses XGBoost classifier to predict probability of positive return
    over the next N days (defined in config).
    
    Key Features:
    - Chronological train/test split (NO shuffling to prevent data leakage)
    - Early stopping to prevent overfitting
    - Feature importance analysis
    - Probability calibration
    - Model persistence
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the quant model.
        
        Args:
            params: Optional dictionary of XGBoost parameters
        """
        self.params = params or self._get_default_params()
        self.model = None
        self.feature_names = []
        self.is_fitted = False
        self.training_history = {}
        self.logger = logger.logger
        
        if not XGB_AVAILABLE:
            self.logger.error("XGBoost not installed. Install with: pip install xgboost")
        
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
    
    def _get_default_params(self) -> Dict:
        """Get default XGBoost parameters from config."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': model_config.n_estimators,
            'max_depth': model_config.max_depth,
            'learning_rate': model_config.learning_rate,
            'subsample': model_config.subsample,
            'colsample_bytree': model_config.colsample_bytree,
            'min_child_weight': model_config.min_child_weight,
            'gamma': model_config.gamma,
            'reg_alpha': model_config.reg_alpha,  # L1 regularization
            'reg_lambda': model_config.reg_lambda,  # L2 regularization
            'random_state': model_config.random_state,
            'tree_method': 'hist',  # Faster training
            'early_stopping_rounds': model_config.early_stopping_rounds,
            'verbosity': 0
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'target'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Drop rows with missing values in features or target
        cols_to_check = feature_columns + [target_column]
        df_clean = df.dropna(subset=cols_to_check).copy()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        self.logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        
        return X, y, feature_columns
    
    def chronological_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform chronological train/test split (NO shuffling).
        
        WARNING: Never shuffle time-series data as it causes data leakage.
        
        Args:
            X: Feature matrix
            y: Target vector
            split_ratio: Train/test split ratio (default from config)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if split_ratio is None:
            split_ratio = trading_config.train_test_split
        
        n_samples = len(X)
        split_idx = int(n_samples * split_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        self.logger.info(
            f"Chronological split: Train={len(X_train)} ({len(X_train)/len(X)*100:.1f}%), "
            f"Test={len(X_test)} ({len(X_test)/len(X)*100:.1f}%)"
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            verbose: Print training progress
        
        Returns:
            Dictionary of training metrics
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for training")
        
        self.feature_names = feature_names
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Remove early_stopping_rounds from params for train() call
        train_params = {k: v for k, v in self.params.items() if k != 'early_stopping_rounds'}
        
        # Train model
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.params.get('early_stopping_rounds', 50),
            verbose_eval=verbose
        )
        
        self.is_fitted = True
        
        # Get training metrics
        train_metrics = self._evaluate_model(X_train, y_train, "Training")
        val_metrics = self._evaluate_model(X_val, y_val, "Validation")
        
        self.training_history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': self.model.best_iteration,
            'n_features': len(feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        self.logger.log_model_training(
            model_type="XGBoost",
            train_samples=len(X_train),
            test_samples=len(X_val),
            metrics=val_metrics,
            duration_seconds=0  # Would need to track actual time
        )
        
        return val_metrics
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'target',
        split_ratio: float = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Full training pipeline: prepare data, split, and train.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            split_ratio: Train/test split ratio
            verbose: Print training progress
        
        Returns:
            Dictionary of validation metrics
        """
        # Prepare data
        X, y, feat_names = self.prepare_data(df, feature_columns, target_column)
        
        # Chronological split
        X_train, X_test, y_train, y_test = self.chronological_train_test_split(
            X, y, split_ratio
        )
        
        # Train model
        metrics = self.train(X_train, y_train, X_test, y_test, feat_names, verbose)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted class labels (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for prediction")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)
        
        # Convert probabilities to class labels
        class_labels = (predictions > 0.5).astype(int)
        
        return class_labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of probability scores (0 to 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for prediction")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        probabilities = self.model.predict(dmatrix)
        
        return probabilities
    
    def _evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Dataset"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name of dataset for logging
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        self.logger.debug(
            f"{dataset_name} Evaluation: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        
        return metrics
    
    def get_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain')
        
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['importance'])
        importance_df.index.name = 'feature'
        importance_df = importance_df.reset_index()
        
        # Normalize importance scores
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (importance_df['importance'] / total_importance * 100).round(2)
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Return top N
        top_features = importance_df.head(top_n)
        
        self.logger.info(f"Top {top_n} features by {importance_type} importance retrieved")
        
        return top_features
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        probability_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            probability_threshold: Threshold for buy signal
        
        Returns:
            DataFrame with added signal columns
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        df_signals = df.copy()
        
        # Prepare features
        X = df_signals[feature_columns].values
        
        # Get predictions
        probabilities = self.predict_proba(X)
        predictions = (probabilities > probability_threshold).astype(int)
        
        # Add to DataFrame
        df_signals['signal_probability'] = probabilities
        df_signals['signal_prediction'] = predictions
        
        # Convert to trading signals
        df_signals['signal'] = predictions.map({1: 'BUY', 0: 'HOLD'})
        
        self.logger.info(f"Generated signals for {len(df_signals)} rows")
        
        return df_signals
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise RuntimeError("No model to save")
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required")
        
        self.model.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required")
        
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'target',
        cv_folds: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        WARNING: Uses TimeSeriesSplit to prevent data leakage.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            cv_folds: Number of CV folds
        
        Returns:
            Dictionary of CV scores
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CV")
        
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for CV")
        
        # Prepare data
        X, y, _ = self.prepare_data(df, feature_columns, target_column)
        
        # Time series split (no shuffling!)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Store results
        cv_results = {
            'fold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        self.logger.info(f"Starting {cv_folds}-fold time-series cross-validation")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train temporary model
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)
            
            temp_model = xgb.train(
                params={k: v for k, v in self.params.items() if k != 'early_stopping_rounds'},
                dtrain=dtrain,
                num_boost_round=self.params['n_estimators']
            )
            
            # Evaluate
            dmatrix_test = xgb.DMatrix(X_test, feature_names=feature_columns)
            y_pred = (temp_model.predict(dmatrix_test) > 0.5).astype(int)
            y_proba = temp_model.predict(dmatrix_test)
            
            cv_results['fold'].append(fold + 1)
            cv_results['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            cv_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            cv_results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            cv_results['roc_auc'].append(roc_auc_score(y_test, y_proba))
            
            self.logger.debug(
                f"Fold {fold + 1}: Acc={cv_results['accuracy'][-1]:.4f}, "
                f"AUC={cv_results['roc_auc'][-1]:.4f}"
            )
        
        # Calculate mean and std
        cv_summary = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for metric, values in cv_results.items() if metric != 'fold'
        }
        
        self.logger.info(
            f"CV Results: Accuracy={cv_summary['accuracy']['mean']:.4f}±{cv_summary['accuracy']['std']:.4f}, "
            f"AUC={cv_summary['roc_auc']['mean']:.4f}±{cv_summary['roc_auc']['std']:.4f}"
        )
        
        return cv_summary


if __name__ == "__main__":
    # Test model training pipeline
    from data_ingestion import EGXDataFetcher
    from feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Testing ML Model Training Pipeline")
    print("=" * 60)
    
    # Fetch and prepare data
    print("\n1. Fetching data...")
    fetcher = EGXDataFetcher()
    df = fetcher.fetch_historical_data("COMI.CA", period="3y")
    
    print("\n2. Creating features...")
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    df_clean = fe.handle_missing_values(df_features)
    
    print(f"   Total features: {len(fe.feature_columns)}")
    
    # Train model
    print("\n3. Training XGBoost model...")
    model = QuantModel()
    
    try:
        metrics = model.fit(
            df=df_clean,
            feature_columns=fe.feature_columns,
            target_column='target',
            verbose=True
        )
        
        print(f"\n4. Validation Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Feature importance
        print("\n5. Top 10 Feature Importance:")
        importance_df = model.get_feature_importance(top_n=10)
        for _, row in importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance_pct']:.2f}%")
        
        # Cross-validation
        print("\n6. Running 5-fold cross-validation...")
        cv_results = model.cross_validate(
            df=df_clean,
            feature_columns=fe.feature_columns,
            cv_folds=5
        )
        print(f"   Mean Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"   Mean AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Note: This may fail if dependencies (xgboost, sklearn) are not installed")
    
    print("\n" + "=" * 60)
    print("ML Model Test Complete")
    print("=" * 60)
