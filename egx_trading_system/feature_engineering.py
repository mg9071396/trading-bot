"""
Feature Engineering Module for EGX Trading System.

Creates technical indicators, statistical features, and merges multimodal data.
Handles normalization, missing values, and feature validation.

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import warnings

from logger import TradingLogger
from config import trading_config

logger = TradingLogger("feature_engineering")


class FeatureEngineer:
    """
    Generates technical and statistical features for EGX trading signals.
    
    Features include:
    - Technical Indicators: RSI, MACD, Bollinger Bands, ATR, ADX
    - Volume Features: Volume MA, OBV, VWAP
    - Statistical Features: Returns, Volatility, Skewness, Kurtosis
    - Price Patterns: Support/Resistance, Higher Highs/Lower Lows
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize feature engineer with configurable lookback periods.
        
        Args:
            lookback_periods: Dictionary of indicator periods
        """
        self.lookback_periods = lookback_periods or {
            'rsi': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bbands': 20,
            'atr': 14,
            'adx': 14,
            'volume_ma': 20,
            'volatility': 20,
            'momentum': 10
        }
        self.logger = logger.logger
        self.feature_columns = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all engineered features
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Work on a copy to avoid modifying original data
        features_df = df.copy()
        
        # Price-based features
        features_df = self._add_return_features(features_df)
        features_df = self._add_volatility_features(features_df)
        
        # Technical indicators
        features_df = self._add_rsi(features_df)
        features_df = self._add_macd(features_df)
        features_df = self._add_bollinger_bands(features_df)
        features_df = self._add_atr(features_df)
        features_df = self._add_adx(features_df)
        
        # Volume features
        features_df = self._add_volume_features(features_df)
        features_df = self._add_obv(features_df)
        features_df = self._add_vwap(features_df)
        
        # Statistical features
        features_df = self._add_statistical_features(features_df)
        
        # Lagged features (for time-series modeling)
        features_df = self._add_lagged_features(features_df, lags=[1, 2, 3, 5])
        
        # Target variable (future return)
        features_df = self._add_target_variable(features_df)
        
        # Store feature column names
        self.feature_columns = [
            col for col in features_df.columns 
            if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'target']
        ]
        
        self.logger.info(f"Created {len(self.feature_columns)} features")
        
        return features_df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various return calculations."""
        # Simple returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_21d'] = df['close'].pct_change(21)  # Monthly
        
        # Log returns (more statistically sound)
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5))
        
        # Gap (overnight return)
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Intraday return
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_pct'] = df['hl_range'].rolling(5).mean()
        
        self.logger.debug("Added return features")
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures."""
        period = self.lookback_periods['volatility']
        
        # Historical volatility (rolling std of returns)
        df['volatility'] = df['log_return_1d'].rolling(window=period).std() * np.sqrt(252)
        
        # Rolling volatility ratios
        df['vol_ratio_short_long'] = (
            df['log_return_1d'].rolling(5).std() / 
            df['log_return_1d'].rolling(20).std()
        )
        
        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['high'] / df['low'])) ** 2).rolling(window=period).mean()
        ) * np.sqrt(252)
        
        self.logger.debug("Added volatility features")
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI)."""
        period = self.lookback_periods['rsi']
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI divergences
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        self.logger.debug(f"Added RSI ({period}-day)")
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        fast = self.lookback_periods['macd_fast']
        slow = self.lookback_periods['macd_slow']
        signal = self.lookback_periods['macd_signal']
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_cross_above'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        df['macd_cross_below'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
        
        self.logger.debug(f"Added MACD ({fast}/{slow}/{signal})")
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        period = self.lookback_periods['bbands']
        std_dev = 2
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * bb_std)
        
        # Bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Squeeze indicator (narrow bands)
        df['bb_squeeze'] = (df['bb_bandwidth'] < df['bb_bandwidth'].rolling(20).quantile(0.1)).astype(int)
        
        self.logger.debug(f"Added Bollinger Bands ({period}-day, {std_dev} std)")
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)."""
        period = self.lookback_periods['atr']
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        self.logger.debug(f"Added ATR ({period}-day)")
        return df
    
    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        period = self.lookback_periods['adx']
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM and TR
        atr = true_range.rolling(window=period).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).sum() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).sum() / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        
        # Trend strength
        df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)
        
        self.logger.debug(f"Added ADX ({period}-day)")
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        period = self.lookback_periods['volume_ma']
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume spike detection
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
        
        # Money flow (simplified)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = typical_price * df['volume']
        
        self.logger.debug(f"Added volume features ({period}-day MA)")
        return df
    
    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume (OBV)."""
        obv = [0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        
        # OBV trend
        df['obv_trend'] = df['obv'].rolling(10).mean() - df['obv'].rolling(30).mean()
        
        self.logger.debug("Added OBV")
        return df
    
    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).cumsum()
        cumulative_vol = df['volume'].cumsum()
        
        df['vwap'] = cumulative_tp_vol / cumulative_vol
        
        # VWAP deviation
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        self.logger.debug("Added VWAP")
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        period = 21  # Monthly window
        
        # Rolling skewness and kurtosis
        df['skewness'] = df['log_return_1d'].rolling(window=period).skew()
        df['kurtosis'] = df['log_return_1d'].rolling(window=period).kurt()
        
        # Sharpe ratio (rolling)
        rf_daily = 0.18 / 252  # Assuming 18% annual risk-free rate
        excess_return = df['log_return_1d'] - rf_daily
        df['sharpe_rolling'] = (excess_return.rolling(period).mean() / 
                                excess_return.rolling(period).std()) * np.sqrt(252)
        
        # Calmar ratio approximation (return / max drawdown)
        rolling_max = df['close'].rolling(window=period, min_periods=1).max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        df['max_drawdown_rolling'] = drawdown.rolling(period).min()
        
        self.logger.debug(f"Added statistical features ({period}-day window)")
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Add lagged versions of key features."""
        if lags is None:
            lags = [1, 2, 3, 5]
        
        lag_cols = ['return_1d', 'rsi', 'macd', 'volume_ratio', 'news_sentiment']
        
        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        self.logger.debug(f"Added lagged features for lags: {lags}")
        return df
    
    def _add_target_variable(self, df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
        """
        Add target variable for prediction.
        
        Target: Binary classification of positive return over next N days
        """
        if horizon is None:
            horizon = trading_config.prediction_horizon
        
        # Future return over prediction horizon
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Binary target: 1 if positive return, 0 otherwise
        # Handle NaN values by setting to -1 (will be dropped later)
        df['target'] = np.nan
        valid_mask = ~df['future_return'].isna()
        df.loc[valid_mask, 'target'] = (df.loc[valid_mask, 'future_return'] > 0).astype(int)
        df.loc[~valid_mask, 'target'] = -1  # Placeholder for invalid rows
        
        # Multi-class target for more nuanced prediction (simplified to avoid categorical issues)
        df['target_class'] = 0  # Default
        valid_idx = df[valid_mask].index
        for idx in valid_idx:
            fr = df.loc[idx, 'future_return']
            if fr <= -0.05:
                df.loc[idx, 'target_class'] = 0  # Strong negative
            elif fr <= 0:
                df.loc[idx, 'target_class'] = 1  # Weak negative
            elif fr <= 0.05:
                df.loc[idx, 'target_class'] = 2  # Weak positive
            else:
                df.loc[idx, 'target_class'] = 3  # Strong positive
        
        self.logger.debug(f"Added target variable (horizon: {horizon} days)")
        return df
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'ffill_bfill',
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values in the feature set.
        
        Args:
            df: DataFrame with features
            method: Imputation method ('ffill_bfill', 'mean', 'median', 'drop')
            threshold: Max fraction of missing values allowed per column
        
        Returns:
            DataFrame with imputed missing values
        """
        df_clean = df.copy()
        
        # Check missing percentage per column
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.warning(
                f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}"
            )
            df_clean = df_clean.drop(columns=cols_to_drop)
            self.feature_columns = [c for c in self.feature_columns if c not in cols_to_drop]
        
        # Apply imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'ffill_bfill':
            df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
        elif method == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif method == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=self.feature_columns)
        
        # Final check for remaining NaNs
        remaining_nans = df_clean[self.feature_columns].isnull().sum().sum()
        if remaining_nans > 0:
            self.logger.warning(f"Still have {remaining_nans} missing values after imputation")
            # Last resort: drop rows with any NaN
            df_clean = df_clean.dropna(subset=self.feature_columns)
        
        self.logger.info(f"Handled missing values using method: {method}")
        
        return df_clean
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        fit_window: int = 252
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features for model training.
        
        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', 'robust')
            fit_window: Window for computing normalization parameters
        
        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        df_norm = df.copy()
        norm_params = {}
        
        for col in self.feature_columns:
            if col not in df_norm.columns:
                continue
            
            if method == 'zscore':
                # Rolling z-score normalization
                mean = df_norm[col].rolling(window=fit_window).mean()
                std = df_norm[col].rolling(window=fit_window).std()
                df_norm[col] = (df_norm[col] - mean) / std
                
                norm_params[col] = {'method': 'zscore', 'window': fit_window}
                
            elif method == 'minmax':
                # Rolling min-max scaling to [0, 1]
                min_val = df_norm[col].rolling(window=fit_window).min()
                max_val = df_norm[col].rolling(window=fit_window).max()
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-8)
                
                norm_params[col] = {'method': 'minmax', 'window': fit_window}
                
            elif method == 'robust':
                # Robust scaling using median and IQR
                median = df_norm[col].rolling(window=fit_window).median()
                q1 = df_norm[col].rolling(window=fit_window).quantile(0.25)
                q3 = df_norm[col].rolling(window=fit_window).quantile(0.75)
                iqr = q3 - q1
                df_norm[col] = (df_norm[col] - median) / (iqr + 1e-8)
                
                norm_params[col] = {'method': 'robust', 'window': fit_window}
        
        self.logger.info(f"Normalized {len(norm_params)} features using {method} method")
        
        return df_norm, norm_params
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality and detect issues.
        
        Returns:
            Validation report dictionary
        """
        report = {
            'total_features': len(self.feature_columns),
            'missing_values': {},
            'constant_features': [],
            'high_correlation_pairs': [],
            'outlier_counts': {},
            'validation_passed': True
        }
        
        # Check missing values
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = missing
        
        # Check constant features
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            if df[col].nunique() == 1:
                report['constant_features'].append(col)
        
        # Check high correlations
        numeric_df = df[self.feature_columns].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr = [
                (col1, col2, corr_matrix.loc[col1, col2])
                for col1 in upper_tri.columns
                for col2 in upper_tri.index
                if upper_tri.loc[col1, col2] > 0.95
            ]
            report['high_correlation_pairs'] = high_corr
        
        # Count outliers (using IQR method)
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                report['outlier_counts'][col] = int(outliers)
        
        # Determine if validation passed
        if (report['constant_features'] or 
            report['high_correlation_pairs'] or 
            sum(report['missing_values'].values()) > len(df) * 0.1):
            report['validation_passed'] = False
        
        self.logger.info(
            f"Feature validation: {'PASSED' if report['validation_passed'] else 'FAILED'}"
        )
        
        return report


if __name__ == "__main__":
    # Test feature engineering pipeline
    from data_ingestion import EGXDataFetcher
    
    print("=" * 60)
    print("Testing Feature Engineering Pipeline")
    print("=" * 60)
    
    # Fetch sample data
    fetcher = EGXDataFetcher()
    df = fetcher.fetch_historical_data("COMI.CA", period="2y")
    print(f"\n1. Raw data shape: {df.shape}")
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    print(f"\n2. Features created: {len(fe.feature_columns)}")
    print(f"   Feature columns: {fe.feature_columns[:10]}...")
    
    # Handle missing values
    df_clean = fe.handle_missing_values(df_features, method='ffill_bfill')
    print(f"\n3. After cleaning: {df_clean.shape}")
    
    # Normalize features
    df_norm, norm_params = fe.normalize_features(df_clean, method='zscore')
    print(f"\n4. Features normalized")
    
    # Validate features
    validation_report = fe.validate_features(df_norm)
    print(f"\n5. Validation Report:")
    print(f"   Total features: {validation_report['total_features']}")
    print(f"   Constant features: {len(validation_report['constant_features'])}")
    print(f"   High correlation pairs: {len(validation_report['high_correlation_pairs'])}")
    print(f"   Validation passed: {validation_report['validation_passed']}")
    
    # Show sample rows
    print(f"\n6. Sample data (last 5 rows):")
    print(df_norm[['date', 'close', 'rsi', 'macd', 'target']].tail())
    
    print("\n" + "=" * 60)
    print("Feature Engineering Test Complete")
    print("=" * 60)
