"""
Configuration module for EGX Trading System.

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence. Emerging markets like EGX have unique risks.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EGXTradingConfig(BaseModel):
    """
    Configuration for Egyptian Exchange (EGX) trading parameters.
    
    EGX Specifics:
    - Trading Hours: 10:00 AM - 2:30 PM EET (Sunday-Thursday)
    - Settlement: T+2
    - Tick Size: Varies by security, typically 0.01 EGP
    - Circuit Breakers: ±5% daily price limits on most stocks
    """
    
    # Trading Hours (EET - Egypt Standard Time)
    market_open: time = Field(default=time(10, 0), description="EGX market open time (10:00 AM EET)")
    market_close: time = Field(default=time(14, 30), description="EGX market close time (2:30 PM EET)")
    
    # Position Sizing & Risk Limits
    max_position_size_pct: float = Field(default=0.05, ge=0, le=1, description="Max 5% of portfolio per position")
    max_portfolio_drawdown: float = Field(default=0.15, ge=0, le=1, description="Max 15% drawdown before halting trades")
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1, description="Default 5% stop-loss")
    take_profit_pct: float = Field(default=0.10, ge=0, le=1, description="Default 10% take-profit")
    
    # Transaction Costs (EGX-specific)
    commission_rate: float = Field(default=0.0020, ge=0, description="0.20% commission per trade (typical EGX range: 0.15%-0.25%)")
    slippage_rate: float = Field(default=0.001, ge=0, description="0.1% slippage model for lower liquidity")
    stamp_duty: float = Field(default=0.0015, ge=0, description="0.15% stamp duty on sell orders (EGX requirement)")
    
    # Data & Model Parameters
    lookback_days: int = Field(default=252, ge=1, description="One year of trading days for lookback")
    prediction_horizon: int = Field(default=5, ge=1, description="Predict return over next N days")
    train_test_split: float = Field(default=0.8, ge=0.5, le=0.95, description="Chronological train/test split")
    
    # Sentiment Thresholds
    sentiment_buy_threshold: float = Field(default=0.3, ge=-1, le=1, description="Buy if sentiment > 0.3")
    sentiment_sell_threshold: float = Field(default=-0.5, ge=-1, le=1, description="Reject trade if sentiment < -0.5")
    
    # Kelly Criterion Parameters
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Fractional Kelly (25% to reduce volatility)")
    max_kelly_position: float = Field(default=0.10, ge=0, le=1, description="Max 10% position even if Kelly suggests more")
    
    # API Keys (loaded from environment)
    yahoo_api_key: Optional[str] = Field(default=None, description="Yahoo Finance API key (if needed)")
    news_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("NEWS_API_KEY"), description="Arabic News API key")
    broker_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("BROKER_API_KEY"), description="Broker API key")
    broker_secret: Optional[str] = Field(default_factory=lambda: os.getenv("BROKER_SECRET"), description="Broker API secret")
    thndr_phone: Optional[str] = Field(default_factory=lambda: os.getenv("THNDR_PHONE"), description="Thndr phone number for real-time data")
    
    @validator('commission_rate')
    def validate_commission(cls, v):
        """Ensure commission is within realistic EGX bounds."""
        if not (0.0015 <= v <= 0.0030):
            raise ValueError("Commission rate should be between 0.15% and 0.30% for EGX")
        return v
    
    @validator('max_position_size_pct')
    def validate_position_size(cls, v):
        """Enforce strict position sizing for risk management."""
        if v > 0.05:
            raise ValueError("Position size must not exceed 5% for diversification")
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            time: lambda v: v.strftime("%H:%M")
        }


class ModelConfig(BaseModel):
    """Configuration for machine learning models."""
    
    # XGBoost Parameters
    n_estimators: int = Field(default=500, ge=100, le=2000)
    max_depth: int = Field(default=6, ge=3, le=15)
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.3)
    subsample: float = Field(default=0.8, ge=0.5, le=1.0)
    colsample_bytree: float = Field(default=0.8, ge=0.5, le=1.0)
    min_child_weight: int = Field(default=5, ge=1, le=10)
    gamma: float = Field(default=0.1, ge=0, le=1)
    reg_alpha: float = Field(default=0.1, ge=0, le=10)  # L1 regularization
    reg_lambda: float = Field(default=1.0, ge=0, le=10)  # L2 regularization
    
    # Prevent overfitting
    early_stopping_rounds: int = Field(default=50, ge=10, le=200)
    eval_metric: str = Field(default="logloss", description="Evaluation metric for classification")
    
    # Cross-validation
    cv_folds: int = Field(default=5, ge=3, le=10)
    random_state: int = Field(default=42, description="For reproducibility")
    
    class Config:
        arbitrary_types_allowed = True


class BacktestConfig(BaseModel):
    """Configuration for backtesting engine."""
    
    # Initial capital
    initial_capital: float = Field(default=1000000.0, ge=10000, description="Starting capital in EGP")
    
    # Benchmark
    benchmark_symbol: str = Field(default="EGX30.CA", description="EGX30 index as benchmark")
    
    # Risk-free rate (Egyptian Treasury Bill rate approx.)
    risk_free_rate: float = Field(default=0.18, ge=0, le=1, description="Annual risk-free rate (~18% for Egypt)")
    
    # Commission and slippage override (can override EGXTradingConfig)
    use_custom_commission: bool = Field(default=False)
    custom_commission: float = Field(default=0.0020, ge=0)
    
    # Output metrics
    output_dir: str = Field(default="./backtest_results", description="Directory for backtest reports")
    
    class Config:
        arbitrary_types_allowed = True


# Global configuration instances
trading_config = EGXTradingConfig()
model_config = ModelConfig()
backtest_config = BacktestConfig()


def get_config() -> dict:
    """Return all configurations as a dictionary."""
    return {
        "trading": trading_config.dict(),
        "model": model_config.dict(),
        "backtest": backtest_config.dict()
    }


if __name__ == "__main__":
    # Test configuration loading
    import json
    config = get_config()
    print("Configuration loaded successfully:")
    print(json.dumps(config, indent=2, default=str))
