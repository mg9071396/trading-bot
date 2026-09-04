"""
Logging module for EGX Trading System.

Provides structured logging for data gaps, model predictions, execution errors,
and risk management events.

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "egx_trading",
    log_level: int = logging.INFO,
    log_dir: Optional[str] = "./logs",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a robust logger with both console and file handlers.
    
    Args:
        name: Logger name (typically module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Enable console output
        file_output: Enable file output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter(fmt='%(levelname)-8s | %(message)s'))
        logger.addHandler(console_handler)
    
    # File handler with JSON output for structured logging
    if file_output and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Daily rotating file handler
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log
        error_file = log_path / f"errors_{timestamp}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class TradingLogger:
    """
    Specialized logger for trading system events.
    
    Provides methods for logging specific trading events like:
    - Data ingestion gaps
    - Model predictions
    - Trade signals
    - Risk management rejections
    - Execution errors
    """
    
    def __init__(self, name: str = "egx_trading"):
        self.logger = setup_logger(name)
        self._prediction_log = []
        self._trade_log = []
        self._error_log = []
    
    def log_data_gap(self, symbol: str, gap_type: str, details: dict):
        """Log missing or incomplete data."""
        self.logger.warning(
            f"DATA_GAP | Symbol: {symbol} | Type: {gap_type} | Details: {json.dumps(details)}"
        )
    
    def log_prediction(
        self,
        symbol: str,
        prediction: float,
        probability: float,
        features_used: list,
        timestamp: datetime
    ):
        """Log model prediction for audit trail."""
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "prediction": prediction,
            "probability": probability,
            "features_used": features_used
        }
        self._prediction_log.append(log_entry)
        self.logger.info(
            f"PREDICTION | Symbol: {symbol} | Pred: {prediction:.4f} | Prob: {probability:.4f}"
        )
    
    def log_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: int,
        reason: str,
        sentiment_score: Optional[float] = None
    ):
        """Log trading signal generation."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "price": price,
            "quantity": quantity,
            "reason": reason,
            "sentiment_score": sentiment_score
        }
        self._trade_log.append(log_entry)
        
        sentiment_str = f" | Sentiment: {sentiment_score:.3f}" if sentiment_score else ""
        self.logger.info(
            f"SIGNAL | {signal} | Symbol: {symbol} | Price: {price:.4f} | Qty: {quantity}{sentiment_str} | Reason: {reason}"
        )
    
    def log_risk_rejection(
        self,
        symbol: str,
        signal: str,
        rejection_reason: str,
        risk_metric: Optional[str] = None,
        risk_value: Optional[float] = None
    ):
        """Log trade rejection by risk manager."""
        metric_str = f" | Metric: {risk_metric}={risk_value}" if risk_value else ""
        self.logger.warning(
            f"RISK_REJECT | Symbol: {symbol} | Signal: {signal} | Reason: {rejection_reason}{metric_str}"
        )
    
    def log_execution(
        self,
        symbol: str,
        action: str,
        quantity: int,
        executed_price: float,
        order_id: Optional[str] = None,
        status: str = "PENDING"
    ):
        """Log order execution."""
        self.logger.info(
            f"EXECUTION | {status} | {action} | Symbol: {symbol} | Qty: {quantity} | Price: {executed_price:.4f} | OrderID: {order_id}"
        )
    
    def log_model_training(
        self,
        model_type: str,
        train_samples: int,
        test_samples: int,
        metrics: dict,
        duration_seconds: float
    ):
        """Log model training completion."""
        self.logger.info(
            f"MODEL_TRAIN | Type: {model_type} | Train: {train_samples} | Test: {test_samples} | "
            f"Metrics: {json.dumps(metrics)} | Duration: {duration_seconds:.2f}s"
        )
    
    def log_backtest_result(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        num_trades: int
    ):
        """Log backtest results."""
        self.logger.info(
            f"BACKTEST | Return: {total_return:.2%} | Sharpe: {sharpe_ratio:.2f} | "
            f"MaxDD: {max_drawdown:.2%} | WinRate: {win_rate:.2%} | Trades: {num_trades}"
        )
    
    def log_error(
        self,
        module: str,
        error_type: str,
        error_message: str,
        context: Optional[dict] = None
    ):
        """Log errors with context."""
        context_str = f" | Context: {json.dumps(context)}" if context else ""
        self._error_log.append({
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        })
        self.logger.error(
            f"ERROR | Module: {module} | Type: {error_type} | Message: {error_message}{context_str}"
        )
    
    def get_prediction_history(self) -> list:
        """Retrieve prediction history."""
        return self._prediction_log.copy()
    
    def get_trade_history(self) -> list:
        """Retrieve trade signal history."""
        return self._trade_log.copy()
    
    def get_error_history(self) -> list:
        """Retrieve error history."""
        return self._error_log.copy()
    
    def flush_logs(self):
        """Clear in-memory logs (useful for long-running processes)."""
        self._prediction_log.clear()
        self._trade_log.clear()
        self._error_log.clear()
        self.logger.info("Logs flushed from memory")


# Global logger instance
logger = TradingLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module."""
    if name:
        return setup_logger(name)
    return logger.logger


if __name__ == "__main__":
    # Test logging functionality
    test_logger = TradingLogger("test_egx")
    test_logger.log_data_gap("COMI.CA", "OHLCV", {"missing_dates": ["2024-01-15", "2024-01-16"]})
    test_logger.log_prediction("COMI.CA", 0.65, 0.78, ["RSI", "MACD"], datetime.now())
    test_logger.log_signal("COMI.CA", "BUY", 45.50, 1000, "ML signal + positive sentiment", 0.65)
    test_logger.log_risk_rejection("HRHO.CA", "BUY", "Exceeds position limit", "position_pct", 0.08)
    test_logger.log_error("data_ingestion", "APIError", "Failed to fetch data", {"symbol": "ETEL.CA"})
