"""
Real-Time Trading Module for EGX Trading System.

This module enables live trading with real-time data feeds:
1. Real-time price data fetching (via yfinance or custom API)
2. Live signal generation
3. Paper trading execution
4. Real-time monitoring and logging

WARNING: =====================================================================
THIS IS FOR PAPER TRADING AND TESTING PURPOSES ONLY.

DO NOT use this for live trading without:
- Extensive testing in paper trading mode
- Integration with licensed Egyptian broker APIs
- Proper risk management and compliance review
===============================================================================

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import json
import threading
from enum import Enum

from config import trading_config, get_config
from logger import TradingLogger
from data_ingestion import EGXDataFetcher
from feature_engineering import FeatureEngineer
from model import QuantModel
from risk_and_execution import TradingSystem, SignalType

logger = TradingLogger("realtime_trading")


class MarketStatus(Enum):
    """EGX Market Status."""
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    HALTED = "HALTED"
    WEEKEND = "WEEKEND"


class RealTimeTrader:
    """
    Real-time trading engine for EGX securities.
    
    Features:
    - Real-time price monitoring
    - Live signal generation
    - Paper trading execution
    - Risk management integration
    - Performance tracking
    """
    
    def __init__(
        self,
        symbols: List[str],
        model_path: Optional[str] = None,
        initial_capital: float = 1000000,
        paper_trading: bool = True,
        update_interval: int = 60,  # seconds
        data_dir: str = "./realtime_data"
    ):
        """
        Initialize real-time trader.
        
        Args:
            symbols: List of EGX symbols to trade
            model_path: Path to pre-trained model (if None, will train on startup)
            initial_capital: Starting capital in EGP
            paper_trading: If True, simulate trades (recommended)
            update_interval: Data refresh interval in seconds
            data_dir: Directory for storing real-time data
        """
        self.symbols = symbols
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        self.update_interval = update_interval
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_fetcher = EGXDataFetcher(cache_dir=str(self.data_dir / "cache"))
        self.feature_engineer = FeatureEngineer()
        self.trading_system = TradingSystem(
            portfolio_value=initial_capital,
            paper_trading=paper_trading
        )
        
        # State variables
        self.models: Dict[str, QuantModel] = {}
        self.latest_prices: Dict[str, float] = {}
        self.latest_features: Dict[str, pd.DataFrame] = {}
        self.signals: Dict[str, str] = {}
        self.positions: Dict[str, int] = {symbol: 0 for symbol in symbols}
        self.trades: List[Dict[str, Any]] = []
        self.is_running = False
        self.market_status = MarketStatus.CLOSED
        
        # Historical data for feature calculation
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        self.logger = logger.logger
        self.logger.info(f"RealTimeTrader initialized for {len(symbols)} symbols")
        self.logger.info(f"Paper Trading Mode: {paper_trading}")
    
    def initialize_models(self, lookback_days: int = 252) -> bool:
        """
        Initialize or load models for all symbols.
        
        Args:
            lookback_days: Number of days of historical data for training
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Initializing models...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in self.symbols:
            try:
                # Fetch historical data
                self.logger.info(f"Fetching data for {symbol}...")
                df = self.data_fetcher.fetch_historical_data(
                    symbol, start_date, end_date
                )
                
                if len(df) < 100:
                    self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    continue
                
                self.historical_data[symbol] = df
                
                # Create features
                df_features = self.feature_engineer.create_all_features(df)
                df_clean = self.feature_engineer.handle_missing_values(df_features)
                
                if len(df_clean) < 100:
                    self.logger.warning(f"Insufficient features for {symbol} after cleaning")
                    continue
                
                # Train model
                self.logger.info(f"Training model for {symbol}...")
                model = QuantModel()
                metrics = model.fit(
                    df=df_clean,
                    feature_columns=self.feature_engineer.feature_columns,
                    target_column='target',
                    verbose=False
                )
                
                self.models[symbol] = model
                self.latest_features[symbol] = df_clean
                
                self.logger.info(
                    f"✓ Model trained for {symbol} | "
                    f"Accuracy: {metrics.get('accuracy', 0):.4f} | "
                    f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to initialize model for {symbol}: {str(e)}")
                continue
        
        if not self.models:
            self.logger.error("No models could be initialized")
            return False
        
        self.logger.info(f"Successfully initialized {len(self.models)} models")
        return True
    
    def check_market_status(self) -> MarketStatus:
        """
        Check current EGX market status.
        
        EGX Trading Hours:
        - Sunday-Thursday: 10:00 AM - 2:30 PM (Cairo Time, UTC+2)
        - Friday-Saturday: Closed
        """
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        # Check weekend (Friday=4, Saturday=5 in Python)
        if weekday >= 4:  # Friday or Saturday
            return MarketStatus.WEEKEND
        
        # Convert to Cairo time (simplified - should use pytz in production)
        cairo_hour = hour - 2  # Approximate UTC to Cairo conversion
        
        # Check market hours (10:00 - 14:30 Cairo time)
        if cairo_hour < 10:
            return MarketStatus.PRE_MARKET
        elif cairo_hour >= 14 or (cairo_hour == 14 and minute > 30):
            return MarketStatus.CLOSED
        else:
            return MarketStatus.OPEN
    
    def fetch_realtime_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch latest real-time data for a symbol.
        
        Args:
            symbol: EGX symbol
            
        Returns:
            DataFrame with latest price data or None
        """
        try:
            # Fetch recent data (last 5 days to ensure we have enough for features)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            df = self.data_fetcher.fetch_historical_data(
                symbol, start_date, end_date, use_cache=False
            )
            
            if df is None or len(df) == 0:
                self.logger.warning(f"No real-time data for {symbol}")
                return None
            
            # Get latest price
            latest_row = df.iloc[-1]
            self.latest_prices[symbol] = latest_row['close']
            
            # Update historical data
            if symbol in self.historical_data:
                # Append new data
                self.historical_data[symbol] = pd.concat(
                    [self.historical_data[symbol], df],
                    ignore_index=True
                ).drop_duplicates(subset=['date'], keep='last')
            else:
                self.historical_data[symbol] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None
    
    def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: EGX symbol
            
        Returns:
            Signal dictionary or None
        """
        if symbol not in self.models:
            self.logger.warning(f"No model available for {symbol}")
            return None
        
        if symbol not in self.historical_data:
            self.logger.warning(f"No historical data for {symbol}")
            return None
        
        try:
            model = self.models[symbol]
            df = self.historical_data[symbol]
            
            # Get latest row
            latest_df = df.iloc[-1:].copy()
            
            # Create features for latest data
            latest_features = self.feature_engineer.create_all_features(latest_df)
            latest_clean = self.feature_engineer.handle_missing_values(latest_features)
            
            if len(latest_clean) == 0:
                self.logger.warning(f"Cannot create features for {symbol}")
                return None
            
            # Generate signal
            signal_df = model.generate_signals(
                latest_clean,
                self.feature_engineer.feature_columns
            )
            
            signal = signal_df['signal'].iloc[0]
            probability = signal_df.get('signal_probability', pd.Series([0.5])).iloc[0]
            price = self.latest_prices.get(symbol, latest_clean['close'].iloc[0])
            
            # Get sentiment score if available
            sentiment_score = latest_clean.get('news_sentiment', pd.Series([0])).iloc[0]
            
            signal_result = {
                'symbol': symbol,
                'signal': signal,
                'probability': probability,
                'price': price,
                'timestamp': datetime.now(),
                'sentiment_score': sentiment_score
            }
            
            self.signals[symbol] = signal
            
            return signal_result
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def execute_trade(self, signal_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a trade based on signal.
        
        Args:
            signal_info: Signal dictionary from generate_signal
            
        Returns:
            Trade result dictionary or None
        """
        symbol = signal_info['symbol']
        signal = signal_info['signal']
        price = signal_info['price']
        probability = signal_info['probability']
        
        # Convert signal to SignalType
        if signal == 'BUY':
            signal_type = SignalType.BUY
        elif signal == 'SELL':
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate position size using risk management
        suggested_quantity = int(self.initial_capital * 0.1 / price)  # 10% allocation
        
        result = self.trading_system.process_signal(
            signal=signal_type,
            symbol=symbol,
            price=price,
            suggested_quantity=suggested_quantity,
            sentiment_score=signal_info.get('sentiment_score', 0),
            win_rate=0.55,  # Should come from model metrics
            win_loss_ratio=1.5
        )
        
        if result['status'] == 'APPROVED':
            quantity = result['approved_quantity']
            
            # Execute trade (paper trading)
            if self.paper_trading:
                if signal_type == SignalType.BUY:
                    cost = quantity * price
                    if cost <= self.trading_system.portfolio_value:
                        self.positions[symbol] += quantity
                        self.trading_system.portfolio_value -= cost
                        
                        trade_record = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'type': 'BUY',
                            'quantity': quantity,
                            'price': price,
                            'cost': cost,
                            'status': 'EXECUTED'
                        }
                        self.trades.append(trade_record)
                        
                        self.logger.info(
                            f"📈 BUY EXECUTED: {quantity} {symbol} @ {price:.2f} EGP | "
                            f"Cost: {cost:,.2f} EGP"
                        )
                        
                        return trade_record
                    else:
                        self.logger.warning(f"Insufficient capital for BUY {symbol}")
                        return None
                        
                elif signal_type == SignalType.SELL:
                    if self.positions[symbol] > 0:
                        sell_quantity = min(quantity, self.positions[symbol])
                        revenue = sell_quantity * price
                        self.positions[symbol] -= sell_quantity
                        self.trading_system.portfolio_value += revenue
                        
                        trade_record = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'type': 'SELL',
                            'quantity': sell_quantity,
                            'price': price,
                            'revenue': revenue,
                            'status': 'EXECUTED'
                        }
                        self.trades.append(trade_record)
                        
                        self.logger.info(
                            f"📉 SELL EXECUTED: {sell_quantity} {symbol} @ {price:.2f} EGP | "
                            f"Revenue: {revenue:,.2f} EGP"
                        )
                        
                        return trade_record
                    else:
                        self.logger.warning(f"No position to sell for {symbol}")
                        return None
            else:
                # Live trading mode (requires broker integration)
                self.logger.warning("Live trading not implemented - requires broker API")
                return None
        else:
            self.logger.info(
                f"Trade REJECTED for {symbol}: {result.get('rejection_reasons', ['Unknown'])}"
            )
            return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        total_value = self.trading_system.portfolio_value
        
        # Add current position values
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in self.latest_prices:
                total_value += quantity * self.latest_prices[symbol]
        
        return {
            'cash': self.trading_system.portfolio_value,
            'positions': self.positions,
            'latest_prices': self.latest_prices,
            'total_value': total_value,
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'timestamp': datetime.now()
        }
    
    def save_snapshot(self):
        """Save current state snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': self.get_portfolio_summary(),
            'signals': self.signals,
            'recent_trades': self.trades[-10:]  # Last 10 trades
        }
        
        # Convert timestamps to strings
        snapshot['portfolio']['timestamp'] = snapshot['portfolio']['timestamp'].isoformat()
        snapshot['recent_trades'] = [
            {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in trade.items()}
            for trade in snapshot['recent_trades']
        ]
        
        snapshot_file = self.data_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        self.logger.debug(f"Snapshot saved to {snapshot_file}")
    
    def run_iteration(self) -> Dict[str, Any]:
        """
        Run one iteration of the trading loop.
        
        Returns:
            Dictionary with iteration results
        """
        results = {
            'timestamp': datetime.now(),
            'market_status': self.check_market_status(),
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': []
        }
        
        # Check market status
        if results['market_status'] == MarketStatus.WEEKEND:
            self.logger.info("Market is closed (weekend)")
            return results
        
        # Fetch real-time data and generate signals
        for symbol in self.symbols:
            if symbol not in self.models:
                continue
            
            # Fetch latest data
            df = self.fetch_realtime_data(symbol)
            if df is None:
                results['errors'].append(f"Failed to fetch data for {symbol}")
                continue
            
            # Generate signal
            signal = self.generate_signal(symbol)
            if signal:
                results['signals_generated'] += 1
                self.logger.info(
                    f"Signal for {symbol}: {signal['signal']} | "
                    f"Price: {signal['price']:.2f} | "
                    f"Probability: {signal['probability']:.2%}"
                )
                
                # Execute trade if signal is not HOLD
                if signal['signal'] != 'HOLD':
                    trade = self.execute_trade(signal)
                    if trade:
                        results['trades_executed'] += 1
        
        # Save snapshot
        self.save_snapshot()
        
        return results
    
    def run(self, duration_hours: Optional[float] = None):
        """
        Run the real-time trading loop.
        
        Args:
            duration_hours: How long to run (None for indefinite)
        """
        self.is_running = True
        self.logger.info("=" * 70)
        self.logger.info("STARTING REAL-TIME TRADING")
        self.logger.info("=" * 70)
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Initial Capital: {self.initial_capital:,.0f} EGP")
        self.logger.info(f"Update Interval: {self.update_interval} seconds")
        self.logger.info(f"Duration: {duration_hours or 'Indefinite'} hours")
        self.logger.info("=" * 70)
        
        start_time = datetime.now()
        iteration_count = 0
        
        try:
            while self.is_running:
                # Check if duration limit reached
                if duration_hours:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        self.logger.info(f"Duration limit reached ({duration_hours} hours)")
                        break
                
                # Run iteration
                iteration_count += 1
                self.logger.info(f"\n{'='*70}\nIteration {iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                results = self.run_iteration()
                
                # Print summary
                portfolio = self.get_portfolio_summary()
                self.logger.info(f"\nPortfolio Value: {portfolio['total_value']:,.2f} EGP")
                self.logger.info(f"Total Return: {portfolio['total_return']:.2%}")
                self.logger.info(f"Signals Generated: {results['signals_generated']}")
                self.logger.info(f"Trades Executed: {results['trades_executed']}")
                
                if results['errors']:
                    self.logger.warning(f"Errors: {results['errors']}")
                
                # Wait for next iteration
                if self.is_running:
                    self.logger.info(f"\nWaiting {self.update_interval} seconds...")
                    time.sleep(self.update_interval)
                    
        except KeyboardInterrupt:
            self.logger.info("\nTrading interrupted by user")
        except Exception as e:
            self.logger.error(f"Trading error: {str(e)}")
        finally:
            self.is_running = False
            self.logger.info("=" * 70)
            self.logger.info("TRADING STOPPED")
            self.logger.info("=" * 70)
            
            # Final summary
            portfolio = self.get_portfolio_summary()
            self.logger.info(f"Final Portfolio Value: {portfolio['total_value']:,.2f} EGP")
            self.logger.info(f"Total Return: {portfolio['total_return']:.2%}")
            self.logger.info(f"Total Trades: {len(self.trades)}")
    
    def stop(self):
        """Stop the trading loop."""
        self.is_running = False
        self.logger.info("Stopping trading loop...")


def main():
    """Main function to run real-time trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EGX Real-Time Trading System")
    parser.add_argument(
        "--symbols",
        type=str,
        default="COMI.CA,ETEL.CA,HRHO.CA",
        help="Comma-separated list of EGX symbols"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000000,
        help="Initial capital in EGP"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Enable paper trading mode (default)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading mode (requires broker integration)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Trading duration in hours (None for indefinite)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Lookback days for model training"
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    paper_trading = args.paper or not args.live
    
    # Initialize trader
    trader = RealTimeTrader(
        symbols=symbols,
        initial_capital=args.capital,
        paper_trading=paper_trading,
        update_interval=args.interval
    )
    
    # Initialize models
    if not trader.initialize_models(lookback_days=args.lookback):
        print("Failed to initialize models. Exiting.")
        return
    
    # Run trading
    trader.run(duration_hours=args.duration)


if __name__ == "__main__":
    main()
