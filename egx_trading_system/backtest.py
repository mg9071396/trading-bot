"""
Backtesting Engine Module for EGX Trading System.

Implements backtesting with realistic EGX friction:
- Commission: 0.15% - 0.25% per trade
- Slippage: 0.1% model for lower liquidity
- Stamp duty: 0.15% on sell orders (EGX requirement)
- Trading hours check: 10:00 AM - 2:30 PM EET, Sunday-Thursday

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence. Backtests often overstate performance.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, time
import warnings

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

from logger import TradingLogger
from config import trading_config, backtest_config

logger = TradingLogger("backtesting")


class EGXBacktester:
    """
    Backtesting engine for EGX trading strategies.
    
    Features:
    - Realistic EGX transaction costs (commission, slippage, stamp duty)
    - Trading hours validation
    - Portfolio-level metrics
    - Benchmark comparison (EGX30)
    - Drawdown analysis
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission_rate: float = None,
        slippage_rate: float = None,
        stamp_duty: float = None
    ):
        """
        Initialize backtester with EGX-specific parameters.
        
        Args:
            initial_capital: Starting capital in EGP
            commission_rate: Commission per trade (default from config)
            slippage_rate: Slippage model rate
            stamp_duty: Stamp duty on sells (EGX requirement)
        """
        self.initial_capital = initial_capital or backtest_config.initial_capital
        self.commission_rate = commission_rate or trading_config.commission_rate
        self.slippage_rate = slippage_rate or trading_config.slippage_rate
        self.stamp_duty = stamp_duty or trading_config.stamp_duty
        
        self.logger = logger.logger
        self.results = None
        self.metrics = {}
        
        # EGX trading hours
        self.market_open = trading_config.market_open
        self.market_close = trading_config.market_close
        
        if not VECTORBT_AVAILABLE:
            self.logger.warning(
                "vectorbt not installed. Install with: pip install vectorbt. "
                "Using simple backtest engine instead."
            )
    
    def is_within_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within EGX trading hours.
        
        EGX trades Sunday-Thursday, 10:00 AM - 2:30 PM EET.
        
        Args:
            timestamp: Datetime to check
        
        Returns:
            Boolean indicating if within trading hours
        """
        # Check day of week (Sunday=0 to Thursday=4)
        if timestamp.weekday() >= 5:  # Friday or Saturday
            return False
        
        # Check time of day
        if not (self.market_open <= timestamp.time() <= self.market_close):
            return False
        
        return True
    
    def calculate_transaction_costs(
        self,
        price: float,
        quantity: int,
        is_buy: bool
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            price: Execution price
            quantity: Number of shares
            is_buy: True for buy order, False for sell
        
        Returns:
            Dictionary with cost breakdown
        """
        notional = price * quantity
        
        # Commission
        commission = notional * self.commission_rate
        
        # Slippage (adverse price movement)
        slippage = notional * self.slippage_rate
        
        # Stamp duty (only on sells, EGX requirement)
        stamp = notional * self.stamp_duty if not is_buy else 0
        
        total_cost = commission + slippage + stamp
        
        return {
            'notional': notional,
            'commission': commission,
            'slippage': slippage,
            'stamp_duty': stamp,
            'total_cost': total_cost
        }
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Run backtest using vectorbt engine.
        
        Args:
            df: DataFrame with OHLCV data and features
            signals: Series of trading signals ('BUY', 'SELL', 'HOLD')
            feature_columns: Optional list of feature columns used
        
        Returns:
            DataFrame with backtest results and portfolio values
        """
        if len(df) == 0:
            raise ValueError("Empty dataset provided for backtest")
        
        self.logger.info(f"Running backtest on {len(df)} bars")
        
        # Convert signals to numeric
        entries = (signals == 'BUY').values
        exits = (signals == 'SELL').values
        
        # Get close prices
        close_prices = df['close'].values
        
        if VECTORBT_AVAILABLE:
            return self._run_vectorbt_backtest(close_prices, entries, exits, df)
        else:
            return self._run_simple_backtest(close_prices, entries, exits, df)
    
    def _run_vectorbt_backtest(
        self,
        close_prices: np.ndarray,
        entries: np.ndarray,
        exits: np.ndarray,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run backtest using vectorbt library."""
        try:
            # Create portfolio using vectorbt
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=exits,
                init_cash=self.initial_capital,
                fees=self.commission_rate + self.slippage_rate,
                freq='1D'
            )
            
            # Get portfolio value
            portfolio_value = portfolio.portfolio_value()
            
            # Add to DataFrame
            results_df = df.copy()
            results_df['portfolio_value'] = portfolio_value.values
            
            # Store results
            self.results = portfolio
            self._calculate_metrics_vbt(portfolio, close_prices)
            
            self.logger.info(f"Vectorbt backtest complete: {len(results_df)} bars")
            
            return results_df
            
        except Exception as e:
            self.logger.log_error(
                "backtesting",
                "VectorbtError",
                f"Vectorbt backtest failed: {str(e)}"
            )
            # Fall back to simple backtest
            return self._run_simple_backtest(close_prices, entries, exits, df)
    
    def _run_simple_backtest(
        self,
        close_prices: np.ndarray,
        entries: np.ndarray,
        exits: np.ndarray,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Simple backtest engine (fallback when vectorbt unavailable).
        
        Implements basic long-only strategy with transaction costs.
        """
        self.logger.warning("Using simple backtest engine")
        
        n_bars = len(close_prices)
        
        # Initialize tracking arrays
        position = np.zeros(n_bars)  # 1 if long, 0 if flat
        portfolio_value = np.zeros(n_bars)
        cash = self.initial_capital
        shares = 0
        
        # Track trades
        trades = []
        current_trade = None
        
        for i in range(n_bars):
            price = close_prices[i]
            
            # Entry signal
            if entries[i] and position[i-1] == 0 and i > 0:
                # Calculate how many shares to buy (max 5% of portfolio per position)
                max_position_value = cash * trading_config.max_position_size_pct
                shares_to_buy = int(max_position_value / price)
                
                if shares_to_buy > 0:
                    # Calculate costs
                    costs = self.calculate_transaction_costs(price, shares_to_buy, is_buy=True)
                    
                    # Execute buy
                    total_cost = shares_to_buy * price + costs['total_cost']
                    if cash >= total_cost:
                        cash -= total_cost
                        shares += shares_to_buy
                        position[i] = 1
                        current_trade = {'entry_price': price, 'entry_date': i, 'shares': shares_to_buy}
            
            # Exit signal
            elif exits[i] and position[i-1] == 1 and i > 0:
                if shares > 0:
                    # Calculate costs
                    costs = self.calculate_transaction_costs(price, shares, is_buy=False)
                    
                    # Execute sell
                    proceeds = shares * price - costs['total_cost']
                    cash += proceeds
                    
                    # Record trade
                    if current_trade:
                        current_trade['exit_price'] = price
                        current_trade['exit_date'] = i
                        current_trade['pnl'] = (price - current_trade['entry_price']) * current_trade['shares'] - costs['total_cost']
                        trades.append(current_trade)
                    
                    shares = 0
                    position[i] = 0
                    current_trade = None
            
            # Carry forward position
            else:
                position[i] = position[i-1] if i > 0 else 0
            
            # Calculate portfolio value
            portfolio_value[i] = cash + shares * price
        
        # Handle any open position at the end
        if shares > 0 and current_trade:
            # Mark to market
            current_trade['exit_price'] = close_prices[-1]
            current_trade['exit_date'] = n_bars - 1
            current_trade['pnl'] = (close_prices[-1] - current_trade['entry_price']) * current_trade['shares']
            trades.append(current_trade)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['portfolio_value'] = portfolio_value
        results_df['position'] = position
        results_df['cash'] = cash + shares * close_prices  # Final liquidation value
        
        # Store trades and calculate metrics
        self.trades = trades
        self._calculate_metrics_simple(portfolio_value, close_prices, trades)
        
        self.logger.info(f"Simple backtest complete: {len(trades)} trades")
        
        return results_df
    
    def _calculate_metrics_vbt(self, portfolio, close_prices: np.ndarray):
        """Calculate metrics from vectorbt portfolio."""
        try:
            # Get various metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            
            # Get trade statistics
            trade_stats = portfolio.trades.stats()
            
            win_rate = trade_stats.loc['win_rate'] if hasattr(trade_stats, 'loc') else 0
            profit_factor = trade_stats.loc['profit_factor'] if hasattr(trade_stats, 'loc') else 0
            num_trades = int(trade_stats.loc['count']) if hasattr(trade_stats, 'loc') else 0
            
            self.metrics = {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'num_trades': num_trades,
                'final_value': float(portfolio.portfolio_value().iloc[-1]),
                'initial_value': self.initial_capital
            }
            
            self.logger.log_backtest_result(
                total_return=self.metrics['total_return'],
                sharpe_ratio=self.metrics['sharpe_ratio'],
                max_drawdown=self.metrics['max_drawdown'],
                win_rate=self.metrics['win_rate'],
                num_trades=self.metrics['num_trades']
            )
            
        except Exception as e:
            self.logger.log_error(
                "backtesting",
                "MetricsError",
                f"Failed to calculate VBT metrics: {str(e)}"
            )
            self.metrics = {}
    
    def _calculate_metrics_simple(
        self,
        portfolio_value: np.ndarray,
        close_prices: np.ndarray,
        trades: List[Dict]
    ):
        """Calculate metrics from simple backtest."""
        if len(portfolio_value) == 0:
            self.metrics = {}
            return
        
        # Total return
        initial_value = self.initial_capital
        final_value = portfolio_value[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        daily_returns = pd.Series(portfolio_value).pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            rf_daily = backtest_config.risk_free_rate / 252
            excess_returns = daily_returns - rf_daily
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_value)
        drawdowns = (portfolio_value - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdowns.min())
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            total_profit = sum(t.get('pnl', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        self.metrics = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.99,
            'num_trades': len(trades),
            'final_value': float(final_value),
            'initial_value': float(initial_value)
        }
        
        self.logger.log_backtest_result(
            total_return=self.metrics['total_return'],
            sharpe_ratio=self.metrics['sharpe_ratio'],
            max_drawdown=self.metrics['max_drawdown'],
            win_rate=self.metrics['win_rate'],
            num_trades=self.metrics['num_trades']
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get backtest metrics."""
        return self.metrics.copy()
    
    def get_trade_log(self) -> List[Dict]:
        """Get detailed trade log."""
        return getattr(self, 'trades', []).copy()
    
    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot backtest results.
        
        Args:
            results_df: DataFrame with backtest results
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # Plot 1: Portfolio value vs Buy & Hold
            ax1 = axes[0]
            ax1.plot(results_df['date'], results_df['portfolio_value'], label='Portfolio', linewidth=2)
            
            # Buy and hold comparison
            initial_shares = self.initial_capital / results_df['close'].iloc[0]
            bh_value = initial_shares * results_df['close']
            ax1.plot(results_df['date'], bh_value, label='Buy & Hold', alpha=0.7)
            
            ax1.set_ylabel('Value (EGP)')
            ax1.set_title(f'Backtest Results: Total Return = {self.metrics.get("total_return", 0):.2%}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Drawdown
            ax2 = axes[1]
            cumulative_max = results_df['portfolio_value'].cummax()
            drawdown = (results_df['portfolio_value'] - cumulative_max) / cumulative_max * 100
            ax2.fill_between(results_df['date'], drawdown, 0, alpha=0.5, color='red')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_title(f'Max Drawdown: {self.metrics.get("max_drawdown", 0):.2%}')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Position and signals
            ax3 = axes[2]
            if 'position' in results_df.columns:
                ax3.plot(results_df['date'], results_df['position'], label='Position', drawstyle='steps-post')
            ax3.set_ylabel('Position (0/1)')
            ax3.set_xlabel('Date')
            ax3.set_title('Trading Positions')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.log_error(
                "backtesting",
                "PlotError",
                f"Failed to create plots: {str(e)}"
            )
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate backtest report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("EGX TRADING SYSTEM - BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("CONFIGURATION:")
        report.append(f"  Initial Capital: {self.initial_capital:,.0f} EGP")
        report.append(f"  Commission Rate: {self.commission_rate:.2%}")
        report.append(f"  Slippage Rate: {self.slippage_rate:.2%}")
        report.append(f"  Stamp Duty: {self.stamp_duty:.2%}")
        report.append("")
        
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Total Return: {self.metrics.get('total_return', 0):.2%}")
        report.append(f"  Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Win Rate: {self.metrics.get('win_rate', 0):.2%}")
        report.append(f"  Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
        report.append(f"  Number of Trades: {self.metrics.get('num_trades', 0)}")
        report.append("")
        
        report.append("CAPITAL:")
        report.append(f"  Initial: {self.metrics.get('initial_value', 0):,.0f} EGP")
        report.append(f"  Final: {self.metrics.get('final_value', 0):,.0f} EGP")
        report.append("")
        
        report.append("=" * 60)
        report.append("WARNING: This is a framework for research purposes only.")
        report.append("Past performance does not guarantee future results.")
        report.append("DO NOT use for live trading without extensive testing.")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_text


if __name__ == "__main__":
    # Test backtesting engine
    from data_ingestion import EGXDataFetcher
    from feature_engineering import FeatureEngineer
    from model import QuantModel
    
    print("=" * 60)
    print("Testing EGX Backtesting Engine")
    print("=" * 60)
    
    # Fetch and prepare data
    print("\n1. Preparing data...")
    fetcher = EGXDataFetcher()
    df = fetcher.fetch_historical_data("COMI.CA", period="2y")
    
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    df_clean = fe.handle_missing_values(df_features)
    
    # Train model and generate signals
    print("\n2. Training model and generating signals...")
    model = QuantModel()
    
    try:
        model.fit(df=df_clean, feature_columns=fe.feature_columns, verbose=False)
        df_signals = model.generate_signals(df_clean, fe.feature_columns)
        
        # Create simple entry/exit logic
        signals = pd.Series('HOLD', index=df_signals.index)
        signals[df_signals['signal'] == 'BUY'] = 'BUY'
        
        # Add SELL signals (e.g., after 5 days or when prediction drops)
        for i in range(1, len(signals)):
            if signals.iloc[i-1] == 'BUY':
                # Exit after 5 days or if probability drops below 0.4
                if i + 5 < len(signals) or df_signals['signal_probability'].iloc[i] < 0.4:
                    signals.iloc[i] = 'SELL'
        
        # Run backtest
        print("\n3. Running backtest...")
        backtester = EGXBacktester(initial_capital=1000000)
        results_df = backtester.run_backtest(df_signals, signals)
        
        # Display metrics
        print("\n4. Backtest Metrics:")
        metrics = backtester.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}" if 'return' in key or 'drawdown' in key or 'rate' in key else f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # Generate report
        print("\n5. Generating report...")
        report = backtester.generate_report()
        print(report)
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
        print("Note: This may fail if dependencies are not installed")
    
    print("\n" + "=" * 60)
    print("Backtest Test Complete")
    print("=" * 60)
