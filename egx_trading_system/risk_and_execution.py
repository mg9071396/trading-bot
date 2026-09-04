"""
Risk Management and Execution Module for EGX Trading System.

Implements strict risk controls:
- Position sizing limits (max 5% per position)
- Maximum drawdown circuit breaker (halt at 15% drawdown)
- Sentiment-based filtering (reject trades with strongly negative sentiment)
- Kelly Criterion for optimal position sizing

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence. Risk management is critical for survival.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import json

from logger import TradingLogger
from config import trading_config

# Initialize logger
trading_logger = TradingLogger("risk_management")
logger = trading_logger.logger


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskCheckResult:
    """Result of a risk check."""
    
    def __init__(
        self,
        passed: bool,
        signal: SignalType,
        symbol: str,
        rejection_reasons: List[str] = None,
        adjusted_quantity: int = None
    ):
        self.passed = passed
        self.signal = signal
        self.symbol = symbol
        self.rejection_reasons = rejection_reasons or []
        self.adjusted_quantity = adjusted_quantity
    
    def __bool__(self):
        return self.passed
    
    def to_dict(self) -> Dict:
        return {
            'passed': self.passed,
            'signal': self.signal.value,
            'symbol': self.symbol,
            'rejection_reasons': self.rejection_reasons,
            'adjusted_quantity': self.adjusted_quantity
        }


class RiskManager:
    """
    Risk management system for EGX trading.
    
    Implements multiple layers of risk control:
    1. Position sizing limits
    2. Portfolio drawdown circuit breaker
    3. Sentiment-based filtering
    4. Kelly Criterion optimization
    5. Concentration limits
    6. Daily loss limits
    """
    
    def __init__(
        self,
        portfolio_value: float = None,
        max_position_pct: float = None,
        max_drawdown_pct: float = None,
        sentiment_threshold: float = None
    ):
        """
        Initialize risk manager.
        
        Args:
            portfolio_value: Current portfolio value
            max_position_pct: Maximum position size as % of portfolio
            max_drawdown_pct: Maximum drawdown before halting trades
            sentiment_threshold: Minimum sentiment score to allow trades
        """
        self.portfolio_value = portfolio_value or trading_config.initial_capital
        self.max_position_pct = max_position_pct or trading_config.max_position_size_pct
        self.max_drawdown_pct = max_drawdown_pct or trading_config.max_portfolio_drawdown
        self.sentiment_threshold = sentiment_threshold or trading_config.sentiment_sell_threshold
        
        self.logger = trading_logger
        
        # Track state
        self.current_drawdown = 0.0
        self.peak_value = self.portfolio_value
        self.positions = {}  # symbol -> quantity
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.halted = False  # Trading halt flag
        
        # Kelly Criterion parameters
        self.kelly_fraction = trading_config.kelly_fraction
        self.max_kelly_position = trading_config.max_kelly_position
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and track drawdown."""
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        
        # Update peak and drawdown
        if new_value > self.peak_value:
            self.peak_value = new_value
        
        self.current_drawdown = (self.peak_value - new_value) / self.peak_value
        
        # Check if trading should be halted
        if self.current_drawdown >= self.max_drawdown_pct:
            self.halted = True
            self.logger.log_risk_rejection(
                "PORTFOLIO",
                "ALL",
                f"Trading halted: Drawdown {self.current_drawdown:.2%} exceeds limit {self.max_drawdown_pct:.2%}",
                "drawdown",
                self.current_drawdown
            )
    
    def check_signal(
        self,
        signal: SignalType,
        symbol: str,
        price: float,
        suggested_quantity: int,
        sentiment_score: float = 0.0,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.0
    ) -> RiskCheckResult:
        """
        Perform comprehensive risk check on a trading signal.
        
        Args:
            signal: Buy/Sell/Hold signal
            symbol: Stock symbol
            price: Current price
            suggested_quantity: Suggested number of shares
            sentiment_score: News sentiment score (-1 to 1)
            win_rate: Historical win rate for this signal type
            avg_win_loss_ratio: Average win/loss ratio
        
        Returns:
            RiskCheckResult with approval/rejection decision
        """
        rejection_reasons = []
        adjusted_quantity = suggested_quantity
        
        # Check if trading is halted
        if self.halted:
            return RiskCheckResult(
                passed=False,
                signal=signal,
                symbol=symbol,
                rejection_reasons=["Trading halted due to risk limits"],
                adjusted_quantity=0
            )
        
        # Only check buy signals (sell signals are always allowed for risk reduction)
        if signal == SignalType.BUY:
            # 1. Check position size limit
            position_check = self._check_position_size(symbol, price, suggested_quantity)
            if not position_check['passed']:
                rejection_reasons.append(position_check['reason'])
                adjusted_quantity = position_check.get('adjusted_quantity', 0)
            
            # 2. Check drawdown circuit breaker
            if self.current_drawdown >= self.max_drawdown_pct * 0.8:  # Warning level
                rejection_reasons.append(f"Drawdown warning: {self.current_drawdown:.2%}")
            
            if self.current_drawdown >= self.max_drawdown_pct:
                rejection_reasons.append(
                    f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}"
                )
            
            # 3. Check sentiment threshold
            if sentiment_score < self.sentiment_threshold:
                rejection_reasons.append(
                    f"Sentiment too negative: {sentiment_score:.2f} < {self.sentiment_threshold}"
                )
            
            # 4. Apply Kelly Criterion for position sizing
            kelly_size = self._calculate_kelly_position(
                win_rate=win_rate,
                win_loss_ratio=avg_win_loss_ratio,
                price=price
            )
            
            if kelly_size < suggested_quantity:
                # Kelly criterion suggests smaller position
                adjusted_quantity = min(adjusted_quantity, kelly_size) if adjusted_quantity else kelly_size
            
            # 5. Check concentration limits
            concentration_check = self._check_concentration(symbol, price, adjusted_quantity)
            if not concentration_check['passed']:
                rejection_reasons.append(concentration_check['reason'])
        
        # Determine if signal passes
        passed = len(rejection_reasons) == 0
        
        if not passed and signal == SignalType.BUY:
            self.logger.log_risk_rejection(
                symbol=symbol,
                signal=signal.value,
                rejection_reason="; ".join(rejection_reasons),
                risk_metric="sentiment",
                risk_value=sentiment_score
            )
        
        return RiskCheckResult(
            passed=passed,
            signal=signal,
            symbol=symbol,
            rejection_reasons=rejection_reasons,
            adjusted_quantity=adjusted_quantity if passed else 0
        )
    
    def _check_position_size(
        self,
        symbol: str,
        price: float,
        quantity: int
    ) -> Dict[str, Any]:
        """Check if position size is within limits."""
        position_value = price * quantity
        max_position_value = self.portfolio_value * self.max_position_pct
        
        # Check existing position
        existing_quantity = self.positions.get(symbol, 0)
        existing_value = existing_quantity * price if existing_quantity else 0
        
        total_value = existing_value + position_value
        
        if total_value > max_position_value:
            # Calculate maximum allowed additional quantity
            remaining_value = max_position_value - existing_value
            adjusted_quantity = int(remaining_value / price) if remaining_value > 0 else 0
            
            return {
                'passed': False,
                'reason': f"Position would exceed {self.max_position_pct:.0%} limit",
                'adjusted_quantity': adjusted_quantity
            }
        
        return {'passed': True}
    
    def _check_concentration(
        self,
        symbol: str,
        price: float,
        quantity: int
    ) -> Dict[str, Any]:
        """Check sector/industry concentration limits."""
        # Simplified: just check single stock concentration
        position_value = price * quantity
        position_pct = position_value / self.portfolio_value
        
        # Hard limit at 5%
        if position_pct > self.max_position_pct:
            return {
                'passed': False,
                'reason': f"Single stock concentration {position_pct:.1%} exceeds {self.max_position_pct:.0%} limit"
            }
        
        return {'passed': True}
    
    def _calculate_kelly_position(
        self,
        win_rate: float,
        win_loss_ratio: float,
        price: float
    ) -> int:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win probability
            R = Win/Loss ratio
        
        Uses fractional Kelly (25%) to reduce volatility.
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0
        
        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly (reduce by kelly_fraction)
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # Cap at maximum
        kelly_pct = min(kelly_pct, self.max_kelly_position)
        
        # Don't take negative positions
        if kelly_pct <= 0:
            return 0
        
        # Convert to share quantity
        kelly_value = self.portfolio_value * kelly_pct
        kelly_shares = int(kelly_value / price)
        
        return kelly_shares
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary."""
        total_exposure = sum(
            qty * 100  # Approximate with placeholder price
            for qty in self.positions.values()
        )
        
        return {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown_pct,
            'trading_halted': self.halted,
            'num_positions': len(self.positions),
            'total_exposure_estimate': total_exposure,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today
        }
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ):
        """Record executed trade for tracking."""
        if action == 'BUY':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif action == 'SELL':
            self.positions[symbol] = max(0, self.positions.get(symbol, 0) - quantity)
        
        self.trades_today += 1
        
        self.logger.log_execution(
            symbol=symbol,
            action=action,
            quantity=quantity,
            executed_price=price,
            status="EXECUTED"
        )
    
    def reset_daily_counters(self):
        """Reset daily tracking counters (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.logger.info("Daily counters reset")
    
    def emergency_halt(self, reason: str):
        """Immediately halt all trading."""
        self.halted = True
        self.logger.log_risk_rejection(
            symbol="ALL",
            signal="ALL",
            rejection_reason=f"Emergency halt: {reason}"
        )


class ExecutionEngine:
    """
    Placeholder execution engine for order routing.
    
    WARNING: This is a PLACEHOLDER. For production, integrate with
    a licensed Egyptian broker's API such as:
    - EFG Hermes (https://www.efghermes.com/)
    - CI Capital (https://www.cicapital.com.eg/)
    - Interactive Brokers (if supporting international access)
    - Direct EGX membership for institutional traders
    
    DO NOT attempt live trading without proper brokerage integration
    and regulatory compliance.
    """
    
    def __init__(
        self,
        broker_name: str = "PLACEHOLDER",
        api_key: str = None,
        api_secret: str = None,
        paper_trading: bool = True
    ):
        """
        Initialize execution engine.
        
        Args:
            broker_name: Name of broker (placeholder by default)
            api_key: Broker API key
            api_secret: Broker API secret
            paper_trading: If True, simulate orders without executing
        """
        self.broker_name = broker_name
        self.paper_trading = paper_trading
        self.api_key = api_key or trading_config.broker_api_key
        self.api_secret = api_secret or trading_config.broker_secret
        
        self.logger = logger
        self.order_log = []
        
        if not paper_trading:
            self.logger.warning(
                "=" * 60,
                "LIVE TRADING MODE ENABLED",
                "=" * 60,
                "Broker:", broker_name,
                "WARNING: Ensure proper licensing and compliance before live trading!",
                "=" * 60
            )
        else:
            self.logger.info("Paper trading mode enabled - orders will be simulated")
    
    def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float = None,
        time_in_force: str = "DAY"
    ) -> Dict[str, Any]:
        """
        Submit an order to the broker.
        
        Args:
            symbol: EGX symbol (e.g., "COMI.CA")
            action: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MARKET", "LIMIT", "STOP", etc.
            limit_price: Limit price for limit orders
            time_in_force: "DAY", "GTC", "IOC", etc.
        
        Returns:
            Order result dictionary
        """
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        order_details = {
            'order_id': order_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_type': order_type,
            'limit_price': limit_price,
            'time_in_force': time_in_force,
            'timestamp': datetime.now().isoformat(),
            'status': 'PENDING',
            'paper_trade': self.paper_trading
        }
        
        if self.paper_trading:
            # Simulate order execution
            return self._simulate_execution(order_details)
        else:
            # PLACEHOLDER: Real broker integration goes here
            self.logger.warning(
                f"LIVE ORDER SUBMISSION NOT IMPLEMENTED",
                f"Broker: {self.broker_name}",
                f"Order: {order_details}"
            )
            
            # TODO: Implement actual broker API integration
            # Example structure for EFG Hermes API:
            # response = efg_client.place_order(
            #     symbol=symbol,
            #     side=action.lower(),
            #     quantity=quantity,
            #     order_type=order_type.lower(),
            #     limit_price=limit_price,
            #     time_in_force=time_in_force
            # )
            
            return {
                **order_details,
                'status': 'NOT_IMPLEMENTED',
                'message': 'Live trading requires broker API integration'
            }
    
    def _simulate_execution(self, order: Dict) -> Dict:
        """Simulate order execution for paper trading."""
        # Simulate fill at market price (would need real price feed)
        estimated_price = 50.0  # Placeholder
        
        order['status'] = 'FILLED'
        order['filled_quantity'] = order['quantity']
        order['filled_price'] = estimated_price
        order['commission'] = estimated_price * order['quantity'] * trading_config.commission_rate
        order['fill_time'] = datetime.now().isoformat()
        
        self.order_log.append(order)
        
        self.logger.log_execution(
            symbol=order['symbol'],
            action=order['action'],
            quantity=order['filled_quantity'],
            executed_price=order['filled_price'],
            order_id=order['order_id'],
            status='FILLED (SIMULATED)'
        )
        
        return order
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an outstanding order."""
        if self.paper_trading:
            return {
                'order_id': order_id,
                'status': 'CANCELLED',
                'message': 'Order cancelled (simulated)'
            }
        else:
            # TODO: Implement broker-specific cancellation
            return {
                'order_id': order_id,
                'status': 'NOT_IMPLEMENTED',
                'message': 'Live cancellation requires broker API integration'
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order."""
        for order in self.order_log:
            if order['order_id'] == order_id:
                return order
        
        return {
            'order_id': order_id,
            'status': 'NOT_FOUND'
        }
    
    def get_account_info(self) -> Dict:
        """Get account information from broker."""
        if self.paper_trading:
            return {
                'account_type': 'PAPER',
                'broker': self.broker_name,
                'buying_power': trading_config.initial_capital,
                'cash_balance': trading_config.initial_capital,
                'portfolio_value': trading_config.initial_capital
            }
        else:
            # TODO: Implement broker-specific account info
            return {
                'account_type': 'LIVE',
                'broker': self.broker_name,
                'message': 'Live account info requires broker API integration'
            }
    
    def get_order_history(self, days: int = 30) -> List[Dict]:
        """Get order history."""
        return self.order_log.copy()


class TradingSystem:
    """
    Main trading system orchestrator combining risk management and execution.
    
    Flow:
    1. Receive signal from model
    2. Run through risk manager
    3. If approved, send to execution engine
    4. Log and track results
    """
    
    def __init__(
        self,
        portfolio_value: float = None,
        paper_trading: bool = True
    ):
        """Initialize trading system."""
        self.portfolio_value = portfolio_value or trading_config.initial_capital
        self.risk_manager = RiskManager(portfolio_value=self.portfolio_value)
        self.execution_engine = ExecutionEngine(paper_trading=paper_trading)
        self.logger = logger
        
        self.logger.info(
            f"Trading System initialized | Paper Trading: {paper_trading} | "
            f"Portfolio Value: {self.portfolio_value:,.0f} EGP"
        )
    
    def process_signal(
        self,
        signal: SignalType,
        symbol: str,
        price: float,
        suggested_quantity: int,
        sentiment_score: float = 0.0,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process a trading signal through risk management to execution.
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            symbol: Stock symbol
            price: Current price
            suggested_quantity: Model-suggested quantity
            sentiment_score: News sentiment score
            win_rate: Historical win rate
            win_loss_ratio: Win/loss ratio
        
        Returns:
            Result dictionary with full audit trail
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'original_signal': signal.value,
            'price': price,
            'suggested_quantity': suggested_quantity,
            'sentiment_score': sentiment_score
        }
        
        # Step 1: Risk check
        risk_result = self.risk_manager.check_signal(
            signal=signal,
            symbol=symbol,
            price=price,
            suggested_quantity=suggested_quantity,
            sentiment_score=sentiment_score,
            win_rate=win_rate,
            avg_win_loss_ratio=win_loss_ratio
        )
        
        result['risk_check'] = risk_result.to_dict()
        
        if not risk_result.passed:
            result['status'] = 'REJECTED'
            result['rejection_reasons'] = risk_result.rejection_reasons
            self.logger.log_risk_rejection(
                symbol=symbol,
                signal=signal.value,
                rejection_reason="; ".join(risk_result.rejection_reasons)
            )
            return result
        
        # Step 2: Execute order (if not HOLD)
        if signal == SignalType.HOLD:
            result['status'] = 'NO_ACTION'
            return result
        
        order_result = self.execution_engine.submit_order(
            symbol=symbol,
            action=signal.value,
            quantity=risk_result.adjusted_quantity,
            order_type="MARKET"
        )
        
        result['order'] = order_result
        result['status'] = order_result.get('status', 'UNKNOWN')
        
        # Step 3: Record trade in risk manager
        if order_result.get('status') == 'FILLED':
            self.risk_manager.record_trade(
                symbol=symbol,
                action=signal.value,
                quantity=order_result.get('filled_quantity', 0),
                price=order_result.get('filled_price', price)
            )
        
        return result
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value in risk manager."""
        self.risk_manager.update_portfolio_value(new_value)
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            'risk_manager': self.risk_manager.get_position_summary(),
            'execution_engine': self.execution_engine.get_account_info(),
            'trading_halted': self.risk_manager.halted
        }


if __name__ == "__main__":
    # Test risk management and execution
    print("=" * 60)
    print("Testing Risk Management & Execution System")
    print("=" * 60)
    
    # Initialize trading system
    print("\n1. Initializing trading system...")
    system = TradingSystem(portfolio_value=1000000, paper_trading=True)
    
    # Test various scenarios
    print("\n2. Testing risk checks...")
    
    # Scenario 1: Normal buy signal
    print("\n   Scenario 1: Normal BUY signal")
    result = system.process_signal(
        signal=SignalType.BUY,
        symbol="COMI.CA",
        price=45.50,
        suggested_quantity=1000,
        sentiment_score=0.65,
        win_rate=0.55,
        win_loss_ratio=1.2
    )
    print(f"   Status: {result['status']}")
    if result.get('rejection_reasons'):
        print(f"   Rejections: {result['rejection_reasons']}")
    
    # Scenario 2: Negative sentiment
    print("\n   Scenario 2: BUY with negative sentiment")
    result = system.process_signal(
        signal=SignalType.BUY,
        symbol="HRHO.CA",
        price=32.00,
        suggested_quantity=500,
        sentiment_score=-0.7,  # Very negative
        win_rate=0.55,
        win_loss_ratio=1.2
    )
    print(f"   Status: {result['status']}")
    print(f"   Rejections: {result.get('rejection_reasons', [])}")
    
    # Scenario 3: Excessive position size
    print("\n   Scenario 3: BUY with excessive position size")
    result = system.process_signal(
        signal=SignalType.BUY,
        symbol="ETEL.CA",
        price=25.00,
        suggested_quantity=100000,  # Too large
        sentiment_score=0.5,
        win_rate=0.55,
        win_loss_ratio=1.2
    )
    print(f"   Status: {result['status']}")
    print(f"   Adjusted quantity: {result['risk_check'].get('adjusted_quantity', 'N/A')}")
    
    # Get system status
    print("\n3. System Status:")
    status = system.get_system_status()
    print(f"   Trading Halted: {status['trading_halted']}")
    print(f"   Risk Manager Summary: {json.dumps(status['risk_manager'], indent=2)}")
    
    print("\n" + "=" * 60)
    print("Risk Management Test Complete")
    print("=" * 60)
