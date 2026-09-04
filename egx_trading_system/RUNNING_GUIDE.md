# EGX Real-Time Trading System - Testing Guide

## Overview

This guide explains how to run and test the real-time algorithmic trading system for the Egyptian Exchange (EGX).

## ⚠️ IMPORTANT DISCLAIMER

**THIS SYSTEM IS FOR PAPER TRADING AND EDUCATIONAL PURPOSES ONLY**

- Do NOT use for live trading without extensive testing
- Past performance does not guarantee future results
- Emerging markets like EGX have unique risks
- You are solely responsible for your trading decisions

---

## Prerequisites

### 1. Install Dependencies

```bash
cd /workspace/egx_trading_system
pip install -r requirements.txt
```

For basic testing, minimum requirements:
```bash
pip install pandas numpy pydantic yfinance xgboost scikit-learn
```

### 2. Verify Installation

```bash
python -c "import pandas; import xgboost; import sklearn; print('✓ All dependencies installed')"
```

---

## How to Run the Real-Time Trading System

### Quick Start (Paper Trading)

Run a **5-minute test** with default symbols:

```bash
cd /workspace/egx_trading_system
python realtime_trading.py --duration 0.083
```

Note: 0.083 hours = 5 minutes (for quick testing)

### Full Command Options

```bash
python realtime_trading.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA" \
  --capital 1000000 \
  --paper \
  --interval 60 \
  --duration 1.0 \
  --lookback 252
```

#### Parameters Explained:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--symbols` | COMI.CA,ETEL.CA,HRHO.CA | Comma-separated EGX symbols |
| `--capital` | 1,000,000 | Initial capital in EGP |
| `--paper` | True | Paper trading mode (safe) |
| `--live` | False | Live trading (requires broker API) |
| `--interval` | 60 | Update interval in seconds |
| `--duration` | None | Run duration in hours (None=indefinite) |
| `--lookback` | 252 | Days of historical data for training |

---

## Testing Strategy

### Phase 1: Basic Functionality Test (5-10 minutes)

**Objective:** Verify the system runs without errors

```bash
# Run for 5 minutes with 2 symbols
python realtime_trading.py \
  --symbols "COMI.CA,ETEL.CA" \
  --duration 0.083 \
  --interval 30
```

**What to check:**
- ✓ Models initialize successfully
- ✓ Data is fetched (or mock data generated)
- ✓ Signals are generated
- ✓ No critical errors in logs
- ✓ Snapshots are saved to `./realtime_data/`

### Phase 2: Extended Paper Trading (1-4 hours)

**Objective:** Observe signal generation and simulated trades

```bash
# Run for 2 hours
python realtime_trading.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA,SWDY.CA" \
  --duration 2.0 \
  --interval 60 \
  --capital 1000000
```

**What to monitor:**
- Signal frequency and types (BUY/SELL/HOLD)
- Trade execution logic
- Portfolio value changes
- Risk management rejections

### Phase 3: Full Day Simulation (Market Hours)

**Objective:** Test across different market conditions

```bash
# Run for full EGX trading session (4.5 hours)
# EGX: Sunday-Thursday, 10:00-14:30 Cairo Time
python realtime_trading.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA,SWDY.CA,AAIC.CA" \
  --duration 4.5 \
  --interval 120 \
  --capital 5000000
```

**What to analyze:**
- Performance during market open/close
- Response to price movements
- Drawdown patterns
- Win rate and Sharpe ratio

---

## How to Evaluate Performance

### 1. Check Log Output

The system prints real-time logs showing:
```
📈 BUY EXECUTED: 1500 COMI.CA @ 85.50 EGP | Cost: 128,250.00 EGP
📉 SELL EXECUTED: 1500 COMI.CA @ 87.20 EGP | Revenue: 130,800.00 EGP

Portfolio Value: 1,025,450.00 EGP
Total Return: 2.55%
Signals Generated: 5
Trades Executed: 2
```

### 2. Review Snapshot Files

Snapshots are saved to `./realtime_data/`:

```bash
ls -la ./realtime_data/
cat ./realtime_data/snapshot_20241215_143022.json
```

Each snapshot contains:
- Current portfolio value
- Open positions
- Recent trades
- Latest signals

### 3. Calculate Key Metrics

After running, calculate:

**Return Metrics:**
- Total Return = (Final Value - Initial Capital) / Initial Capital
- Annualized Return (if running extended period)

**Risk Metrics:**
- Maximum Drawdown
- Sharpe Ratio (risk-adjusted return)
- Win Rate = Winning Trades / Total Trades

**Trading Metrics:**
- Number of trades
- Average trade size
- Holding period

---

## Alternative Testing Methods

### Method 1: Run Backtest First (Recommended)

Before real-time testing, run historical backtest:

```bash
python main.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA" \
  --start-date "2023-01-01" \
  --end-date "2024-12-01" \
  --initial-capital 1000000
```

This validates the strategy on historical data first.

### Method 2: Interactive Python Session

For detailed inspection:

```python
from realtime_trading import RealTimeTrader

# Initialize trader
trader = RealTimeTrader(
    symbols=["COMI.CA", "ETEL.CA"],
    initial_capital=1000000,
    paper_trading=True,
    update_interval=60
)

# Initialize models
trader.initialize_models(lookback_days=252)

# Run single iteration
results = trader.run_iteration()
print(results)

# Check portfolio
portfolio = trader.get_portfolio_summary()
print(f"Portfolio Value: {portfolio['total_value']:,.2f} EGP")
print(f"Positions: {portfolio['positions']}")
```

### Method 3: Jupyter Notebook Analysis

Create a notebook for interactive analysis:

```python
import pandas as pd
import json
from pathlib import Path

# Load snapshots
snapshot_files = list(Path('./realtime_data').glob('snapshot_*.json'))
snapshots = [json.load(open(f)) for f in sorted(snapshot_files)]

# Analyze performance
portfolio_values = [s['portfolio']['total_value'] for s in snapshots]
returns = [(v - portfolio_values[0]) / portfolio_values[0] for v in portfolio_values]

# Plot
import matplotlib.pyplot as plt
plt.plot(returns)
plt.xlabel('Time')
plt.ylabel('Return')
plt.title('Real-Time Trading Performance')
plt.show()
```

---

## Troubleshooting

### Issue: "No data found for symbol"

**Solution:** The system will generate mock data for testing. For real data:
- Ensure internet connection
- Check if symbol exists on Yahoo Finance
- Consider using a local EGX data provider

### Issue: "Insufficient data for model training"

**Solution:** Increase lookback period:
```bash
python realtime_trading.py --lookback 500
```

### Issue: "Market is closed"

**Solution:** EGX is open Sunday-Thursday, 10:00-14:30 Cairo Time. The system will still run but won't execute trades when closed.

### Issue: High memory usage

**Solution:** Reduce number of symbols or lookback period:
```bash
python realtime_trading.py --symbols "COMI.CA,ETEL.CA" --lookback 120
```

---

## Expected Output Example

```
======================================================================
STARTING REAL-TIME TRADING
======================================================================
Symbols: ['COMI.CA', 'ETEL.CA', 'HRHO.CA']
Initial Capital: 1,000,000 EGP
Update Interval: 60 seconds
Duration: 0.5 hours
======================================================================

======================================================================
Iteration 1 - 2024-12-15 11:30:00
Signal for COMI.CA: BUY | Price: 85.50 | Probability: 62.3%
📈 BUY EXECUTED: 1500 COMI.CA @ 85.50 EGP | Cost: 128,250.00 EGP

Portfolio Value: 998,450.00 EGP
Total Return: -0.15%
Signals Generated: 3
Trades Executed: 1

Waiting 60 seconds...

======================================================================
Iteration 2 - 2024-12-15 11:31:00
Signal for ETEL.CA: HOLD | Price: 42.30 | Probability: 51.2%

Portfolio Value: 998,450.00 EGP
Total Return: -0.15%
Signals Generated: 3
Trades Executed: 0

Waiting 60 seconds...
```

---

## Next Steps After Testing

1. **Extended Paper Trading:** Run for 3-6 months minimum
2. **Performance Analysis:** Compare against benchmark (EGX30 index)
3. **Risk Assessment:** Evaluate maximum drawdown tolerance
4. **Broker Integration:** Connect to licensed Egyptian broker API
5. **Compliance Review:** Ensure regulatory compliance
6. **Gradual Live Deployment:** Start with small capital

---

## Additional Resources

- **Backtesting:** Use `main.py` for historical backtesting
- **Feature Engineering:** See `feature_engineering.py`
- **Risk Management:** See `risk_and_execution.py`
- **Model Details:** See `model.py`

## Support

For issues or questions:
1. Check log files in `./logs/`
2. Review snapshot data in `./realtime_data/`
3. Validate configuration in `config.py`

---

**Remember:** Algorithmic trading involves substantial risk. Always start with paper trading and never risk more than you can afford to lose.
