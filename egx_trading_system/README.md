# EGX Algorithmic Trading System

A comprehensive, modular, production-ready Python framework for multimodal algorithmic trading targeting the Egyptian Exchange (EGX).

## ⚠️ IMPORTANT WARNING

**THIS IS A RESEARCH FRAMEWORK ONLY - NOT FOR LIVE TRADING**

This system is designed for educational and research purposes. DO NOT use for live trading without:

- Extensive paper-trading (minimum 3-6 months)
- Thorough backtesting across multiple market conditions
- Financial due diligence and regulatory compliance review
- Integration with licensed Egyptian broker APIs (EFG Hermes, CI Capital, etc.)
- Proper risk management infrastructure

Emerging markets like the Egyptian Exchange (EGX) have unique risks:
- Lower liquidity and higher slippage
- Circuit breakers and trading halts
- Currency risk (EGP volatility)
- Political and economic instability
- Limited data availability and quality issues

## Features

### Multimodal Data Ingestion
- **OHLCV Data**: Historical price data from Yahoo Finance (placeholder for EGX API)
- **Arabic News Sentiment**: AraBERT-based sentiment analysis for Arabic financial news
- **Alternative Data**: Whisper transcription for video/audio content (earnings calls, CNBC Arabia)

### Feature Engineering
- Technical Indicators: RSI, MACD, Bollinger Bands, ATR, ADX
- Volume Analysis: OBV, VWAP, Volume spikes
- Statistical Features: Volatility, skewness, kurtosis, rolling Sharpe
- Lagged features for time-series modeling

### Machine Learning
- XGBoost classifier for predicting positive returns
- Strict chronological train/test split (NO shuffling to prevent data leakage)
- Time-series cross-validation
- Feature importance analysis

### Backtesting
- Realistic EGX friction: 0.20% commission, 0.1% slippage, 0.15% stamp duty
- Trading hours validation (10:00 AM - 2:30 PM EET, Sunday-Thursday)
- Comprehensive metrics: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor

### Risk Management
- Position sizing limits (max 5% per position)
- Drawdown circuit breaker (halt at 15% drawdown)
- Sentiment-based filtering
- Kelly Criterion optimization
- Concentration limits

## Project Structure

```
egx_trading_system/
├── config.py              # Configuration with pydantic validation
├── logger.py              # Structured logging system
├── data_ingestion.py      # Data fetchers (OHLCV, news, alternative)
├── feature_engineering.py # Technical indicators and features
├── model.py               # XGBoost ML model
├── backtest.py            # Backtesting engine
├── risk_and_execution.py  # Risk management and execution placeholder
├── main.py                # Main orchestration script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Basic Installation (Core Functionality)

```bash
cd egx_trading_system
pip install -r requirements.txt
```

This installs core dependencies for basic functionality.

### Full Installation (All Features)

```bash
pip install pandas numpy pydantic python-dotenv yfinance xgboost scikit-learn
pip install transformers torch  # For Arabic NLP
pip install openai-whisper  # For audio/video transcription
pip install vectorbt  # For advanced backtesting
```

## Quick Start

### Run Full Pipeline

```bash
python main.py --symbols COMI.CA ETEL.CA HRHO.CA --start-date 2022-01-01 --end-date 2024-01-01
```

### Individual Module Testing

```bash
# Test data ingestion
python data_ingestion.py

# Test feature engineering
python feature_engineering.py

# Test model training
python model.py

# Test backtesting
python backtest.py

# Test risk management
python risk_and_execution.py
```

## Configuration

Edit `config.py` or create a `.env` file for API keys:

```env
NEWS_API_KEY=your_arabic_news_api_key
BROKER_API_KEY=your_broker_api_key
BROKER_SECRET=your_broker_secret
```

### Key Configuration Parameters

```python
# EGX Trading Parameters
market_open = "10:00"  # EET
market_close = "14:30"  # EET
commission_rate = 0.0020  # 0.20%
slippage_rate = 0.001  # 0.1%
stamp_duty = 0.0015  # 0.15% on sells

# Risk Limits
max_position_size_pct = 0.05  # 5% max per position
max_portfolio_drawdown = 0.15  # 15% max drawdown
sentiment_sell_threshold = -0.5  # Reject trades below this
```

## Usage Examples

### Basic Pipeline

```python
from main import run_full_pipeline

results = run_full_pipeline(
    symbols=["COMI.CA", "ETEL.CA"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=1000000,
    paper_trading=True
)
```

### Custom Model Training

```python
from data_ingestion import EGXDataFetcher
from feature_engineering import FeatureEngineer
from model import QuantModel

# Fetch data
fetcher = EGXDataFetcher()
df = fetcher.fetch_historical_data("COMI.CA", period="2y")

# Create features
fe = FeatureEngineer()
df_features = fe.create_all_features(df)
df_clean = fe.handle_missing_values(df_features)

# Train model
model = QuantModel()
metrics = model.fit(df=df_clean, feature_columns=fe.feature_columns)

# Generate signals
signals_df = model.generate_signals(df_clean, fe.feature_columns)
```

### Risk Management Check

```python
from risk_and_execution import TradingSystem, SignalType

system = TradingSystem(portfolio_value=1000000, paper_trading=True)

result = system.process_signal(
    signal=SignalType.BUY,
    symbol="COMI.CA",
    price=45.50,
    suggested_quantity=1000,
    sentiment_score=0.65,
    win_rate=0.55,
    win_loss_ratio=1.2
)

if result['status'] == 'REJECTED':
    print(f"Trade rejected: {result['rejection_reasons']}")
```

## EGX-Specific Considerations

### Trading Hours
- **Market Open**: 10:00 AM EET
- **Market Close**: 2:30 PM EET
- **Trading Days**: Sunday through Thursday
- **Weekend**: Friday and Saturday

### Transaction Costs
- **Commission**: 0.15% - 0.25% per trade (varies by broker)
- **Slippage**: ~0.1% (higher for illiquid stocks)
- **Stamp Duty**: 0.15% on sell orders (EGX requirement)

### Settlement
- **Settlement Cycle**: T+2 (trade date plus 2 business days)
- **Currency**: Egyptian Pound (EGP)

### Liquidity Considerations
Many EGX stocks have lower liquidity compared to developed markets:
- Use limit orders instead of market orders in production
- Implement volume-based position sizing
- Monitor bid-ask spreads carefully

## Production Deployment Checklist

Before considering any form of live deployment:

- [ ] Replace mock data fetchers with real EGX data provider API
- [ ] Fine-tune AraBERT on Egyptian financial news corpus
- [ ] Integrate with licensed broker API (EFG Hermes, CI Capital)
- [ ] Complete 3-6 months of paper trading
- [ ] Conduct out-of-sample testing
- [ ] Implement proper error handling and monitoring
- [ ] Set up alerting systems
- [ ] Obtain regulatory approvals
- [ ] Establish compliance procedures
- [ ] Create disaster recovery plan

## Disclaimer

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors are not responsible for any financial losses, damages, or liabilities arising from the use of this software.

Trading in financial markets involves substantial risk of loss. Past performance does not guarantee future results. Always consult with qualified financial advisors and regulatory experts before engaging in algorithmic trading.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP-8 style guidelines
- New features include tests
- Documentation is updated
- No "holy grail" overfitting strategies

## Contact

For questions or issues, please open a GitHub issue.

---

**Remember**: Algorithmic trading is hard. Emerging markets are harder. Never risk more than you can afford to lose.
