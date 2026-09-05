# Thndr Real-Time Signal Generator Guide

## Overview

This system integrates **Thndr's unofficial API** to provide **REAL-TIME EGX signals** while keeping you safe from account bans. The system generates trading signals that you manually execute on the Thndr mobile app.

## ⚠️ Important Warnings

1. **SIGNALS ONLY** - This system does NOT execute trades automatically
2. **Manual Execution Required** - You must open the Thndr app and execute trades yourself
3. **Unofficial API** - Uses reverse-engineered endpoints; Thndr may change them anytime
4. **Rate Limiting** - Built-in safeguards prevent bans (3-second minimum between requests)
5. **Paper Trading First** - Test extensively before using real money

## Setup Instructions

### 1. Install Dependencies

```bash
cd /workspace/egx_trading_system
pip install requests pandas numpy python-dotenv
```

### 2. Configure Your Phone Number (Optional but Recommended)

Create a `.env` file in the project directory:

```bash
echo "THNDR_PHONE=+201XXXXXXXXX" > .env
```

Replace `+201XXXXXXXXX` with your actual Thndr-registered phone number.

**Why add your phone number?**
- Authenticated users get better rate limits
- Session tokens are cached for 24 hours (no repeated SMS codes)
- More reliable data access

### 3. Run the Signal Generator

#### Basic Usage (Guest Mode - No Authentication)

```bash
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA,HRHO.CA" --interval 60 --duration 1.0
```

#### With Authentication (Recommended)

```bash
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA" --interval 30 --duration 2.0 --phone "+201234567890"
```

**First-time authentication:**
1. Run the script with your phone number
2. You'll receive an SMS code from Thndr
3. Enter the 4-digit code when prompted
4. Token is cached for 24 hours (no need to re-enter)

## Command-Line Options

| Option | Description | Default | Min Value |
|--------|-------------|---------|-----------|
| `--symbols` | Comma-separated EGX symbols | COMI.CA,ETEL.CA,HRHO.CA | - |
| `--interval` | Update frequency in seconds | 60 | 3 (to avoid bans) |
| `--duration` | How long to run in hours | 1.0 | 0 (infinite) |
| `--phone` | Thndr phone number | None | - |

### Example Commands

**Monitor 3 stocks every 60 seconds for 1 hour:**
```bash
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA,HRHO.CA" --interval 60 --duration 1.0
```

**Monitor 5 stocks every 30 seconds for 4 hours (with auth):**
```bash
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA,HRHO.CA,FWRY.CA,SWDY.CA" --interval 30 --duration 4.0 --phone "+201234567890"
```

**Run indefinitely during market hours (stop with Ctrl+C):**
```bash
python thndr_signal_generator.py --symbols "COMI.CA" --interval 10 --duration 0 --phone "+201234567890"
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  YOUR COMPUTER                           │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────────┐    │
│  │ ThndrDataFetcher │─────>│ SignalGenerator      │    │
│  │ (Real-time data) │      │ (ML + Rules)         │    │
│  └──────────────────┘      └──────────────────────┘    │
│                            │                             │
│                            ▼                             │
│                    ┌──────────────┐                     │
│                    │ DISPLAY SIGNAL│                     │
│                    └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
           │
           │ You read signal
           ▼
┌─────────────────────────────────────────────────────────┐
│              THNDR MOBILE APP (YOUR PHONE)               │
│                                                          │
│  YOU MANUALLY EXECUTE:                                   │
│  - Open Thndr app                                        │
│  - Search stock symbol                                   │
│  - Tap BUY/SELL                                          │
│  - Enter quantity                                        │
│  - Confirm order                                         │
└─────────────────────────────────────────────────────────┘
```

### Safety Features

1. **Rate Limiting**: Minimum 3 seconds between API requests
2. **Token Caching**: Authentication tokens saved for 24 hours
3. **Market Hours Check**: Only fetches data when EGX is open
4. **Mock Data Fallback**: If API fails, shows mock data (clearly labeled)
5. **No Auto-Execution**: Impossible to accidentally trade

## Signal Interpretation

The system generates three types of signals:

### BUY Signal
```
SYMBOL: COMI.CA
PRICE: 85.50 EGP (+2.50%)
SIGNAL: BUY (confidence: 75%)
REASON: Strong momentum: +2.50%; Oversold: 6.2% below 30-day average
ACTION: Open Thndr app and BUY COMI.CA
```

**What to do:**
1. Open Thndr app immediately
2. Search for "COMI" or "Commercial International Bank"
3. Tap "BUY"
4. Enter desired quantity (use your own position sizing rules)
5. Review current price matches signal price
6. Confirm order

### SELL Signal
```
SYMBOL: ETEL.CA
PRICE: 42.30 EGP (-3.10%)
SIGNAL: SELL (confidence: 62%)
REASON: Strong downtrend: -3.10%; Overbought: 7.5% above average
ACTION: Open Thndr app and SELL ETEL.CA
```

**What to do:**
1. Open Thndr app
2. Find your existing ETEL position
3. Tap "SELL"
4. Enter quantity to sell
5. Confirm order

### HOLD Signal
```
SYMBOL: HRHO.CA
PRICE: 18.90 EGP (+0.30%)
SIGNAL: HOLD (confidence: 0%)
REASON: Price change: +0.30%
ACTION: Open Thndr app and MONITOR HRHO.CA
```

**What to do:**
- No action required
- Continue monitoring
- Wait for stronger signal

## Testing Strategy

### Phase 1: Paper Trading (Week 1-2)

1. **Run without phone number** (guest mode):
   ```bash
   python thndr_signal_generator.py --symbols "COMI.CA" --interval 60 --duration 0.5
   ```

2. **Log all signals** in a spreadsheet:
   - Timestamp
   - Symbol
   - Signal (BUY/SELL/HOLD)
   - Confidence
   - Price at signal
   - What you would have done

3. **Do NOT execute any real trades**
4. **Track hypothetical performance**

### Phase 2: Small Position Testing (Week 3-4)

1. **Add your phone number** for authenticated access:
   ```bash
   echo "THNDR_PHONE=+201XXXXXXXXX" >> .env
   ```

2. **Execute smallest possible positions** (10-100 shares)
3. **Compare signal price vs execution price** (slippage)
4. **Track actual P&L** vs hypothetical

### Phase 3: Normal Trading (After 1 Month)

1. **Increase position sizes gradually**
2. **Integrate with your risk management rules**
3. **Monitor for API changes** (Thndr may update their app)

## Troubleshooting

### Problem: "Failed to request OTP"

**Cause:** Phone number format incorrect or Thndr server issue

**Solution:**
- Ensure format: `+201XXXXXXXXX` (include country code)
- Wait 5 minutes and retry
- Check if Thndr app works normally on your phone

### Problem: "Token expired"

**Cause:** Cached token older than 24 hours

**Solution:**
- Delete cache: `rm -rf thndr_cache/`
- Re-run with phone number to get new token
- Enter SMS code when prompted

### Problem: "MOCK_DATA" status

**Cause:** API returned error, showing fake data instead

**Solution:**
- Check internet connection
- Verify Thndr service is up (try opening app)
- Reduce request frequency (increase `--interval`)
- Accept that real-time data unavailable, use for testing only

### Problem: Signals seem delayed

**Cause:** Rate limiting or network latency

**Solution:**
- Increase `--interval` to reduce API load
- Use authenticated mode (better priority)
- Check if market is actually open (EGX: Sun-Thu, 10AM-2:30PM)

## EGX Market Hours

The system automatically checks market status:

- **Sunday - Thursday**: 10:00 AM - 2:30 PM EET (Open)
- **Friday - Saturday**: Closed (Weekend)
- **Egyptian Holidays**: Closed

If market is closed, the system will wait and show status messages.

## Advanced: Integrating with Your ML Model

The `SignalGenerator` class uses simple momentum rules by default. To integrate your XGBoost model:

```python
from thndr_signal_generator import ThndrDataFetcher, SignalGenerator
from model import EGXTradingModel  # Your existing model

# Initialize
fetcher = ThndrDataFetcher(phone_number="+201234567890")
model = EGXTradingModel()
model.load_model("path/to/your/model.pkl")

# Get live price
live_data = fetcher.get_live_price("COMI.CA")

# Get historical data for features
from data_ingestion import EGXDataFetcher
data_fetcher = EGXDataFetcher()
hist_df = data_fetcher.fetch_historical_data("COMI.CA", period="6mo")

# Generate features and predict
features = engineer_features(hist_df)  # Your feature engineering
prediction = model.predict(features)

# Generate signal based on prediction
if prediction > 0.7:
    signal = "BUY"
elif prediction < 0.3:
    signal = "SELL"
else:
    signal = "HOLD"

print(f"ML Signal: {signal} (confidence: {prediction})")
# Manually execute on Thndr app
```

## Legal & Compliance Disclaimer

⚠️ **IMPORTANT:**

1. This tool uses **unofficial API endpoints** reverse-engineered from the Thndr mobile app
2. Thndr's Terms of Service may prohibit automated access
3. You are responsible for compliance with:
   - Thndr's Terms of Service
   - Egyptian Financial Regulatory Authority (FRA) rules
   - EGX trading regulations
4. **DO NOT** use for high-frequency trading or large volumes
5. Consider contacting Thndr for official API access if you plan serious algorithmic trading

## Alternative: Official Broker APIs

For production trading, consider these official options:

1. **EFG Hermes** - Institutional FIX API (requires $50k+ capital)
2. **CI Capital** - May provide API access to active traders
3. **Mubasher Trade** - Most tech-savvy retail broker, ask about API
4. **Interactive Brokers** - If you have international access

## Support & Updates

- **Logs**: Check `logs/thndr_data_YYYYMMDD.log` for detailed logs
- **Cache**: Session tokens stored in `thndr_cache/`
- **Updates**: Thndr may change API; monitor for breaking changes

## Quick Start Summary

```bash
# 1. Navigate to project
cd /workspace/egx_trading_system

# 2. Add your phone number (optional but recommended)
echo "THNDR_PHONE=+201234567890" > .env

# 3. Run signal generator
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA" --interval 60 --duration 1.0

# 4. Watch signals appear
# 5. Open Thndr app when you see BUY/SELL signal
# 6. Manually execute the trade
# 7. Monitor your positions
```

---

**Remember: This is a SIGNAL GENERATOR only. You are responsible for all trading decisions and executions.**
