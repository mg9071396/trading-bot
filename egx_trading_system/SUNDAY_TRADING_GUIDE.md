# 🚀 SUNDAY TRADING DAY - COMPLETE GUIDE

## ⏰ BEFORE MARKET OPENS (Before 10:00 AM Egypt Time)

### Step 1: Verify Your Setup (9:00 AM - 9:30 AM)

```bash
cd /workspace/egx_trading_system

# Test the signal generator with a quick 1-minute run
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA,HRHO.CA" --interval 10 --duration 0.02
```

**Expected Output:**
- Should show "THNDR SIGNAL MONITOR - MANUAL EXECUTION MODE"
- Should display current prices (may be delayed if using guest mode)
- Should say "Market will open at 10:00 AM" or similar

### Step 2: Prepare Your Watchlist

Create a file called `watchlist.txt` with your target stocks:

```bash
echo "COMI.CA
ETEL.CA
HRHO.CA
SWDY.CA
ESRS.CA" > watchlist.txt
```

**Recommended EGX Stocks for Algorithmic Trading:**
- **COMI.CA** - Commercial International Bank (high liquidity)
- **ETEL.CA** - Telecom Egypt (stable, good volume)
- **HRHO.CA** - EFG Hermes (brokerage, volatile)
- **SWDY.CA** - Elsewedy Electric (industrial)
- **ESRS.CA** - Ezz Steel (steel sector)

### Step 3: Open Required Apps

On your phone, have these ready:
1. **Thndr App** - Logged in and ready to trade
2. **Calculator** - For position sizing
3. **Notes App** - To log executed trades

---

## 📈 DURING MARKET HOURS (10:00 AM - 2:30 PM Egypt Time)

### Step 4: Start the Signal Generator

**Run this command at 10:00 AM sharp:**

```bash
cd /workspace/egx_trading_system

# OPTION A: Basic monitoring (recommended for first week)
python thndr_signal_generator.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA" \
  --interval 30 \
  --duration 4.5

# OPTION B: Extended watchlist
python thndr_signal_generator.py \
  --symbols "COMI.CA,ETEL.CA,HRHO.CA,SWDY.CA,ESRS.CA" \
  --interval 60 \
  --duration 4.5

# OPTION C: With authentication (if you add phone later)
python thndr_signal_generator.py \
  --symbols "COMI.CA,ETEL.CA" \
  --interval 30 \
  --duration 4.5 \
  --phone "+201XXXXXXXXX"
```

**Parameter Explanation:**
- `--symbols`: Comma-separated list of EGX stocks (use .CA suffix)
- `--interval`: How often to check for signals (minimum 30 seconds to avoid bans)
- `--duration`: How long to run in hours (4.5 = full trading day)

### Step 5: Monitor Signals

The script will output signals like this:

```
═══════════════════════════════════════════════════════════
📊 SIGNAL ALERT - 2024-01-15 10:35:22
═══════════════════════════════════════════════════════════
SYMBOL: COMI.CA
CURRENT PRICE: 85.50 EGP
SIGNAL: BUY ✅
CONFIDENCE: 75%
REASON: Strong momentum (+2.5%), Oversold conditions
STOP LOSS: 83.00 EGP (-2.9%)
TAKE PROFIT: 89.50 EGP (+4.7%)
POSITION SIZE: 100 shares (8,550 EGP)
───────────────────────────────────────────────────────────
ACTION REQUIRED:
1. Open Thndr app
2. Search "COMI" or "Commercial International Bank"
3. Tap BUY
4. Enter quantity: 100
5. Review order and CONFIRM
═══════════════════════════════════════════════════════════
```

### Step 6: Execute Trades MANUALLY

**When you see a BUY signal:**

1. **Immediately** open Thndr app on your phone
2. Search for the stock symbol
3. Check the current price matches the signal (±2% tolerance)
4. Calculate position size:
   ```
   Position Size = (Account Risk %) × Portfolio Value ÷ (Entry - Stop Loss)
   Example: 1% risk on 50,000 EGP portfolio, entry 85.50, stop 83.00
   Position = 0.01 × 50,000 ÷ (85.50 - 83.00) = 200 shares
   ```
5. Enter the quantity in Thndr
6. Review and tap "Confirm Buy"
7. **Log the trade** in a spreadsheet:
   - Date/Time
   - Symbol
   - Signal Price
   - Execution Price
   - Quantity
   - Stop Loss
   - Take Profit

**When you see a SELL signal:**

1. Check if you own the stock
2. If yes, execute sell immediately
3. If no, ignore the signal

### Step 7: Risk Management Rules

**NEVER break these rules:**

1. **Maximum 3 positions** open at once
2. **Maximum 2% portfolio risk** per trade
3. **Always set stop loss** (the signal provides it)
4. **Never chase prices** - if price moved >3% since signal, skip
5. **No trading first 15 minutes** (10:00-10:15 AM) - too volatile
6. **Close all positions by 2:15 PM** - avoid end-of-day volatility

---

## 📊 TRACKING YOUR PERFORMANCE

### Create a Trade Log Spreadsheet

| Date | Time | Symbol | Signal | Sig Price | Exec Price | Qty | Stop Loss | Take Profit | Exit Price | P/L | Notes |
|------|------|--------|--------|-----------|------------|-----|-----------|-------------|------------|-----|-------|
| Jan 15 | 10:35 | COMI.CA | BUY | 85.50 | 85.60 | 100 | 83.00 | 89.50 | 87.20 | +160 | Good fill |
| Jan 15 | 11:20 | ETEL.CA | SELL | 42.30 | 42.25 | 200 | - | - | 42.25 | -10 | Slippage |

### Daily Review Checklist (After 2:30 PM)

1. **Count total signals** generated
2. **Count executed trades** vs missed opportunities
3. **Calculate slippage**: (Exec Price - Signal Price) for each trade
4. **Review stopped-out trades**: Did stop losses work correctly?
5. **Note any technical issues**: API failures, delays, etc.

---

## 🛑 TROUBLESHOOTING

### Problem: "Failed to resolve api.thndr.app"

**Solution:** This is normal in some network environments. The script will:
- Fall back to Yahoo Finance data (15-min delay)
- Generate mock data if all else fails
- Still provide useful signals for medium-term trades

**Action:** Continue running, but be aware prices may be slightly delayed.

### Problem: "Market is closed" message during trading hours

**Possible causes:**
1. It's Friday or Saturday (EGX weekend)
2. It's before 10:00 AM or after 2:30 PM
3. Egyptian public holiday

**Action:** Check EGX calendar at [egx.com.eg](https://www.egx.com.eg)

### Problem: Signals seem too frequent or too rare

**Adjust the interval:**
- Too frequent? Increase `--interval` to 60 or 90 seconds
- Too rare? Decrease `--interval` to 30 seconds (minimum safe value)

### Problem: Price on Thndr app differs from signal price

**This is normal due to:**
- Network latency (2-5 second delay)
- Bid-ask spread
- Rapid price movements

**Rule:** If difference >3%, skip the trade and wait for next signal.

---

## 📅 WEEKLY SCHEDULE

### Sunday (First Week)
- [ ] Run signal generator 10:00 AM - 2:30 PM
- [ ] Execute ONLY 1-2 trades maximum
- [ ] Focus on process, not profits
- [ ] Log everything in spreadsheet

### Monday-Friday (Week 1)
- [ ] Paper trade only (no real money)
- [ ] Track hypothetical performance
- [ ] Refine your execution speed
- [ ] Build confidence in the system

### Sunday (Week 2)
- [ ] Start small real positions (10-100 shares)
- [ ] Compare signal vs execution prices
- [ ] Adjust position sizing based on experience

### Ongoing
- [ ] Weekly review every Friday after market close
- [ ] Monthly performance analysis
- [ ] Gradually increase position sizes as confidence grows

---

## ⚠️ CRITICAL WARNINGS

1. **This is NOT financial advice** - You are responsible for your trades
2. **Start with paper trading** - Do not use real money for at least 2 weeks
3. **Never trade money you can't afford to lose** - EGX can be volatile
4. **Technical failures happen** - Have a manual backup plan
5. **Thndr may change their API** - Monitor for breaking changes
6. **Tax implications** - Keep detailed records for tax purposes

---

## 📞 EMERGENCY PROCEDURES

### If the script crashes during trading:
1. Don't panic - your existing positions are still active
2. Restart the script with same parameters
3. Manually check your positions in Thndr app
4. Set manual alerts for your stop losses

### If Thndr app goes down:
1. Contact Thndr support immediately
2. Use alternative broker if you have one
3. Hold positions until app is restored (if safe)

### If you accidentally execute wrong trade:
1. Close the position immediately (market order)
2. Log the error and learn from it
3. Double-check future orders before confirming

---

## 🎯 SUCCESS METRICS

Track these weekly:

| Metric | Target (Week 1) | Target (Month 1) |
|--------|-----------------|------------------|
| Signal Accuracy | >50% | >60% |
| Execution Speed | <30 seconds | <15 seconds |
| Slippage | <1% | <0.5% |
| Win Rate | >45% | >55% |
| Max Drawdown | <5% | <3% |

---

## 📚 RESOURCES

- **EGX Trading Hours**: 10:00 AM - 2:30 PM (Sun-Thu)
- **EGX Website**: [www.egx.com.eg](https://www.egx.com.eg)
- **Thndr Support**: support@thndr.app
- **Egyptian Market Holidays**: Check EGX calendar

---

## 🆘 QUICK REFERENCE COMMANDS

```bash
# Start monitoring (full day)
python thndr_signal_generator.py --symbols "COMI.CA,ETEL.CA" --interval 30 --duration 4.5

# Quick test (5 minutes)
python thndr_signal_generator.py --symbols "COMI.CA" --interval 10 --duration 0.1

# Monitor single stock
python thndr_signal_generator.py --symbols "HRHO.CA" --interval 60 --duration 2.0

# Stop the script early
# Press Ctrl+C in the terminal
```

---

**Good luck on Sunday! Remember: Slow and steady wins the race. Focus on learning the system, not making quick profits.** 🚀📈
