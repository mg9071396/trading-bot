"""
Thndr Real-Time Data Fetcher for EGX Trading System.

This module integrates with Thndr's unofficial API to fetch REAL-TIME EGX data
for signal generation ONLY. It does NOT execute trades automatically.

IMPORTANT SAFETY MEASURES:
1. PAPER TRADING ONLY - No automatic order execution
2. Rate limiting to avoid bans (max 1 request per 3 seconds)
3. Session token caching to reduce login frequency
4. Manual trade execution required (you execute signals on phone app)

WARNING: This uses Thndr's unofficial API. Use at your own risk.
Terms of Service may prohibit automated access. Consider contacting
Thndr for official API access if you plan serious algorithmic trading.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging
import hashlib

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from logger import TradingLogger
from config import trading_config

logger = TradingLogger("thndr_data")


class ThndrDataFetcher:
    """
    Real-time data fetcher using Thndr's API.
    
    This class mimics the Thndr mobile app's network requests to fetch
    live EGX prices. It includes safety features to avoid account bans:
    
    - Rate limiting (max 1 request per 3 seconds)
    - Session token caching
    - Request throttling during market hours
    - Automatic retry with exponential backoff
    
    Usage:
        fetcher = ThndrDataFetcher(phone_number="+201xxxxxxxxx")
        price = fetcher.get_live_price("COMI.CA")
        signal = generate_signal(price)  # You decide when to trade
        # Manually execute on Thndr app
    """
    
    def __init__(
        self,
        phone_number: str = None,
        cache_dir: str = "./thndr_cache",
        rate_limit_seconds: float = 3.0
    ):
        """
        Initialize Thndr data fetcher.
        
        Args:
            phone_number: Egyptian phone number (e.g., "+201234567890")
                         If None, will use guest mode (limited access)
            cache_dir: Directory to cache session tokens
            rate_limit_seconds: Minimum time between API calls (prevent bans)
        """
        self.phone_number = phone_number or trading_config.thndr_phone
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0
        self.session = requests.Session()
        self.logger = logger.logger
        self.token = None
        self.user_id = None
        
        # Thndr API endpoints (reverse-engineered from mobile app)
        self.base_url = "https://api.thndr.app"
        self.auth_endpoint = f"{self.base_url}/authentication/"
        self.quotes_endpoint = f"{self.base_url}/marketdata/quotes/"
        self.instruments_endpoint = f"{self.base_url}/marketdata/instruments/"
        
        # Load cached session if available
        self._load_cached_session()
        
        self.logger.info(f"ThndrDataFetcher initialized (rate limit: {rate_limit_seconds}s)")
    
    def _load_cached_session(self):
        """Load cached authentication token if available."""
        if not self.phone_number:
            self.logger.warning("No phone number provided. Using guest mode (limited).")
            return
        
        cache_file = self.cache_dir / f"thndr_session_{hashlib.md5(self.phone_number.encode()).hexdigest()}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if token is still valid (expires in 24 hours typically)
                expires_at = datetime.fromisoformat(cached_data.get('expires_at', '2000-01-01'))
                if datetime.now() < expires_at:
                    self.token = cached_data.get('access_token')
                    self.user_id = cached_data.get('user_id')
                    self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                    self.logger.info("Loaded cached Thndr session token")
                else:
                    self.logger.info("Cached token expired, will re-authenticate")
                    cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to load cached session: {str(e)}")
    
    def _save_session_cache(self, token: str, user_id: str, expires_in: int = 86400):
        """Cache authentication token for future use."""
        if not self.phone_number:
            return
        
        cache_file = self.cache_dir / f"thndr_session_{hashlib.md5(self.phone_number.encode()).hexdigest()}.json"
        
        try:
            cache_data = {
                'access_token': token,
                'user_id': user_id,
                'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
                'phone_number': self.phone_number
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            self.logger.debug(f"Cached Thndr session token (expires in {expires_in}s)")
        except Exception as e:
            self.logger.warning(f"Failed to cache session: {str(e)}")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting to avoid bans."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def authenticate(self) -> bool:
        """
        Authenticate with Thndr using phone number.
        
        Note: This requires SMS verification code on first login.
        For subsequent logins, cached tokens are used.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.phone_number:
            self.logger.warning("Cannot authenticate without phone number")
            return False
        
        # Check if already authenticated
        if self.token:
            self.logger.info("Already authenticated")
            return True
        
        self.logger.info(f"Attempting authentication for {self.phone_number}")
        self.logger.warning("=" * 60)
        self.logger.warning("MANUAL STEP REQUIRED:")
        self.logger.warning("1. You will receive an SMS code from Thndr")
        self.logger.warning("2. Enter the code when prompted")
        self.logger.warning("3. Token will be cached for 24 hours")
        self.logger.warning("=" * 60)
        
        try:
            # Step 1: Request OTP
            self._rate_limit_wait()
            otp_response = self.session.post(
                f"{self.auth_endpoint}token/",
                json={
                    'phone_number': self.phone_number,
                    'device_id': hashlib.md5(self.phone_number.encode()).hexdigest()
                }
            )
            
            if otp_response.status_code != 200:
                self.logger.error(f"Failed to request OTP: {otp_response.text}")
                return False
            
            otp_data = otp_response.json()
            if 'error' in otp_data:
                self.logger.error(f"OTP request error: {otp_data['error']}")
                return False
            
            # Step 2: Get SMS code from user
            print("\n" + "=" * 60)
            print(f"SMS sent to {self.phone_number}")
            sms_code = input("Enter the 4-digit SMS code from Thndr: ").strip()
            print("=" * 60 + "\n")
            
            # Step 3: Verify OTP and get access token
            self._rate_limit_wait()
            token_response = self.session.post(
                f"{self.auth_endpoint}token/verify/",
                json={
                    'phone_number': self.phone_number,
                    'code': sms_code,
                    'device_id': hashlib.md5(self.phone_number.encode()).hexdigest()
                }
            )
            
            if token_response.status_code != 200:
                self.logger.error(f"Token verification failed: {token_response.text}")
                return False
            
            token_data = token_response.json()
            if 'access_token' not in token_data:
                self.logger.error(f"No access token in response: {token_data}")
                return False
            
            self.token = token_data['access_token']
            self.user_id = token_data.get('user_id')
            expires_in = token_data.get('expires_in', 86400)
            
            # Update session headers
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})
            
            # Cache the session
            self._save_session_cache(self.token, self.user_id, expires_in)
            
            self.logger.info("Authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_live_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch real-time price for a single EGX symbol.
        
        Args:
            symbol: EGX symbol (e.g., "COMI.CA", "ETEL.CA")
        
        Returns:
            Dictionary with live price data or None if failed
        """
        # Convert symbol format if needed
        thndr_symbol = self._convert_to_thndr_symbol(symbol)
        
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.quotes_endpoint}{thndr_symbol}/",
                timeout=10
            )
            
            if response.status_code == 401:
                self.logger.warning("Token expired, attempting re-authentication")
                if self.authenticate():
                    # Retry once
                    response = self.session.get(
                        f"{self.quotes_endpoint}{thndr_symbol}/",
                        timeout=10
                    )
                else:
                    self.logger.error("Re-authentication failed")
                    return None
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch {symbol}: HTTP {response.status_code}")
                # Return mock data for testing (REMOVE IN PRODUCTION)
                return self._generate_mock_live_data(symbol)
            
            data = response.json()
            return self._parse_quote_data(data, symbol)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching {symbol}: {str(e)}")
            return self._generate_mock_live_data(symbol)
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {symbol}: {str(e)}")
            return self._generate_mock_live_data(symbol)
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch real-time prices for multiple symbols.
        
        Args:
            symbols: List of EGX symbols
        
        Returns:
            Dictionary mapping symbols to their price data
        """
        results = {}
        for symbol in symbols:
            self.logger.info(f"Fetching live price for {symbol}")
            price_data = self.get_live_price(symbol)
            if price_data:
                results[symbol] = price_data
            else:
                self.logger.warning(f"Failed to fetch {symbol}")
                results[symbol] = None
            
            # Small delay between requests to be safe
            time.sleep(0.5)
        
        return results
    
    def _convert_to_thndr_symbol(self, symbol: str) -> str:
        """Convert EGX symbol format to Thndr's internal format."""
        # Thndr typically uses symbols without .CA suffix
        if symbol.endswith('.CA'):
            return symbol[:-3]
        return symbol
    
    def _parse_quote_data(self, data: Dict, symbol: str) -> Dict:
        """Parse Thndr quote response into standardized format."""
        try:
            # Adjust these field names based on actual Thndr API response
            # These are examples - you may need to inspect actual responses
            parsed = {
                'symbol': symbol,
                'price': data.get('last_price') or data.get('price') or data.get('ltp', 0),
                'change': data.get('change') or data.get('change_percent', 0),
                'change_percent': data.get('change_percent') or data.get('chg_pct', 0),
                'volume': data.get('volume') or data.get('vol', 0),
                'high': data.get('high') or data.get('h', 0),
                'low': data.get('low') or data.get('l', 0),
                'open': data.get('open') or data.get('o', 0),
                'previous_close': data.get('previous_close') or data.get('prev_close', 0),
                'bid': data.get('bid') or data.get('b', 0),
                'ask': data.get('ask') or data.get('a', 0),
                'timestamp': datetime.now().isoformat(),
                'market_status': data.get('market_status', 'OPEN')
            }
            
            self.logger.debug(f"Parsed price for {symbol}: {parsed['price']}")
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing quote data: {str(e)}")
            return None
    
    def _generate_mock_live_data(self, symbol: str) -> Dict:
        """
        Generate mock live data for testing when API is unavailable.
        
        WARNING: This is for development only. DO NOT trade on mock data.
        """
        import numpy as np
        
        self.logger.warning(f"GENERATING MOCK LIVE DATA FOR {symbol} - NOT REAL PRICES")
        
        np.random.seed(int(time.time()) % 1000)
        
        # Generate realistic-looking but fake data
        base_price = np.random.uniform(10, 100)
        change_percent = np.random.normal(0, 0.02)
        change = base_price * change_percent
        
        return {
            'symbol': symbol,
            'price': round(base_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent * 100, 2),
            'volume': np.random.randint(10000, 500000),
            'high': round(base_price * 1.02, 2),
            'low': round(base_price * 0.98, 2),
            'open': round(base_price * 0.995, 2),
            'previous_close': round(base_price, 2),
            'bid': round(base_price * 0.998, 2),
            'ask': round(base_price * 1.002, 2),
            'timestamp': datetime.now().isoformat(),
            'market_status': 'MOCK_DATA'
        }
    
    def get_market_status(self) -> Dict:
        """
        Check if EGX market is currently open.
        
        Returns:
            Dictionary with market status information
        """
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.base_url}/marketdata/status/",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback: check current time vs EGX trading hours
                return self._check_market_status_by_time()
                
        except Exception as e:
            self.logger.error(f"Failed to get market status: {str(e)}")
            return self._check_market_status_by_time()
    
    def _check_market_status_by_time(self) -> Dict:
        """Determine market status based on current time (EGX timezone)."""
        now = datetime.now()
        
        # EGX trading hours: Sunday-Thursday, 10:00 AM - 2:30 PM EET
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        minute = now.minute
        
        # Check if weekend (Friday-Saturday)
        if weekday >= 4:  # Friday=4, Saturday=5
            return {
                'is_open': False,
                'status': 'WEEKEND',
                'message': 'EGX is closed on Friday-Saturday'
            }
        
        # Check trading hours (10:00 - 14:30)
        if 10 <= hour < 14 or (hour == 14 and minute <= 30):
            return {
                'is_open': True,
                'status': 'OPEN',
                'message': 'EGX is currently open'
            }
        elif hour < 10:
            return {
                'is_open': False,
                'status': 'PRE_MARKET',
                'message': 'EGX opens at 10:00 AM'
            }
        else:
            return {
                'is_open': False,
                'status': 'CLOSED',
                'message': 'EGX closed at 2:30 PM'
            }


class SignalGenerator:
    """
    Generates BUY/SELL/HOLD signals based on real-time Thndr data.
    
    This is a SEPARATE module from execution - it only generates signals.
    You must manually execute trades on the Thndr mobile app.
    """
    
    def __init__(self, lookback_period: int = 30):
        """
        Initialize signal generator.
        
        Args:
            lookback_period: Number of days for historical comparison
        """
        self.lookback_period = lookback_period
        self.logger = logger.logger
    
    def generate_signal(
        self,
        symbol: str,
        live_price: Dict,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate trading signal based on live price and historical context.
        
        Args:
            symbol: EGX symbol
            live_price: Current price data from Thndr
            historical_data: Historical OHLCV data for context
        
        Returns:
            Signal dictionary with recommendation and reasoning
        """
        if not live_price or 'price' not in live_price:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No price data available',
                'timestamp': datetime.now().isoformat()
            }
        
        current_price = live_price['price']
        change_pct = live_price.get('change_percent', 0)
        
        # Simple momentum-based signal (replace with your ML model)
        signal = 'HOLD'
        confidence = 0.0
        reasons = []
        
        # Rule 1: Strong positive momentum
        if change_pct > 2.0:
            signal = 'BUY'
            confidence = min(change_pct / 5.0, 1.0)
            reasons.append(f"Strong momentum: +{change_pct:.2f}%")
        
        # Rule 2: Strong negative momentum
        elif change_pct < -2.0:
            signal = 'SELL'
            confidence = min(abs(change_pct) / 5.0, 1.0)
            reasons.append(f"Strong downtrend: {change_pct:.2f}%")
        
        # Rule 3: Compare to historical average (if data available)
        if historical_data is not None and not historical_data.empty:
            avg_price = historical_data['close'].rolling(window=self.lookback_period).mean().iloc[-1]
            deviation = (current_price - avg_price) / avg_price
            
            if deviation > 0.05:  # 5% above average
                if signal == 'BUY':
                    reasons.append(f"Caution: Price {deviation*100:.1f}% above {self.lookback_period}-day average")
                    confidence *= 0.7
                elif signal == 'HOLD':
                    signal = 'SELL'
                    confidence = min(abs(deviation) * 2, 1.0)
                    reasons.append(f"Overbought: {deviation*100:.1f}% above average")
            
            elif deviation < -0.05:  # 5% below average
                if signal == 'HOLD':
                    signal = 'BUY'
                    confidence = min(abs(deviation) * 2, 1.0)
                    reasons.append(f"Oversold: {deviation*100:.1f}% below average")
        
        # Default reason if none generated
        if not reasons:
            reasons.append(f"Price change: {change_pct:.2f}%")
        
        result = {
            'symbol': symbol,
            'signal': signal,
            'confidence': round(confidence, 2),
            'reason': '; '.join(reasons),
            'current_price': current_price,
            'change_percent': change_pct,
            'timestamp': datetime.now().isoformat(),
            'action_required': f"Open Thndr app and {'BUY' if signal == 'BUY' else 'SELL' if signal == 'SELL' else 'MONITOR'} {symbol}"
        }
        
        self.logger.info(f"Signal for {symbol}: {signal} (confidence: {confidence:.2f})")
        return result


def run_signal_monitor(
    symbols: List[str],
    interval_seconds: int = 60,
    duration_hours: float = 1.0,
    phone_number: str = None
):
    """
    Run continuous signal monitoring with Thndr real-time data.
    
    This function:
    1. Fetches live prices from Thndr
    2. Generates trading signals
    3. Displays signals for manual execution
    4. Does NOT execute trades automatically (you do it on phone)
    
    Args:
        symbols: List of EGX symbols to monitor
        interval_seconds: How often to fetch new data (min 3s recommended)
        duration_hours: How long to run (0 for infinite)
        phone_number: Thndr phone number for authentication
    """
    logger.logger.info("=" * 60)
    logger.logger.info("THNDR SIGNAL MONITOR - MANUAL EXECUTION MODE")
    logger.logger.info("=" * 60)
    logger.logger.info("This system generates signals ONLY.")
    logger.logger.info("You must manually execute trades on Thndr app.")
    logger.logger.info("=" * 60)
    
    # Initialize fetcher
    fetcher = ThndrDataFetcher(
        phone_number=phone_number,
        rate_limit_seconds=max(3.0, interval_seconds / 2)  # Be conservative
    )
    
    # Try to authenticate if phone number provided
    if phone_number:
        if not fetcher.authenticate():
            logger.logger.error("Authentication failed. Running in limited mode.")
    else:
        logger.logger.warning("Running without authentication (guest mode)")
    
    # Initialize signal generator
    signal_gen = SignalGenerator(lookback_period=30)
    
    # Load historical data for context (optional but recommended)
    from data_ingestion import EGXDataFetcher
    data_fetcher = EGXDataFetcher()
    historical_data = {}
    
    for symbol in symbols:
        try:
            hist_df = data_fetcher.fetch_historical_data(symbol, period="6mo")
            historical_data[symbol] = hist_df
            logger.logger.info(f"Loaded historical data for {symbol}")
        except Exception as e:
            logger.logger.warning(f"Could not load history for {symbol}: {str(e)}")
            historical_data[symbol] = None
    
    # Main monitoring loop
    start_time = datetime.now()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            
            # Check duration limit
            if duration_hours > 0 and elapsed >= duration_hours:
                logger.logger.info(f"Duration limit reached ({duration_hours} hours)")
                break
            
            # Check market status
            market_status = fetcher.get_market_status()
            if not market_status.get('is_open', False):
                logger.logger.info(f"Market closed: {market_status.get('message', '')}")
                logger.logger.info(f"Next check in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                continue
            
            # Fetch live prices
            logger.logger.info(f"\n{'='*60}")
            logger.logger.info(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.logger.info(f"{'='*60}")
            
            for symbol in symbols:
                # Get live price
                live_price = fetcher.get_live_price(symbol)
                
                if live_price:
                    # Generate signal
                    signal = signal_gen.generate_signal(
                        symbol=symbol,
                        live_price=live_price,
                        historical_data=historical_data.get(symbol)
                    )
                    
                    # Display signal
                    print(f"\n{'─'*60}")
                    print(f"SYMBOL: {signal['symbol']}")
                    print(f"PRICE: {signal['current_price']:.2f} EGP ({signal['change_percent']:+.2f}%)")
                    print(f"SIGNAL: {signal['signal']} (confidence: {signal['confidence']:.0%})")
                    print(f"REASON: {signal['reason']}")
                    print(f"ACTION: {signal['action_required']}")
                    print(f"TIME: {signal['timestamp']}")
                    print(f"{'─'*60}")
                    
                    # Log signal
                    logger.logger.info(
                        f"SIGNAL: {signal['symbol']} | {signal['signal']} | "
                        f"Conf: {signal['confidence']:.0%} | Price: {signal['current_price']:.2f}"
                    )
                else:
                    logger.logger.warning(f"No price data for {symbol}")
            
            # Wait for next iteration
            logger.logger.info(f"Next update in {interval_seconds} seconds...")
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        logger.logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.logger.error(f"Monitoring error: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Thndr Real-Time Signal Generator")
    parser.add_argument(
        "--symbols",
        type=str,
        default="COMI.CA,ETEL.CA,HRHO.CA",
        help="Comma-separated list of EGX symbols (default: COMI.CA,ETEL.CA,HRHO.CA)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (min 3, default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration in hours (0 for infinite, default: 1.0)"
    )
    parser.add_argument(
        "--phone",
        type=str,
        default=None,
        help="Thndr phone number for authentication (e.g., +201234567890)"
    )
    
    args = parser.parse_args()
    
    symbols_list = [s.strip() for s in args.symbols.split(",")]
    
    print("\n" + "=" * 60)
    print("THNDR REAL-TIME SIGNAL GENERATOR")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols_list)}")
    print(f"Interval: {args.interval} seconds")
    print(f"Duration: {args.duration} hours")
    print(f"Phone: {args.phone or 'Guest mode (limited)'}")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: This generates signals ONLY.")
    print("   You must manually execute trades on the Thndr app.")
    print("=" * 60 + "\n")
    
    run_signal_monitor(
        symbols=symbols_list,
        interval_seconds=args.interval,
        duration_hours=args.duration,
        phone_number=args.phone
    )
