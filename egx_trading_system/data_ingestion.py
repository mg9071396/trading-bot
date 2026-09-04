"""
Data Ingestion Module for EGX Trading System.

Handles multimodal data ingestion:
1. Historical OHLCV data (Yahoo Finance or custom EGX API)
2. Arabic news sentiment analysis (AraBERT)
3. Alternative data from video/audio transcription (Whisper)

WARNING: This is a framework for research and development purposes only.
DO NOT use for live trading without extensive paper-trading, backtesting,
and financial due diligence. The data fetchers use placeholder/mock APIs
that must be replaced with real Egyptian financial data providers.

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ARABERT_AVAILABLE = True
except ImportError:
    ARABERT_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from logger import TradingLogger
from config import trading_config

logger = TradingLogger("data_ingestion")


class EGXDataFetcher:
    """
    Fetches historical OHLCV data for EGX-listed securities.
    
    EGX Symbol Format: SYMBOL.CA (e.g., COMI.CA for Commercial International Bank)
    
    Note: Yahoo Finance has limited EGX coverage. For production, replace with
    a local Egyptian data provider API (e.g., Mubasher, EFG Hermes, or direct EGX feed).
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.logger
        
        # EGX trading calendar (simplified - should be expanded for production)
        self.trading_days_per_year = 252  # Approximate for EGX (Sunday-Thursday)
        
    def _get_yahoo_symbol(self, symbol: str) -> str:
        """Convert EGX symbol format to Yahoo Finance format if needed."""
        # Yahoo Finance uses .CA for Cairo Exchange
        if not symbol.endswith('.CA'):
            symbol = f"{symbol}.CA"
        return symbol
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "max",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.
        
        Args:
            symbol: EGX symbol (e.g., "COMI.CA", "ETEL.CA", "HRHO.CA")
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj_Close, Volume
        
        Raises:
            ValueError: If symbol is invalid or data cannot be fetched
        """
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        cache_file = self.cache_dir / f"{yahoo_symbol.replace('.', '_')}_historical.parquet"
        
        # Check cache first
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self.logger.info(f"Loaded cached data for {symbol}: {len(df)} rows")
                return df
            except Exception as e:
                self.logger.warning(f"Cache read failed for {symbol}: {str(e)}")
        
        # Fetch from Yahoo Finance (placeholder - replace with real EGX API in production)
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance not installed. Install with: pip install yfinance")
            raise ImportError("yfinance is required for data fetching")
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                # Generate mock data for demonstration (REMOVE IN PRODUCTION)
                self.logger.warning(f"No data found for {yahoo_symbol}. Generating mock data for testing.")
                df = self._generate_mock_egx_data(symbol, periods=500)
            else:
                self.logger.info(f"Fetched {len(df)} rows for {symbol} from Yahoo Finance")
            
            # Clean and standardize data
            df = self._clean_ohlcv_data(df, symbol)
            
            # Cache the data
            if use_cache:
                df.to_parquet(cache_file)
                self.logger.debug(f"Cached data for {symbol}")
            
            return df
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch data for {symbol}",
                extra={"error_type": type(e).__name__, "error_message": str(e)}
            )
            # Return mock data for testing (REMOVE IN PRODUCTION)
            self.logger.warning(f"Returning mock data for {symbol} due to fetch error")
            return self._generate_mock_egx_data(symbol, periods=500)
    
    def _clean_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize OHLCV data."""
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to standard format
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Handle missing values
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        missing_before = df[numeric_cols].isnull().sum().sum()
        
        if missing_before > 0:
            # Forward fill then backward fill
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            
            # If still missing, use median imputation
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            self.logger.log_data_gap(
                symbol, "OHLCV",
                {"missing_values": int(missing_before), "imputation_method": "ffill+bfill+median"}
            )
        
        # Validate data quality
        if (df['volume'] <= 0).any():
            self.logger.warning(f"Zero or negative volume detected for {symbol}")
            df.loc[df['volume'] <= 0, 'volume'] = df['volume'].median()
        
        # Ensure OHLC logic: High >= Low, High >= Open, High >= Close, etc.
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            self.logger.warning(f"Invalid OHLC relationships detected for {symbol}, correcting...")
            # Simple correction: swap if needed
            df.loc[invalid_ohlc, ['high', 'low']] = df.loc[invalid_ohlc, [['high', 'low']]].max(axis=1), df.loc[invalid_ohlc, [['high', 'low']]].min(axis=1)
        
        return df
    
    def _generate_mock_egx_data(self, symbol: str, periods: int = 500) -> pd.DataFrame:
        """
        Generate mock EGX data for testing purposes.
        
        WARNING: This is for demonstration only. REPLACE with real data in production.
        """
        self.logger.warning(f"GENERATING MOCK DATA FOR {symbol} - NOT FOR LIVE TRADING")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate dates (EGX trades Sunday-Thursday)
        end_date = datetime.now()
        dates = []
        current_date = end_date - timedelta(days=int(periods * 1.5))  # Buffer for weekends
        
        while len(dates) < periods:
            # Skip Fridays and Saturdays (EGX weekend)
            if current_date.weekday() < 5:  # Sunday=0 to Thursday=4
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Generate realistic price series with mean reversion
        base_price = np.random.uniform(10, 100)  # Typical EGX stock price range
        returns = np.random.normal(0.0005, 0.025, periods)  # Daily return distribution
        
        # Add some momentum and mean reversion
        returns = returns + 0.05 * np.roll(returns, 1) - 0.03 * (np.arange(periods) % 20 - 10) / 10
        returns[0] = 0  # First return is 0
        
        close_prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC from close prices
        daily_volatility = np.abs(np.random.normal(0.015, 0.005, periods))
        high_prices = close_prices * (1 + daily_volatility)
        low_prices = close_prices * (1 - daily_volatility)
        open_prices = low_prices + np.random.uniform(0.3, 0.7, periods) * (high_prices - low_prices)
        
        # Generate volume (lower liquidity typical for EGX)
        base_volume = np.random.randint(10000, 500000)
        volume = base_volume * np.random.lognormal(0, 0.5, periods)
        
        df = pd.DataFrame({
            'date': dates[:periods],
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'adj_close': close_prices,  # Simplified
            'volume': volume.astype(int),
            'symbol': symbol
        })
        
        return df
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.fetch_historical_data(symbol, start_date, end_date)
            except Exception as e:
                self.logger.log_error(
                    "data_ingestion",
                    "MultiSymbolFetchError",
                    f"Failed to fetch {symbol}",
                    {"error": str(e)}
                )
        return data_dict


class NewsSentimentFetcher:
    """
    Fetches Arabic financial news and computes sentiment scores using AraBERT.
    
    AraBERT is a pre-trained BERT model for Arabic language understanding.
    This implementation uses a sentiment analysis model fine-tuned for Arabic.
    
    Note: For production, integrate with real Arabic financial news APIs:
    - Mubasher (https://www.mubasher.info/)
    - Reuters Arabic
    - Al Arabiya Business
    """
    
    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        """
        Initialize AraBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name for Arabic BERT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.logger = logger.logger
        
        # Sentiment label mapping (adjust based on actual model)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        if ARABERT_AVAILABLE:
            self._load_model()
        else:
            self.logger.warning("transformers/torch not available. Using mock sentiment analysis.")
    
    def _load_model(self):
        """Load AraBERT model and tokenizer."""
        try:
            self.logger.info(f"Loading AraBERT model: {self.model_name}")
            # Use a sentiment analysis model if available, otherwise use base model
            # For production, fine-tune on Egyptian financial news
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Try to load a sentiment classification head
            # For demo, we'll use a simple approach
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # negative, neutral, positive
            )
            self.logger.info("AraBERT model loaded successfully")
        except Exception as e:
            self.logger.log_error(
                "sentiment_analysis",
                "ModelLoadError",
                f"Failed to load AraBERT: {str(e)}"
            )
            self.model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of Arabic text.
        
        Args:
            text: Arabic text to analyze
        
        Returns:
            Dictionary with sentiment_score (-1 to 1) and confidence
        """
        if not text or not isinstance(text, str):
            return {"sentiment_score": 0.0, "confidence": 0.0, "label": "unknown"}
        
        # Mock sentiment analysis if model not available
        if not ARABERT_AVAILABLE or self.model is None:
            return self._mock_sentiment_analysis(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            
            # Convert to sentiment score (-1 to 1)
            neg_prob = probabilities[0].item()
            neu_prob = probabilities[1].item()
            pos_prob = probabilities[2].item()
            
            # Weighted sentiment score
            sentiment_score = pos_prob - neg_prob
            confidence = max(probabilities).item()
            label = self.label_map[torch.argmax(probabilities).item()]
            
            return {
                "sentiment_score": round(sentiment_score, 4),
                "confidence": round(confidence, 4),
                "label": label,
                "probabilities": {
                    "negative": round(neg_prob, 4),
                    "neutral": round(neu_prob, 4),
                    "positive": round(pos_prob, 4)
                }
            }
            
        except Exception as e:
            self.logger.log_error(
                "sentiment_analysis",
                "InferenceError",
                f"Sentiment analysis failed: {str(e)}",
                {"text_length": len(text)}
            )
            return {"sentiment_score": 0.0, "confidence": 0.0, "label": "error"}
    
    def _mock_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Mock sentiment analysis for testing.
        
        WARNING: Replace with real AraBERT inference in production.
        """
        # Simple keyword-based mock (for demonstration only)
        positive_keywords = ['ربح', 'نمو', 'إيجابي', 'صعود', 'ارتفاع', 'مكسب', 'تفوق']
        negative_keywords = ['خسارة', 'انخفاض', 'سلبي', 'هبوط', 'تراجع', 'خسر', 'فشل']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
        
        # Add some noise
        score = np.clip(score + np.random.normal(0, 0.1), -1, 1)
        
        return {
            "sentiment_score": round(score, 4),
            "confidence": 0.5,
            "label": "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral"),
            "mock": True
        }
    
    def fetch_news_and_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        sources: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch news articles for a symbol and compute sentiment scores.
        
        Args:
            symbol: EGX symbol
            start_date: Start date for news collection
            end_date: End date for news collection
            sources: List of news sources to query
        
        Returns:
            DataFrame with columns: date, headline, text, sentiment_score, confidence
        """
        self.logger.info(f"Fetching news for {symbol} from {start_date} to {end_date}")
        
        # Placeholder: Generate mock news data
        # In production, replace with actual API calls to Arabic news providers
        news_data = self._generate_mock_news(symbol, start_date, end_date)
        
        # Analyze sentiment for each article
        sentiment_scores = []
        for idx, row in news_data.iterrows():
            sentiment_result = self.analyze_sentiment(row['text'])
            sentiment_scores.append(sentiment_result['sentiment_score'])
        
        news_data['sentiment_score'] = sentiment_scores
        
        self.logger.info(f"Analyzed sentiment for {len(news_data)} news articles")
        
        return news_data
    
    def _generate_mock_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        num_articles: int = 50
    ) -> pd.DataFrame:
        """Generate mock Arabic news articles for testing."""
        np.random.seed(42)
        
        # Sample Arabic headlines and texts (for demonstration)
        arabic_headlines = [
            f"شركة {symbol} تحقق أرباحاً قوية في الربع الأخير",
            f"تحليل: أداء إيجابي متوقع لسهم {symbol}",
            f"{symbol} تعلن عن توسعات جديدة في السوق المصري",
            f"خسائر غير متوقعة تؤثر على سهم {symbol}",
            f"توصية بشراء سهم {symbol} من محللين ماليين",
            f"{symbol} تواجه تحديات في السوق الإقليمي",
            f"ارتفاع حجم التداولات على سهم {symbol}",
            f"تقرير: آفاق نمو واعدة لشركة {symbol}",
        ]
        
        arabic_texts = [
            f"أعلنت شركة {symbol} عن نتائج مالية إيجابية للربع الرابع، حيث حققت نمواً في الأرباح بنسبة 15% مقارنة بالعام السابق. ويتوقع المحللون استمرار هذا الزخم الإيجابي.",
            f"شهد سهم {symbol} تداولات نشطة اليوم وسط تفاؤل المستثمرين بالأداء المستقبلي للشركة. وأوصى عدة محللين ماليين بالشراء عند المستويات الحالية.",
            f"تواجه {symbol} بعض التحديات في السوق المحلي نتيجة الظروف الاقتصادية الراهنة، مما أدى إلى تراجع الأرباح بنسبة 8% مقارنة بالفترة المماثلة من العام الماضي.",
            f"أعلنت إدارة {symbol} عن خطة توسع استراتيجية تشمل دخول أسواق جديدة وإطلاق منتجات مبتكرة، مما يعزز آفاق النمو على المدى الطويل.",
        ]
        
        # Generate random dates within range
        date_range = (end_date - start_date).days
        dates = [
            start_date + timedelta(days=np.random.randint(0, max(1, date_range)))
            for _ in range(num_articles)
        ]
        
        news_data = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'headline': [np.random.choice(arabic_headlines) for _ in range(num_articles)],
            'text': [np.random.choice(arabic_texts) for _ in range(num_articles)],
            'source': np.random.choice(['Mubasher', 'Reuters Arabic', 'Al Arabiya'], num_articles)
        })
        
        news_data = news_data.sort_values('date').reset_index(drop=True)
        
        self.logger.warning(f"Generated {num_articles} MOCK news articles for {symbol}")
        
        return news_data


class AlternativeDataFetcher:
    """
    Fetches and processes alternative data from video/audio content.
    
    Uses OpenAI Whisper for transcription of financial videos/audio clips
    (e.g., earnings calls, CNBC Arabia segments, analyst presentations).
    
    Note: Video transcription is computationally expensive and may introduce noise.
    Start with text-based sentiment analysis before adding this module.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper transcription model.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       Larger models are more accurate but slower
        """
        self.model_size = model_size
        self.model = None
        self.logger = logger.logger
        
        if WHISPER_AVAILABLE:
            self._load_model()
        else:
            self.logger.warning("Whisper not available. Install with: pip install openai-whisper")
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.log_error(
                "alternative_data",
                "WhisperLoadError",
                f"Failed to load Whisper: {str(e)}"
            )
            self.model = None
    
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: str = "ar"
    ) -> Dict[str, any]:
        """
        Transcribe audio/video file using Whisper.
        
        Args:
            audio_path: Path to audio/video file
            language: Language code ("ar" for Arabic, "en" for English)
        
        Returns:
            Dictionary with transcription, detected language, and segments
        """
        if not WHISPER_AVAILABLE or self.model is None:
            self.logger.error("Whisper model not available")
            return self._mock_transcription(str(audio_path))
        
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.logger.info(f"Transcribing {audio_path.name} with Whisper ({self.model_size})")
            
            # Transcribe with options
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False
            )
            
            # Extract key information
            transcription = {
                "text": result["text"],
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "duration": result.get("duration", 0),
                "source_file": str(audio_path)
            }
            
            self.logger.info(f"Transcription complete: {len(result['text'])} characters")
            
            return transcription
            
        except Exception as e:
            self.logger.log_error(
                "alternative_data",
                "TranscriptionError",
                f"Transcription failed: {str(e)}",
                {"audio_path": str(audio_path)}
            )
            return self._mock_transcription(str(audio_path))
    
    def transcribe_url(
        self,
        url: str,
        download_first: bool = True
    ) -> Dict[str, any]:
        """
        Transcribe audio/video from URL (e.g., YouTube, direct link).
        
        Args:
            url: URL to audio/video content
            download_first: Whether to download before transcription
        
        Returns:
            Transcription dictionary
        """
        # Placeholder for URL download and transcription
        # In production, use yt-dlp or similar to download, then transcribe
        self.logger.info(f"Processing URL: {url}")
        
        if download_first:
            self.logger.warning("URL download not implemented. Use local files for now.")
        
        return self._mock_transcription(url)
    
    def extract_financial_entities(self, transcription: Dict[str, any]) -> Dict[str, list]:
        """
        Extract key financial entities from transcribed text.
        
        Args:
            transcription: Output from transcribe_audio/transcribe_url
        
        Returns:
            Dictionary with extracted entities (revenues, profits, guidance, etc.)
        """
        text = transcription.get("text", "")
        
        if not text:
            return {}
        
        # Simple keyword-based extraction (for production, use NER model)
        entities = {
            "revenue_mentions": [],
            "profit_mentions": [],
            "guidance_mentions": [],
            "risk_factors": [],
            "positive_statements": [],
            "negative_statements": []
        }
        
        # Arabic financial keywords
        revenue_keywords = ['إيرادات', 'مبيعات', 'دخل', 'عائدات']
        profit_keywords = ['أرباح', 'صافي ربح', 'مكاسب', 'عائد']
        guidance_keywords = ['توقعات', 'توجيهات', 'متوقع', 'نتوقع']
        risk_keywords = ['مخاطر', 'تحديات', 'صعوبات', 'تهديدات']
        
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue
            
            if any(kw in sentence_lower for kw in revenue_keywords):
                entities["revenue_mentions"].append(sentence)
            if any(kw in sentence_lower for kw in profit_keywords):
                entities["profit_mentions"].append(sentence)
            if any(kw in sentence_lower for kw in guidance_keywords):
                entities["guidance_mentions"].append(sentence)
            if any(kw in sentence_lower for kw in risk_keywords):
                entities["risk_factors"].append(sentence)
            
            # Simple sentiment classification
            positive_words = ['نمو', 'إيجابي', 'تحسن', 'زيادة', 'تفوق']
            negative_words = ['انخفاض', 'سلبي', 'تراجع', 'خسارة', 'تدهور']
            
            if any(word in sentence_lower for word in positive_words):
                entities["positive_statements"].append(sentence)
            elif any(word in sentence_lower for word in negative_words):
                entities["negative_statements"].append(sentence)
        
        self.logger.info(f"Extracted entities: {sum(len(v) for v in entities.values())} mentions")
        
        return entities
    
    def _mock_transcription(self, source: str) -> Dict[str, any]:
        """Generate mock transcription for testing."""
        mock_text = """
        أعلنت الشركة عن نتائج مالية إيجابية للربع الرابع، حيث حققت نمواً في الإيرادات بنسبة 12%.
        وتوقع الإدارة استمرار هذا الزخم الإيجابي خلال العام القادم مع التوسع في الأسواق الجديدة.
        ومع ذلك، هناك بعض التحديات المتعلقة بارتفاع تكاليف التشغيل والظروف الاقتصادية الإقليمية.
        """
        
        self.logger.warning(f"Generated MOCK transcription for {source}")
        
        return {
            "text": mock_text.strip(),
            "language": "ar",
            "segments": [],
            "duration": 0,
            "source_file": source,
            "mock": True
        }


class MultimodalDataMerger:
    """
    Merges data from multiple sources (OHLCV, news sentiment, alternative data).
    
    Aligns timestamps and handles missing data appropriately.
    """
    
    def __init__(self):
        self.logger = logger.logger
    
    def merge_all_data(
        self,
        ohlcv_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
        alternative_data: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Merge all data sources into a single DataFrame.
        
        Args:
            ohlcv_data: OHLCV price data
            news_data: News sentiment data (optional)
            alternative_data: Alternative data from transcriptions (optional)
        
        Returns:
            Merged DataFrame aligned by date
        """
        merged = ohlcv_data.copy()
        
        # Ensure date column is datetime
        if 'date' in merged.columns:
            merged['date'] = pd.to_datetime(merged['date'])
        
        # Merge news sentiment data
        if news_data is not None and not news_data.empty:
            news_data['date'] = pd.to_datetime(news_data['date'])
            
            # Aggregate daily sentiment (mean of all articles per day)
            daily_sentiment = news_data.groupby(news_data['date'].dt.date).agg({
                'sentiment_score': 'mean',
                'symbol': 'first'
            }).reset_index()
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.rename(columns={'sentiment_score': 'news_sentiment'})
            
            merged = merged.merge(daily_sentiment, on='date', how='left')
            self.logger.info(f"Merged news sentiment: {len(daily_sentiment)} days")
        else:
            merged['news_sentiment'] = 0.0
            self.logger.warning("No news data provided, setting sentiment to 0")
        
        # Forward-fill sentiment (last known sentiment carries forward)
        merged['news_sentiment'] = merged['news_sentiment'].ffill().fillna(0)
        
        # Add alternative data flags if available
        if alternative_data:
            # Create binary flags for alternative data presence
            merged['has_alternative_data'] = 0
            # Could add more sophisticated merging based on timestamps
            self.logger.info(f"Incorporated {len(alternative_data)} alternative data points")
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Merged dataset shape: {merged.shape}")
        
        return merged


if __name__ == "__main__":
    # Test data ingestion pipeline
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("Testing EGX Data Ingestion Pipeline")
    print("=" * 60)
    
    # Test OHLCV data fetching
    fetcher = EGXDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("\n1. Fetching OHLCV data...")
    ohlcv_df = fetcher.fetch_historical_data("COMI.CA", start_date, end_date)
    print(f"   Shape: {ohlcv_df.shape}")
    print(f"   Columns: {list(ohlcv_df.columns)}")
    print(f"   Date range: {ohlcv_df['date'].min()} to {ohlcv_df['date'].max()}")
    
    # Test sentiment analysis
    print("\n2. Testing sentiment analysis...")
    sentiment_fetcher = NewsSentimentFetcher()
    test_arabic_text = "الشركة تحقق أرباحاً قوية ونمواً في الإيرادات"
    sentiment = sentiment_fetcher.analyze_sentiment(test_arabic_text)
    print(f"   Text: {test_arabic_text}")
    print(f"   Sentiment Score: {sentiment['sentiment_score']}")
    print(f"   Label: {sentiment['label']}")
    
    # Test news fetching
    print("\n3. Fetching mock news data...")
    news_df = sentiment_fetcher.fetch_news_and_sentiment("COMI.CA", start_date, end_date)
    print(f"   Shape: {news_df.shape}")
    print(f"   Avg Sentiment: {news_df['sentiment_score'].mean():.3f}")
    
    # Test alternative data
    print("\n4. Testing alternative data transcription...")
    alt_fetcher = AlternativeDataFetcher()
    transcription = alt_fetcher._mock_transcription("test_audio.mp3")
    entities = alt_fetcher.extract_financial_entities(transcription)
    print(f"   Transcription length: {len(transcription['text'])} chars")
    print(f"   Entities extracted: {sum(len(v) for v in entities.values())} mentions")
    
    # Test data merging
    print("\n5. Merging all data sources...")
    merger = MultimodalDataMerger()
    merged_df = merger.merge_all_data(ohlcv_df, news_df)
    print(f"   Merged shape: {merged_df.shape}")
    print(f"   Columns: {list(merged_df.columns)}")
    
    print("\n" + "=" * 60)
    print("Data Ingestion Test Complete")
    print("=" * 60)
