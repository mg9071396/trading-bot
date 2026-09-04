"""
Main Orchestration Script for EGX Trading System.

This script orchestrates the complete pipeline:
1. Data ingestion (OHLCV, news sentiment, alternative data)
2. Feature engineering
3. Model training and signal generation
4. Backtesting with realistic EGX friction
5. Risk management validation

WARNING: =====================================================================
THIS IS A FRAMEWORK FOR RESEARCH AND DEVELOPMENT PURPOSES ONLY.

DO NOT use this system for live trading without:
- Extensive paper-trading (minimum 3-6 months)
- Thorough backtesting across multiple market conditions
- Financial due diligence and regulatory compliance review
- Integration with licensed Egyptian broker APIs
- Proper risk management testing

Emerging markets like the Egyptian Exchange (EGX) have unique risks including:
- Lower liquidity and higher slippage
- Circuit breakers and trading halts
- Currency risk (EGP volatility)
- Political and economic instability
- Limited data availability and quality issues

By using this framework, you acknowledge that:
- Past performance does not guarantee future results
- All trading involves substantial risk of loss
- You are solely responsible for your trading decisions
===============================================================================

Author: Quantitative Developer & ML Engineer
Date: 2024
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Import all modules
from config import trading_config, model_config, backtest_config, get_config
from logger import TradingLogger
from data_ingestion import EGXDataFetcher, NewsSentimentFetcher, AlternativeDataFetcher, MultimodalDataMerger
from feature_engineering import FeatureEngineer
from model import QuantModel
from backtest import EGXBacktester
from risk_and_execution import TradingSystem, SignalType

# Initialize global logger
logger = TradingLogger("egx_trading_main")


def run_full_pipeline(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = None,
    paper_trading: bool = True,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run the complete EGX trading system pipeline.
    
    Args:
        symbols: List of EGX symbols to trade (e.g., ["COMI.CA", "ETEL.CA"])
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Starting capital in EGP
        paper_trading: If True, simulate trades
        output_dir: Directory for output files
    
    Returns:
        Dictionary with pipeline results
    """
    # Default symbols (EGX blue chips)
    if symbols is None:
        symbols = ["COMI.CA", "ETEL.CA", "HRHO.CA", "SWDY.CA", "AAIC.CA"]
    
    # Set dates
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=365*2)  # 2 years of data
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    initial_capital = initial_capital or backtest_config.initial_capital
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EGX ALGORITHMIC TRADING SYSTEM - FULL PIPELINE")
    print("=" * 70)
    print(f"\nRun Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: {initial_capital:,.0f} EGP")
    print(f"Paper Trading: {paper_trading}")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'start_date': start_dt.isoformat(),
            'end_date': end_dt.isoformat(),
            'initial_capital': initial_capital
        },
        'data': {},
        'features': {},
        'model': {},
        'backtest': {},
        'risk_checks': []
    }
    
    # =========================================================================
    # STEP 1: DATA INGESTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA INGESTION")
    print("=" * 70)
    
    # Fetch OHLCV data
    print("\n[1.1] Fetching historical OHLCV data...")
    ohlcv_fetcher = EGXDataFetcher()
    
    all_ohlcv_data = {}
    for symbol in symbols:
        try:
            df = ohlcv_fetcher.fetch_historical_data(symbol, start_dt, end_dt)
            all_ohlcv_data[symbol] = df
            print(f"   ✓ {symbol}: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
        except Exception as e:
            print(f"   ✗ {symbol}: Failed - {str(e)}")
            logger.log_error("data_ingestion", "FetchError", str(e), {"symbol": symbol})
    
    if not all_ohlcv_data:
        raise ValueError("Failed to fetch data for any symbol")
    
    # Fetch news sentiment data
    print("\n[1.2] Fetching news sentiment data...")
    sentiment_fetcher = NewsSentimentFetcher()
    
    all_news_data = {}
    for symbol in symbols:
        try:
            news_df = sentiment_fetcher.fetch_news_and_sentiment(symbol, start_dt, end_dt)
            all_news_data[symbol] = news_df
            avg_sentiment = news_df['sentiment_score'].mean()
            print(f"   ✓ {symbol}: {len(news_df)} articles, avg sentiment: {avg_sentiment:.3f}")
        except Exception as e:
            print(f"   ✗ {symbol}: Failed - {str(e)}")
    
    # Merge data
    print("\n[1.3] Merging multimodal data...")
    merger = MultimodalDataMerger()
    
    merged_data = {}
    for symbol in symbols:
        ohlcv = all_ohlcv_data[symbol]
        news = all_news_data.get(symbol)
        
        merged = merger.merge_all_data(ohlcv, news)
        merged_data[symbol] = merged
        print(f"   ✓ {symbol}: {len(merged)} rows after merge")
    
    results['data'] = {
        'symbols_processed': len(symbols),
        'total_ohlcv_rows': sum(len(df) for df in all_ohlcv_data.values()),
        'total_news_articles': sum(len(df) for df in all_news_data.values())
    }
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    feature_engineer = FeatureEngineer()
    featured_data = {}
    
    for symbol in symbols:
        df = merged_data[symbol]
        
        try:
            # Create features
            df_features = feature_engineer.create_all_features(df)
            
            # Handle missing values
            df_clean = feature_engineer.handle_missing_values(df_features)
            
            featured_data[symbol] = df_clean
            
            print(f"\n   ✓ {symbol}:")
            print(f"      Features created: {len(feature_engineer.feature_columns)}")
            print(f"      Rows after cleaning: {len(df_clean)}")
            
            # Validate features
            validation = feature_engineer.validate_features(df_clean)
            if not validation['validation_passed']:
                print(f"      ⚠ Validation warnings:")
                if validation['constant_features']:
                    print(f"         - Constant features: {len(validation['constant_features'])}")
                if validation['high_correlation_pairs']:
                    print(f"         - High correlation pairs: {len(validation['high_correlation_pairs'])}")
            
        except Exception as e:
            print(f"   ✗ {symbol}: Feature engineering failed - {str(e)}")
            logger.log_error("feature_engineering", "FeatureError", str(e), {"symbol": symbol})
    
    results['features'] = {
        'num_features': len(feature_engineer.feature_columns),
        'feature_columns': feature_engineer.feature_columns[:10],  # First 10
        'symbols_processed': len(featured_data)
    }
    
    # =========================================================================
    # STEP 3: MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)
    
    models = {}
    model_metrics = {}
    
    for symbol in symbols:
        if symbol not in featured_data:
            continue
        
        df = featured_data[symbol]
        
        try:
            print(f"\n   Training model for {symbol}...")
            
            model = QuantModel()
            
            # Train model
            metrics = model.fit(
                df=df,
                feature_columns=feature_engineer.feature_columns,
                target_column='target',
                verbose=False
            )
            
            models[symbol] = model
            model_metrics[symbol] = metrics
            
            print(f"   ✓ {symbol} model trained:")
            print(f"      Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"      Precision: {metrics.get('precision', 0):.4f}")
            print(f"      Recall: {metrics.get('recall', 0):.4f}")
            print(f"      ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            
            # Get feature importance
            importance_df = model.get_feature_importance(top_n=5)
            top_features = importance_df.head(3)['feature'].tolist()
            print(f"      Top features: {', '.join(top_features)}")
            
        except Exception as e:
            print(f"   ✗ {symbol}: Model training failed - {str(e)}")
            logger.log_error("model_training", "TrainError", str(e), {"symbol": symbol})
    
    results['model'] = {
        'symbols_trained': len(models),
        'metrics': model_metrics
    }
    
    # =========================================================================
    # STEP 4: BACKTESTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: BACKTESTING")
    print("=" * 70)
    
    backtest_results = {}
    
    for symbol in symbols:
        if symbol not in models or symbol not in featured_data:
            continue
        
        model = models[symbol]
        df = featured_data[symbol]
        
        try:
            print(f"\n   Backtesting {symbol}...")
            
            # Generate signals
            df_signals = model.generate_signals(df, feature_engineer.feature_columns)
            
            # Create signal series for backtester
            signals = pd.Series('HOLD', index=df_signals.index)
            signals[df_signals['signal'] == 'BUY'] = 'BUY'
            
            # Simple exit logic: exit after N days
            hold_period = trading_config.prediction_horizon
            for i in range(hold_period, len(signals)):
                if signals.iloc[i - hold_period] == 'BUY':
                    signals.iloc[i] = 'SELL'
            
            # Run backtest
            backtester = EGXBacktester(initial_capital=initial_capital / len(symbols))
            results_df = backtester.run_backtest(df_signals, signals)
            
            metrics = backtester.get_metrics()
            backtest_results[symbol] = {
                'metrics': metrics,
                'results_df': results_df
            }
            
            print(f"   ✓ {symbol} backtest complete:")
            print(f"      Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"      Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"      Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"      Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"      Number of Trades: {metrics.get('num_trades', 0)}")
            
        except Exception as e:
            print(f"   ✗ {symbol}: Backtest failed - {str(e)}")
            logger.log_error("backtesting", "BacktestError", str(e), {"symbol": symbol})
    
    # Aggregate metrics
    if backtest_results:
        total_return = sum(r['metrics'].get('total_return', 0) for r in backtest_results.values())
        avg_sharpe = sum(r['metrics'].get('sharpe_ratio', 0) for r in backtest_results.values()) / len(backtest_results)
        total_trades = sum(r['metrics'].get('num_trades', 0) for r in backtest_results.values())
        
        print(f"\n   AGGREGATE RESULTS:")
        print(f"      Combined Return: {total_return:.2%}")
        print(f"      Average Sharpe: {avg_sharpe:.2f}")
        print(f"      Total Trades: {total_trades}")
    
    results['backtest'] = {
        'symbols_backtested': len(backtest_results),
        'aggregate_return': total_return if backtest_results else 0,
        'average_sharpe': avg_sharpe if backtest_results else 0,
        'total_trades': total_trades if backtest_results else 0
    }
    
    # =========================================================================
    # STEP 5: RISK MANAGEMENT VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: RISK MANAGEMENT VALIDATION")
    print("=" * 70)
    
    trading_system = TradingSystem(portfolio_value=initial_capital, paper_trading=paper_trading)
    
    print("\n   Testing risk management on recent signals...")
    
    # Test a few signals through risk management
    risk_test_results = []
    
    for symbol in list(models.keys())[:3]:  # Test first 3 symbols
        model = models[symbol]
        df = featured_data[symbol]
        
        # Get last row's signal
        last_row = df.iloc[-1:]
        df_signals = model.generate_signals(last_row, feature_engineer.feature_columns)
        
        signal = df_signals['signal'].iloc[0]
        probability = df_signals.get('signal_probability', [0.5]).iloc[0]
        price = df['close'].iloc[-1]
        
        # Convert to SignalType
        if signal == 'BUY':
            signal_type = SignalType.BUY
        elif signal == 'SELL':
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Get sentiment score
        sentiment_score = df.get('news_sentiment', pd.Series([0])).iloc[-1]
        
        # Process through risk manager
        result = trading_system.process_signal(
            signal=signal_type,
            symbol=symbol,
            price=price,
            suggested_quantity=1000,
            sentiment_score=sentiment_score,
            win_rate=model_metrics.get(symbol, {}).get('accuracy', 0.5),
            win_loss_ratio=1.5
        )
        
        risk_test_results.append(result)
        
        status_icon = "✓" if result['status'] != 'REJECTED' else "✗"
        print(f"   {status_icon} {symbol}: {signal} @ {price:.2f} EGP -> {result['status']}")
        
        if result.get('rejection_reasons'):
            for reason in result['rejection_reasons']:
                print(f"      Rejection: {reason}")
    
    results['risk_checks'] = risk_test_results
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_path / f"pipeline_summary_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = json.loads(json.dumps(results, default=str))
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n   ✓ Summary saved to: {summary_file}")
    
    # Save backtest reports
    for symbol, br in backtest_results.items():
        if 'metrics' in br:
            report_file = output_path / f"backtest_report_{symbol}_{timestamp}.txt"
            backtester = EGXBacktester()
            backtester.metrics = br['metrics']
            report = backtester.generate_report()
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"   ✓ Backtest report saved: {report_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults Summary:")
    print(f"   Symbols Processed: {results['data']['symbols_processed']}")
    print(f"   Models Trained: {results['model']['symbols_trained']}")
    print(f"   Symbols Backtested: {results['backtest']['symbols_backtested']}")
    print(f"   Aggregate Return: {results['backtest']['aggregate_return']:.2%}")
    print(f"   Average Sharpe: {results['backtest']['average_sharpe']:.2f}")
    print(f"\nOutput Directory: {output_path.absolute()}")
    
    print("\n" + "=" * 70)
    print("WARNING: THIS IS A RESEARCH FRAMEWORK ONLY")
    print("=" * 70)
    print("""
IMPORTANT REMINDERS:
• This system requires extensive paper-trading before any live deployment
• Backtests often overstate performance - expect lower returns in live trading
• EGX has unique risks: lower liquidity, circuit breakers, currency risk
• Always use proper position sizing and risk management
• Consult with financial advisors and regulatory experts before live trading
• Past performance does not guarantee future results
""")
    
    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="EGX Algorithmic Trading System - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbols COMI.CA ETEL.CA HRHO.CA
  python main.py --start-date 2023-01-01 --end-date 2024-01-01
  python main.py --initial-capital 500000 --paper-trading
  python main.py --output-dir ./my_results

WARNING: This is for research purposes only. Do not use for live trading
without extensive testing and regulatory compliance.
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='EGX symbols to trade (e.g., COMI.CA ETEL.CA HRHO.CA)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=None,
        help='Initial capital in EGP (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        default=True,
        help='Enable paper trading mode (default: True)'
    )
    
    parser.add_argument(
        '--live-trading',
        action='store_true',
        default=False,
        help='Enable live trading mode (DANGEROUS - requires broker integration)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory for output files'
    )
    
    args = parser.parse_args()
    
    # Override paper_trading if --live-trading is specified
    if args.live_trading:
        print("\n" + "!" * 70)
        print("LIVE TRADING MODE REQUESTED")
        print("!" * 70)
        print("""
WARNING: Live trading requires:
1. Licensed broker API integration (EFG Hermes, CI Capital, etc.)
2. Regulatory compliance approval
3. Extensive paper-trading validation (minimum 3-6 months)
4. Proper risk management infrastructure

This framework does NOT include production broker integration.
Switching to paper trading mode for safety.
""")
        print("!" * 70 + "\n")
    
    # Run pipeline
    try:
        results = run_full_pipeline(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            paper_trading=True,  # Always use paper trading
            output_dir=args.output_dir
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\nPipeline failed with error: {str(e)}")
        logger.log_error("main", "PipelineError", str(e))
        return 1


if __name__ == "__main__":
    exit(main())
