import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from libs.data_fetcher import DataFetcher
from libs.technical_analyzer import TechnicalAnalyzer
from libs.fundamental_analyzer import FundamentalAnalyzer
from libs.volume_profile_analyzer import VolumeProfileAnalyzer
from libs.signal_analyzer import SignalAnalyzer

# Load environment variables
load_dotenv()

def analyze_coin(symbol, timeframe=None, limit=None):
    """
    Analyze a single coin using all analyzers
    
    Args:
        symbol (str): Coin symbol (e.g., 'BTC/USDT')
        timeframe (str): Timeframe for analysis
        limit (int): Number of candles to fetch
        
    Returns:
        dict: Analysis results
    """
    # Use environment variables if not specified
    timeframe = timeframe or os.getenv('DEFAULT_TIMEFRAME', '1h')
    limit = limit or int(os.getenv('DEFAULT_LIMIT', '500'))
    
    # Format symbol for ccxt
    if '/' not in symbol:
        symbol = f"{symbol[:-4]}/{symbol[-4:]}"
    
    print(f"\n{'='*50}")
    print(f"Analyzing {symbol}...")
    print(f"{'='*50}")
    
    # Initialize analyzers
    data_fetcher = DataFetcher(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    technical_analyzer = TechnicalAnalyzer()
    fundamental_analyzer = FundamentalAnalyzer()
    volume_analyzer = VolumeProfileAnalyzer()
    signal_analyzer = SignalAnalyzer()
    
    # Fetch data
    print(f"\nFetching {symbol} data...")
    df = data_fetcher.get_historical_data(symbol, timeframe, periods=limit)
    if df is None or df.empty:
        print(f"Failed to fetch data for {symbol}")
        return None
    
    # Technical Analysis
    print("\nPerforming Technical Analysis...")
    df = technical_analyzer.analyze_trend(df)
    df = technical_analyzer.analyze_momentum(df)
    df = technical_analyzer.analyze_volatility(df)
    df = technical_analyzer.detect_candle_patterns(df)
    
    # Get Fibonacci levels
    fib_levels = technical_analyzer.get_fibonacci_levels(df)
    
    # Volume Profile Analysis
    print("\nPerforming Volume Profile Analysis...")
    volume_analysis = volume_analyzer.analyze(df)
    volume_levels = {
        'poc': volume_analysis['vpoc'],
        'value_area_high': volume_analysis['vah'],
        'value_area_low': volume_analysis['val']
    }
    
    # Fundamental Analysis
    print("\nPerforming Fundamental Analysis...")
    df = fundamental_analyzer.analyze_volume(df)
    df = fundamental_analyzer.analyze_trend(df)
    fundamental_metrics = fundamental_analyzer.get_fundamental_metrics(df)
    
    # Signal Analysis
    print("\nGenerating Trading Signals...")
    df = signal_analyzer.analyze_data(df)
    signals_df = signal_analyzer.detect_signals(df)
    
    # Convert signals DataFrame to list of dictionaries
    signals = []
    if not signals_df.empty:
        signal_rows = signals_df[signals_df['pre_pump_signal']]
        for _, row in signal_rows.iterrows():
            signals.append({
                'type': 'Pre-Pump',
                'strength': row['signal_strength'],
                'price': row['close'],
                'volume': row['volume'],
                'patterns': [
                    pattern for pattern in ['hammer', 'morning_star', 'engulfing', 'doji']
                    if row.get(pattern, False)
                ]
            })
    
    # Prepare results
    results = {
        'symbol': symbol,
        'current_price': df['close'].iloc[-1],
        'technical_indicators': {
            'trend_strength': df['trend_strength'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'macd_signal': df['macd_signal'].iloc[-1],
            'bb_upper': df['bb_upper'].iloc[-1],
            'bb_lower': df['bb_lower'].iloc[-1],
            'atr': df['atr'].iloc[-1]
        },
        'fibonacci_levels': fib_levels,
        'volume_profile': volume_levels,
        'fundamental_metrics': fundamental_metrics,
        'signals': signals
    }
    
    return results

def print_analysis_results(results):
    """Print analysis results in a formatted way"""
    if results is None:
        return
    
    print(f"\n{'='*50}")
    print(f"Analysis Results for {results['symbol']}")
    print(f"{'='*50}")
    
    print(f"\nCurrent Price: ${results['current_price']:,.2f}")
    
    print("\nTechnical Indicators:")
    tech = results['technical_indicators']
    print(f"  Trend Strength: {tech['trend_strength']:.2f}")
    print(f"  RSI: {tech['rsi']:.2f}")
    print(f"  MACD: {tech['macd']:.2f}")
    print(f"  MACD Signal: {tech['macd_signal']:.2f}")
    print(f"  Bollinger Bands: {tech['bb_lower']:.2f} - {tech['bb_upper']:.2f}")
    print(f"  ATR: {tech['atr']:.2f}")
    
    print("\nFibonacci Levels:")
    fib = results['fibonacci_levels']
    print("  Retracement Levels:")
    for level, price in fib['retracement'].items():
        print(f"    {level}: ${price:,.2f}")
    print("  Extension Levels:")
    for level, price in fib['extension'].items():
        print(f"    {level}: ${price:,.2f}")
    print(f"  Current Position: {fib['position']:.3f}")
    
    print("\nVolume Profile:")
    vol = results['volume_profile']
    print(f"  POC: ${vol['poc']:,.2f}")
    print(f"  Value Area High: ${vol['value_area_high']:,.2f}")
    print(f"  Value Area Low: ${vol['value_area_low']:,.2f}")
    
    print("\nFundamental Metrics:")
    fund = results['fundamental_metrics']
    print(f"  Market Cap: ${fund['market_cap']:,.2f}")
    print(f"  Liquidity 24h: ${fund['liquidity_24h']:,.2f}")
    print(f"  Volume Trend: {fund['volume_trend']:.2f}")
    print(f"  Fundamental Score: {fund['fundamental_score']:.2f}")
    print(f"  Volume Score: {fund['volume_score']:.2f}")
    
    print("\nTrading Signals:")
    signals = results['signals']
    if signals:
        for signal in signals:
            print(f"  {signal['type']} Signal:")
            print(f"    Strength: {signal['strength']:.2f}")
            print(f"    Price: ${signal['price']:,.2f}")
            print(f"    Volume: {signal['volume']:,.2f}")
            print(f"    Patterns: {', '.join(signal['patterns'])}")
    else:
        print("  No significant signals detected")

def main():
    """Main function to analyze multiple coins"""
    # Get symbols from environment variable
    default_symbols = os.getenv('DEFAULT_SYMBOLS', 'BTCUSDT,ETHUSDT,DOGEUSDT,TRUMPUSDT')
    coins = default_symbols.split(',')
    
    # Analyze each coin
    for coin in coins:
        results = analyze_coin(coin)
        print_analysis_results(results)

if __name__ == "__main__":
    main() 