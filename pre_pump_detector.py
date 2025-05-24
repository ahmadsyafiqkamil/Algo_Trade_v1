import pandas as pd
from datetime import datetime
from libs.data_fetcher import DataFetcher
from libs.signal_analyzer import SignalAnalyzer
from libs.pump_ranking_analyzer import PumpRankingAnalyzer

class PrePumpDetector:
    def __init__(self, api_key, api_secret):
        """
        Initialize PrePumpDetector with DataFetcher, SignalAnalyzer, and PumpRankingAnalyzer
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
        """
        self.data_fetcher = DataFetcher(api_key, api_secret)
        self.signal_analyzer = SignalAnalyzer()
        self.ranking_analyzer = PumpRankingAnalyzer()
    
    def analyze_symbol(self, symbol, timeframe='1h'):
        """
        Analyze a symbol for pre-pump signals and ranking
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe
            
        Returns:
            tuple: (DataFrame with analysis results, ranking analysis)
        """
        # Fetch data using DataFetcher
        df = self.data_fetcher.get_historical_data(symbol, timeframe)
        if df is None:
            return None, None
        
        # Perform technical analysis
        df = self.signal_analyzer.analyze_data(df)
        
        # Detect signals
        df = self.signal_analyzer.detect_signals(df)
        
        # Calculate pump ranking
        ranking_analysis = self.ranking_analyzer.analyze_symbol(df)
        
        return df, ranking_analysis

def main():
    # Your API credentials
    api_key = "jd9uzMfTNlHAsRwzFTpWsepPQ5lcAd4uyD0GFKW0ddZLo4sqPXMIeqN3hjH4KdPi"
    api_secret = "aKINVDzGQFQ8L46fLPISjGK5RiamDtiLIKF6JatXKrTkQTPYUG3ctLAmcRgpqCOc"
    
    # Initialize detector
    detector = PrePumpDetector(api_key, api_secret)
    
    # Get all USDT pairs
    print("Fetching all USDT pairs from Binance...")
    symbols = detector.data_fetcher.get_all_usdt_pairs()
    
    if not symbols:
        print("No USDT pairs found or error occurred")
        return
        
    print(f"Found {len(symbols)} USDT pairs")
    
    # Create a list to store all signals and rankings
    all_signals = []
    all_rankings = []
    
    # Analyze each symbol
    for i, symbol in enumerate(symbols, 1):
        print(f"\nAnalyzing {symbol} ({i}/{len(symbols)})...")
        
        # Analyze symbol
        df, ranking = detector.analyze_symbol(symbol)
        
        if df is not None and ranking is not None:
            # Get the latest signals
            latest_signals = df[df['pre_pump_signal']].tail(5)
            
            if not latest_signals.empty:
                print(f"\nPre-pump signals detected for {symbol}:")
                for _, signal in latest_signals.iterrows():
                    # Format and print signal information
                    signal_info = detector.signal_analyzer.format_signal_output(signal)
                    for key, value in signal_info.items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                    print("---")
                
                # Add signals to the list
                latest_signals['symbol'] = symbol
                all_signals.append(latest_signals)
                
                # Add ranking to the list
                ranking['symbol'] = symbol
                all_rankings.append(ranking)
    
    # Save all signals to CSV
    if all_signals:
        combined_signals = pd.concat(all_signals)
        filename = detector.signal_analyzer.save_signals_to_csv(combined_signals)
        print(f"\nAll signals saved to {filename}")
        
        # Save rankings to CSV and display sorted rankings
        rankings_df = pd.DataFrame(all_rankings)
        
        # Sort rankings by total score in descending order
        rankings_df['total_score'] = rankings_df['scores'].apply(lambda x: x['total_score'])
        rankings_df = rankings_df.sort_values(by='total_score', ascending=False)
        
        # Display rankings
        print("\n=== RANKINGS FROM HIGHEST TO LOWEST ===")
        for idx, (_, row) in enumerate(rankings_df.iterrows(), 1):
            print(f"\nRank #{idx}: {row['symbol']}")
            print(detector.ranking_analyzer.format_ranking_output(row))
            print("-" * 50)
        
        # Save to CSV
        rankings_filename = f"pump_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rankings_df.to_csv(rankings_filename, index=False)
        print(f"\nAll rankings saved to {rankings_filename}")
    else:
        print("\nNo pre-pump signals detected for any symbol")

if __name__ == "__main__":
    main() 