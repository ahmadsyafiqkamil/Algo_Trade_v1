import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from .technical_analyzer import TechnicalAnalyzer
from .volume_profile_analyzer import VolumeProfileAnalyzer
from .fundamental_analyzer import FundamentalAnalyzer

class SignalAnalyzer:
    def __init__(self):
        """Initialize SignalAnalyzer with analyzers"""
        self.technical_analyzer = TechnicalAnalyzer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        
        # Signal thresholds
        self.trend_threshold = 60  # Minimum trend strength
        self.volume_threshold = 1.5  # Minimum volume multiplier
        self.rsi_oversold = 30  # RSI oversold threshold
        self.rsi_overbought = 70  # RSI overbought threshold
        self.macd_threshold = 0  # MACD signal threshold
        self.volatility_threshold = 0.02  # 2% minimum volatility
        
        # Scoring weights
        self.weights = {
            'technical': 0.40,      # 40%
            'fundamental': 0.30,    # 30%
            'volume_profile': 0.20, # 20%
            'patterns': 0.10        # 10%
        }
    
    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete analysis on the data
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with all analysis results
        """
        # Technical analysis
        df = self.technical_analyzer.analyze_trend(df)
        df = self.technical_analyzer.detect_candle_patterns(df)
        df = self.technical_analyzer.analyze_volume(df)
        
        # Fundamental analysis
        df = self.fundamental_analyzer.analyze_trend(df)
        df = self.fundamental_analyzer.analyze_volume(df)
        
        # Volume profile analysis
        volume_profile = self.volume_analyzer.analyze(df)
        df['vwap'] = volume_profile['vwap']
        df['vpoc'] = volume_profile['vpoc']
        df['vah'] = volume_profile['vah']
        df['val'] = volume_profile['val']
        df['va_position'] = volume_profile['va_position']
        
        return df

    def calculate_signal_strength(self, row: pd.Series) -> float:
        """
        Calculate the strength of pre-pump signal (0-100%)
        
        Args:
            row: DataFrame row containing indicator values
            
        Returns:
            float: Signal strength percentage
        """
        # Initialize score components
        technical_score = self._calculate_technical_score(row)
        fundamental_score = self._calculate_fundamental_score(row)
        volume_profile_score = self._calculate_vp_score(row)
        pattern_score = self._calculate_pattern_score(row)
        
        # Calculate total score (weighted average)
        total_score = (
            technical_score * self.weights['technical'] +
            fundamental_score * self.weights['fundamental'] +
            volume_profile_score * self.weights['volume_profile'] +
            pattern_score * self.weights['patterns']
        )
        
        return total_score * 100  # Convert to percentage
    
    def _calculate_technical_score(self, row: pd.Series) -> float:
        """Calculate technical score based on multiple indicators"""
        score = 0
        
        # Trend strength
        if row['trend_strength'] > 80:
            score += 0.4
        elif row['trend_strength'] > 60:
            score += 0.3
            
        # RSI
        if row['rsi'] < self.rsi_oversold:
            score += 0.3
        elif row['rsi'] < 40:
            score += 0.2
            
        # MACD
        if row['macd'] > row['macd_signal'] and row['macd'] > 0:
            score += 0.3
        elif row['macd'] > row['macd_signal']:
            score += 0.2
            
        return min(score, 1.0)
    
    def _calculate_fundamental_score(self, row: pd.Series) -> float:
        """Calculate fundamental score"""
        return row['fundamental_score']
    
    def _calculate_vp_score(self, row: pd.Series) -> float:
        """Calculate volume profile score"""
        score = 0
        
        if row['va_position'] < 0.2:  # Price below Value Area
            score += 0.4
        elif row['va_position'] < 0.4:
            score += 0.3
        elif row['va_position'] > 0.8:  # Price above Value Area
            score += 0.2
            
        # VWAP position
        if row['close'] > row['vwap']:
            score += 0.2
            
        return min(score, 1.0)
    
    def _calculate_pattern_score(self, row: pd.Series) -> float:
        """Calculate pattern score based on candlestick patterns"""
        score = 0
        
        if row['hammer'] or row['morning_star']:
            score += 0.4
        elif row['engulfing'] == 1:  # Bullish engulfing
            score += 0.3
        elif row['doji']:
            score += 0.2
            
        return min(score, 1.0)

    def detect_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect pre-pump signals based on technical and fundamental analysis
        
        Args:
            df (pandas.DataFrame): DataFrame with analysis results
            
        Returns:
            pandas.DataFrame: DataFrame with pre-pump signals
        """
        # Create conditions for pre-pump detection
        conditions = (
            # Technical conditions
            (df['trend_strength'] >= self.trend_threshold) &
            (df['rsi'] < self.rsi_overbought) &
            (df['macd'] > df['macd_signal']) &
            
            # Volume conditions
            (df['volume_trend'] >= self.volume_threshold) &
            (df['volume'] > df['volume'].rolling(20).mean()) &
            
            # Pattern conditions
            (
                (df['hammer']) |
                (df['morning_star']) |
                (df['engulfing'] == 1)
            ) &
            
            # Volatility check
            (df['volatility'] >= self.volatility_threshold) &
            
            # Volume profile conditions
            (df['va_position'] < 0.4) &
            (df['close'] > df['vwap']) &
            
            # Fundamental conditions
            (df['fundamental_score'] > 0.5)
        )
        
        # Add pre-pump signal column
        df['pre_pump_signal'] = conditions
        
        # Calculate signal strength for rows with pre-pump signals
        df['signal_strength'] = 0.0
        signal_rows = df[df['pre_pump_signal']].index
        for idx in signal_rows:
            df.loc[idx, 'signal_strength'] = self.calculate_signal_strength(df.loc[idx])
        
        return df

    def format_signal_output(self, signal: pd.Series) -> Dict:
        """
        Format signal information for display
        
        Args:
            signal: DataFrame row containing signal data
            
        Returns:
            dict: Formatted signal information
        """
        output = {
            'time': signal['timestamp'],
            'price': signal['close'],
            'trend_strength': f"{signal['trend_strength']:.1f}%",
            'volume_trend': f"{signal['volume_trend']:.2f}x",
            'signal_strength': f"{signal['signal_strength']:.1f}%",
            'rsi': f"{signal['rsi']:.1f}",
            'macd': f"{signal['macd']:.8f}",
            'volatility': f"{signal['volatility']*100:.1f}%",
            'vwap': f"{signal['vwap']:.8f}",
            'vpoc': f"{signal['vpoc']:.8f}",
            'vah': f"{signal['vah']:.8f}",
            'val': f"{signal['val']:.8f}",
            'va_position': f"{signal['va_position']:.2f}",
            'fundamental_score': f"{signal['fundamental_score']:.2f}"
        }
        
        # Add detected patterns
        patterns = []
        if signal['hammer']: patterns.append("Hammer")
        if signal['morning_star']: patterns.append("Morning Star")
        if signal['engulfing'] == 1: patterns.append("Bullish Engulfing")
        if signal['doji']: patterns.append("Doji")
        if patterns:
            output['patterns'] = ', '.join(patterns)
            
        return output

    def save_signals_to_csv(self, signals: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save detected signals to CSV file
        
        Args:
            signals (pandas.DataFrame): DataFrame containing signals
            filename (str, optional): Custom filename. If None, generates timestamp-based name
            
        Returns:
            str: Name of the saved file
        """
        if filename is None:
            filename = f"all_pre_pump_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # Add metadata
        metadata = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signals': len(signals),
            'average_signal_strength': signals['signal_strength'].mean(),
            'min_signal_strength': signals['signal_strength'].min(),
            'max_signal_strength': signals['signal_strength'].max()
        }
        
        # Save signals
        signals.to_csv(filename, index=False)
        
        # Save metadata
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return filename 