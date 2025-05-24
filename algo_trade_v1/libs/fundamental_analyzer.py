import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class FundamentalAnalyzer:
    def __init__(self):
        """Initialize FundamentalAnalyzer with default parameters"""
        # Fundamental analysis thresholds
        self.volume_threshold = 1.5  # Minimum volume multiplier
        self.market_cap_threshold = 1000000  # Minimum market cap in USD
        self.liquidity_threshold = 100000  # Minimum 24h volume in USD
        
        # Scoring weights
        self.weights = {
            'market_cap': 0.30,      # 30%
            'liquidity': 0.25,       # 25%
            'volume': 0.20,          # 20%
            'holders': 0.15,         # 15%
            'social_metrics': 0.10    # 10%
        }
    
    def analyze_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze fundamental trend indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with fundamental trend analysis
        """
        # Calculate market metrics
        df['market_cap'] = df['close'] * df['volume']  # Simplified market cap
        df['liquidity_24h'] = df['volume'] * df['close']  # 24h volume in USD
        
        # Calculate trend metrics
        df['market_cap_trend'] = df['market_cap'].pct_change(periods=20)
        df['liquidity_trend'] = df['liquidity_24h'].pct_change(periods=20)
        
        # Calculate fundamental score
        df['fundamental_score'] = self._calculate_fundamental_score(df)
        
        return df
    
    def analyze_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze fundamental volume indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with fundamental volume analysis
        """
        # Calculate volume metrics
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_std_20'] = df['volume'].rolling(window=20).std()
        
        # Calculate volume trend
        df['volume_trend'] = df['volume'] / df['volume_ma_20']
        
        # Calculate volume score
        df['volume_score'] = self._calculate_volume_score(df)
        
        return df
    
    def _calculate_fundamental_score(self, df: pd.DataFrame) -> float:
        """
        Calculate fundamental score based on multiple metrics
        
        Args:
            df (pandas.DataFrame): DataFrame with fundamental metrics
            
        Returns:
            float: Fundamental score (0-1)
        """
        score = 0
        
        # Market cap score
        if df['market_cap'].iloc[-1] > self.market_cap_threshold:
            score += self.weights['market_cap']
            
        # Liquidity score
        if df['liquidity_24h'].iloc[-1] > self.liquidity_threshold:
            score += self.weights['liquidity']
            
        # Volume trend score
        if df['volume_trend'].iloc[-1] > self.volume_threshold:
            score += self.weights['volume']
            
        # Additional metrics can be added here
        # For example: holder distribution, social metrics, etc.
        
        return min(score, 1.0)
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calculate volume score based on volume analysis
        
        Args:
            df (pandas.DataFrame): DataFrame with volume metrics
            
        Returns:
            float: Volume score (0-1)
        """
        score = 0
        
        # Volume trend score
        if df['volume_trend'].iloc[-1] >= 2.0:
            score += 0.4
        elif df['volume_trend'].iloc[-1] >= 1.5:
            score += 0.3
        elif df['volume_trend'].iloc[-1] >= 1.2:
            score += 0.2
            
        # Volume consistency score
        if df['volume_std_20'].iloc[-1] < df['volume_ma_20'].iloc[-1] * 0.5:
            score += 0.3
        elif df['volume_std_20'].iloc[-1] < df['volume_ma_20'].iloc[-1]:
            score += 0.2
            
        return min(score, 1.0)
    
    def get_fundamental_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Get current fundamental metrics
        
        Args:
            df (pandas.DataFrame): DataFrame with fundamental metrics
            
        Returns:
            dict: Current fundamental metrics
        """
        return {
            'market_cap': df['market_cap'].iloc[-1],
            'liquidity_24h': df['liquidity_24h'].iloc[-1],
            'volume_trend': df['volume_trend'].iloc[-1],
            'fundamental_score': df['fundamental_score'].iloc[-1],
            'volume_score': df['volume_score'].iloc[-1]
        } 