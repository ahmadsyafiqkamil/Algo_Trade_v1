import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class VolumeProfileAnalyzer:
    def __init__(self, num_bins: int = 24):
        """
        Initialize VolumeProfileAnalyzer
        
        Args:
            num_bins (int): Number of price bins for volume profile calculation
        """
        self.num_bins = num_bins
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Calculate Volume Profile and related indicators
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Tuple containing:
            - price_bins: Array of price levels
            - volume_profile: Array of volume at each price level
            - vpoc: Volume Point of Control price
            - vah: Value Area High
            - val: Value Area Low
        """
        # Create price bins
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / self.num_bins
        price_bins = np.arange(df['low'].min(), df['high'].max() + bin_size, bin_size)
        
        # Calculate volume profile
        volume_profile = np.zeros(len(price_bins) - 1)
        for i in range(len(df)):
            low_idx = np.searchsorted(price_bins, df['low'].iloc[i])
            high_idx = np.searchsorted(price_bins, df['high'].iloc[i])
            volume = df['volume'].iloc[i]
            
            # Distribute volume across price bins
            for j in range(low_idx, high_idx):
                if j < len(volume_profile):
                    volume_profile[j] += volume
        
        # Find VPOC (Volume Point of Control)
        vpoc_idx = np.argmax(volume_profile)
        vpoc = (price_bins[vpoc_idx] + price_bins[vpoc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        # Find Value Area High and Low
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        included_bins = set()
        
        for idx in sorted_indices:
            cumulative_volume += volume_profile[idx]
            included_bins.add(idx)
            if cumulative_volume >= target_volume:
                break
        
        included_bins = sorted(list(included_bins))
        val = price_bins[min(included_bins)]
        vah = price_bins[max(included_bins) + 1]
        
        return price_bins, volume_profile, vpoc, vah, val
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform complete volume profile analysis
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Dict containing all volume profile indicators
        """
        # Calculate VWAP
        df['vwap'] = self.calculate_vwap(df)
        
        # Calculate Volume Profile and related indicators
        price_bins, volume_profile, vpoc, vah, val = self.calculate_volume_profile(df)
        
        # Calculate current price position relative to Value Area
        current_price = df['close'].iloc[-1]
        va_position = (current_price - val) / (vah - val) if vah != val else 0.5
        
        return {
            'vwap': df['vwap'].iloc[-1],
            'vpoc': vpoc,
            'vah': vah,
            'val': val,
            'va_position': va_position,
            'price_bins': price_bins,
            'volume_profile': volume_profile
        }
    
    def get_visible_range_volume_profile(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Calculate Visible Range Volume Profile for recent price action
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            lookback (int): Number of periods to look back
            
        Returns:
            Dict containing VRVP analysis
        """
        recent_df = df.tail(lookback)
        return self.analyze(recent_df) 