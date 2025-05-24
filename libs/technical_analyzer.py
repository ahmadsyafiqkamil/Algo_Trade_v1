import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, MFIIndicator, ChaikinMoneyFlowIndicator

class TechnicalAnalyzer:
    def __init__(self):
        """Initialize TechnicalAnalyzer with default parameters"""
        # Indicator parameters
        self.rsi_period = 14
        self.stoch_period = 14
        self.stoch_smooth = 3
        self.bb_period = 20
        self.bb_std = 2
        self.atr_period = 14
        self.vwap_period = 14
        self.mfi_period = 14
        self.cmf_period = 20
        
        # Fibonacci levels
        self.fib_retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        self.fib_extension_levels = [1.272, 1.618, 2.618, 3.618, 4.236]
        
        # Scoring weights
        self.weights = {
            'trend': 0.30,      # 30%
            'momentum': 0.25,    # 25%
            'volatility': 0.20,  # 20%
            'volume': 0.15,      # 15%
            'patterns': 0.10     # 10%
        }

    def analyze_trend(self, df):
        """
        Analyze price trend using multiple indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with trend indicators
        """
        # Moving Averages
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX for trend strength
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # Fibonacci levels
        df = self.calculate_fibonacci_levels(df)
        
        # Trend Strength
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        return df

    def calculate_fibonacci_levels(self, df, window=20):
        """
        Calculate Fibonacci Retracement and Extension levels
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            window (int): Window size for finding swing highs and lows
            
        Returns:
            pandas.DataFrame: DataFrame with Fibonacci levels
        """
        # Find swing highs and lows
        df['swing_high'] = df['high'].rolling(window=window, center=True).max()
        df['swing_low'] = df['low'].rolling(window=window, center=True).min()
        
        # Calculate price range
        df['price_range'] = df['swing_high'] - df['swing_low']
        
        # Calculate Fibonacci Retracement levels
        for level in self.fib_retracement_levels:
            if level == 0:
                df[f'fib_retracement_{int(level*1000)}'] = df['swing_high']
            elif level == 1:
                df[f'fib_retracement_{int(level*1000)}'] = df['swing_low']
            else:
                df[f'fib_retracement_{int(level*1000)}'] = df['swing_high'] - (df['price_range'] * level)
        
        # Calculate Fibonacci Extension levels
        for level in self.fib_extension_levels:
            df[f'fib_extension_{int(level*1000)}'] = df['swing_low'] - (df['price_range'] * (level - 1))
        
        # Calculate current Fibonacci position
        df['fib_position'] = self._calculate_fib_position(df)
        
        return df

    def _calculate_fib_position(self, df):
        """
        Calculate current price position relative to Fibonacci levels
        
        Args:
            df (pandas.DataFrame): DataFrame with Fibonacci levels
            
        Returns:
            float: Position relative to Fibonacci levels (0-1)
        """
        current_price = df['close'].iloc[-1]
        swing_high = df['swing_high'].iloc[-1]
        swing_low = df['swing_low'].iloc[-1]
        
        if current_price > swing_high:
            # Price is above swing high, calculate extension position
            extension_range = current_price - swing_high
            total_range = swing_high - swing_low
            return 1 + (extension_range / total_range)
        elif current_price < swing_low:
            # Price is below swing low, calculate extension position
            extension_range = swing_low - current_price
            total_range = swing_high - swing_low
            return -(extension_range / total_range)
        else:
            # Price is between swing high and low, calculate retracement position
            return (swing_high - current_price) / (swing_high - swing_low)

    def get_fibonacci_levels(self, df):
        """
        Get current Fibonacci levels
        
        Args:
            df (pandas.DataFrame): DataFrame with Fibonacci levels
            
        Returns:
            dict: Current Fibonacci levels
        """
        current_price = df['close'].iloc[-1]
        levels = {
            'retracement': {},
            'extension': {},
            'position': df['fib_position'].iloc[-1]
        }
        
        # Get retracement levels
        for level in self.fib_retracement_levels:
            level_key = f'fib_retracement_{int(level*1000)}'
            if level_key in df.columns:
                levels['retracement'][f'{level:.3f}'] = df[level_key].iloc[-1]
        
        # Get extension levels
        for level in self.fib_extension_levels:
            level_key = f'fib_extension_{int(level*1000)}'
            if level_key in df.columns:
                levels['extension'][f'{level:.3f}'] = df[level_key].iloc[-1]
        
        return levels

    def analyze_momentum(self, df):
        """
        Analyze price momentum using multiple indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with momentum indicators
        """
        # RSI
        df['rsi'] = RSIIndicator(df['close'], window=self.rsi_period).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], 
                                   window=self.stoch_period, 
                                   smooth_window=self.stoch_smooth)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Rate of Change
        df['roc'] = ROCIndicator(df['close']).roc()
        
        return df

    def analyze_volatility(self, df):
        """
        Analyze price volatility using multiple indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with volatility indicators
        """
        # Bollinger Bands
        bollinger = BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Average True Range
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 
                                   window=self.atr_period).average_true_range()
        
        # Donchian Channel
        donchian = DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = donchian.donchian_channel_hband()
        df['dc_lower'] = donchian.donchian_channel_lband()
        df['dc_middle'] = donchian.donchian_channel_mband()
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df

    def analyze_volume(self, df):
        """
        Analyze volume patterns
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with volume analysis
        """
        # Volume Moving Average
        df['volume_sma_20'] = SMAIndicator(df['volume'], window=20).sma_indicator()
        
        # Volume trend
        df['volume_trend'] = df['volume'] / df['volume_sma_20']
        
        # Volume price trend
        df['volume_price_trend'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'],
            -df['volume']
        )
        
        # VWAP
        df['vwap'] = VolumeWeightedAveragePrice(
            df['high'], df['low'], df['close'], df['volume'],
            window=self.vwap_period
        ).volume_weighted_average_price()
        
        # Money Flow Index
        df['mfi'] = MFIIndicator(
            df['high'], df['low'], df['close'], df['volume'],
            window=self.mfi_period
        ).money_flow_index()
        
        # Chaikin Money Flow
        df['cmf'] = ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume'],
            window=self.cmf_period
        ).chaikin_money_flow()
        
        return df

    def _calculate_trend_strength(self, df):
        """
        Calculate trend strength based on multiple indicators
        
        Returns:
            float: Trend strength score (0-100)
        """
        strength = 0
        
        # Price above/below moving averages
        if df['close'].iloc[-1] > df['sma_20'].iloc[-1]:
            strength += 20
        if df['close'].iloc[-1] > df['sma_50'].iloc[-1]:
            strength += 20
        if df['close'].iloc[-1] > df['sma_200'].iloc[-1]:
            strength += 20
            
        # MACD signal
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            strength += 20
            
        # Moving average alignment
        if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]:
            strength += 20
            
        return strength

    def detect_candle_patterns(self, df):
        """
        Detect common candle patterns
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with pattern detection results
        """
        # Calculate body and shadows
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_size'] = abs(df['body'])
        
        # Detect patterns
        df['doji'] = self._is_doji(df)
        df['hammer'] = self._is_hammer(df)
        df['shooting_star'] = self._is_shooting_star(df)
        df['engulfing'] = self._is_engulfing(df)
        df['morning_star'] = self._is_morning_star(df)
        df['evening_star'] = self._is_evening_star(df)
        
        return df

    def _is_doji(self, df):
        """Detect doji pattern"""
        return (abs(df['body']) <= 0.1 * (df['high'] - df['low']))

    def _is_hammer(self, df):
        """Detect hammer pattern"""
        return (
            (df['lower_shadow'] > 2 * df['body_size']) &
            (df['upper_shadow'] < df['body_size']) &
            (df['body'] > 0)
        )

    def _is_shooting_star(self, df):
        """Detect shooting star pattern"""
        return (
            (df['upper_shadow'] > 2 * df['body_size']) &
            (df['lower_shadow'] < df['body_size']) &
            (df['body'] < 0)
        )

    def _is_engulfing(self, df):
        """Detect bullish/bearish engulfing pattern"""
        bullish_engulfing = (
            (df['body'].shift(1) < 0) &
            (df['body'] > 0) &
            (df['body_size'] > df['body_size'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )
        
        bearish_engulfing = (
            (df['body'].shift(1) > 0) &
            (df['body'] < 0) &
            (df['body_size'] > df['body_size'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )
        
        return pd.Series(
            np.where(bullish_engulfing, 1,
                    np.where(bearish_engulfing, -1, 0)),
            index=df.index
        )

    def _is_morning_star(self, df):
        """Detect morning star pattern"""
        return (
            (df['body'].shift(2) < 0) &
            (abs(df['body'].shift(1)) < 0.3 * df['body_size'].shift(2)) &
            (df['body'] > 0) &
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)
        )

    def _is_evening_star(self, df):
        """Detect evening star pattern"""
        return (
            (df['body'].shift(2) > 0) &
            (abs(df['body'].shift(1)) < 0.3 * df['body_size'].shift(2)) &
            (df['body'] < 0) &
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)
        )

    def get_support_resistance(self, df, window=20):
        """
        Calculate support and resistance levels
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            window (int): Window size for local extrema
            
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        # Find local minima and maxima
        df['local_min'] = df['low'].rolling(window=window, center=True).min()
        df['local_max'] = df['high'].rolling(window=window, center=True).max()
        
        # Get current support and resistance
        current_price = df['close'].iloc[-1]
        support_levels = df[df['local_min'] < current_price]['local_min'].unique()
        resistance_levels = df[df['local_max'] > current_price]['local_max'].unique()
        
        return support_levels, resistance_levels 