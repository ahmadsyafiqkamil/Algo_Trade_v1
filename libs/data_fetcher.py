import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional, Union
import websocket
import threading
import logging
from functools import lru_cache
import ssl
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataFetcher:
    def __init__(self, api_key: str, api_secret: str, cache_dir: str = "cache"):
        """
        Initialize Binance connection and DataFetcher with enhanced features
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            cache_dir (str): Directory for caching data
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot market
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        
        # Setup cache
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize websocket
        self.ws = None
        self.ws_thread = None
        self.live_data = {}
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # WebSocket ping/pong
        self.last_ping_time = 0
        self.ping_interval = 30  # seconds
        
    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol to CCXT format (e.g., BTCUSDT -> BTC/USDT)
        
        Args:
            symbol (str): Raw symbol string
            
        Returns:
            str: Formatted symbol string
        """
        if '/' not in symbol:
            return f"{symbol[:-4]}/{symbol[-4:]}"
        return symbol
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API restrictions"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """Get cache file path for a symbol and timeframe"""
        return os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(symbol, timeframe)
        df.to_csv(cache_path, index=False)
        self.logger.info(f"Data cached for {symbol} {timeframe}")
    
    def _load_from_cache(self, symbol: str, timeframe: str, max_age_hours: int = 1) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists and is not too old"""
        cache_path = self._get_cache_path(symbol, timeframe)
        
        if not os.path.exists(cache_path):
            return None
            
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > (max_age_hours * 3600):
            return None
            
        try:
            df = pd.read_csv(cache_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.logger.info(f"Data loaded from cache for {symbol} {timeframe}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            return None
    
    @lru_cache(maxsize=100)
    def get_all_usdt_pairs(self) -> List[str]:
        """
        Get all available USDT trading pairs from Binance with caching
        
        Returns:
            list: List of USDT trading pairs
        """
        try:
            self._rate_limit()
            markets = self.exchange.load_markets()
            
            usdt_pairs = [symbol for symbol in markets.keys() 
                         if symbol.endswith('/USDT') and 
                         markets[symbol]['active'] and
                         markets[symbol]['futures']]
            
            return sorted(usdt_pairs)
        except Exception as e:
            self.logger.error(f"Error fetching USDT pairs: {str(e)}")
            return []
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                          periods: int = 1000, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a symbol with caching support
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT' or 'BTC/USDT')
            timeframe (str): Candle timeframe (e.g., '1m', '5m', '1h', '1d')
            periods (int): Number of periods to fetch
            use_cache (bool): Whether to use cached data
            
        Returns:
            pandas.DataFrame: DataFrame with historical data
        """
        # Format symbol
        symbol = self._format_symbol(symbol)
        
        if use_cache:
            cached_data = self._load_from_cache(symbol, timeframe)
            if cached_data is not None:
                return cached_data
        
        try:
            self._rate_limit()
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=periods)
            
            if not ohlcv:
                self.logger.warning(f"No data received for {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save to cache
            self._save_to_cache(df, symbol, timeframe)
            
            return df
        except ccxt.RateLimitExceeded:
            self.logger.warning("Rate limit exceeded, waiting before retry...")
            time.sleep(60)  # Wait for 1 minute
            return self.get_historical_data(symbol, timeframe, periods, use_cache)
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            return None
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def start_websocket(self, symbols: List[str], on_message=None):
        """
        Start websocket connection for real-time data with enhanced security
        
        Args:
            symbols (List[str]): List of symbols to subscribe to
            on_message (callable): Callback function for handling messages
        """
        def on_open(ws):
            self.logger.info("WebSocket connection opened")
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Subscribe to klines for each symbol
            for symbol in symbols:
                symbol = symbol.replace('/', '').lower()
                subscribe_message = {
                    "method": "SUBSCRIBE",
                    "params": [f"{symbol}@kline_1m"],
                    "id": 1
                }
                ws.send(json.dumps(subscribe_message))
            
            # Start ping thread
            self._start_ping_thread(ws)
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {str(error)}")
            self.is_connected = False
            
            # Handle reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                self.logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
                time.sleep(self.reconnect_delay)
                self._reconnect_websocket(symbols, on_message)
            else:
                self.logger.error("Max reconnection attempts reached")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.is_connected = False
            
            # Handle reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                self.logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
                time.sleep(self.reconnect_delay)
                self._reconnect_websocket(symbols, on_message)
        
        def on_message_wrapper(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                else:
                    self._handle_websocket_message(data)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding message: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
        
        def on_ping(ws, message):
            self.logger.debug("Ping received")
            ws.send_pong(message)
        
        def on_pong(ws, message):
            self.logger.debug("Pong received")
            self.last_ping_time = time.time()
        
        # Initialize WebSocket connection with SSL context
        websocket.enableTrace(True)
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        self.ws = websocket.WebSocketApp(
            "wss://fstream.binance.com/ws",
            on_open=on_open,
            on_message=on_message_wrapper,
            on_error=on_error,
            on_close=on_close,
            on_ping=on_ping,
            on_pong=on_pong,
            header={
                'User-Agent': 'Mozilla/5.0',
                'Origin': 'https://www.binance.com'
            }
        )
        
        # Start WebSocket in a separate thread with SSL context
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={
                'sslopt': {
                    'cert_reqs': ssl.CERT_REQUIRED,
                    'ca_certs': certifi.where(),
                    'check_hostname': True
                }
            }
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def _start_ping_thread(self, ws):
        """Start a thread to send periodic pings"""
        def ping_loop():
            while self.is_connected:
                try:
                    if time.time() - self.last_ping_time > self.ping_interval:
                        ws.send_ping()
                        self.last_ping_time = time.time()
                except Exception as e:
                    self.logger.error(f"Error sending ping: {str(e)}")
                time.sleep(1)
        
        ping_thread = threading.Thread(target=ping_loop)
        ping_thread.daemon = True
        ping_thread.start()
    
    def _reconnect_websocket(self, symbols: List[str], on_message=None):
        """Handle WebSocket reconnection"""
        self.stop_websocket()
        time.sleep(self.reconnect_delay)
        self.start_websocket(symbols, on_message)
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        self.is_connected = False
        self.logger.info("WebSocket connection stopped")
    
    def _handle_websocket_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        try:
            if 'k' in data:
                kline = data['k']
                symbol = data['s']
                
                # Update live data
                if symbol not in self.live_data:
                    self.live_data[symbol] = []
                
                self.live_data[symbol].append({
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                })
                
                # Keep only last 1000 candles
                if len(self.live_data[symbol]) > 1000:
                    self.live_data[symbol] = self.live_data[symbol][-1000:]
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def get_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get live data for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            pandas.DataFrame: DataFrame with live data
        """
        symbol = symbol.replace('/', '').upper()
        if symbol in self.live_data:
            return pd.DataFrame(self.live_data[symbol])
        return None

    def fetch_klines(self, symbol, timeframe='1h', limit=500):
        """
        Fetch klines (candlestick) data from Binance
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            timeframe (str): Kline interval (e.g., '1h', '4h', '1d')
            limit (int): Number of klines to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Format symbol for CCXT
            symbol = self._format_symbol(symbol)
            
            # Fetch OHLCV data from Binance
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Save to cache
            self._save_to_cache(df, symbol, timeframe)
            
            return df
            
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_klines(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Error fetching klines: {str(e)}")
            return None
            
    def fetch_ticker(self, symbol):
        """
        Fetch current ticker data for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            return None
            
    def fetch_order_book(self, symbol, limit=100):
        """
        Fetch order book data for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            limit (int): Number of orders to fetch
            
        Returns:
            dict: Order book data
        """
        try:
            depth = self.exchange.fetch_order_book(symbol)
            return depth
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            return None
            
    def fetch_recent_trades(self, symbol, limit=500):
        """
        Fetch recent trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            limit (int): Number of trades to fetch
            
        Returns:
            list: Recent trades
        """
        try:
            trades = self.exchange.fetch_recent_trades(symbol)
            return trades
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_recent_trades(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {str(e)}")
            return None
            
    def fetch_historical_trades(self, symbol, limit=500):
        """
        Fetch historical trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            limit (int): Number of trades to fetch
            
        Returns:
            list: Historical trades
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_historical_trades(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching historical trades: {str(e)}")
            return None
            
    def fetch_aggregate_trades(self, symbol, limit=500):
        """
        Fetch aggregate trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            limit (int): Number of trades to fetch
            
        Returns:
            list: Aggregate trades
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_aggregate_trades(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching aggregate trades: {str(e)}")
            return None
            
    def fetch_24h_ticker(self, symbol):
        """
        Fetch 24-hour ticker data for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: 24-hour ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_24h_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching 24h ticker: {str(e)}")
            return None
            
    def fetch_symbol_info(self, symbol):
        """
        Fetch symbol information
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Symbol information
        """
        try:
            info = self.exchange.fetch_symbol_info(symbol)
            return info
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait for 1 minute
            return self.fetch_symbol_info(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching symbol info: {str(e)}")
            return None 