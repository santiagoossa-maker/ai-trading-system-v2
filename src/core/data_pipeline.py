"""
Real-time Data Pipeline for MT5 Integration
Handles multi-asset, multi-timeframe data collection and processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import pickle
import yaml
import os
from datetime import datetime, timedelta

# Optional imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

# Asset lot sizes from requirements
LOTES = {
    "1HZ75V": 0.05, "R_75": 0.01, "R_100": 0.5, "1HZ100V": 0.2,
    "R_50": 4.0, "1HZ50V": 0.01, "R_25": 0.5, "R_10": 0.5,
    "1HZ10V": 0.5, "1HZ25V": 0.01, "stpRNG": 0.1, "stpRNG2": 0.1,
    "stpRNG3": 0.1, "stpRNG4": 0.1, "stpRNG5": 0.1
}

# Timeframe mappings (with fallback values when MT5 is not available)
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
else:
    # Fallback constants when MT5 is not available
    TIMEFRAME_MAP = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'M30': 30,
        'H1': 16385,
        'H4': 16388,
        'D1': 16408
    }

@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime

@dataclass
class BarData:
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime

class DataBuffer:
    """Thread-safe data buffer for storing historical data"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = {}
        self.lock = threading.RLock()
    
    def add_bar(self, symbol: str, timeframe: str, bar: BarData):
        """Add a new bar to the buffer"""
        with self.lock:
            key = f"{symbol}_{timeframe}"
            if key not in self.data:
                self.data[key] = deque(maxlen=self.max_size)
            self.data[key].append(bar)
    
    def get_bars(self, symbol: str, timeframe: str, count: int = None) -> List[BarData]:
        """Get bars for a symbol and timeframe"""
        with self.lock:
            key = f"{symbol}_{timeframe}"
            if key not in self.data:
                return []
            
            bars = list(self.data[key])
            if count:
                return bars[-count:]
            return bars
    
    def get_dataframe(self, symbol: str, timeframe: str, count: int = None) -> pd.DataFrame:
        """Get bars as a pandas DataFrame"""
        bars = self.get_bars(symbol, timeframe, count)
        if not bars:
            return pd.DataFrame()
        
        data = {
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars],
        }
        
        df = pd.DataFrame(data, index=[bar.timestamp for bar in bars])
        return df

class IndicatorCalculator:
    """Calculates technical indicators in parallel"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.cache = {}
        self.cache_duration = 60  # seconds
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, np.ndarray]:
        """Calculate multiple indicators for a dataframe"""
        if df.empty or len(df) < 50:
            return {}
        
        cache_key = f"{symbol}_{timeframe}_{len(df)}_{df.iloc[-1].name}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if current_time - cached_time < self.cache_duration:
                return cached_data
        
        try:
            import talib
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            indicators = {}
            
            # Moving averages
            for period in [8, 13, 21, 34, 50, 89]:
                if len(close) >= period:
                    indicators[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                    indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # Oscillators
            if len(close) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
            
            if len(close) >= 14:
                indicators['rsi'] = talib.RSI(close, timeperiod=14)
                indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
                indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
            
            # Stochastic
            if len(close) >= 14:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['stoch_k'] = slowk
                indicators['stoch_d'] = slowd
            
            # Cache the results
            self.cache[cache_key] = (indicators, current_time)
            
            # Clean old cache entries
            if len(self.cache) > 100:
                oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:20]
                for key in oldest_keys:
                    del self.cache[key]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}_{timeframe}: {str(e)}")
            return {}

class MT5Connection:
    """Manages MT5 connection and data retrieval"""
    
    def __init__(self, login: str = None, password: str = None, server: str = None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.symbols_info = {}
    
    def connect(self) -> bool:
        """Connect to MT5"""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 module not available")
            return False
            
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
                return False
            
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login failed, error code = {mt5.last_error()}")
                    return False
            
            self.connected = True
            logger.info("Connected to MT5")
            
            # Get symbol information
            self._load_symbols_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def _load_symbols_info(self):
        """Load symbol information"""
        try:
            for symbol in LOTES.keys():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    self.symbols_info[symbol] = {
                        'point': symbol_info.point,
                        'digits': symbol_info.digits,
                        'spread': symbol_info.spread,
                        'trade_contract_size': symbol_info.trade_contract_size,
                        'min_lot': symbol_info.volume_min,
                        'max_lot': symbol_info.volume_max,
                        'lot_step': symbol_info.volume_step
                    }
                else:
                    logger.warning(f"Could not get info for symbol {symbol}")
        except Exception as e:
            logger.error(f"Error loading symbols info: {str(e)}")
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 1000) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            if not self.connected:
                return pd.DataFrame()
            
            tf = TIMEFRAME_MAP.get(timeframe)
            if not tf:
                logger.error(f"Unknown timeframe: {timeframe}")
                return pd.DataFrame()
            
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None:
                logger.error(f"Failed to get rates for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_tick_data(self, symbol: str) -> Optional[TickData]:
        """Get latest tick data for a symbol"""
        try:
            if not self.connected:
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
            
            return TickData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                timestamp=datetime.fromtimestamp(tick.time)
            )
            
        except Exception as e:
            logger.error(f"Error getting tick data for {symbol}: {str(e)}")
            return None

class DataPipeline:
    """
    Main data pipeline that orchestrates data collection, processing and caching
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.mt5 = MT5Connection(
            login=self.config.get('mt5', {}).get('login'),
            password=self.config.get('mt5', {}).get('password'),
            server=self.config.get('mt5', {}).get('server')
        )
        
        self.data_buffer = DataBuffer(max_size=self.config.get('buffer_size', 10000))
        self.indicator_calculator = IndicatorCalculator(
            max_workers=self.config.get('max_workers', 4)
        )
        
        # Redis cache for real-time data
        self.redis_client = None
        self._init_redis()
        
        # Threading controls
        self.running = False
        self.threads = {}
        self.update_frequency = self.config.get('update_frequency', 1)  # seconds
        
        # Asset categories
        self.volatility_indices = ["R_75", "R_100", "R_50", "R_25", "R_10"]
        self.hz_indices = ["1HZ75V", "1HZ100V", "1HZ50V", "1HZ10V", "1HZ25V"]
        self.step_indices = ["stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"]
        self.all_symbols = list(LOTES.keys())
        
        # Timeframes to monitor
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'mt5': {
                'login': None,
                'password': None,
                'server': None
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'buffer_size': 10000,
            'max_workers': 4,
            'update_frequency': 1,
            'historical_days': 30,
            'cache_duration': 300
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        
        return default_config
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Caching disabled.")
            self.redis_client = None
            return
            
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {str(e)}. Caching disabled.")
            self.redis_client = None
    
    def start(self) -> bool:
        """Start the data pipeline"""
        try:
            # Connect to MT5
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5")
                return False
            
            self.running = True
            
            # Load initial historical data
            self._load_historical_data()
            
            # Start real-time data collection threads
            self._start_realtime_threads()
            
            logger.info("Data pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting data pipeline: {str(e)}")
            return False
    
    def stop(self):
        """Stop the data pipeline"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=5)
        
        # Disconnect from MT5
        self.mt5.disconnect()
        
        logger.info("Data pipeline stopped")
    
    def _load_historical_data(self):
        """Load historical data for all symbols and timeframes"""
        logger.info("Loading historical data...")
        
        def load_symbol_data(symbol, timeframe):
            try:
                df = self.mt5.get_historical_data(symbol, timeframe, count=1000)
                if not df.empty:
                    # Convert to BarData objects and store in buffer
                    for idx, row in df.iterrows():
                        bar = BarData(
                            symbol=symbol,
                            timeframe=timeframe,
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume'],
                            timestamp=idx
                        )
                        self.data_buffer.add_bar(symbol, timeframe, bar)
                    
                    # Cache in Redis
                    self._cache_dataframe(symbol, timeframe, df)
                    
                    logger.debug(f"Loaded {len(df)} bars for {symbol} {timeframe}")
                else:
                    logger.warning(f"No data for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol} {timeframe}: {str(e)}")
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            futures = []
            for symbol in self.all_symbols:
                for timeframe in self.timeframes:
                    future = executor.submit(load_symbol_data, symbol, timeframe)
                    futures.append(future)
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in historical data loading: {str(e)}")
        
        logger.info("Historical data loading completed")
    
    def _start_realtime_threads(self):
        """Start real-time data collection threads"""
        # Start tick data collection thread
        tick_thread = threading.Thread(target=self._tick_data_collector, daemon=True)
        tick_thread.start()
        self.threads['tick_collector'] = tick_thread
        
        # Start bar data update thread for each timeframe
        for timeframe in ['M1', 'M5']:  # Only monitor high-frequency timeframes in real-time
            bar_thread = threading.Thread(
                target=self._bar_data_collector, 
                args=(timeframe,), 
                daemon=True
            )
            bar_thread.start()
            self.threads[f'bar_collector_{timeframe}'] = bar_thread
        
        # Start indicator calculation thread
        indicator_thread = threading.Thread(target=self._indicator_calculator_worker, daemon=True)
        indicator_thread.start()
        self.threads['indicator_calculator'] = indicator_thread
    
    def _tick_data_collector(self):
        """Collect tick data for all symbols"""
        logger.info("Started tick data collector")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect ticks for all symbols in parallel
                with ThreadPoolExecutor(max_workers=len(self.all_symbols)) as executor:
                    futures = {
                        executor.submit(self.mt5.get_tick_data, symbol): symbol 
                        for symbol in self.all_symbols
                    }
                    
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            tick = future.result()
                            if tick:
                                self._cache_tick_data(tick)
                        except Exception as e:
                            logger.error(f"Error collecting tick for {symbol}: {str(e)}")
                
                # Maintain update frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_frequency - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in tick data collector: {str(e)}")
                time.sleep(1)
    
    def _bar_data_collector(self, timeframe: str):
        """Collect bar data for a specific timeframe"""
        logger.info(f"Started bar data collector for {timeframe}")
        
        # Calculate update interval based on timeframe
        update_intervals = {
            'M1': 60,   # 1 minute
            'M5': 300,  # 5 minutes
            'M15': 900, # 15 minutes
            'M30': 1800,# 30 minutes
            'H1': 3600, # 1 hour
            'H4': 14400,# 4 hours
            'D1': 86400 # 1 day
        }
        
        interval = update_intervals.get(timeframe, 300)
        
        while self.running:
            try:
                for symbol in self.all_symbols:
                    try:
                        # Get latest bar
                        df = self.mt5.get_historical_data(symbol, timeframe, count=1)
                        if not df.empty:
                            latest_row = df.iloc[-1]
                            
                            bar = BarData(
                                symbol=symbol,
                                timeframe=timeframe,
                                open=latest_row['open'],
                                high=latest_row['high'],
                                low=latest_row['low'],
                                close=latest_row['close'],
                                volume=latest_row['volume'],
                                timestamp=df.index[-1]
                            )
                            
                            # Check if this is a new bar
                            existing_bars = self.data_buffer.get_bars(symbol, timeframe, count=1)
                            if not existing_bars or existing_bars[-1].timestamp < bar.timestamp:
                                self.data_buffer.add_bar(symbol, timeframe, bar)
                                logger.debug(f"New bar added: {symbol} {timeframe}")
                    
                    except Exception as e:
                        logger.error(f"Error collecting bar data for {symbol} {timeframe}: {str(e)}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in bar data collector for {timeframe}: {str(e)}")
                time.sleep(60)
    
    def _indicator_calculator_worker(self):
        """Calculate indicators for all symbol-timeframe combinations"""
        logger.info("Started indicator calculator worker")
        
        while self.running:
            try:
                for symbol in self.all_symbols:
                    for timeframe in self.timeframes:
                        try:
                            df = self.data_buffer.get_dataframe(symbol, timeframe, count=200)
                            if not df.empty and len(df) >= 50:
                                indicators = self.indicator_calculator.calculate_indicators(df, symbol, timeframe)
                                if indicators:
                                    self._cache_indicators(symbol, timeframe, indicators)
                        except Exception as e:
                            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {str(e)}")
                
                time.sleep(30)  # Update indicators every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in indicator calculator worker: {str(e)}")
                time.sleep(60)
    
    def _cache_tick_data(self, tick: TickData):
        """Cache tick data in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"tick:{tick.symbol}"
            data = {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'timestamp': tick.timestamp.isoformat()
            }
            self.redis_client.setex(key, 60, pickle.dumps(data))  # 1 minute expiry
        except Exception as e:
            logger.error(f"Error caching tick data: {str(e)}")
    
    def _cache_dataframe(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Cache dataframe in Redis"""
        if not self.redis_client or df.empty:
            return
        
        try:
            key = f"bars:{symbol}:{timeframe}"
            self.redis_client.setex(key, 3600, pickle.dumps(df))  # 1 hour expiry
        except Exception as e:
            logger.error(f"Error caching dataframe: {str(e)}")
    
    def _cache_indicators(self, symbol: str, timeframe: str, indicators: Dict[str, np.ndarray]):
        """Cache indicators in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"indicators:{symbol}:{timeframe}"
            self.redis_client.setex(key, 1800, pickle.dumps(indicators))  # 30 minutes expiry
        except Exception as e:
            logger.error(f"Error caching indicators: {str(e)}")
    
    def get_latest_data(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get latest data for a symbol and timeframe"""
        try:
            # Try Redis cache first
            if self.redis_client:
                key = f"bars:{symbol}:{timeframe}"
                cached_data = self.redis_client.get(key)
                if cached_data:
                    df = pickle.loads(cached_data)
                    return df.tail(count) if len(df) > count else df
            
            # Fallback to buffer
            return self.data_buffer.get_dataframe(symbol, timeframe, count)
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        if timeframes is None:
            timeframes = self.timeframes
        
        data = {}
        for timeframe in timeframes:
            df = self.get_latest_data(symbol, timeframe)
            if not df.empty:
                data[timeframe] = df
        
        return data
    
    def get_all_symbols_data(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Get data for all symbols for a specific timeframe"""
        data = {}
        for symbol in self.all_symbols:
            df = self.get_latest_data(symbol, timeframe)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def get_asset_category_data(self, category: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Get data for a specific asset category"""
        category_symbols = {
            'volatility': self.volatility_indices,
            'hz': self.hz_indices,
            'step': self.step_indices
        }
        
        symbols = category_symbols.get(category, [])
        data = {}
        
        for symbol in symbols:
            df = self.get_latest_data(symbol, timeframe)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick data for a symbol"""
        try:
            # Try Redis cache first
            if self.redis_client:
                key = f"tick:{symbol}"
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    return TickData(
                        symbol=symbol,
                        bid=data['bid'],
                        ask=data['ask'],
                        last=data['last'],
                        volume=data['volume'],
                        timestamp=datetime.fromisoformat(data['timestamp'])
                    )
            
            # Fallback to direct MT5 call
            return self.mt5.get_tick_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting latest tick for {symbol}: {str(e)}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            'mt5_connected': self.mt5.connected,
            'redis_connected': self.redis_client is not None,
            'running': self.running,
            'active_threads': len([t for t in self.threads.values() if t.is_alive()]),
            'symbols_count': len(self.all_symbols),
            'timeframes_count': len(self.timeframes),
            'buffer_size': sum(len(self.data_buffer.get_bars(symbol, tf)) 
                             for symbol in self.all_symbols 
                             for tf in self.timeframes)
        }

if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    try:
        if pipeline.start():
            print("Data pipeline started successfully")
            
            # Let it run for a bit
            time.sleep(10)
            
            # Get some data
            r75_data = pipeline.get_multi_timeframe_data("R_75")
            print(f"R_75 data available for timeframes: {list(r75_data.keys())}")
            
            # Get system status
            status = pipeline.get_system_status()
            print(f"System status: {status}")
            
        else:
            print("Failed to start data pipeline")
    
    finally:
        pipeline.stop()
        print("Data pipeline stopped")