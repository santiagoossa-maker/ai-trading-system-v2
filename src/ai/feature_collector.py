"""
AI Feature Collector System
Comprehensive feature engineering for machine learning models
Supports multi-timeframe analysis and advanced indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    macd_configs: List[Tuple[int, int, int]] = None
    rsi_periods: List[int] = None
    bollinger_configs: List[Tuple[int, float]] = None
    adx_periods: List[int] = None
    stochastic_configs: List[Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 8, 13, 21, 34, 50, 89, 144, 200]
        if self.ema_periods is None:
            self.ema_periods = [5, 8, 13, 21, 34, 50, 89]
        if self.macd_configs is None:
            self.macd_configs = [(12, 26, 9), (8, 21, 5), (5, 13, 3)]
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21, 28]
        if self.bollinger_configs is None:
            self.bollinger_configs = [(20, 2.0), (13, 1.5), (34, 2.5)]
        if self.adx_periods is None:
            self.adx_periods = [14, 21, 28]
        if self.stochastic_configs is None:
            self.stochastic_configs = [(14, 3, 3), (21, 5, 5)]

class FeatureCollector:
    """
    Advanced feature collector for AI trading system
    Generates comprehensive features for machine learning models
    """
    
    def __init__(self, indicator_config: Optional[IndicatorConfig] = None):
        self.config = indicator_config or IndicatorConfig()
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        self.cache = {}
        
    def collect_all_features(self, data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """
        Collect all features for a given symbol across all timeframes
        
        Args:
            data: Dictionary with timeframe as key and OHLCV data as value
            symbol: Trading symbol
            
        Returns:
            DataFrame with all features
        """
        try:
            features_list = []
            
            # Process each timeframe
            for timeframe in self.timeframes:
                if timeframe in data and not data[timeframe].empty:
                    tf_features = self._process_timeframe(data[timeframe], timeframe, symbol)
                    features_list.append(tf_features)
            
            if not features_list:
                logger.warning(f"No valid data for symbol {symbol}")
                return pd.DataFrame()
            
            # Combine all timeframe features
            combined_features = pd.concat(features_list, axis=1)
            
            # Add cross-timeframe features
            cross_tf_features = self._generate_cross_timeframe_features(data, symbol)
            combined_features = pd.concat([combined_features, cross_tf_features], axis=1)
            
            # Add market regime features
            regime_features = self._detect_market_regime(data, symbol)
            combined_features = pd.concat([combined_features, regime_features], axis=1)
            
            # Add pattern recognition features
            pattern_features = self._generate_pattern_features(data, symbol)
            combined_features = pd.concat([combined_features, pattern_features], axis=1)
            
            return combined_features.dropna()
            
        except Exception as e:
            logger.error(f"Error collecting features for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _process_timeframe(self, df: pd.DataFrame, timeframe: str, symbol: str) -> pd.DataFrame:
        """Process single timeframe data and generate features"""
        try:
            features = pd.DataFrame(index=df.index)
            prefix = f"{timeframe}_{symbol}"
            
            # Basic price features
            features[f'{prefix}_open'] = df['open']
            features[f'{prefix}_high'] = df['high']
            features[f'{prefix}_low'] = df['low']
            features[f'{prefix}_close'] = df['close']
            features[f'{prefix}_volume'] = df.get('volume', 0)
            
            # Price-based features
            features[f'{prefix}_hl_ratio'] = df['high'] / df['low']
            features[f'{prefix}_oc_ratio'] = df['open'] / df['close']
            features[f'{prefix}_body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features[f'{prefix}_upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
            features[f'{prefix}_lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
            
            # Moving averages
            features = self._add_moving_averages(features, df, prefix)
            
            # Oscillators
            features = self._add_oscillators(features, df, prefix)
            
            # Volatility indicators
            features = self._add_volatility_indicators(features, df, prefix)
            
            # Momentum indicators
            features = self._add_momentum_indicators(features, df, prefix)
            
            # Volume indicators (if available)
            if 'volume' in df.columns:
                features = self._add_volume_indicators(features, df, prefix)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing timeframe {timeframe} for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_moving_averages(self, features: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add moving average indicators and derived features"""
        try:
            close = df['close'].values
            
            if not TALIB_AVAILABLE:
                # Fallback implementation using pandas
                for period in self.config.sma_periods:
                    if len(close) >= period:
                        sma = pd.Series(close).rolling(window=period).mean().values
                        features[f'{prefix}_sma_{period}'] = sma
                        features[f'{prefix}_close_sma_{period}_ratio'] = close / sma
                        features[f'{prefix}_close_sma_{period}_distance'] = (close - sma) / sma * 100
                
                for period in self.config.ema_periods:
                    if len(close) >= period:
                        ema = pd.Series(close).ewm(span=period).mean().values
                        features[f'{prefix}_ema_{period}'] = ema
                        features[f'{prefix}_close_ema_{period}_ratio'] = close / ema
                        features[f'{prefix}_close_ema_{period}_distance'] = (close - ema) / ema * 100
                
                return features
            
            # TA-Lib implementation
            # Simple Moving Averages
            for period in self.config.sma_periods:
                if len(close) >= period:
                    sma = talib.SMA(close, timeperiod=period)
                    features[f'{prefix}_sma_{period}'] = sma
                    features[f'{prefix}_close_sma_{period}_ratio'] = close / sma
                    features[f'{prefix}_close_sma_{period}_distance'] = (close - sma) / sma * 100
            
            # Exponential Moving Averages
            for period in self.config.ema_periods:
                if len(close) >= period:
                    ema = talib.EMA(close, timeperiod=period)
                    features[f'{prefix}_ema_{period}'] = ema
                    features[f'{prefix}_close_ema_{period}_ratio'] = close / ema
                    features[f'{prefix}_close_ema_{period}_distance'] = (close - ema) / ema * 100
            
            # Moving average crossovers
            if len(close) >= max(self.config.sma_periods[:2]):
                sma_fast = talib.SMA(close, timeperiod=self.config.sma_periods[0])
                sma_slow = talib.SMA(close, timeperiod=self.config.sma_periods[1])
                features[f'{prefix}_sma_crossover'] = np.where(sma_fast > sma_slow, 1, -1)
                
            if len(close) >= max(self.config.ema_periods[:2]):
                ema_fast = talib.EMA(close, timeperiod=self.config.ema_periods[0])
                ema_slow = talib.EMA(close, timeperiod=self.config.ema_periods[1])
                features[f'{prefix}_ema_crossover'] = np.where(ema_fast > ema_slow, 1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")
            return features
    
    def _add_oscillators(self, features: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add oscillator indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # MACD variants
            for fast, slow, signal in self.config.macd_configs:
                if len(close) >= slow:
                    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fast, 
                                                          slowperiod=slow, signalperiod=signal)
                    suffix = f"{fast}_{slow}_{signal}"
                    features[f'{prefix}_macd_{suffix}'] = macd
                    features[f'{prefix}_macd_signal_{suffix}'] = macdsignal
                    features[f'{prefix}_macd_histogram_{suffix}'] = macdhist
                    features[f'{prefix}_macd_crossover_{suffix}'] = np.where(macd > macdsignal, 1, -1)
            
            # RSI variants
            for period in self.config.rsi_periods:
                if len(close) >= period:
                    rsi = talib.RSI(close, timeperiod=period)
                    features[f'{prefix}_rsi_{period}'] = rsi
                    features[f'{prefix}_rsi_{period}_overbought'] = np.where(rsi > 70, 1, 0)
                    features[f'{prefix}_rsi_{period}_oversold'] = np.where(rsi < 30, 1, 0)
                    features[f'{prefix}_rsi_{period}_normalized'] = (rsi - 50) / 50
            
            # Stochastic oscillator variants
            for k_period, d_period, smooth in self.config.stochastic_configs:
                if len(close) >= k_period:
                    slowk, slowd = talib.STOCH(high, low, close, 
                                             fastk_period=k_period, 
                                             slowk_period=smooth, 
                                             slowd_period=d_period)
                    suffix = f"{k_period}_{d_period}_{smooth}"
                    features[f'{prefix}_stoch_k_{suffix}'] = slowk
                    features[f'{prefix}_stoch_d_{suffix}'] = slowd
                    features[f'{prefix}_stoch_crossover_{suffix}'] = np.where(slowk > slowd, 1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding oscillators: {str(e)}")
            return features
    
    def _add_volatility_indicators(self, features: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add volatility indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Bollinger Bands variants
            for period, std_dev in self.config.bollinger_configs:
                if len(close) >= period:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period, 
                                                               nbdevup=std_dev, nbdevdn=std_dev)
                    suffix = f"{period}_{std_dev}"
                    features[f'{prefix}_bb_upper_{suffix}'] = bb_upper
                    features[f'{prefix}_bb_middle_{suffix}'] = bb_middle
                    features[f'{prefix}_bb_lower_{suffix}'] = bb_lower
                    features[f'{prefix}_bb_position_{suffix}'] = (close - bb_lower) / (bb_upper - bb_lower)
                    features[f'{prefix}_bb_width_{suffix}'] = (bb_upper - bb_lower) / bb_middle
                    features[f'{prefix}_bb_squeeze_{suffix}'] = np.where(
                        (bb_upper - bb_lower) < np.roll(bb_upper - bb_lower, 20), 1, 0)
            
            # Average True Range variants
            for period in self.config.adx_periods:
                if len(close) >= period:
                    atr = talib.ATR(high, low, close, timeperiod=period)
                    features[f'{prefix}_atr_{period}'] = atr
                    features[f'{prefix}_atr_{period}_normalized'] = atr / close
            
            # Volatility measures
            for window in [10, 20, 50]:
                if len(close) >= window:
                    returns = np.diff(np.log(close))
                    vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
                    features[f'{prefix}_volatility_{window}'] = np.concatenate([[np.nan], vol])
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volatility indicators: {str(e)}")
            return features
    
    def _add_momentum_indicators(self, features: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # ADX variants
            for period in self.config.adx_periods:
                if len(close) >= period:
                    adx = talib.ADX(high, low, close, timeperiod=period)
                    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
                    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
                    
                    features[f'{prefix}_adx_{period}'] = adx
                    features[f'{prefix}_plus_di_{period}'] = plus_di
                    features[f'{prefix}_minus_di_{period}'] = minus_di
                    features[f'{prefix}_di_diff_{period}'] = plus_di - minus_di
            
            # Rate of Change
            for period in [5, 10, 20]:
                if len(close) >= period:
                    roc = talib.ROC(close, timeperiod=period)
                    features[f'{prefix}_roc_{period}'] = roc
            
            # Momentum
            for period in [5, 10, 20]:
                if len(close) >= period:
                    momentum = talib.MOM(close, timeperiod=period)
                    features[f'{prefix}_momentum_{period}'] = momentum
            
            # Williams %R
            for period in [14, 21]:
                if len(close) >= period:
                    willr = talib.WILLR(high, low, close, timeperiod=period)
                    features[f'{prefix}_willr_{period}'] = willr
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {str(e)}")
            return features
    
    def _add_volume_indicators(self, features: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            
            # Volume SMA
            for period in [10, 20, 50]:
                if len(volume) >= period:
                    vol_sma = talib.SMA(volume, timeperiod=period)
                    features[f'{prefix}_volume_sma_{period}'] = vol_sma
                    features[f'{prefix}_volume_ratio_{period}'] = volume / vol_sma
            
            # On Balance Volume
            if len(close) >= 2:
                obv = talib.OBV(close, volume)
                features[f'{prefix}_obv'] = obv
            
            # Volume Rate of Change
            for period in [5, 10]:
                if len(volume) >= period:
                    vol_roc = talib.ROC(volume, timeperiod=period)
                    features[f'{prefix}_volume_roc_{period}'] = vol_roc
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {str(e)}")
            return features
    
    def _generate_cross_timeframe_features(self, data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Generate features that compare across timeframes"""
        try:
            features = pd.DataFrame()
            
            # Get the main timeframe (typically M5 or M15)
            main_tf = 'M5' if 'M5' in data else list(data.keys())[0]
            if main_tf not in data or data[main_tf].empty:
                return features
            
            main_data = data[main_tf]
            features = pd.DataFrame(index=main_data.index)
            
            # Compare trend direction across timeframes
            trend_directions = {}
            for tf in ['M1', 'M5', 'M15', 'M30', 'H1']:
                if tf in data and not data[tf].empty:
                    close = data[tf]['close'].values
                    if len(close) >= 20:
                        sma_short = talib.SMA(close, timeperiod=8)
                        sma_long = talib.SMA(close, timeperiod=20)
                        trend = np.where(sma_short > sma_long, 1, -1)
                        trend_directions[tf] = trend[-1] if len(trend) > 0 else 0
            
            # Trend alignment score
            if trend_directions:
                features[f'{symbol}_trend_alignment'] = sum(trend_directions.values()) / len(trend_directions)
                features[f'{symbol}_trend_consensus'] = len([v for v in trend_directions.values() if v != 0])
            
            # Volatility comparison
            volatilities = {}
            for tf in ['M5', 'M15', 'M30', 'H1']:
                if tf in data and not data[tf].empty:
                    close = data[tf]['close'].values
                    if len(close) >= 20:
                        atr = talib.ATR(data[tf]['high'].values, data[tf]['low'].values, close, timeperiod=14)
                        volatilities[tf] = atr[-1] if len(atr) > 0 else 0
            
            if len(volatilities) >= 2:
                vol_values = list(volatilities.values())
                features[f'{symbol}_volatility_ratio'] = max(vol_values) / min(vol_values) if min(vol_values) > 0 else 1
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating cross-timeframe features: {str(e)}")
            return pd.DataFrame()
    
    def _detect_market_regime(self, data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Detect current market regime"""
        try:
            features = pd.DataFrame()
            
            # Use M15 timeframe for regime detection
            main_tf = 'M15' if 'M15' in data else list(data.keys())[0]
            if main_tf not in data or data[main_tf].empty:
                return features
            
            df = data[main_tf]
            features = pd.DataFrame(index=df.index)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            if len(close) >= 50:
                # ADX for trend strength
                adx = talib.ADX(high, low, close, timeperiod=14)
                features[f'{symbol}_regime_adx'] = adx
                features[f'{symbol}_regime_trending'] = np.where(adx > 25, 1, 0)
                features[f'{symbol}_regime_ranging'] = np.where(adx < 20, 1, 0)
                
                # Choppiness Index
                atr = talib.ATR(high, low, close, timeperiod=14)
                hl_range = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
                chop = 100 * np.log10(pd.Series(atr).rolling(14).sum() / 
                                     (pd.Series(hl_range).rolling(14).max() - 
                                      pd.Series(hl_range).rolling(14).min())) / np.log10(14)
                features[f'{symbol}_choppiness'] = chop
                features[f'{symbol}_regime_choppy'] = np.where(chop > 61.8, 1, 0)
                
                # Volatility regime
                vol_20 = pd.Series(atr).rolling(20).mean()
                vol_50 = pd.Series(atr).rolling(50).mean()
                features[f'{symbol}_volatility_regime'] = vol_20 / vol_50
                features[f'{symbol}_high_volatility'] = np.where(vol_20 > vol_50 * 1.2, 1, 0)
                
                # Support/Resistance levels
                highs = pd.Series(high).rolling(20).max()
                lows = pd.Series(low).rolling(20).min()
                features[f'{symbol}_near_resistance'] = np.where(close > highs * 0.99, 1, 0)
                features[f'{symbol}_near_support'] = np.where(close < lows * 1.01, 1, 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return pd.DataFrame()
    
    def _generate_pattern_features(self, data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Generate candlestick pattern recognition features"""
        try:
            features = pd.DataFrame()
            
            # Use M5 timeframe for pattern recognition
            main_tf = 'M5' if 'M5' in data else list(data.keys())[0]
            if main_tf not in data or data[main_tf].empty:
                return features
            
            df = data[main_tf]
            features = pd.DataFrame(index=df.index)
            
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            if len(close_prices) >= 5:
                # Candlestick patterns
                patterns = {
                    'doji': talib.CDLDOJI,
                    'hammer': talib.CDLHAMMER,
                    'shooting_star': talib.CDLSHOOTINGSTAR,
                    'engulfing': talib.CDLENGULFING,
                    'harami': talib.CDLHARAMI,
                    'piercing': talib.CDLPIERCING,
                    'dark_cloud': talib.CDLDARKCLOUDCOVER,
                    'morning_star': talib.CDLMORNINGSTAR,
                    'evening_star': talib.CDLEVENINGSTAR
                }
                
                for pattern_name, pattern_func in patterns.items():
                    try:
                        pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                        features[f'{symbol}_pattern_{pattern_name}'] = pattern_result
                    except:
                        features[f'{symbol}_pattern_{pattern_name}'] = 0
                
                # Fractal patterns
                features[f'{symbol}_fractal_high'] = self._detect_fractals(high_prices, mode='high')
                features[f'{symbol}_fractal_low'] = self._detect_fractals(low_prices, mode='low')
                
                # Gap detection
                gaps = np.where(open_prices[1:] > high_prices[:-1], 1, 
                               np.where(open_prices[1:] < low_prices[:-1], -1, 0))
                features[f'{symbol}_gap'] = np.concatenate([[0], gaps])
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating pattern features: {str(e)}")
            return pd.DataFrame()
    
    def _detect_fractals(self, prices: np.ndarray, mode: str = 'high', period: int = 5) -> np.ndarray:
        """Detect fractal highs/lows"""
        try:
            fractals = np.zeros(len(prices))
            half_period = period // 2
            
            for i in range(half_period, len(prices) - half_period):
                if mode == 'high':
                    if all(prices[i] >= prices[i-j] for j in range(1, half_period + 1)) and \
                       all(prices[i] >= prices[i+j] for j in range(1, half_period + 1)):
                        fractals[i] = 1
                else:  # mode == 'low'
                    if all(prices[i] <= prices[i-j] for j in range(1, half_period + 1)) and \
                       all(prices[i] <= prices[i+j] for j in range(1, half_period + 1)):
                        fractals[i] = -1
            
            return fractals
            
        except Exception as e:
            logger.error(f"Error detecting fractals: {str(e)}")
            return np.zeros(len(prices))
    
    def generate_target_variables(self, data: Dict[str, pd.DataFrame], symbol: str, 
                                look_ahead_periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Generate target variables for machine learning
        
        Args:
            data: Price data
            symbol: Trading symbol
            look_ahead_periods: Periods to look ahead for targets
            
        Returns:
            DataFrame with target variables
        """
        try:
            # Use M5 timeframe for targets
            main_tf = 'M5' if 'M5' in data else list(data.keys())[0]
            if main_tf not in data or data[main_tf].empty:
                return pd.DataFrame()
            
            df = data[main_tf]
            targets = pd.DataFrame(index=df.index)
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            for periods in look_ahead_periods:
                if len(close) > periods:
                    # Future returns
                    future_returns = (np.roll(close, -periods) - close) / close * 100
                    targets[f'{symbol}_return_{periods}'] = future_returns
                    
                    # Binary targets (profitable or not)
                    targets[f'{symbol}_profitable_{periods}'] = np.where(future_returns > 0, 1, 0)
                    
                    # Maximum favorable excursion
                    future_highs = pd.Series(high).rolling(periods).max().shift(-periods)
                    mfe = (future_highs - close) / close * 100
                    targets[f'{symbol}_mfe_{periods}'] = mfe
                    
                    # Maximum adverse excursion
                    future_lows = pd.Series(low).rolling(periods).min().shift(-periods)
                    mae = (close - future_lows) / close * 100
                    targets[f'{symbol}_mae_{periods}'] = mae
                    
                    # Risk-reward ratio
                    targets[f'{symbol}_risk_reward_{periods}'] = mfe / (mae + 0.001)  # Avoid division by zero
            
            return targets.iloc[:-max(look_ahead_periods)]  # Remove last rows without valid targets
            
        except Exception as e:
            logger.error(f"Error generating target variables: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    collector = FeatureCollector()
    
    # Sample data structure (in real implementation, this comes from MT5)
    sample_data = {
        'M5': pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 1000)
        })
    }
    
    # Generate features
    features = collector.collect_all_features(sample_data, "R_75")
    targets = collector.generate_target_variables(sample_data, "R_75")
    
    print(f"Generated {features.shape[1]} features and {targets.shape[1]} targets")
    print("Sample features:", features.columns[:10].tolist())
    print("Sample targets:", targets.columns.tolist())