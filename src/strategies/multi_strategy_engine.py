"""
Multi-Strategy Trading Engine
Executes 5 different trading strategies simultaneously with intelligent signal aggregation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import yaml
import os

# Optional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

logger = logging.getLogger(__name__)

# Import the new SMA MACD strategy
try:
    from .sma_macd_strategy import SMAMACDStrategy, SignalType, TradingSignal as BaseTradingSignal
    SMA_STRATEGY_AVAILABLE = True
except ImportError:
    SMA_STRATEGY_AVAILABLE = False
    
    # Fallback definitions
    class SignalType(Enum):
        BUY = 1
        SELL = -1
        HOLD = 0

class StrategyType(Enum):
    SMA_MACD = "sma_macd"
    EMA_MACD = "ema_macd"
    TRIPLE_EMA = "triple_ema"
    ADAPTIVE_VOLATILITY = "adaptive_volatility"
    BOLLINGER_RSI = "bollinger_rsi"

@dataclass
class TradingSignal:
    strategy: StrategyType
    signal: SignalType
    strength: float  # 0.0 to 1.0
    price: float
    timestamp: pd.Timestamp
    timeframe: str
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class AggregatedSignal:
    final_signal: SignalType
    strength: float
    confidence: float
    contributing_strategies: List[StrategyType]
    signals: List[TradingSignal]
    price: float
    timestamp: pd.Timestamp

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.signals_history = []
        
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Generate trading signal - to be implemented by subclasses"""
        raise NotImplementedError
        
    def validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validate signal quality - to be implemented by subclasses"""
        return True

class SMAMACDStrategy(BaseStrategy):
    """Original SMA8/50 + MACD strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SMA_MACD", config)
        self.sma_fast = config.get('sma_fast', 8)
        self.sma_slow = config.get('sma_slow', 50)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        try:
            if len(data) < self.sma_slow + 10:
                return None
                
            close = data['close'].values
            
            if TALIB_AVAILABLE:
                # Calculate SMAs
                sma_fast = talib.SMA(close, timeperiod=self.sma_fast)
                sma_slow = talib.SMA(close, timeperiod=self.sma_slow)
                
                # Calculate MACD
                macd, macd_signal_line, macd_histogram = talib.MACD(
                    close, fastperiod=self.macd_fast, 
                    slowperiod=self.macd_slow, signalperiod=self.macd_signal
                )
            else:
                # Fallback implementation using pandas
                close_series = pd.Series(close)
                sma_fast = close_series.rolling(window=self.sma_fast).mean().values
                sma_slow = close_series.rolling(window=self.sma_slow).mean().values
                
                # Simple MACD implementation
                ema_fast = close_series.ewm(span=self.macd_fast).mean()
                ema_slow = close_series.ewm(span=self.macd_slow).mean()
                macd = (ema_fast - ema_slow).values
                macd_signal_line = pd.Series(macd).ewm(span=self.macd_signal).mean().values
                macd_histogram = macd - macd_signal_line
            
            if any(np.isnan([sma_fast[-1], sma_slow[-1], macd[-1], macd_signal_line[-1]])):
                return None
            
            # Generate signals
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # SMA crossover condition
            sma_bullish = sma_fast[-1] > sma_slow[-1]
            sma_bearish = sma_fast[-1] < sma_slow[-1]
            
            # MACD condition
            macd_bullish = macd[-1] > macd_signal_line[-1] and macd_histogram[-1] > 0
            macd_bearish = macd[-1] < macd_signal_line[-1] and macd_histogram[-1] < 0
            
            # Combined signal logic
            if sma_bullish and macd_bullish:
                signal_type = SignalType.BUY
                strength = min(1.0, abs(macd_histogram[-1]) * 10)
                confidence = 0.8
            elif sma_bearish and macd_bearish:
                signal_type = SignalType.SELL
                strength = min(1.0, abs(macd_histogram[-1]) * 10)
                confidence = 0.8
            
            # Additional confirmation
            if signal_type != SignalType.HOLD:
                # Check for recent crossover
                if len(sma_fast) >= 3:
                    recent_sma_cross = (sma_fast[-1] > sma_slow[-1]) != (sma_fast[-3] > sma_slow[-3])
                    if recent_sma_cross:
                        confidence += 0.1
                
                # Check MACD momentum
                if len(macd_histogram) >= 2:
                    macd_momentum = macd_histogram[-1] > macd_histogram[-2]
                    if (signal_type == SignalType.BUY and macd_momentum) or \
                       (signal_type == SignalType.SELL and not macd_momentum):
                        confidence += 0.1
            
            return TradingSignal(
                strategy=StrategyType.SMA_MACD,
                signal=signal_type,
                strength=strength,
                price=close[-1],
                timestamp=data.index[-1],
                timeframe=timeframe,
                confidence=min(1.0, confidence),
                metadata={
                    'sma_fast': sma_fast[-1],
                    'sma_slow': sma_slow[-1],
                    'macd': macd[-1],
                    'macd_signal': macd_signal_line[-1],
                    'macd_histogram': macd_histogram[-1]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in SMA MACD strategy: {str(e)}")
            return None

class EMAMACDStrategy(BaseStrategy):
    """Reactive EMA8/21 + MACD strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EMA_MACD", config)
        self.ema_fast = config.get('ema_fast', 8)
        self.ema_slow = config.get('ema_slow', 21)
        self.macd_fast = config.get('macd_fast', 8)
        self.macd_slow = config.get('macd_slow', 21)
        self.macd_signal = config.get('macd_signal', 5)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        try:
            if len(data) < self.ema_slow + 10:
                return None
                
            close = data['close'].values
            
            # Calculate EMAs
            ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
            ema_slow = talib.EMA(close, timeperiod=self.ema_slow)
            
            # Calculate MACD with faster settings
            macd, macd_signal_line, macd_histogram = talib.MACD(
                close, fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, signalperiod=self.macd_signal
            )
            
            if any(np.isnan([ema_fast[-1], ema_slow[-1], macd[-1], macd_signal_line[-1]])):
                return None
            
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # EMA crossover condition (more reactive)
            ema_bullish = ema_fast[-1] > ema_slow[-1]
            ema_bearish = ema_fast[-1] < ema_slow[-1]
            
            # MACD condition with faster parameters
            macd_bullish = macd[-1] > macd_signal_line[-1] and macd[-1] > 0
            macd_bearish = macd[-1] < macd_signal_line[-1] and macd[-1] < 0
            
            # Price position relative to EMAs
            price_above_emas = close[-1] > ema_fast[-1] > ema_slow[-1]
            price_below_emas = close[-1] < ema_fast[-1] < ema_slow[-1]
            
            if ema_bullish and macd_bullish and price_above_emas:
                signal_type = SignalType.BUY
                strength = min(1.0, (ema_fast[-1] - ema_slow[-1]) / ema_slow[-1] * 100)
                confidence = 0.75
            elif ema_bearish and macd_bearish and price_below_emas:
                signal_type = SignalType.SELL
                strength = min(1.0, (ema_slow[-1] - ema_fast[-1]) / ema_slow[-1] * 100)
                confidence = 0.75
            
            # Additional confirmation for EMA strategy
            if signal_type != SignalType.HOLD:
                # Check EMA slope
                if len(ema_fast) >= 3:
                    ema_slope = (ema_fast[-1] - ema_fast[-3]) / ema_fast[-3]
                    if (signal_type == SignalType.BUY and ema_slope > 0) or \
                       (signal_type == SignalType.SELL and ema_slope < 0):
                        confidence += 0.15
            
            return TradingSignal(
                strategy=StrategyType.EMA_MACD,
                signal=signal_type,
                strength=strength,
                price=close[-1],
                timestamp=data.index[-1],
                timeframe=timeframe,
                confidence=min(1.0, confidence),
                metadata={
                    'ema_fast': ema_fast[-1],
                    'ema_slow': ema_slow[-1],
                    'macd': macd[-1],
                    'macd_signal': macd_signal_line[-1],
                    'ema_slope': (ema_fast[-1] - ema_fast[-3]) / ema_fast[-3] if len(ema_fast) >= 3 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error in EMA MACD strategy: {str(e)}")
            return None

class TripleEMAStrategy(BaseStrategy):
    """Triple EMA strategy for strong trends"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TRIPLE_EMA", config)
        self.ema_fast = config.get('ema_fast', 5)
        self.ema_medium = config.get('ema_medium', 13)
        self.ema_slow = config.get('ema_slow', 34)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        try:
            if len(data) < self.ema_slow + 10:
                return None
                
            close = data['close'].values
            
            # Calculate Triple EMAs
            ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
            ema_medium = talib.EMA(close, timeperiod=self.ema_medium)
            ema_slow = talib.EMA(close, timeperiod=self.ema_slow)
            
            if any(np.isnan([ema_fast[-1], ema_medium[-1], ema_slow[-1]])):
                return None
            
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Perfect alignment for strong trends
            perfect_bullish = ema_fast[-1] > ema_medium[-1] > ema_slow[-1] and close[-1] > ema_fast[-1]
            perfect_bearish = ema_fast[-1] < ema_medium[-1] < ema_slow[-1] and close[-1] < ema_fast[-1]
            
            if perfect_bullish:
                signal_type = SignalType.BUY
                # Calculate strength based on EMA separation
                separation = ((ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]) * 100
                strength = min(1.0, separation * 2)
                confidence = 0.9
            elif perfect_bearish:
                signal_type = SignalType.SELL
                separation = ((ema_slow[-1] - ema_fast[-1]) / ema_slow[-1]) * 100
                strength = min(1.0, separation * 2)
                confidence = 0.9
            
            # Additional confirmation
            if signal_type != SignalType.HOLD and len(close) >= 5:
                # Check if trend is accelerating
                prev_fast = ema_fast[-5] if len(ema_fast) >= 5 else ema_fast[-1]
                prev_slow = ema_slow[-5] if len(ema_slow) >= 5 else ema_slow[-1]
                
                if signal_type == SignalType.BUY:
                    current_spread = ema_fast[-1] - ema_slow[-1]
                    prev_spread = prev_fast - prev_slow
                    if current_spread > prev_spread:
                        confidence += 0.1
                elif signal_type == SignalType.SELL:
                    current_spread = ema_slow[-1] - ema_fast[-1]
                    prev_spread = prev_slow - prev_fast
                    if current_spread > prev_spread:
                        confidence += 0.1
            
            return TradingSignal(
                strategy=StrategyType.TRIPLE_EMA,
                signal=signal_type,
                strength=strength,
                price=close[-1],
                timestamp=data.index[-1],
                timeframe=timeframe,
                confidence=min(1.0, confidence),
                metadata={
                    'ema_fast': ema_fast[-1],
                    'ema_medium': ema_medium[-1],
                    'ema_slow': ema_slow[-1],
                    'perfect_alignment': perfect_bullish or perfect_bearish
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Triple EMA strategy: {str(e)}")
            return None

class AdaptiveVolatilityStrategy(BaseStrategy):
    """Adaptive strategy that changes parameters based on volatility"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ADAPTIVE_VOLATILITY", config)
        self.base_period = config.get('base_period', 20)
        self.volatility_multiplier = config.get('volatility_multiplier', 0.5)
        self.min_period = config.get('min_period', 8)
        self.max_period = config.get('max_period', 50)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        try:
            if len(data) < self.max_period + 20:
                return None
                
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate volatility using ATR
            atr = talib.ATR(high, low, close, timeperiod=14)
            current_atr = atr[-1]
            avg_atr = np.mean(atr[-20:])
            
            if np.isnan(current_atr) or np.isnan(avg_atr):
                return None
            
            # Adapt periods based on volatility
            volatility_ratio = current_atr / avg_atr
            adaptive_period = int(self.base_period * (1 + (volatility_ratio - 1) * self.volatility_multiplier))
            adaptive_period = max(self.min_period, min(self.max_period, adaptive_period))
            
            # Calculate adaptive SMAs
            sma_fast = talib.SMA(close, timeperiod=max(5, adaptive_period // 3))
            sma_slow = talib.SMA(close, timeperiod=adaptive_period)
            
            # Calculate RSI with adaptive period
            rsi = talib.RSI(close, timeperiod=max(7, adaptive_period // 2))
            
            if any(np.isnan([sma_fast[-1], sma_slow[-1], rsi[-1]])):
                return None
            
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Adaptive signal logic
            sma_bullish = sma_fast[-1] > sma_slow[-1]
            sma_bearish = sma_fast[-1] < sma_slow[-1]
            
            # Volatility-adjusted RSI levels
            if volatility_ratio > 1.2:  # High volatility
                rsi_oversold = 25
                rsi_overbought = 75
            elif volatility_ratio < 0.8:  # Low volatility
                rsi_oversold = 35
                rsi_overbought = 65
            else:  # Normal volatility
                rsi_oversold = 30
                rsi_overbought = 70
            
            if sma_bullish and rsi[-1] < rsi_overbought and close[-1] > sma_fast[-1]:
                signal_type = SignalType.BUY
                strength = min(1.0, (sma_fast[-1] - sma_slow[-1]) / sma_slow[-1] * 100 * volatility_ratio)
                confidence = 0.7
            elif sma_bearish and rsi[-1] > rsi_oversold and close[-1] < sma_fast[-1]:
                signal_type = SignalType.SELL
                strength = min(1.0, (sma_slow[-1] - sma_fast[-1]) / sma_slow[-1] * 100 * volatility_ratio)
                confidence = 0.7
            
            # Volatility confirmation
            if signal_type != SignalType.HOLD:
                if 0.8 <= volatility_ratio <= 1.5:  # Optimal volatility range
                    confidence += 0.2
                elif volatility_ratio > 2.0:  # Too volatile
                    confidence -= 0.3
            
            return TradingSignal(
                strategy=StrategyType.ADAPTIVE_VOLATILITY,
                signal=signal_type,
                strength=strength,
                price=close[-1],
                timestamp=data.index[-1],
                timeframe=timeframe,
                confidence=max(0.0, min(1.0, confidence)),
                metadata={
                    'adaptive_period': adaptive_period,
                    'volatility_ratio': volatility_ratio,
                    'atr': current_atr,
                    'rsi': rsi[-1],
                    'sma_fast': sma_fast[-1],
                    'sma_slow': sma_slow[-1]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Adaptive Volatility strategy: {str(e)}")
            return None

class BollingerRSIStrategy(BaseStrategy):
    """Bollinger Bands + RSI reversal strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BOLLINGER_RSI", config)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        try:
            if len(data) < self.bb_period + 10:
                return None
                
            close = data['close'].values
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=self.bb_period, 
                nbdevup=self.bb_std, nbdevdn=self.bb_std
            )
            
            # Calculate RSI
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            
            if any(np.isnan([bb_upper[-1], bb_lower[-1], rsi[-1]])):
                return None
            
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # Bollinger Band position
            bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Reversal signals
            oversold_reversal = (close[-1] <= bb_lower[-1] and rsi[-1] <= self.rsi_oversold and 
                               close[-1] > close[-2])  # Price bouncing from lower band
            
            overbought_reversal = (close[-1] >= bb_upper[-1] and rsi[-1] >= self.rsi_overbought and 
                                 close[-1] < close[-2])  # Price rejecting upper band
            
            if oversold_reversal:
                signal_type = SignalType.BUY
                strength = min(1.0, (1 - bb_position) + (1 - rsi[-1]/100))
                confidence = 0.8
            elif overbought_reversal:
                signal_type = SignalType.SELL
                strength = min(1.0, bb_position + (rsi[-1]/100))
                confidence = 0.8
            
            # Additional confirmation
            if signal_type != SignalType.HOLD:
                # Check for band squeeze (low volatility before breakout)
                bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                avg_bb_width = np.mean([(bb_upper[i] - bb_lower[i]) / bb_middle[i] 
                                      for i in range(-20, -1) if not np.isnan(bb_middle[i])])
                
                if bb_width < avg_bb_width * 0.8:  # Band squeeze
                    confidence += 0.1
                
                # RSI divergence check
                if len(close) >= 10 and len(rsi) >= 10:
                    price_higher = close[-1] > close[-10]
                    rsi_lower = rsi[-1] < rsi[-10]
                    price_lower = close[-1] < close[-10]
                    rsi_higher = rsi[-1] > rsi[-10]
                    
                    if (signal_type == SignalType.BUY and price_lower and rsi_higher) or \
                       (signal_type == SignalType.SELL and price_higher and rsi_lower):
                        confidence += 0.15
            
            return TradingSignal(
                strategy=StrategyType.BOLLINGER_RSI,
                signal=signal_type,
                strength=strength,
                price=close[-1],
                timestamp=data.index[-1],
                timeframe=timeframe,
                confidence=min(1.0, confidence),
                metadata={
                    'bb_upper': bb_upper[-1],
                    'bb_middle': bb_middle[-1],
                    'bb_lower': bb_lower[-1],
                    'bb_position': bb_position,
                    'rsi': rsi[-1],
                    'bb_width': (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Bollinger RSI strategy: {str(e)}")
            return None

class MultiStrategyEngine:
    """
    Multi-strategy engine that runs 5 strategies simultaneously
    and provides intelligent signal aggregation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.strategies = {}
        self.signal_history = []
        self.aggregation_weights = {}
        self.load_configuration(config_path)
        self.initialize_strategies()
        
    def load_configuration(self, config_path: Optional[str]):
        """Load strategy configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'strategies': {
                    'sma_macd': {'enabled': True, 'weight': 0.25},
                    'ema_macd': {'enabled': True, 'weight': 0.25},
                    'triple_ema': {'enabled': True, 'weight': 0.20},
                    'adaptive_volatility': {'enabled': True, 'weight': 0.15},
                    'bollinger_rsi': {'enabled': True, 'weight': 0.15}
                },
                'aggregation': {
                    'min_strategies': 2,
                    'min_confidence': 0.6,
                    'consensus_threshold': 0.7
                }
            }
    
    def initialize_strategies(self):
        """Initialize all strategy instances"""
        strategy_configs = self.config.get('strategies', {})
        
        if strategy_configs.get('sma_macd', {}).get('enabled', True):
            # We'll handle SMA MACD in the generate_signals method
            self.aggregation_weights[StrategyType.SMA_MACD] = strategy_configs.get('sma_macd', {}).get('weight', 0.25)
        
        if strategy_configs.get('ema_macd', {}).get('enabled', True):
            self.strategies[StrategyType.EMA_MACD] = EMAMACDStrategy(strategy_configs.get('ema_macd', {}))
            self.aggregation_weights[StrategyType.EMA_MACD] = strategy_configs.get('ema_macd', {}).get('weight', 0.25)
        
        if strategy_configs.get('triple_ema', {}).get('enabled', True):
            self.strategies[StrategyType.TRIPLE_EMA] = TripleEMAStrategy(strategy_configs.get('triple_ema', {}))
            self.aggregation_weights[StrategyType.TRIPLE_EMA] = strategy_configs.get('triple_ema', {}).get('weight', 0.20)
        
        if strategy_configs.get('adaptive_volatility', {}).get('enabled', True):
            self.strategies[StrategyType.ADAPTIVE_VOLATILITY] = AdaptiveVolatilityStrategy(strategy_configs.get('adaptive_volatility', {}))
            self.aggregation_weights[StrategyType.ADAPTIVE_VOLATILITY] = strategy_configs.get('adaptive_volatility', {}).get('weight', 0.15)
        
        if strategy_configs.get('bollinger_rsi', {}).get('enabled', True):
            self.strategies[StrategyType.BOLLINGER_RSI] = BollingerRSIStrategy(strategy_configs.get('bollinger_rsi', {}))
            self.aggregation_weights[StrategyType.BOLLINGER_RSI] = strategy_configs.get('bollinger_rsi', {}).get('weight', 0.15)
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate signals from market data (updated interface for trading system)
        
        Args:
            market_data: Dictionary with symbol data from MarketDataEngine
            
        Returns:
            List of trading signals
        """
        all_signals = []
        
        if not market_data:
            return all_signals
        
        for symbol, data in market_data.items():
            df = data.get('data')
            timeframe = data.get('timeframe', 'M1')
            
            if df is not None and not df.empty:
                # Use new SMA MACD strategy if available
                if SMA_STRATEGY_AVAILABLE:
                    try:
                        sma_strategy = SMAMACDStrategy()
                        signal = sma_strategy.generate_signal(df, symbol)
                        if signal and signal.signal.value != 0:  # Not HOLD
                            # Convert to multi-strategy format
                            trading_signal = TradingSignal(
                                strategy=StrategyType.SMA_MACD,
                                signal=signal.signal,
                                strength=signal.strength,
                                price=signal.price,
                                timestamp=signal.timestamp,
                                timeframe=timeframe,
                                confidence=signal.confidence,
                                metadata=signal.metadata
                            )
                            all_signals.append(trading_signal)
                    except Exception as e:
                        logger.error(f"Error with new SMA MACD strategy for {symbol}: {str(e)}")
                
                # Generate signals from legacy strategies
                legacy_signals = self.generate_signals_legacy(df, symbol, timeframe)
                all_signals.extend(legacy_signals)
        
        return all_signals
    
    def generate_signals_legacy(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Generate signals from all enabled strategies (legacy method)"""
        signals = []
        
        # Skip SMA_MACD if new strategy is available (already handled)
        strategies_to_use = {k: v for k, v in self.strategies.items() 
                           if not (SMA_STRATEGY_AVAILABLE and k == StrategyType.SMA_MACD)}
        
        with ThreadPoolExecutor(max_workers=len(strategies_to_use)) as executor:
            future_to_strategy = {
                executor.submit(strategy.generate_signal, data, symbol, timeframe): strategy_type
                for strategy_type, strategy in strategies_to_use.items()
            }
            
            for future in future_to_strategy:
                try:
                    signal = future.result()
                    if signal and signal.signal != SignalType.HOLD:
                        signals.append(signal)
                except Exception as e:
                    strategy_type = future_to_strategy[future]
                    logger.error(f"Error generating signal for {strategy_type}: {str(e)}")
        
        return signals
    
    def aggregate_signals(self, signals: List[TradingSignal]) -> Optional[AggregatedSignal]:
        """Aggregate multiple strategy signals into a single trading decision"""
        if not signals:
            return None
        
        aggregation_config = self.config.get('aggregation', {})
        min_strategies = aggregation_config.get('min_strategies', 2)
        min_confidence = aggregation_config.get('min_confidence', 0.6)
        consensus_threshold = aggregation_config.get('consensus_threshold', 0.7)
        
        # Separate buy and sell signals
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for signal in buy_signals:
            weight = self.aggregation_weights.get(signal.strategy, 0.1)
            buy_score += weight * signal.strength * signal.confidence
            total_weight += weight
        
        for signal in sell_signals:
            weight = self.aggregation_weights.get(signal.strategy, 0.1)
            sell_score += weight * signal.strength * signal.confidence
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Normalize scores
        buy_score = buy_score / total_weight if buy_signals else 0
        sell_score = sell_score / total_weight if sell_signals else 0
        
        # Determine final signal
        final_signal = SignalType.HOLD
        final_strength = 0.0
        final_confidence = 0.0
        contributing_strategies = []
        
        if buy_score > sell_score and buy_score > consensus_threshold and len(buy_signals) >= min_strategies:
            final_signal = SignalType.BUY
            final_strength = buy_score
            final_confidence = np.mean([s.confidence for s in buy_signals])
            contributing_strategies = [s.strategy for s in buy_signals]
        elif sell_score > buy_score and sell_score > consensus_threshold and len(sell_signals) >= min_strategies:
            final_signal = SignalType.SELL
            final_strength = sell_score
            final_confidence = np.mean([s.confidence for s in sell_signals])
            contributing_strategies = [s.strategy for s in sell_signals]
        
        # Check minimum confidence requirement
        if final_confidence < min_confidence:
            final_signal = SignalType.HOLD
        
        if final_signal == SignalType.HOLD:
            return None
        
        return AggregatedSignal(
            final_signal=final_signal,
            strength=final_strength,
            confidence=final_confidence,
            contributing_strategies=contributing_strategies,
            signals=signals,
            price=signals[0].price,
            timestamp=signals[0].timestamp
        )
    
    def process_symbol(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[AggregatedSignal]:
        """Process a single symbol and return aggregated signal"""
        try:
            # Generate signals from all strategies
            signals = self.generate_signals(data, symbol, timeframe)
            
            if not signals:
                return None
            
            # Aggregate signals
            aggregated_signal = self.aggregate_signals(signals)
            
            # Store in history
            if aggregated_signal:
                self.signal_history.append(aggregated_signal)
                # Keep only last 1000 signals
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]
            
            return aggregated_signal
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return None
    
    def get_strategy_performance(self) -> Dict[StrategyType, Dict[str, float]]:
        """Get performance metrics for each strategy"""
        performance = {}
        
        for strategy_type in self.strategies.keys():
            strategy_signals = [s for s in self.signal_history 
                              if strategy_type in s.contributing_strategies]
            
            if strategy_signals:
                avg_confidence = np.mean([s.confidence for s in strategy_signals])
                avg_strength = np.mean([s.strength for s in strategy_signals])
                signal_count = len(strategy_signals)
                
                performance[strategy_type] = {
                    'avg_confidence': avg_confidence,
                    'avg_strength': avg_strength,
                    'signal_count': signal_count,
                    'contribution_rate': signal_count / len(self.signal_history) if self.signal_history else 0
                }
            else:
                performance[strategy_type] = {
                    'avg_confidence': 0.0,
                    'avg_strength': 0.0,
                    'signal_count': 0,
                    'contribution_rate': 0.0
                }
        
        return performance

if __name__ == "__main__":
    # Example usage
    engine = MultiStrategyEngine()
    
    # Sample data
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='5T'))
    
    # Process symbol
    aggregated_signal = engine.process_symbol(sample_data, "R_75", "M5")
    
    if aggregated_signal:
        print(f"Signal: {aggregated_signal.final_signal.name}")
        print(f"Strength: {aggregated_signal.strength:.3f}")
        print(f"Confidence: {aggregated_signal.confidence:.3f}")
        print(f"Contributing strategies: {[s.name for s in aggregated_signal.contributing_strategies]}")
    else:
        print("No signal generated")
    
    # Get performance metrics
    performance = engine.get_strategy_performance()
    for strategy, metrics in performance.items():
        print(f"{strategy.name}: {metrics}")