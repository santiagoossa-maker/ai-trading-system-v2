"""
SMA/MACD Trading Strategy
Simple Moving Average crossover with MACD confirmation strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    signal: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    symbol: str
    timeframe: str
    metadata: Dict[str, Any] = None

class SMAMACDStrategy:
    """
    Simple Moving Average + MACD Trading Strategy
    
    Strategy Logic:
    - SMA8 and SMA50 for trend direction
    - MACD for momentum confirmation
    - Buy when: SMA8 > SMA50 AND MACD > Signal AND MACD Histogram > 0
    - Sell when: SMA8 < SMA50 AND MACD < Signal AND MACD Histogram < 0
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with configuration"""
        if config is None:
            config = {}
        
        # SMA parameters
        self.sma_fast = config.get('sma_fast', 8)
        self.sma_slow = config.get('sma_slow', 50)
        
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # Signal filtering
        self.min_bars = config.get('min_bars', 60)
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Strategy state
        self.last_signal = None
        self.signal_history = []
        
        logger.info(f"SMA/MACD Strategy initialized with SMA({self.sma_fast},{self.sma_slow}) and MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2.0 / (period + 1.0)
        ema = np.full(len(prices), np.nan)
        
        # Initialize with first SMA value
        ema[period - 1] = np.mean(prices[:period])
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal line, and Histogram"""
        # Calculate EMAs for MACD
        ema_fast = self.calculate_ema(prices, self.macd_fast)
        ema_slow = self.calculate_ema(prices, self.macd_slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = self.calculate_ema(macd_line[~np.isnan(macd_line)], self.macd_signal)
        
        # Pad signal line to match original length
        signal_padded = np.full(len(prices), np.nan)
        valid_start = len(prices) - len(signal_line)
        signal_padded[valid_start:] = signal_line
        
        # Histogram
        histogram = macd_line - signal_padded
        
        return macd_line, signal_padded, histogram
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators"""
        try:
            close = data['close'].values
            
            # Check if we have enough data
            if len(close) < max(self.sma_slow, self.macd_slow) + self.macd_signal:
                logger.warning(f"Insufficient data: {len(close)} bars, need at least {max(self.sma_slow, self.macd_slow) + self.macd_signal}")
                return {}
            
            # Calculate SMAs
            sma_fast = self.calculate_sma(close, self.sma_fast)
            sma_slow = self.calculate_sma(close, self.sma_slow)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.calculate_macd(close)
            
            return {
                'sma_fast': sma_fast,
                'sma_slow': sma_slow,
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram,
                'close': close
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str = 'M5') -> Optional[TradingSignal]:
        """
        Generate trading signal based on SMA/MACD strategy
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            TradingSignal or None
        """
        try:
            if data.empty or len(data) < self.min_bars:
                logger.debug(f"Insufficient data for signal generation: {len(data)} bars")
                return None
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if not indicators:
                return None
            
            # Get latest values
            idx = -1  # Latest bar
            
            sma_fast = indicators['sma_fast'][idx]
            sma_slow = indicators['sma_slow'][idx]
            macd = indicators['macd'][idx]
            macd_signal = indicators['macd_signal'][idx]
            macd_histogram = indicators['macd_histogram'][idx]
            close = indicators['close'][idx]
            
            # Check for valid values
            if any(np.isnan([sma_fast, sma_slow, macd, macd_signal, macd_histogram])):
                logger.debug("Invalid indicator values (NaN)")
                return None
            
            # Generate signal
            signal_type = SignalType.HOLD
            strength = 0.0
            confidence = 0.0
            
            # SMA crossover conditions
            sma_bullish = sma_fast > sma_slow
            sma_bearish = sma_fast < sma_slow
            
            # MACD conditions
            macd_bullish = macd > macd_signal and macd_histogram > 0
            macd_bearish = macd < macd_signal and macd_histogram < 0
            
            # Combined signal logic
            if sma_bullish and macd_bullish:
                signal_type = SignalType.BUY
                # Calculate strength based on indicator separation
                sma_strength = (sma_fast - sma_slow) / sma_slow
                macd_strength = abs(macd_histogram) / max(abs(macd), 0.0001)
                strength = min(1.0, (sma_strength + macd_strength) * 10)
                confidence = 0.7
                
            elif sma_bearish and macd_bearish:
                signal_type = SignalType.SELL
                # Calculate strength based on indicator separation
                sma_strength = (sma_slow - sma_fast) / sma_slow
                macd_strength = abs(macd_histogram) / max(abs(macd), 0.0001)
                strength = min(1.0, (sma_strength + macd_strength) * 10)
                confidence = 0.7
            
            # Additional confirmation checks
            if signal_type != SignalType.HOLD:
                # Check for recent crossover (increases confidence)
                if len(indicators['sma_fast']) >= 3:
                    prev_sma_cross = (indicators['sma_fast'][-2] > indicators['sma_slow'][-2])
                    curr_sma_cross = (sma_fast > sma_slow)
                    if prev_sma_cross != curr_sma_cross:  # Recent crossover
                        confidence += 0.1
                
                # Check MACD momentum
                if len(indicators['macd_histogram']) >= 2:
                    prev_histogram = indicators['macd_histogram'][-2]
                    if not np.isnan(prev_histogram):
                        macd_momentum = macd_histogram > prev_histogram
                        if (signal_type == SignalType.BUY and macd_momentum) or \
                           (signal_type == SignalType.SELL and not macd_momentum):
                            confidence += 0.1
                
                # Check price position relative to SMAs
                if signal_type == SignalType.BUY and close > sma_fast:
                    confidence += 0.1
                elif signal_type == SignalType.SELL and close < sma_fast:
                    confidence += 0.1
            
            # Filter weak signals
            if confidence < self.min_confidence:
                signal_type = SignalType.HOLD
            
            # Only return signal if it's not HOLD
            if signal_type == SignalType.HOLD:
                return None
            
            # Create signal
            signal = TradingSignal(
                signal=signal_type,
                strength=min(1.0, max(0.0, strength)),
                confidence=min(1.0, max(0.0, confidence)),
                price=close,
                timestamp=data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                metadata={
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_histogram,
                    'sma_bullish': sma_bullish,
                    'sma_bearish': sma_bearish,
                    'macd_bullish': macd_bullish,
                    'macd_bearish': macd_bearish
                }
            )
            
            # Store signal
            self.last_signal = signal
            self.signal_history.append(signal)
            
            # Keep history manageable
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            logger.info(f"Generated {signal.signal.name} signal for {symbol} with strength {signal.strength:.3f} and confidence {signal.confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def get_current_indicators(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get current indicator values for analysis"""
        try:
            indicators = self.calculate_indicators(data)
            if not indicators:
                return None
            
            return {
                'sma_fast': indicators['sma_fast'][-1],
                'sma_slow': indicators['sma_slow'][-1],
                'macd': indicators['macd'][-1],
                'macd_signal': indicators['macd_signal'][-1],
                'macd_histogram': indicators['macd_histogram'][-1],
                'close': indicators['close'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error getting current indicators: {str(e)}")
            return None
    
    def get_signal_history(self, limit: int = 10) -> List[TradingSignal]:
        """Get recent signal history"""
        return self.signal_history[-limit:] if self.signal_history else []
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration"""
        return {
            'name': 'SMA_MACD',
            'sma_fast': self.sma_fast,
            'sma_slow': self.sma_slow,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'min_bars': self.min_bars,
            'min_confidence': self.min_confidence
        }

# Convenience functions
def create_strategy(config: Optional[Dict[str, Any]] = None) -> SMAMACDStrategy:
    """Create a new SMA/MACD strategy instance"""
    return SMAMACDStrategy(config)

def quick_signal(data: pd.DataFrame, symbol: str, timeframe: str = 'M5', 
                config: Optional[Dict[str, Any]] = None) -> Optional[TradingSignal]:
    """Quick signal generation without managing strategy instance"""
    strategy = SMAMACDStrategy(config)
    return strategy.generate_signal(data, symbol, timeframe)

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5T')
    
    # Generate trending data
    base_price = 100
    trend = np.linspace(0, 10, 100)  # Upward trend
    noise = np.random.normal(0, 0.5, 100)
    prices = base_price + trend + noise
    
    # Create OHLCV data
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.1, 100),
        'high': prices + np.abs(np.random.normal(0, 0.2, 100)),
        'low': prices - np.abs(np.random.normal(0, 0.2, 100)),
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    print("Testing SMA/MACD Strategy")
    print("=" * 40)
    
    # Test strategy
    strategy = SMAMACDStrategy()
    
    print(f"Strategy config: {strategy.get_strategy_config()}")
    print()
    
    # Generate signal
    signal = strategy.generate_signal(sample_data, "TEST_SYMBOL", "M5")
    
    if signal:
        print(f"Generated Signal:")
        print(f"  Type: {signal.signal.name}")
        print(f"  Strength: {signal.strength:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Price: {signal.price:.2f}")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Timeframe: {signal.timeframe}")
        print(f"  Metadata: {signal.metadata}")
    else:
        print("No signal generated")
    
    # Test indicators
    print("\nCurrent Indicators:")
    indicators = strategy.get_current_indicators(sample_data)
    if indicators:
        for name, value in indicators.items():
            print(f"  {name}: {value:.5f}")
    else:
        print("  No indicators available")
    
    print("\nTesting complete!")