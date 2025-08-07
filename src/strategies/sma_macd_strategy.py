"""
Basic SMA8/50 + MACD Strategy Implementation
The core strategy that combines Simple Moving Averages with MACD for signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal: SignalType
    strength: float  # 0.0 to 1.0
    price: float
    symbol: str
    timestamp: pd.Timestamp
    strategy: str = "SMA8_50_MACD"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

class SMAMACDStrategy:
    """
    SMA8/50 + MACD Strategy
    
    Entry Signals:
    - BUY: SMA8 > SMA50 AND MACD > Signal Line AND MACD > 0
    - SELL: SMA8 < SMA50 AND MACD < Signal Line AND MACD < 0
    
    This is a trend-following strategy that combines:
    1. SMA crossover for trend direction
    2. MACD for momentum confirmation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy
        
        Args:
            config: Strategy configuration parameters
        """
        # Default parameters
        self.config = config or {}
        self.sma_fast = self.config.get('sma_fast', 8)
        self.sma_slow = self.config.get('sma_slow', 50)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        # Signal thresholds
        self.min_strength = self.config.get('min_strength', 0.6)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        logger.info(f"SMA MACD Strategy initialized: SMA({self.sma_fast},{self.sma_slow}), MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with indicators added
        """
        if df is None or df.empty or len(df) < self.sma_slow:
            return df
        
        df = df.copy()
        
        try:
            # Calculate SMAs
            df['sma_fast'] = df['close'].rolling(window=self.sma_fast).mean()
            df['sma_slow'] = df['close'].rolling(window=self.sma_slow).mean()
            
            # Calculate MACD
            df['ema_fast'] = df['close'].ewm(span=self.macd_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=self.macd_slow).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Calculate additional indicators for signal strength
            df['sma_diff'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow'] * 100
            df['macd_momentum'] = df['macd'] - df['macd'].shift(1)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on current market data
        
        Args:
            df: DataFrame with OHLC data and indicators
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None if no signal
        """
        if df is None or df.empty or len(df) < self.sma_slow + 1:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        if df is None or df.empty:
            return None
        
        # Get latest values
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        try:
            # Check for required indicators
            required_fields = ['sma_fast', 'sma_slow', 'macd', 'macd_signal', 'close']
            if not all(field in latest.index and pd.notna(latest[field]) for field in required_fields):
                return None
            
            # Current values
            sma_fast = latest['sma_fast']
            sma_slow = latest['sma_slow']
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            close_price = latest['close']
            
            # Previous values for trend confirmation
            prev_sma_fast = previous['sma_fast'] if 'sma_fast' in previous.index else sma_fast
            prev_sma_slow = previous['sma_slow'] if 'sma_slow' in previous.index else sma_slow
            
            # Signal conditions
            sma_bullish = sma_fast > sma_slow
            sma_bearish = sma_fast < sma_slow
            macd_bullish = macd > macd_signal and macd > 0
            macd_bearish = macd < macd_signal and macd < 0
            
            # Trend confirmation (SMA crossover)
            sma_cross_up = (sma_fast > sma_slow) and (prev_sma_fast <= prev_sma_slow)
            sma_cross_down = (sma_fast < sma_slow) and (prev_sma_fast >= prev_sma_slow)
            
            # Calculate signal strength and confidence
            strength = 0.0
            confidence = 0.0
            signal_type = SignalType.HOLD
            
            # BUY Signal
            if sma_bullish and macd_bullish:
                signal_type = SignalType.BUY
                
                # Base strength from conditions
                strength = 0.6
                
                # Bonus for crossover
                if sma_cross_up:
                    strength += 0.2
                
                # MACD momentum bonus
                if macd > 0 and latest.get('macd_momentum', 0) > 0:
                    strength += 0.1
                
                # SMA separation bonus
                sma_separation = abs(sma_fast - sma_slow) / sma_slow
                if sma_separation > 0.001:  # 0.1%
                    strength += 0.1
                
                confidence = min(strength, 1.0)
            
            # SELL Signal
            elif sma_bearish and macd_bearish:
                signal_type = SignalType.SELL
                
                # Base strength from conditions
                strength = 0.6
                
                # Bonus for crossover
                if sma_cross_down:
                    strength += 0.2
                
                # MACD momentum bonus
                if macd < 0 and latest.get('macd_momentum', 0) < 0:
                    strength += 0.1
                
                # SMA separation bonus
                sma_separation = abs(sma_fast - sma_slow) / sma_slow
                if sma_separation > 0.001:  # 0.1%
                    strength += 0.1
                
                confidence = min(strength, 1.0)
            
            # Filter weak signals
            if strength < self.min_strength or confidence < self.min_confidence:
                signal_type = SignalType.HOLD
                strength = 0.0
                confidence = 0.0
            
            # Create signal if not HOLD
            if signal_type != SignalType.HOLD:
                return TradingSignal(
                    signal=signal_type,
                    strength=strength,
                    price=close_price,
                    symbol=symbol,
                    timestamp=pd.Timestamp.now(),
                    strategy="SMA8_50_MACD",
                    confidence=confidence,
                    metadata={
                        'sma_fast': sma_fast,
                        'sma_slow': sma_slow,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'sma_bullish': sma_bullish,
                        'sma_bearish': sma_bearish,
                        'macd_bullish': macd_bullish,
                        'macd_bearish': macd_bearish,
                        'sma_cross_up': sma_cross_up,
                        'sma_cross_down': sma_cross_down
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            'name': 'SMA8/50 + MACD',
            'type': 'Trend Following',
            'parameters': {
                'sma_fast': self.sma_fast,
                'sma_slow': self.sma_slow,
                'macd_fast': self.macd_fast,
                'macd_slow': self.macd_slow,
                'macd_signal': self.macd_signal,
                'min_strength': self.min_strength,
                'min_confidence': self.min_confidence
            },
            'description': 'Combines SMA crossover with MACD confirmation for trend-following signals'
        }