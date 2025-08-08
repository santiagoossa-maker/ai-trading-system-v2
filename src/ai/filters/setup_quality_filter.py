"""
Setup Quality Filter
Advanced analysis of trading setup quality using divergences, S/R levels, and confluences
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

# Technical analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class SetupQuality(Enum):
    """Trading setup quality classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    INVALID = "invalid"

class DivergenceType(Enum):
    """Types of divergences"""
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    NO_DIVERGENCE = "no_divergence"

class SupportResistanceLevel(Enum):
    """Support/Resistance level strength"""
    MAJOR = "major"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    WEAK = "weak"

@dataclass
class DivergenceSignal:
    """Container for divergence analysis"""
    type: DivergenceType
    strength: float  # 0-1
    lookback_periods: int
    price_points: Tuple[float, float]
    indicator_points: Tuple[float, float]
    confidence: float

@dataclass
class SupportResistance:
    """Container for S/R level analysis"""
    level: float
    level_type: str  # 'support' or 'resistance'
    strength: SupportResistanceLevel
    touches: int
    distance: float  # Distance from current price
    age: int  # Periods since level was established
    confidence: float

@dataclass
class SetupAnalysis:
    """Complete trading setup quality analysis"""
    overall_quality: SetupQuality
    quality_score: float  # 0-100
    confidence: float
    
    # Component analyses
    divergences: List[DivergenceSignal]
    support_resistance: List[SupportResistance]
    confluences: List[str]
    
    # Quality factors
    trend_alignment: float
    volume_confirmation: float
    momentum_strength: float
    risk_reward_potential: float
    
    # Metadata
    analysis_timestamp: pd.Timestamp
    data_quality: float
    setup_metadata: Dict[str, Any]

class SetupQualityFilter:
    """
    Advanced setup quality filter that analyzes divergences, support/resistance levels,
    confluences, and other factors to determine trading setup quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the setup quality filter
        
        Args:
            config: Configuration parameters for setup analysis
        """
        self.config = config or self._get_default_config()
        self.analysis_history = []
        self.support_resistance_levels = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for setup analysis"""
        return {
            'divergence': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'lookback_min': 10,
                'lookback_max': 50,
                'min_strength': 0.3
            },
            'support_resistance': {
                'lookback_period': 100,
                'min_touches': 2,
                'touch_tolerance': 0.002,  # 0.2% tolerance
                'major_level_min_touches': 4,
                'level_age_weight': 0.8
            },
            'confluence': {
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                'moving_averages': [20, 50, 100, 200],
                'confluence_distance': 0.005,  # 0.5% distance for confluence
                'min_confluences': 2
            },
            'volume': {
                'volume_sma_period': 20,
                'volume_spike_threshold': 1.5,
                'volume_trend_period': 10
            },
            'momentum': {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_zero_line_weight': 0.3
            },
            'quality_weights': {
                'divergence': 0.25,
                'support_resistance': 0.25,
                'confluence': 0.20,
                'trend_alignment': 0.15,
                'volume': 0.10,
                'momentum': 0.05
            }
        }
    
    def analyze_setup_quality(self, data: pd.DataFrame, 
                            signal_direction: int = 0,
                            timeframe: str = "M5") -> SetupAnalysis:
        """
        Perform comprehensive setup quality analysis
        
        Args:
            data: OHLCV price data
            signal_direction: 1 for buy setup, -1 for sell setup, 0 for neutral
            timeframe: Data timeframe
            
        Returns:
            Complete setup quality analysis
        """
        try:
            if len(data) < 50:
                logger.warning("Insufficient data for setup quality analysis")
                return self._create_default_analysis()
            
            # Analyze divergences
            divergences = self._analyze_divergences(data, signal_direction)
            
            # Analyze support/resistance levels
            support_resistance = self._analyze_support_resistance(data)
            
            # Analyze confluences
            confluences = self._analyze_confluences(data, support_resistance)
            
            # Calculate component scores
            trend_alignment = self._calculate_trend_alignment(data, signal_direction)
            volume_confirmation = self._calculate_volume_confirmation(data, signal_direction)
            momentum_strength = self._calculate_momentum_strength(data, signal_direction)
            risk_reward_potential = self._calculate_risk_reward_potential(
                data, support_resistance, signal_direction
            )
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                divergences, support_resistance, confluences,
                trend_alignment, volume_confirmation, momentum_strength
            )
            
            # Determine quality classification
            overall_quality = self._classify_setup_quality(quality_score)
            
            # Calculate confidence
            confidence = self._calculate_setup_confidence(data, quality_score)
            
            # Assess data quality
            data_quality = self._assess_data_quality(data)
            
            analysis = SetupAnalysis(
                overall_quality=overall_quality,
                quality_score=quality_score,
                confidence=confidence,
                divergences=divergences,
                support_resistance=support_resistance,
                confluences=confluences,
                trend_alignment=trend_alignment,
                volume_confirmation=volume_confirmation,
                momentum_strength=momentum_strength,
                risk_reward_potential=risk_reward_potential,
                analysis_timestamp=pd.Timestamp.now(),
                data_quality=data_quality,
                setup_metadata=self._generate_setup_metadata(data, signal_direction)
            )
            
            # Store in history
            self.analysis_history.append({
                'timestamp': data.index[-1],
                'quality': overall_quality,
                'score': quality_score,
                'confidence': confidence,
                'signal_direction': signal_direction
            })
            
            # Keep only recent history
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-1000:]
            
            logger.debug(f"Setup quality: {overall_quality.value} "
                        f"(score: {quality_score:.1f}, confidence: {confidence:.3f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in setup quality analysis: {str(e)}")
            return self._create_default_analysis()
    
    def _analyze_divergences(self, data: pd.DataFrame, signal_direction: int) -> List[DivergenceSignal]:
        """Analyze price-indicator divergences"""
        try:
            divergences = []
            
            # RSI Divergence
            rsi_divergence = self._detect_rsi_divergence(data, signal_direction)
            if rsi_divergence.type != DivergenceType.NO_DIVERGENCE:
                divergences.append(rsi_divergence)
            
            # MACD Divergence
            macd_divergence = self._detect_macd_divergence(data, signal_direction)
            if macd_divergence.type != DivergenceType.NO_DIVERGENCE:
                divergences.append(macd_divergence)
            
            # Volume Divergence
            volume_divergence = self._detect_volume_divergence(data, signal_direction)
            if volume_divergence.type != DivergenceType.NO_DIVERGENCE:
                divergences.append(volume_divergence)
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error analyzing divergences: {str(e)}")
            return []
    
    def _detect_rsi_divergence(self, data: pd.DataFrame, signal_direction: int) -> DivergenceSignal:
        """Detect RSI divergences"""
        try:
            period = self.config['divergence']['rsi_period']
            lookback_min = self.config['divergence']['lookback_min']
            lookback_max = self.config['divergence']['lookback_max']
            
            if len(data) < lookback_max:
                return DivergenceSignal(
                    DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
                )
            
            close = data['close'].values
            
            # Calculate RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(close, timeperiod=period)
            else:
                # Fallback RSI calculation
                delta = np.diff(close)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = pd.Series(gain).rolling(period).mean()
                avg_loss = pd.Series(loss).rolling(period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi = np.concatenate([[50], rsi.values])  # Add initial value
            
            # Look for divergence patterns
            for lookback in range(lookback_min, min(lookback_max, len(data))):
                # Recent vs historical extremes
                recent_price = close[-1]
                recent_rsi = rsi[-1]
                historical_price = close[-lookback]
                historical_rsi = rsi[-lookback]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (signal_direction >= 0 and
                    recent_price < historical_price and
                    recent_rsi > historical_rsi and
                    recent_rsi < 50):
                    
                    strength = (historical_rsi - recent_rsi) / 50.0
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BULLISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_rsi, recent_rsi),
                            0.7
                        )
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                elif (signal_direction <= 0 and
                      recent_price > historical_price and
                      recent_rsi < historical_rsi and
                      recent_rsi > 50):
                    
                    strength = (recent_rsi - historical_rsi) / 50.0
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BEARISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_rsi, recent_rsi),
                            0.7
                        )
            
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {str(e)}")
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
    
    def _detect_macd_divergence(self, data: pd.DataFrame, signal_direction: int) -> DivergenceSignal:
        """Detect MACD divergences"""
        try:
            fast = self.config['divergence']['macd_fast']
            slow = self.config['divergence']['macd_slow']
            signal = self.config['divergence']['macd_signal']
            lookback_min = self.config['divergence']['lookback_min']
            lookback_max = self.config['divergence']['lookback_max']
            
            if len(data) < lookback_max:
                return DivergenceSignal(
                    DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
                )
            
            close = data['close'].values
            
            # Calculate MACD
            if TALIB_AVAILABLE:
                macd, macd_signal_line, macd_histogram = talib.MACD(
                    close, fastperiod=fast, slowperiod=slow, signalperiod=signal
                )
            else:
                # Fallback MACD calculation
                ema_fast = pd.Series(close).ewm(span=fast).mean()
                ema_slow = pd.Series(close).ewm(span=slow).mean()
                macd = (ema_fast - ema_slow).values
                macd_signal_line = pd.Series(macd).ewm(span=signal).mean().values
                macd_histogram = macd - macd_signal_line
            
            # Look for divergence patterns using MACD histogram
            for lookback in range(lookback_min, min(lookback_max, len(data))):
                recent_price = close[-1]
                recent_macd = macd_histogram[-1]
                historical_price = close[-lookback]
                historical_macd = macd_histogram[-lookback]
                
                # Skip if MACD values are NaN
                if np.isnan(recent_macd) or np.isnan(historical_macd):
                    continue
                
                # Bullish divergence: price lower low, MACD higher low
                if (signal_direction >= 0 and
                    recent_price < historical_price and
                    recent_macd > historical_macd and
                    recent_macd < 0):
                    
                    strength = abs(recent_macd - historical_macd) / max(abs(recent_macd), abs(historical_macd), 0.001)
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BULLISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_macd, recent_macd),
                            0.8
                        )
                
                # Bearish divergence: price higher high, MACD lower high
                elif (signal_direction <= 0 and
                      recent_price > historical_price and
                      recent_macd < historical_macd and
                      recent_macd > 0):
                    
                    strength = abs(historical_macd - recent_macd) / max(abs(recent_macd), abs(historical_macd), 0.001)
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BEARISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_macd, recent_macd),
                            0.8
                        )
            
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
            
        except Exception as e:
            logger.error(f"Error detecting MACD divergence: {str(e)}")
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
    
    def _detect_volume_divergence(self, data: pd.DataFrame, signal_direction: int) -> DivergenceSignal:
        """Detect volume divergences"""
        try:
            if 'volume' not in data.columns:
                return DivergenceSignal(
                    DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
                )
            
            lookback_min = self.config['divergence']['lookback_min']
            lookback_max = self.config['divergence']['lookback_max']
            
            close = data['close'].values
            volume = data['volume'].values
            
            # Look for volume divergence patterns
            for lookback in range(lookback_min, min(lookback_max, len(data))):
                recent_price = close[-1]
                recent_volume = volume[-1]
                historical_price = close[-lookback]
                historical_volume = volume[-lookback]
                
                # Bearish divergence: price higher high, volume lower
                if (signal_direction <= 0 and
                    recent_price > historical_price and
                    recent_volume < historical_volume):
                    
                    price_change = (recent_price - historical_price) / historical_price
                    volume_change = (historical_volume - recent_volume) / historical_volume
                    strength = min(price_change * volume_change * 10, 1.0)
                    
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BEARISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_volume, recent_volume),
                            0.6
                        )
                
                # Bullish divergence: price lower low, volume higher (selling climax)
                elif (signal_direction >= 0 and
                      recent_price < historical_price and
                      recent_volume > historical_volume):
                    
                    price_change = (historical_price - recent_price) / historical_price
                    volume_change = (recent_volume - historical_volume) / historical_volume
                    strength = min(price_change * volume_change * 10, 1.0)
                    
                    if strength >= self.config['divergence']['min_strength']:
                        return DivergenceSignal(
                            DivergenceType.BULLISH_REGULAR,
                            strength,
                            lookback,
                            (historical_price, recent_price),
                            (historical_volume, recent_volume),
                            0.6
                        )
            
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
            
        except Exception as e:
            logger.error(f"Error detecting volume divergence: {str(e)}")
            return DivergenceSignal(
                DivergenceType.NO_DIVERGENCE, 0.0, 0, (0.0, 0.0), (0.0, 0.0), 0.0
            )
    
    def _analyze_support_resistance(self, data: pd.DataFrame) -> List[SupportResistance]:
        """Analyze support and resistance levels"""
        try:
            lookback = self.config['support_resistance']['lookback_period']
            min_touches = self.config['support_resistance']['min_touches']
            tolerance = self.config['support_resistance']['touch_tolerance']
            
            if len(data) < lookback:
                lookback = len(data)
            
            recent_data = data.tail(lookback)
            current_price = data['close'].iloc[-1]
            
            # Find potential S/R levels using pivot points
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            # Simple pivot detection
            window = 5
            for i in range(window, len(highs) - window):
                # Local high (resistance)
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append((highs[i], i))
                
                # Local low (support)
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append((lows[i], i))
            
            # Group nearby levels and count touches
            sr_levels = []
            
            # Process resistance levels
            for level, age_idx in resistance_levels:
                touches = self._count_level_touches(recent_data, level, tolerance, 'resistance')
                if touches >= min_touches:
                    distance = (level - current_price) / current_price
                    age = len(recent_data) - age_idx
                    strength = self._classify_level_strength(touches, age, distance)
                    confidence = min(touches / 5.0, 1.0)
                    
                    sr_levels.append(SupportResistance(
                        level=level,
                        level_type='resistance',
                        strength=strength,
                        touches=touches,
                        distance=distance,
                        age=age,
                        confidence=confidence
                    ))
            
            # Process support levels
            for level, age_idx in support_levels:
                touches = self._count_level_touches(recent_data, level, tolerance, 'support')
                if touches >= min_touches:
                    distance = (current_price - level) / current_price
                    age = len(recent_data) - age_idx
                    strength = self._classify_level_strength(touches, age, distance)
                    confidence = min(touches / 5.0, 1.0)
                    
                    sr_levels.append(SupportResistance(
                        level=level,
                        level_type='support',
                        strength=strength,
                        touches=touches,
                        distance=distance,
                        age=age,
                        confidence=confidence
                    ))
            
            # Sort by strength and distance
            sr_levels.sort(key=lambda x: (x.strength.value, -x.distance), reverse=True)
            
            return sr_levels[:10]  # Return top 10 levels
            
        except Exception as e:
            logger.error(f"Error analyzing support/resistance: {str(e)}")
            return []
    
    def _count_level_touches(self, data: pd.DataFrame, level: float, 
                           tolerance: float, level_type: str) -> int:
        """Count how many times price touched a S/R level"""
        try:
            touches = 0
            
            if level_type == 'resistance':
                # Count times high was near the level
                for high in data['high']:
                    if abs(high - level) / level <= tolerance:
                        touches += 1
            else:  # support
                # Count times low was near the level
                for low in data['low']:
                    if abs(low - level) / level <= tolerance:
                        touches += 1
            
            return touches
            
        except Exception as e:
            logger.error(f"Error counting level touches: {str(e)}")
            return 0
    
    def _classify_level_strength(self, touches: int, age: int, distance: float) -> SupportResistanceLevel:
        """Classify strength of S/R level"""
        try:
            major_touches = self.config['support_resistance']['major_level_min_touches']
            
            # Base score from touches
            score = touches
            
            # Age factor (older levels are stronger)
            age_factor = min(age / 50.0, 1.0)
            score *= (1 + age_factor)
            
            # Distance factor (closer levels are more relevant)
            distance_factor = max(0.1, 1.0 - abs(distance) * 10)
            score *= distance_factor
            
            if score >= major_touches * 1.5:
                return SupportResistanceLevel.MAJOR
            elif score >= major_touches:
                return SupportResistanceLevel.INTERMEDIATE
            elif score >= 2:
                return SupportResistanceLevel.MINOR
            else:
                return SupportResistanceLevel.WEAK
                
        except Exception as e:
            logger.error(f"Error classifying level strength: {str(e)}")
            return SupportResistanceLevel.WEAK
    
    def _analyze_confluences(self, data: pd.DataFrame, 
                           sr_levels: List[SupportResistance]) -> List[str]:
        """Analyze confluences of technical levels"""
        try:
            confluences = []
            current_price = data['close'].iloc[-1]
            confluence_distance = self.config['confluence']['confluence_distance']
            
            # Check S/R confluence
            nearby_sr = [sr for sr in sr_levels 
                        if abs(sr.level - current_price) / current_price <= confluence_distance]
            
            if len(nearby_sr) >= 2:
                confluences.append(f"Multiple S/R levels confluence ({len(nearby_sr)} levels)")
            
            # Check moving average confluence
            ma_periods = self.config['confluence']['moving_averages']
            nearby_mas = []
            
            for period in ma_periods:
                if len(data) >= period:
                    ma_value = data['close'].rolling(period).mean().iloc[-1]
                    if abs(ma_value - current_price) / current_price <= confluence_distance:
                        nearby_mas.append(f"MA{period}")
            
            if len(nearby_mas) >= 2:
                confluences.append(f"Moving averages confluence ({', '.join(nearby_mas)})")
            
            # Check Fibonacci confluence (if we have a clear swing)
            fib_confluences = self._check_fibonacci_confluence(data, current_price)
            confluences.extend(fib_confluences)
            
            # Check round number confluence
            if self._is_round_number(current_price, confluence_distance):
                confluences.append("Round number confluence")
            
            return confluences
            
        except Exception as e:
            logger.error(f"Error analyzing confluences: {str(e)}")
            return []
    
    def _check_fibonacci_confluence(self, data: pd.DataFrame, current_price: float) -> List[str]:
        """Check for Fibonacci level confluences"""
        try:
            fib_levels = self.config['confluence']['fibonacci_levels']
            confluence_distance = self.config['confluence']['confluence_distance']
            confluences = []
            
            # Find recent swing high and low
            if len(data) < 20:
                return []
            
            recent_data = data.tail(50)
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Calculate Fibonacci retracement levels
            swing_range = swing_high - swing_low
            if swing_range <= 0:
                return []
            
            nearby_fibs = []
            for level in fib_levels:
                fib_price = swing_low + (swing_range * level)
                if abs(fib_price - current_price) / current_price <= confluence_distance:
                    nearby_fibs.append(f"Fib {level:.1%}")
            
            if len(nearby_fibs) >= 1:
                confluences.append(f"Fibonacci confluence ({', '.join(nearby_fibs)})")
            
            return confluences
            
        except Exception as e:
            logger.error(f"Error checking Fibonacci confluence: {str(e)}")
            return []
    
    def _is_round_number(self, price: float, tolerance: float) -> bool:
        """Check if price is near a round number"""
        try:
            # Check for round numbers (10, 50, 100, etc.)
            round_numbers = [10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000]
            
            for round_num in round_numbers:
                if abs(price - round_num) / price <= tolerance:
                    return True
            
            # Check for decade boundaries (1.20, 1.30, etc.)
            if price > 1:
                decimal_part = price - int(price)
                round_decimals = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
                
                for round_dec in round_decimals:
                    if abs(decimal_part - round_dec) <= tolerance * price:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking round number: {str(e)}")
            return False
    
    def _calculate_trend_alignment(self, data: pd.DataFrame, signal_direction: int) -> float:
        """Calculate trend alignment score"""
        try:
            if len(data) < 50:
                return 0.5
            
            close = data['close']
            
            # Multiple timeframe trend analysis
            ema_periods = [8, 21, 50]
            trends = []
            
            for period in ema_periods:
                if len(data) >= period:
                    ema = close.ewm(span=period).mean()
                    trend = 1 if ema.iloc[-1] > ema.iloc[-5] else -1
                    trends.append(trend)
            
            if not trends:
                return 0.5
            
            # Calculate alignment
            if signal_direction > 0:  # Buy signal
                alignment = sum(1 for t in trends if t > 0) / len(trends)
            elif signal_direction < 0:  # Sell signal
                alignment = sum(1 for t in trends if t < 0) / len(trends)
            else:  # Neutral
                alignment = 0.5
            
            return alignment
            
        except Exception as e:
            logger.error(f"Error calculating trend alignment: {str(e)}")
            return 0.5
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame, signal_direction: int) -> float:
        """Calculate volume confirmation score"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return 0.5
            
            volume = data['volume']
            volume_sma_period = self.config['volume']['volume_sma_period']
            
            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(volume_sma_period).mean().iloc[-1]
            
            if avg_volume <= 0:
                return 0.5
            
            volume_ratio = current_volume / avg_volume
            
            # Higher volume generally confirms the move
            if volume_ratio >= 1.5:
                return 0.9
            elif volume_ratio >= 1.2:
                return 0.7
            elif volume_ratio >= 0.8:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {str(e)}")
            return 0.5
    
    def _calculate_momentum_strength(self, data: pd.DataFrame, signal_direction: int) -> float:
        """Calculate momentum strength score"""
        try:
            if len(data) < 20:
                return 0.5
            
            close = data['close'].values
            
            # Calculate RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(close, timeperiod=14)
                current_rsi = rsi[-1]
            else:
                current_rsi = 50.0  # Default
            
            # Calculate momentum score based on RSI and signal direction
            if signal_direction > 0:  # Buy signal
                if current_rsi >= 50:
                    momentum = (current_rsi - 50) / 50
                else:
                    momentum = 0.3  # Oversold can be good for reversal
            elif signal_direction < 0:  # Sell signal
                if current_rsi <= 50:
                    momentum = (50 - current_rsi) / 50
                else:
                    momentum = 0.3  # Overbought can be good for reversal
            else:  # Neutral
                momentum = 0.5
            
            return min(momentum, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum strength: {str(e)}")
            return 0.5
    
    def _calculate_risk_reward_potential(self, data: pd.DataFrame,
                                       sr_levels: List[SupportResistance],
                                       signal_direction: int) -> float:
        """Calculate risk/reward potential based on S/R levels"""
        try:
            if not sr_levels or signal_direction == 0:
                return 0.5
            
            current_price = data['close'].iloc[-1]
            
            if signal_direction > 0:  # Buy signal
                # Find nearest support (stop loss) and resistance (target)
                supports = [sr for sr in sr_levels if sr.level_type == 'support' and sr.level < current_price]
                resistances = [sr for sr in sr_levels if sr.level_type == 'resistance' and sr.level > current_price]
                
                if supports and resistances:
                    nearest_support = max(supports, key=lambda x: x.level)
                    nearest_resistance = min(resistances, key=lambda x: x.level)
                    
                    risk = current_price - nearest_support.level
                    reward = nearest_resistance.level - current_price
                    
                    if risk > 0:
                        rr_ratio = reward / risk
                        # Normalize to 0-1 scale (RR of 2:1 = 0.8, 3:1 = 1.0)
                        return min(rr_ratio / 3.0, 1.0)
            
            elif signal_direction < 0:  # Sell signal
                # Find nearest resistance (stop loss) and support (target)
                resistances = [sr for sr in sr_levels if sr.level_type == 'resistance' and sr.level > current_price]
                supports = [sr for sr in sr_levels if sr.level_type == 'support' and sr.level < current_price]
                
                if resistances and supports:
                    nearest_resistance = min(resistances, key=lambda x: x.level)
                    nearest_support = max(supports, key=lambda x: x.level)
                    
                    risk = nearest_resistance.level - current_price
                    reward = current_price - nearest_support.level
                    
                    if risk > 0:
                        rr_ratio = reward / risk
                        return min(rr_ratio / 3.0, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward potential: {str(e)}")
            return 0.5
    
    def _calculate_quality_score(self, divergences: List[DivergenceSignal],
                               sr_levels: List[SupportResistance],
                               confluences: List[str],
                               trend_alignment: float,
                               volume_confirmation: float,
                               momentum_strength: float) -> float:
        """Calculate overall quality score"""
        try:
            weights = self.config['quality_weights']
            
            # Divergence score
            divergence_score = 0.0
            if divergences:
                avg_strength = np.mean([d.strength for d in divergences])
                avg_confidence = np.mean([d.confidence for d in divergences])
                divergence_score = avg_strength * avg_confidence
            
            # Support/resistance score
            sr_score = 0.0
            if sr_levels:
                strong_levels = [sr for sr in sr_levels if sr.strength in [SupportResistanceLevel.MAJOR, SupportResistanceLevel.INTERMEDIATE]]
                sr_score = min(len(strong_levels) / 3.0, 1.0)
            
            # Confluence score
            confluence_score = min(len(confluences) / 3.0, 1.0)
            
            # Calculate weighted total
            total_score = (
                divergence_score * weights['divergence'] +
                sr_score * weights['support_resistance'] +
                confluence_score * weights['confluence'] +
                trend_alignment * weights['trend_alignment'] +
                volume_confirmation * weights['volume'] +
                momentum_strength * weights['momentum']
            )
            
            return min(total_score * 100, 100.0)  # Convert to 0-100 scale
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 50.0
    
    def _classify_setup_quality(self, score: float) -> SetupQuality:
        """Classify setup quality based on score"""
        if score >= 80:
            return SetupQuality.EXCELLENT
        elif score >= 65:
            return SetupQuality.GOOD
        elif score >= 45:
            return SetupQuality.AVERAGE
        elif score >= 25:
            return SetupQuality.POOR
        else:
            return SetupQuality.INVALID
    
    def _calculate_setup_confidence(self, data: pd.DataFrame, quality_score: float) -> float:
        """Calculate confidence in setup analysis"""
        try:
            confidence_factors = []
            
            # Data quality factor
            if len(data) >= 100:
                confidence_factors.append(0.9)
            elif len(data) >= 50:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Score consistency factor
            if quality_score >= 70:
                confidence_factors.append(0.9)
            elif quality_score >= 50:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Technical indicator availability
            if TALIB_AVAILABLE:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating setup confidence: {str(e)}")
            return 0.5
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess quality of input data"""
        try:
            quality_factors = []
            
            # Completeness
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_factors.append(1.0 - missing_ratio)
            
            # Consistency
            if (data['high'] >= data['low']).all():
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
            
            # Sufficiency
            if len(data) >= 100:
                quality_factors.append(1.0)
            elif len(data) >= 50:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return 0.7
    
    def _generate_setup_metadata(self, data: pd.DataFrame, signal_direction: int) -> Dict[str, Any]:
        """Generate metadata for setup analysis"""
        try:
            return {
                'data_points': len(data),
                'signal_direction': signal_direction,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'talib_available': TALIB_AVAILABLE,
                'indicators_calculated': ['RSI', 'MACD', 'Volume', 'S/R', 'Confluences'],
                'config_version': '1.0'
            }
        except Exception as e:
            logger.error(f"Error generating setup metadata: {str(e)}")
            return {}
    
    def _create_default_analysis(self) -> SetupAnalysis:
        """Create default analysis when calculation fails"""
        return SetupAnalysis(
            overall_quality=SetupQuality.INVALID,
            quality_score=0.0,
            confidence=0.0,
            divergences=[],
            support_resistance=[],
            confluences=[],
            trend_alignment=0.5,
            volume_confirmation=0.5,
            momentum_strength=0.5,
            risk_reward_potential=0.5,
            analysis_timestamp=pd.Timestamp.now(),
            data_quality=0.0,
            setup_metadata={}
        )

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=200, freq='5T')
    
    # Simulate market data with some patterns
    base_price = 100
    trend = np.cumsum(np.random.randn(200) * 0.1) + base_price
    noise = np.random.randn(200) * 0.3
    
    sample_data = pd.DataFrame({
        'open': trend + noise,
        'high': trend + noise + np.abs(np.random.randn(200) * 0.4),
        'low': trend + noise - np.abs(np.random.randn(200) * 0.4),
        'close': trend + noise,
        'volume': np.random.randint(100, 2000, 200)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        ohlc = [
            sample_data['open'].iloc[i],
            sample_data['high'].iloc[i],
            sample_data['low'].iloc[i],
            sample_data['close'].iloc[i]
        ]
        sample_data['high'].iloc[i] = max(ohlc)
        sample_data['low'].iloc[i] = min(ohlc)
    
    # Initialize and test filter
    setup_filter = SetupQualityFilter()
    
    print("Setup Quality Filter Test")
    print("=" * 40)
    
    # Test buy setup analysis
    buy_analysis = setup_filter.analyze_setup_quality(sample_data, signal_direction=1, timeframe="M5")
    
    print(f"Buy Setup Analysis:")
    print(f"  Overall Quality: {buy_analysis.overall_quality.value}")
    print(f"  Quality Score: {buy_analysis.quality_score:.1f}/100")
    print(f"  Confidence: {buy_analysis.confidence:.3f}")
    print(f"  Divergences: {len(buy_analysis.divergences)}")
    print(f"  S/R Levels: {len(buy_analysis.support_resistance)}")
    print(f"  Confluences: {len(buy_analysis.confluences)}")
    print(f"  Trend Alignment: {buy_analysis.trend_alignment:.3f}")
    print(f"  Volume Confirmation: {buy_analysis.volume_confirmation:.3f}")
    print(f"  Risk/Reward Potential: {buy_analysis.risk_reward_potential:.3f}")
    
    if buy_analysis.support_resistance:
        print(f"\n  Top S/R Levels:")
        for i, sr in enumerate(buy_analysis.support_resistance[:3]):
            print(f"    {i+1}. {sr.level_type.title()}: {sr.level:.2f} "
                  f"({sr.strength.value}, {sr.touches} touches)")
    
    if buy_analysis.confluences:
        print(f"\n  Confluences:")
        for conf in buy_analysis.confluences:
            print(f"    - {conf}")
    
    print("\nSetup Quality Filter implementation completed!")