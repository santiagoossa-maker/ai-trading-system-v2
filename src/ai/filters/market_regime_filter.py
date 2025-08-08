"""
Market Regime Filter
Advanced market condition detection using ADX, Choppiness Index, and volatility regimes
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

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_TRENDING = "strong_trending"
    TRENDING = "trending"
    RANGING = "ranging"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"

class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class TrendStrength(Enum):
    """Trend strength classifications"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class RegimeAnalysis:
    """Container for complete regime analysis"""
    primary_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_strength: TrendStrength
    confidence: float
    
    # Regime indicators
    adx_value: float
    choppiness_value: float
    volatility_ratio: float
    trend_direction: int  # 1 for up, -1 for down, 0 for sideways
    
    # Supporting metrics
    regime_persistence: float  # How long this regime has been active
    regime_stability: float    # How stable the regime identification is
    
    # Metadata
    analysis_timeframe: str
    data_quality: float
    regime_metadata: Dict[str, Any]

class MarketRegimeFilter:
    """
    Advanced market regime detection filter combining multiple technical indicators
    to classify market conditions for optimal strategy selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market regime filter
        
        Args:
            config: Configuration parameters for regime detection
        """
        self.config = config or self._get_default_config()
        self.regime_history = []
        self.volatility_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for regime detection"""
        return {
            'adx': {
                'period': 14,
                'trending_threshold': 25,
                'strong_trending_threshold': 40,
                'ranging_threshold': 20
            },
            'choppiness': {
                'period': 14,
                'choppy_threshold': 61.8,
                'trending_threshold': 38.2
            },
            'volatility': {
                'atr_period': 14,
                'lookback_period': 50,
                'high_vol_threshold': 1.5,
                'low_vol_threshold': 0.7
            },
            'trend': {
                'ema_fast': 8,
                'ema_slow': 21,
                'trend_period': 20
            },
            'regime_persistence': {
                'min_periods': 5,
                'stability_period': 10
            }
        }
    
    def analyze_regime(self, data: pd.DataFrame, timeframe: str = "M5") -> RegimeAnalysis:
        """
        Perform comprehensive market regime analysis
        
        Args:
            data: OHLCV price data
            timeframe: Data timeframe
            
        Returns:
            Complete regime analysis
        """
        try:
            if len(data) < 50:
                logger.warning("Insufficient data for regime analysis")
                return self._create_default_analysis(timeframe)
            
            # Calculate core indicators
            adx_value = self._calculate_adx(data)
            choppiness_value = self._calculate_choppiness_index(data)
            volatility_ratio = self._calculate_volatility_ratio(data)
            trend_direction = self._determine_trend_direction(data)
            
            # Determine primary regime
            primary_regime = self._classify_primary_regime(
                adx_value, choppiness_value, volatility_ratio
            )
            
            # Determine volatility regime
            volatility_regime = self._classify_volatility_regime(volatility_ratio)
            
            # Determine trend strength
            trend_strength = self._classify_trend_strength(adx_value, choppiness_value)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(
                adx_value, choppiness_value, volatility_ratio, data
            )
            
            # Calculate regime persistence and stability
            regime_persistence = self._calculate_regime_persistence(primary_regime)
            regime_stability = self._calculate_regime_stability(data)
            
            # Data quality assessment
            data_quality = self._assess_data_quality(data)
            
            analysis = RegimeAnalysis(
                primary_regime=primary_regime,
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                confidence=confidence,
                adx_value=adx_value,
                choppiness_value=choppiness_value,
                volatility_ratio=volatility_ratio,
                trend_direction=trend_direction,
                regime_persistence=regime_persistence,
                regime_stability=regime_stability,
                analysis_timeframe=timeframe,
                data_quality=data_quality,
                regime_metadata=self._generate_metadata(data, adx_value, choppiness_value)
            )
            
            # Store in history
            self.regime_history.append({
                'timestamp': data.index[-1],
                'regime': primary_regime,
                'confidence': confidence,
                'adx': adx_value,
                'choppiness': choppiness_value,
                'volatility_ratio': volatility_ratio
            })
            
            # Keep only recent history
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            logger.debug(f"Regime analysis: {primary_regime.value} "
                        f"(ADX: {adx_value:.1f}, Chop: {choppiness_value:.1f}, "
                        f"Vol: {volatility_ratio:.2f}, Conf: {confidence:.3f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            return self._create_default_analysis(timeframe)
    
    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Calculate Average Directional Index"""
        try:
            period = self.config['adx']['period']
            
            if TALIB_AVAILABLE and len(data) >= period:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                
                adx = talib.ADX(high, low, close, timeperiod=period)
                return float(adx[-1]) if not np.isnan(adx[-1]) else 20.0
            else:
                # Fallback calculation
                return self._calculate_adx_fallback(data, period)
                
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return 20.0
    
    def _calculate_adx_fallback(self, data: pd.DataFrame, period: int) -> float:
        """Fallback ADX calculation without TA-Lib"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            hl = high - low
            hc = np.abs(high - close.shift(1))
            lc = np.abs(low - close.shift(1))
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # Calculate Directional Movement
            plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                              np.maximum(high - high.shift(1), 0), 0)
            minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                               np.maximum(low.shift(1) - low, 0), 0)
            
            # Smooth with EMA
            tr_ema = tr.ewm(span=period).mean()
            plus_dm_ema = pd.Series(plus_dm).ewm(span=period).mean()
            minus_dm_ema = pd.Series(minus_dm).ewm(span=period).mean()
            
            # Calculate DI
            plus_di = 100 * plus_dm_ema / tr_ema
            minus_di = 100 * minus_dm_ema / tr_ema
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period).mean()
            
            return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
            
        except Exception as e:
            logger.error(f"Error in fallback ADX calculation: {str(e)}")
            return 20.0
    
    def _calculate_choppiness_index(self, data: pd.DataFrame) -> float:
        """Calculate Choppiness Index"""
        try:
            period = self.config['choppiness']['period']
            
            if len(data) < period:
                return 50.0
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range calculation
            hl = high - low
            hc = np.abs(high - close.shift(1))
            lc = np.abs(low - close.shift(1))
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # Choppiness Index
            atr_sum = tr.rolling(period).sum()
            high_max = high.rolling(period).max()
            low_min = low.rolling(period).min()
            
            chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
            
            return float(chop.iloc[-1]) if not np.isnan(chop.iloc[-1]) else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating Choppiness Index: {str(e)}")
            return 50.0
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> float:
        """Calculate current volatility relative to historical average"""
        try:
            atr_period = self.config['volatility']['atr_period']
            lookback_period = self.config['volatility']['lookback_period']
            
            if len(data) < max(atr_period, lookback_period):
                return 1.0
            
            # Calculate ATR
            if TALIB_AVAILABLE:
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                atr = talib.ATR(high, low, close, timeperiod=atr_period)
            else:
                # Fallback ATR calculation
                high = data['high']
                low = data['low']
                close = data['close']
                
                hl = high - low
                hc = np.abs(high - close.shift(1))
                lc = np.abs(low - close.shift(1))
                tr = np.maximum(hl, np.maximum(hc, lc))
                atr = tr.rolling(atr_period).mean().values
            
            # Current vs average volatility
            current_atr = atr[-1]
            avg_atr = np.mean(atr[-lookback_period:])
            
            if avg_atr > 0:
                volatility_ratio = current_atr / avg_atr
            else:
                volatility_ratio = 1.0
            
            return float(volatility_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating volatility ratio: {str(e)}")
            return 1.0
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> int:
        """Determine overall trend direction"""
        try:
            ema_fast_period = self.config['trend']['ema_fast']
            ema_slow_period = self.config['trend']['ema_slow']
            
            if len(data) < ema_slow_period:
                return 0
            
            close = data['close']
            
            # Calculate EMAs
            if TALIB_AVAILABLE:
                ema_fast = talib.EMA(close.values, timeperiod=ema_fast_period)
                ema_slow = talib.EMA(close.values, timeperiod=ema_slow_period)
            else:
                ema_fast = close.ewm(span=ema_fast_period).mean().values
                ema_slow = close.ewm(span=ema_slow_period).mean().values
            
            # Determine trend direction
            if ema_fast[-1] > ema_slow[-1]:
                # Additional confirmation: price above both EMAs
                if close.iloc[-1] > ema_fast[-1]:
                    return 1  # Strong uptrend
                else:
                    return 1  # Weak uptrend
            elif ema_fast[-1] < ema_slow[-1]:
                # Additional confirmation: price below both EMAs
                if close.iloc[-1] < ema_fast[-1]:
                    return -1  # Strong downtrend
                else:
                    return -1  # Weak downtrend
            else:
                return 0  # Sideways
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return 0
    
    def _classify_primary_regime(self, adx: float, choppiness: float, volatility_ratio: float) -> MarketRegime:
        """Classify the primary market regime"""
        try:
            # Get thresholds
            adx_trending = self.config['adx']['trending_threshold']
            adx_strong = self.config['adx']['strong_trending_threshold']
            adx_ranging = self.config['adx']['ranging_threshold']
            chop_trending = self.config['choppiness']['trending_threshold']
            chop_choppy = self.config['choppiness']['choppy_threshold']
            vol_high = self.config['volatility']['high_vol_threshold']
            vol_low = self.config['volatility']['low_vol_threshold']
            
            # Strong trending market
            if adx >= adx_strong and choppiness <= chop_trending:
                return MarketRegime.STRONG_TRENDING
            
            # Trending market
            elif adx >= adx_trending and choppiness <= chop_trending + 10:
                return MarketRegime.TRENDING
            
            # Choppy market
            elif choppiness >= chop_choppy:
                return MarketRegime.CHOPPY
            
            # High volatility regime
            elif volatility_ratio >= vol_high:
                if adx >= adx_trending:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regime
            elif volatility_ratio <= vol_low:
                return MarketRegime.LOW_VOLATILITY
            
            # Ranging market
            elif adx <= adx_ranging:
                return MarketRegime.RANGING
            
            # Consolidation (middle ground)
            else:
                return MarketRegime.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"Error classifying primary regime: {str(e)}")
            return MarketRegime.UNKNOWN
    
    def _classify_volatility_regime(self, volatility_ratio: float) -> VolatilityRegime:
        """Classify volatility regime"""
        try:
            vol_high = self.config['volatility']['high_vol_threshold']
            vol_low = self.config['volatility']['low_vol_threshold']
            
            if volatility_ratio >= vol_high * 1.5:
                return VolatilityRegime.EXTREME
            elif volatility_ratio >= vol_high:
                return VolatilityRegime.HIGH
            elif volatility_ratio <= vol_low:
                return VolatilityRegime.LOW
            else:
                return VolatilityRegime.NORMAL
                
        except Exception as e:
            logger.error(f"Error classifying volatility regime: {str(e)}")
            return VolatilityRegime.NORMAL
    
    def _classify_trend_strength(self, adx: float, choppiness: float) -> TrendStrength:
        """Classify trend strength"""
        try:
            # Combined score based on ADX and inverse choppiness
            strength_score = adx - choppiness/2
            
            if strength_score >= 30:
                return TrendStrength.VERY_STRONG
            elif strength_score >= 20:
                return TrendStrength.STRONG
            elif strength_score >= 10:
                return TrendStrength.MODERATE
            elif strength_score >= 0:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK
                
        except Exception as e:
            logger.error(f"Error classifying trend strength: {str(e)}")
            return TrendStrength.MODERATE
    
    def _calculate_regime_confidence(self, adx: float, choppiness: float, 
                                   volatility_ratio: float, data: pd.DataFrame) -> float:
        """Calculate confidence in regime classification"""
        try:
            confidence_factors = []
            
            # ADX confidence (higher ADX = more confident in trend/ranging classification)
            if adx >= 40:
                confidence_factors.append(0.9)
            elif adx >= 25:
                confidence_factors.append(0.7)
            elif adx <= 15:
                confidence_factors.append(0.8)  # Very low ADX is confidently ranging
            else:
                confidence_factors.append(0.5)
            
            # Choppiness confidence (extreme values are more confident)
            if choppiness >= 70 or choppiness <= 30:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Volatility confidence (stable volatility increases confidence)
            if 0.8 <= volatility_ratio <= 1.2:
                confidence_factors.append(0.8)
            elif volatility_ratio >= 2.0 or volatility_ratio <= 0.5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.6)
            
            # Data quality factor
            data_points = len(data)
            if data_points >= 100:
                confidence_factors.append(0.9)
            elif data_points >= 50:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Historical consistency factor
            if len(self.regime_history) >= 5:
                recent_regimes = [h['regime'] for h in self.regime_history[-5:]]
                if len(set(recent_regimes)) <= 2:  # Consistent regimes
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.5)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {str(e)}")
            return 0.5
    
    def _calculate_regime_persistence(self, current_regime: MarketRegime) -> float:
        """Calculate how long the current regime has persisted"""
        try:
            if len(self.regime_history) < 2:
                return 0.0
            
            # Count consecutive periods of the same regime
            consecutive_count = 1
            for i in range(len(self.regime_history) - 2, -1, -1):
                if self.regime_history[i]['regime'] == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            # Normalize to 0-1 scale (max at 20 periods)
            return min(consecutive_count / 20.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating regime persistence: {str(e)}")
            return 0.0
    
    def _calculate_regime_stability(self, data: pd.DataFrame) -> float:
        """Calculate stability of regime indicators"""
        try:
            stability_period = self.config['regime_persistence']['stability_period']
            
            if len(data) < stability_period * 2:
                return 0.5
            
            # Calculate coefficient of variation for ADX over recent periods
            recent_data = data.tail(stability_period)
            adx_values = []
            
            for i in range(len(recent_data) - self.config['adx']['period']):
                window_data = recent_data.iloc[i:i+self.config['adx']['period']]
                adx_val = self._calculate_adx(window_data)
                adx_values.append(adx_val)
            
            if len(adx_values) >= 3:
                adx_mean = np.mean(adx_values)
                adx_std = np.std(adx_values)
                cv = adx_std / adx_mean if adx_mean > 0 else 1.0
                
                # Lower coefficient of variation = higher stability
                stability = max(0.0, 1.0 - cv)
                return min(stability, 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating regime stability: {str(e)}")
            return 0.5
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of input data"""
        try:
            quality_factors = []
            
            # Data completeness
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_factors.append(1.0 - missing_ratio)
            
            # Data consistency (no negative prices, high >= low, etc.)
            consistency_issues = 0
            if (data['high'] < data['low']).any():
                consistency_issues += 1
            if (data['high'] < data['close']).any():
                consistency_issues += 1
            if (data['low'] > data['close']).any():
                consistency_issues += 1
            
            quality_factors.append(max(0.0, 1.0 - consistency_issues * 0.2))
            
            # Data sufficiency
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
    
    def _generate_metadata(self, data: pd.DataFrame, adx: float, choppiness: float) -> Dict[str, Any]:
        """Generate additional metadata for regime analysis"""
        try:
            metadata = {
                'data_points': len(data),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'adx_percentile': self._calculate_percentile(adx, 'adx'),
                'choppiness_percentile': self._calculate_percentile(choppiness, 'choppiness'),
                'indicators_used': ['ADX', 'Choppiness Index', 'ATR', 'EMA'],
                'config_version': '1.0'
            }
            
            # Add recent regime statistics
            if len(self.regime_history) >= 10:
                recent_regimes = [h['regime'].value for h in self.regime_history[-10:]]
                metadata['recent_regime_distribution'] = {
                    regime: recent_regimes.count(regime) for regime in set(recent_regimes)
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {}
    
    def _calculate_percentile(self, value: float, indicator: str) -> float:
        """Calculate percentile of current value relative to history"""
        try:
            if indicator == 'adx':
                historical_values = [h['adx'] for h in self.regime_history[-100:] if 'adx' in h]
            elif indicator == 'choppiness':
                historical_values = [h['choppiness'] for h in self.regime_history[-100:] if 'choppiness' in h]
            else:
                return 50.0
            
            if len(historical_values) >= 10:
                percentile = (np.sum(np.array(historical_values) <= value) / len(historical_values)) * 100
                return float(percentile)
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error calculating percentile: {str(e)}")
            return 50.0
    
    def _create_default_analysis(self, timeframe: str) -> RegimeAnalysis:
        """Create default analysis when calculation fails"""
        return RegimeAnalysis(
            primary_regime=MarketRegime.UNKNOWN,
            volatility_regime=VolatilityRegime.NORMAL,
            trend_strength=TrendStrength.MODERATE,
            confidence=0.0,
            adx_value=20.0,
            choppiness_value=50.0,
            volatility_ratio=1.0,
            trend_direction=0,
            regime_persistence=0.0,
            regime_stability=0.5,
            analysis_timeframe=timeframe,
            data_quality=0.0,
            regime_metadata={}
        )
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of recent regime history"""
        try:
            if not self.regime_history:
                return {}
            
            recent_history = self.regime_history[-50:]  # Last 50 periods
            
            regimes = [h['regime'].value for h in recent_history]
            confidences = [h['confidence'] for h in recent_history]
            
            return {
                'current_regime': regimes[-1] if regimes else 'unknown',
                'regime_distribution': {regime: regimes.count(regime) for regime in set(regimes)},
                'average_confidence': np.mean(confidences),
                'regime_changes': len([i for i in range(1, len(regimes)) if regimes[i] != regimes[i-1]]),
                'most_frequent_regime': max(set(regimes), key=regimes.count) if regimes else 'unknown',
                'analysis_periods': len(recent_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting regime summary: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=200, freq='5T')
    
    # Simulate trending market data
    base_price = 100
    trend = np.cumsum(np.random.randn(200) * 0.1) + base_price
    noise = np.random.randn(200) * 0.5
    
    sample_data = pd.DataFrame({
        'open': trend + noise,
        'high': trend + noise + np.abs(np.random.randn(200) * 0.3),
        'low': trend + noise - np.abs(np.random.randn(200) * 0.3),
        'close': trend + noise,
        'volume': np.random.randint(100, 1000, 200)
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
    regime_filter = MarketRegimeFilter()
    
    print("Market Regime Filter Test")
    print("=" * 40)
    
    # Analyze regime
    analysis = regime_filter.analyze_regime(sample_data, "M5")
    
    print(f"Primary Regime: {analysis.primary_regime.value}")
    print(f"Volatility Regime: {analysis.volatility_regime.value}")
    print(f"Trend Strength: {analysis.trend_strength.value}")
    print(f"Trend Direction: {analysis.trend_direction}")
    print(f"Confidence: {analysis.confidence:.3f}")
    print(f"ADX: {analysis.adx_value:.1f}")
    print(f"Choppiness: {analysis.choppiness_value:.1f}")
    print(f"Volatility Ratio: {analysis.volatility_ratio:.2f}")
    print(f"Data Quality: {analysis.data_quality:.3f}")
    
    # Test multiple periods to see regime evolution
    print("\nRegime Evolution Test:")
    for i in range(50, 200, 30):
        window_data = sample_data.iloc[:i]
        analysis = regime_filter.analyze_regime(window_data, "M5")
        print(f"Period {i}: {analysis.primary_regime.value} "
              f"(ADX: {analysis.adx_value:.1f}, Conf: {analysis.confidence:.2f})")
    
    # Get regime summary
    summary = regime_filter.get_regime_summary()
    print(f"\nRegime Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nMarket Regime Filter implementation completed!")