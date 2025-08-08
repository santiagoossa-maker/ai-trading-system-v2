"""
Temporal Filter
Intelligent time-based filtering for optimal trading sessions, news avoidance, and market hours
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, time, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Major trading sessions"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_TOKYO_LONDON = "tokyo_london_overlap"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OFF_HOURS = "off_hours"

class MarketHoursQuality(Enum):
    """Quality classification for market hours"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    AVOID = "avoid"

class NewsImpact(Enum):
    """News event impact classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass
class TradingHours:
    """Trading hours for different sessions"""
    session: TradingSession
    start_hour: int
    end_hour: int
    timezone: str
    quality_score: float
    volume_multiplier: float
    volatility_multiplier: float

@dataclass
class NewsEvent:
    """Economic news event"""
    timestamp: datetime
    event_name: str
    currency: str
    impact: NewsImpact
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None

@dataclass
class TemporalAnalysis:
    """Complete temporal market analysis"""
    current_session: TradingSession
    session_quality: MarketHoursQuality
    optimal_for_trading: bool
    
    # Time-based factors
    hour_quality_score: float
    session_overlap_bonus: float
    day_of_week_factor: float
    
    # News and events
    upcoming_news_events: List[NewsEvent]
    news_risk_score: float
    recommended_pause_minutes: int
    
    # Session statistics
    expected_volatility: float
    expected_volume: float
    success_rate_historical: float
    
    # Metadata
    analysis_timestamp: datetime
    market_timezone: str
    temporal_metadata: Dict[str, Any]

class TemporalFilter:
    """
    Advanced temporal filter for optimizing trading timing based on:
    - Market sessions and overlaps
    - Historical performance by hour/day
    - Economic news calendar
    - Seasonal patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the temporal filter
        
        Args:
            config: Configuration for temporal analysis
        """
        self.config = config or self._get_default_config()
        self.trading_hours = self._initialize_trading_hours()
        self.news_calendar = []
        self.historical_performance = {}
        self.session_statistics = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for temporal analysis"""
        return {
            'timezone': 'UTC',
            'base_currency': 'USD',
            'trading_hours': {
                'sydney': {'start': 22, 'end': 6, 'quality': 0.6},
                'tokyo': {'start': 0, 'end': 9, 'quality': 0.7},
                'london': {'start': 8, 'end': 16, 'quality': 0.9},
                'new_york': {'start': 13, 'end': 21, 'quality': 0.8},
                'overlap_tokyo_london': {'start': 8, 'end': 9, 'quality': 0.85},
                'overlap_london_ny': {'start': 13, 'end': 16, 'quality': 0.95}
            },
            'news_avoidance': {
                'high_impact_buffer_minutes': 60,
                'medium_impact_buffer_minutes': 30,
                'low_impact_buffer_minutes': 15,
                'currencies_to_monitor': ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
            },
            'day_of_week': {
                'monday': 0.8,      # Slower start
                'tuesday': 1.0,     # Optimal
                'wednesday': 1.0,   # Optimal
                'thursday': 0.9,    # Good
                'friday': 0.7,      # Early close, lower volume
                'saturday': 0.1,    # Weekend
                'sunday': 0.3       # Weekend, some Asian activity
            },
            'hour_quality': {
                # Default quality scores by hour (UTC)
                # These should be updated based on historical analysis
            },
            'volatility_patterns': {
                'high_vol_hours': [8, 9, 13, 14, 15, 16],  # Major session opens/overlaps
                'low_vol_hours': [22, 23, 0, 1, 2, 3, 4, 5],  # Asian session
                'medium_vol_hours': [7, 10, 11, 12, 17, 18, 19, 20, 21]
            }
        }
    
    def _initialize_trading_hours(self) -> Dict[TradingSession, TradingHours]:
        """Initialize trading hours for different sessions"""
        hours_config = self.config['trading_hours']
        
        trading_hours = {}
        
        # Sydney session
        trading_hours[TradingSession.SYDNEY] = TradingHours(
            session=TradingSession.SYDNEY,
            start_hour=hours_config['sydney']['start'],
            end_hour=hours_config['sydney']['end'],
            timezone='Australia/Sydney',
            quality_score=hours_config['sydney']['quality'],
            volume_multiplier=0.7,
            volatility_multiplier=0.8
        )
        
        # Tokyo session
        trading_hours[TradingSession.TOKYO] = TradingHours(
            session=TradingSession.TOKYO,
            start_hour=hours_config['tokyo']['start'],
            end_hour=hours_config['tokyo']['end'],
            timezone='Asia/Tokyo',
            quality_score=hours_config['tokyo']['quality'],
            volume_multiplier=0.8,
            volatility_multiplier=0.9
        )
        
        # London session
        trading_hours[TradingSession.LONDON] = TradingHours(
            session=TradingSession.LONDON,
            start_hour=hours_config['london']['start'],
            end_hour=hours_config['london']['end'],
            timezone='Europe/London',
            quality_score=hours_config['london']['quality'],
            volume_multiplier=1.2,
            volatility_multiplier=1.3
        )
        
        # New York session
        trading_hours[TradingSession.NEW_YORK] = TradingHours(
            session=TradingSession.NEW_YORK,
            start_hour=hours_config['new_york']['start'],
            end_hour=hours_config['new_york']['end'],
            timezone='America/New_York',
            quality_score=hours_config['new_york']['quality'],
            volume_multiplier=1.1,
            volatility_multiplier=1.2
        )
        
        # Overlap sessions
        trading_hours[TradingSession.OVERLAP_TOKYO_LONDON] = TradingHours(
            session=TradingSession.OVERLAP_TOKYO_LONDON,
            start_hour=hours_config['overlap_tokyo_london']['start'],
            end_hour=hours_config['overlap_tokyo_london']['end'],
            timezone='UTC',
            quality_score=hours_config['overlap_tokyo_london']['quality'],
            volume_multiplier=1.4,
            volatility_multiplier=1.5
        )
        
        trading_hours[TradingSession.OVERLAP_LONDON_NY] = TradingHours(
            session=TradingSession.OVERLAP_LONDON_NY,
            start_hour=hours_config['overlap_london_ny']['start'],
            end_hour=hours_config['overlap_london_ny']['end'],
            timezone='UTC',
            quality_score=hours_config['overlap_london_ny']['quality'],
            volume_multiplier=1.5,
            volatility_multiplier=1.6
        )
        
        return trading_hours
    
    def analyze_temporal_conditions(self, current_time: Optional[datetime] = None,
                                  symbol: str = "EURUSD") -> TemporalAnalysis:
        """
        Perform comprehensive temporal analysis for trading conditions
        
        Args:
            current_time: Time to analyze (defaults to current UTC time)
            symbol: Trading symbol for currency-specific analysis
            
        Returns:
            Complete temporal analysis
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            elif current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            
            # Determine current session
            current_session = self._determine_current_session(current_time)
            
            # Calculate session quality
            session_quality = self._calculate_session_quality(current_time, current_session)
            
            # Hour-based quality score
            hour_quality_score = self._calculate_hour_quality(current_time.hour)
            
            # Session overlap bonus
            session_overlap_bonus = self._calculate_overlap_bonus(current_time)
            
            # Day of week factor
            day_of_week_factor = self._calculate_day_of_week_factor(current_time)
            
            # News analysis
            upcoming_news = self._get_upcoming_news_events(current_time, symbol)
            news_risk_score = self._calculate_news_risk_score(upcoming_news, current_time)
            recommended_pause = self._calculate_recommended_pause(upcoming_news, current_time)
            
            # Expected market conditions
            expected_volatility = self._estimate_expected_volatility(current_time, current_session)
            expected_volume = self._estimate_expected_volume(current_time, current_session)
            
            # Historical success rate
            success_rate = self._get_historical_success_rate(current_time, symbol)
            
            # Determine if optimal for trading
            optimal_for_trading = self._is_optimal_for_trading(
                session_quality, hour_quality_score, news_risk_score, day_of_week_factor
            )
            
            analysis = TemporalAnalysis(
                current_session=current_session,
                session_quality=session_quality,
                optimal_for_trading=optimal_for_trading,
                hour_quality_score=hour_quality_score,
                session_overlap_bonus=session_overlap_bonus,
                day_of_week_factor=day_of_week_factor,
                upcoming_news_events=upcoming_news,
                news_risk_score=news_risk_score,
                recommended_pause_minutes=recommended_pause,
                expected_volatility=expected_volatility,
                expected_volume=expected_volume,
                success_rate_historical=success_rate,
                analysis_timestamp=current_time,
                market_timezone=self.config['timezone'],
                temporal_metadata=self._generate_temporal_metadata(current_time, symbol)
            )
            
            logger.debug(f"Temporal analysis: {current_session.value} session, "
                        f"quality: {session_quality.value}, optimal: {optimal_for_trading}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return self._create_default_analysis(current_time)
    
    def _determine_current_session(self, current_time: datetime) -> TradingSession:
        """Determine which trading session is currently active"""
        try:
            hour_utc = current_time.hour
            
            # Check for overlap sessions first (higher priority)
            if 13 <= hour_utc <= 16:  # London-NY overlap
                return TradingSession.OVERLAP_LONDON_NY
            elif 8 <= hour_utc <= 9:  # Tokyo-London overlap
                return TradingSession.OVERLAP_TOKYO_LONDON
            
            # Check individual sessions
            elif 8 <= hour_utc <= 16:  # London session
                return TradingSession.LONDON
            elif 13 <= hour_utc <= 21:  # New York session
                return TradingSession.NEW_YORK
            elif 0 <= hour_utc <= 9:  # Tokyo session
                return TradingSession.TOKYO
            elif hour_utc >= 22 or hour_utc <= 6:  # Sydney session (crosses midnight)
                return TradingSession.SYDNEY
            else:
                return TradingSession.OFF_HOURS
                
        except Exception as e:
            logger.error(f"Error determining current session: {str(e)}")
            return TradingSession.OFF_HOURS
    
    def _calculate_session_quality(self, current_time: datetime, 
                                 session: TradingSession) -> MarketHoursQuality:
        """Calculate quality of current trading session"""
        try:
            if session == TradingSession.OFF_HOURS:
                return MarketHoursQuality.AVOID
            
            # Get base quality from trading hours
            session_info = self.trading_hours.get(session)
            if not session_info:
                return MarketHoursQuality.POOR
            
            base_quality = session_info.quality_score
            
            # Adjust for day of week
            day_factor = self._calculate_day_of_week_factor(current_time)
            adjusted_quality = base_quality * day_factor
            
            # Classify quality
            if adjusted_quality >= 0.9:
                return MarketHoursQuality.EXCELLENT
            elif adjusted_quality >= 0.75:
                return MarketHoursQuality.GOOD
            elif adjusted_quality >= 0.5:
                return MarketHoursQuality.AVERAGE
            elif adjusted_quality >= 0.3:
                return MarketHoursQuality.POOR
            else:
                return MarketHoursQuality.AVOID
                
        except Exception as e:
            logger.error(f"Error calculating session quality: {str(e)}")
            return MarketHoursQuality.AVERAGE
    
    def _calculate_hour_quality(self, hour: int) -> float:
        """Calculate quality score for specific hour"""
        try:
            # Use configured hour quality or calculate based on volatility patterns
            if 'hour_quality' in self.config and str(hour) in self.config['hour_quality']:
                return self.config['hour_quality'][str(hour)]
            
            # Fallback calculation based on volatility patterns
            vol_patterns = self.config['volatility_patterns']
            
            if hour in vol_patterns['high_vol_hours']:
                return 0.9
            elif hour in vol_patterns['medium_vol_hours']:
                return 0.7
            elif hour in vol_patterns['low_vol_hours']:
                return 0.4
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating hour quality: {str(e)}")
            return 0.5
    
    def _calculate_overlap_bonus(self, current_time: datetime) -> float:
        """Calculate bonus for session overlaps"""
        try:
            hour_utc = current_time.hour
            
            # London-NY overlap (most active)
            if 13 <= hour_utc <= 16:
                return 0.2
            
            # Tokyo-London overlap
            elif 8 <= hour_utc <= 9:
                return 0.15
            
            # Minor overlaps
            elif hour_utc in [7, 17]:  # Pre/post overlap hours
                return 0.05
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overlap bonus: {str(e)}")
            return 0.0
    
    def _calculate_day_of_week_factor(self, current_time: datetime) -> float:
        """Calculate factor based on day of week"""
        try:
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = day_names[current_time.weekday()]
            
            return self.config['day_of_week'].get(day_name, 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating day of week factor: {str(e)}")
            return 0.5
    
    def _get_upcoming_news_events(self, current_time: datetime, symbol: str) -> List[NewsEvent]:
        """Get upcoming news events relevant to the symbol"""
        try:
            # This would typically connect to an economic calendar API
            # For now, return sample events based on common patterns
            
            upcoming_events = []
            
            # Extract currencies from symbol
            if len(symbol) >= 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                relevant_currencies = [base_currency, quote_currency]
            else:
                relevant_currencies = ['USD']  # Default for indices
            
            # Generate sample news events (in a real implementation, this would come from an API)
            next_hour = current_time + timedelta(hours=1)
            
            # Sample high-impact events
            if current_time.hour in [8, 13, 15]:  # Common news release times
                for currency in relevant_currencies:
                    if currency in self.config['news_avoidance']['currencies_to_monitor']:
                        # Create sample news event
                        event = NewsEvent(
                            timestamp=next_hour,
                            event_name=f"{currency} Economic Data Release",
                            currency=currency,
                            impact=NewsImpact.MEDIUM,
                            forecast_value=100.0
                        )
                        upcoming_events.append(event)
            
            return upcoming_events
            
        except Exception as e:
            logger.error(f"Error getting upcoming news events: {str(e)}")
            return []
    
    def _calculate_news_risk_score(self, news_events: List[NewsEvent], 
                                 current_time: datetime) -> float:
        """Calculate risk score based on upcoming news"""
        try:
            if not news_events:
                return 0.0
            
            max_risk = 0.0
            
            for event in news_events:
                time_to_event = (event.timestamp - current_time).total_seconds() / 60  # minutes
                
                # Risk decreases with time to event
                if time_to_event <= 0:
                    continue
                
                # Base risk by impact
                if event.impact == NewsImpact.HIGH:
                    base_risk = 0.8
                    buffer_minutes = self.config['news_avoidance']['high_impact_buffer_minutes']
                elif event.impact == NewsImpact.MEDIUM:
                    base_risk = 0.5
                    buffer_minutes = self.config['news_avoidance']['medium_impact_buffer_minutes']
                elif event.impact == NewsImpact.LOW:
                    base_risk = 0.2
                    buffer_minutes = self.config['news_avoidance']['low_impact_buffer_minutes']
                else:
                    base_risk = 0.0
                    buffer_minutes = 0
                
                # Adjust risk based on time proximity
                if time_to_event <= buffer_minutes:
                    time_factor = 1.0 - (time_to_event / buffer_minutes)
                    event_risk = base_risk * time_factor
                    max_risk = max(max_risk, event_risk)
            
            return min(max_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating news risk score: {str(e)}")
            return 0.0
    
    def _calculate_recommended_pause(self, news_events: List[NewsEvent], 
                                   current_time: datetime) -> int:
        """Calculate recommended trading pause in minutes"""
        try:
            if not news_events:
                return 0
            
            max_pause = 0
            
            for event in news_events:
                time_to_event = (event.timestamp - current_time).total_seconds() / 60
                
                if time_to_event <= 0:
                    continue
                
                # Recommended pause based on impact
                if event.impact == NewsImpact.HIGH:
                    pause_minutes = self.config['news_avoidance']['high_impact_buffer_minutes']
                elif event.impact == NewsImpact.MEDIUM:
                    pause_minutes = self.config['news_avoidance']['medium_impact_buffer_minutes']
                elif event.impact == NewsImpact.LOW:
                    pause_minutes = self.config['news_avoidance']['low_impact_buffer_minutes']
                else:
                    pause_minutes = 0
                
                # Only recommend pause if event is soon
                if time_to_event <= pause_minutes:
                    remaining_pause = int(pause_minutes - time_to_event + 15)  # Add 15 min buffer after
                    max_pause = max(max_pause, remaining_pause)
            
            return max_pause
            
        except Exception as e:
            logger.error(f"Error calculating recommended pause: {str(e)}")
            return 0
    
    def _estimate_expected_volatility(self, current_time: datetime, 
                                    session: TradingSession) -> float:
        """Estimate expected volatility for current conditions"""
        try:
            # Base volatility from session
            session_info = self.trading_hours.get(session)
            if session_info:
                base_volatility = session_info.volatility_multiplier
            else:
                base_volatility = 1.0
            
            # Hour-based adjustment
            hour_factor = 1.0
            vol_patterns = self.config['volatility_patterns']
            
            if current_time.hour in vol_patterns['high_vol_hours']:
                hour_factor = 1.3
            elif current_time.hour in vol_patterns['low_vol_hours']:
                hour_factor = 0.7
            
            # Day of week adjustment
            day_factor = self._calculate_day_of_week_factor(current_time)
            
            return base_volatility * hour_factor * day_factor
            
        except Exception as e:
            logger.error(f"Error estimating expected volatility: {str(e)}")
            return 1.0
    
    def _estimate_expected_volume(self, current_time: datetime, 
                                session: TradingSession) -> float:
        """Estimate expected volume for current conditions"""
        try:
            # Base volume from session
            session_info = self.trading_hours.get(session)
            if session_info:
                base_volume = session_info.volume_multiplier
            else:
                base_volume = 1.0
            
            # Overlap bonus
            overlap_bonus = self._calculate_overlap_bonus(current_time)
            
            # Day of week adjustment
            day_factor = self._calculate_day_of_week_factor(current_time)
            
            return base_volume * (1 + overlap_bonus) * day_factor
            
        except Exception as e:
            logger.error(f"Error estimating expected volume: {str(e)}")
            return 1.0
    
    def _get_historical_success_rate(self, current_time: datetime, symbol: str) -> float:
        """Get historical success rate for current time conditions"""
        try:
            # This would typically come from historical analysis
            # For now, return estimated values based on session quality
            
            session = self._determine_current_session(current_time)
            session_info = self.trading_hours.get(session)
            
            if session_info:
                base_success_rate = session_info.quality_score * 0.6 + 0.2  # 20-80% range
            else:
                base_success_rate = 0.5
            
            # Adjust for day of week
            day_factor = self._calculate_day_of_week_factor(current_time)
            
            return base_success_rate * day_factor
            
        except Exception as e:
            logger.error(f"Error getting historical success rate: {str(e)}")
            return 0.5
    
    def _is_optimal_for_trading(self, session_quality: MarketHoursQuality,
                              hour_quality: float, news_risk: float,
                              day_factor: float) -> bool:
        """Determine if current conditions are optimal for trading"""
        try:
            # Avoid trading during poor conditions
            if session_quality in [MarketHoursQuality.AVOID, MarketHoursQuality.POOR]:
                return False
            
            # Avoid trading during high news risk
            if news_risk >= 0.7:
                return False
            
            # Avoid trading during very poor hours
            if hour_quality <= 0.3:
                return False
            
            # Avoid trading on weekends
            if day_factor <= 0.2:
                return False
            
            # Require minimum combined quality
            combined_quality = (
                (session_quality.value == 'excellent') * 0.4 +
                (session_quality.value == 'good') * 0.3 +
                (session_quality.value == 'average') * 0.2 +
                hour_quality * 0.3 +
                (1 - news_risk) * 0.2 +
                day_factor * 0.1
            )
            
            return combined_quality >= 0.6
            
        except Exception as e:
            logger.error(f"Error determining optimal trading conditions: {str(e)}")
            return False
    
    def _generate_temporal_metadata(self, current_time: datetime, symbol: str) -> Dict[str, Any]:
        """Generate metadata for temporal analysis"""
        try:
            return {
                'analysis_time_utc': current_time.isoformat(),
                'symbol': symbol,
                'day_of_week': current_time.strftime('%A'),
                'hour_utc': current_time.hour,
                'timezone': self.config['timezone'],
                'sessions_active': [session.value for session in self.trading_hours.keys()
                                  if self._is_session_active(current_time, session)],
                'config_version': '1.0'
            }
        except Exception as e:
            logger.error(f"Error generating temporal metadata: {str(e)}")
            return {}
    
    def _is_session_active(self, current_time: datetime, session: TradingSession) -> bool:
        """Check if a specific session is currently active"""
        try:
            session_info = self.trading_hours.get(session)
            if not session_info:
                return False
            
            hour = current_time.hour
            start = session_info.start_hour
            end = session_info.end_hour
            
            # Handle sessions that cross midnight
            if start > end:
                return hour >= start or hour <= end
            else:
                return start <= hour <= end
                
        except Exception as e:
            logger.error(f"Error checking session activity: {str(e)}")
            return False
    
    def _create_default_analysis(self, current_time: Optional[datetime] = None) -> TemporalAnalysis:
        """Create default analysis when calculation fails"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        return TemporalAnalysis(
            current_session=TradingSession.OFF_HOURS,
            session_quality=MarketHoursQuality.POOR,
            optimal_for_trading=False,
            hour_quality_score=0.5,
            session_overlap_bonus=0.0,
            day_of_week_factor=0.5,
            upcoming_news_events=[],
            news_risk_score=0.0,
            recommended_pause_minutes=0,
            expected_volatility=1.0,
            expected_volume=1.0,
            success_rate_historical=0.5,
            analysis_timestamp=current_time,
            market_timezone=self.config['timezone'],
            temporal_metadata={}
        )
    
    def add_news_event(self, event: NewsEvent):
        """Add a news event to the calendar"""
        self.news_calendar.append(event)
    
    def update_historical_performance(self, hour: int, day_of_week: str, 
                                    success_rate: float, symbol: str):
        """Update historical performance data"""
        key = f"{symbol}_{day_of_week}_{hour}"
        self.historical_performance[key] = success_rate
    
    def get_optimal_trading_hours(self, symbol: str = "EURUSD") -> List[Tuple[int, float]]:
        """Get list of optimal trading hours with quality scores"""
        try:
            optimal_hours = []
            
            for hour in range(24):
                # Create dummy datetime for the hour
                test_time = datetime.now(pytz.UTC).replace(hour=hour, minute=0, second=0, microsecond=0)
                analysis = self.analyze_temporal_conditions(test_time, symbol)
                
                if analysis.optimal_for_trading:
                    quality_score = analysis.hour_quality_score + analysis.session_overlap_bonus
                    optimal_hours.append((hour, quality_score))
            
            # Sort by quality score
            optimal_hours.sort(key=lambda x: x[1], reverse=True)
            return optimal_hours
            
        except Exception as e:
            logger.error(f"Error getting optimal trading hours: {str(e)}")
            return []
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all trading sessions"""
        try:
            summary = {}
            
            for session, info in self.trading_hours.items():
                summary[session.value] = {
                    'start_hour': info.start_hour,
                    'end_hour': info.end_hour,
                    'quality_score': info.quality_score,
                    'volume_multiplier': info.volume_multiplier,
                    'volatility_multiplier': info.volatility_multiplier,
                    'timezone': info.timezone
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    
    # Initialize temporal filter
    temporal_filter = TemporalFilter()
    
    print("Temporal Filter Test")
    print("=" * 40)
    
    # Test current conditions
    current_analysis = temporal_filter.analyze_temporal_conditions(symbol="EURUSD")
    
    print(f"Current Temporal Analysis:")
    print(f"  Session: {current_analysis.current_session.value}")
    print(f"  Session Quality: {current_analysis.session_quality.value}")
    print(f"  Optimal for Trading: {current_analysis.optimal_for_trading}")
    print(f"  Hour Quality Score: {current_analysis.hour_quality_score:.3f}")
    print(f"  Session Overlap Bonus: {current_analysis.session_overlap_bonus:.3f}")
    print(f"  Day of Week Factor: {current_analysis.day_of_week_factor:.3f}")
    print(f"  News Risk Score: {current_analysis.news_risk_score:.3f}")
    print(f"  Expected Volatility: {current_analysis.expected_volatility:.2f}")
    print(f"  Expected Volume: {current_analysis.expected_volume:.2f}")
    print(f"  Historical Success Rate: {current_analysis.success_rate_historical:.3f}")
    
    if current_analysis.upcoming_news_events:
        print(f"  Upcoming News Events: {len(current_analysis.upcoming_news_events)}")
        for event in current_analysis.upcoming_news_events[:3]:
            print(f"    - {event.event_name} ({event.impact.value})")
    
    if current_analysis.recommended_pause_minutes > 0:
        print(f"  Recommended Pause: {current_analysis.recommended_pause_minutes} minutes")
    
    # Test optimal hours
    print(f"\nOptimal Trading Hours (Top 5):")
    optimal_hours = temporal_filter.get_optimal_trading_hours("EURUSD")
    for i, (hour, quality) in enumerate(optimal_hours[:5]):
        print(f"  {i+1}. {hour:02d}:00 UTC (Quality: {quality:.3f})")
    
    # Test different time periods
    print(f"\nTemporal Analysis at Different Hours:")
    test_times = [8, 13, 16, 22]  # Key market hours
    
    for hour in test_times:
        test_time = datetime.now(pytz.UTC).replace(hour=hour, minute=0, second=0, microsecond=0)
        analysis = temporal_filter.analyze_temporal_conditions(test_time, "EURUSD")
        print(f"  {hour:02d}:00 UTC - {analysis.current_session.value} "
              f"({analysis.session_quality.value}, optimal: {analysis.optimal_for_trading})")
    
    # Session summary
    print(f"\nTrading Sessions Summary:")
    session_summary = temporal_filter.get_session_summary()
    for session, info in session_summary.items():
        print(f"  {session}: {info['start_hour']:02d}:00-{info['end_hour']:02d}:00 "
              f"(Quality: {info['quality_score']:.2f})")
    
    print("\nTemporal Filter implementation completed!")