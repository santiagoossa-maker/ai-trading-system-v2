"""
Test script for intelligent filters
Tests the market regime, setup quality, and temporal filters
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_market_regime_filter():
    """Test market regime filter functionality"""
    print("Testing Market Regime Filter...")
    
    try:
        from ai.filters.market_regime_filter import MarketRegimeFilter, MarketRegime
        
        # Create sample trending market data
        dates = pd.date_range('2024-01-01', periods=200, freq='5T')
        np.random.seed(42)
        
        # Simulate trending market
        trend = np.cumsum(np.random.randn(200) * 0.05) + 100
        volatility = np.random.randn(200) * 0.3
        
        sample_data = pd.DataFrame({
            'open': trend + volatility,
            'high': trend + volatility + np.abs(np.random.randn(200) * 0.2),
            'low': trend + volatility - np.abs(np.random.randn(200) * 0.2),
            'close': trend + volatility,
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
        
        # Test filter
        regime_filter = MarketRegimeFilter()
        analysis = regime_filter.analyze_regime(sample_data, "M5")
        
        # Verify analysis structure
        assert hasattr(analysis, 'primary_regime')
        assert hasattr(analysis, 'volatility_regime')
        assert hasattr(analysis, 'trend_strength')
        assert hasattr(analysis, 'confidence')
        assert hasattr(analysis, 'adx_value')
        assert hasattr(analysis, 'choppiness_value')
        
        # Check that values are in expected ranges
        assert 0 <= analysis.confidence <= 1
        assert 0 <= analysis.adx_value <= 100
        assert 0 <= analysis.choppiness_value <= 100
        assert 0 <= analysis.volatility_ratio <= 10  # Reasonable range
        assert analysis.trend_direction in [-1, 0, 1]
        
        print(f"  Primary Regime: {analysis.primary_regime.value}")
        print(f"  ADX: {analysis.adx_value:.1f}")
        print(f"  Choppiness: {analysis.choppiness_value:.1f}")
        print(f"  Confidence: {analysis.confidence:.3f}")
        
        # Test multiple analyses to check regime evolution
        for i in range(50, 200, 50):
            window_data = sample_data.iloc[:i]
            window_analysis = regime_filter.analyze_regime(window_data, "M5")
            print(f"  Period {i}: {window_analysis.primary_regime.value}")
        
        print("✓ Market Regime Filter test passed")
        return True
        
    except Exception as e:
        print(f"✗ Market Regime Filter test failed: {str(e)}")
        return False

def test_setup_quality_filter():
    """Test setup quality filter functionality"""
    print("Testing Setup Quality Filter...")
    
    try:
        from ai.filters.setup_quality_filter import SetupQualityFilter, SetupQuality
        
        # Create sample market data with some patterns
        dates = pd.date_range('2024-01-01', periods=200, freq='5T')
        np.random.seed(42)
        
        # Create data with potential S/R levels
        base_price = 100
        trend = np.cumsum(np.random.randn(200) * 0.02)
        
        # Add some support/resistance patterns
        support_level = base_price - 2
        resistance_level = base_price + 3
        
        prices = base_price + trend
        
        # Simulate bounces off S/R levels
        for i in range(len(prices)):
            if prices[i] <= support_level:
                prices[i] = support_level + np.random.rand() * 0.5
            elif prices[i] >= resistance_level:
                prices[i] = resistance_level - np.random.rand() * 0.5
        
        noise = np.random.randn(200) * 0.1
        
        sample_data = pd.DataFrame({
            'open': prices + noise,
            'high': prices + noise + np.abs(np.random.randn(200) * 0.15),
            'low': prices + noise - np.abs(np.random.randn(200) * 0.15),
            'close': prices + noise,
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
        
        # Test filter
        setup_filter = SetupQualityFilter()
        
        # Test buy setup
        buy_analysis = setup_filter.analyze_setup_quality(sample_data, signal_direction=1, timeframe="M5")
        
        # Verify analysis structure
        assert hasattr(buy_analysis, 'overall_quality')
        assert hasattr(buy_analysis, 'quality_score')
        assert hasattr(buy_analysis, 'confidence')
        assert hasattr(buy_analysis, 'divergences')
        assert hasattr(buy_analysis, 'support_resistance')
        assert hasattr(buy_analysis, 'confluences')
        
        # Check value ranges
        assert 0 <= buy_analysis.quality_score <= 100
        assert 0 <= buy_analysis.confidence <= 1
        assert 0 <= buy_analysis.trend_alignment <= 1
        assert 0 <= buy_analysis.volume_confirmation <= 1
        assert 0 <= buy_analysis.momentum_strength <= 1
        
        print(f"  Overall Quality: {buy_analysis.overall_quality.value}")
        print(f"  Quality Score: {buy_analysis.quality_score:.1f}/100")
        print(f"  Confidence: {buy_analysis.confidence:.3f}")
        print(f"  Divergences Found: {len(buy_analysis.divergences)}")
        print(f"  S/R Levels Found: {len(buy_analysis.support_resistance)}")
        print(f"  Confluences: {len(buy_analysis.confluences)}")
        
        # Test sell setup
        sell_analysis = setup_filter.analyze_setup_quality(sample_data, signal_direction=-1, timeframe="M5")
        print(f"  Sell Setup Quality: {sell_analysis.overall_quality.value} (Score: {sell_analysis.quality_score:.1f})")
        
        # Test neutral analysis
        neutral_analysis = setup_filter.analyze_setup_quality(sample_data, signal_direction=0, timeframe="M5")
        print(f"  Neutral Analysis Quality: {neutral_analysis.overall_quality.value}")
        
        print("✓ Setup Quality Filter test passed")
        return True
        
    except Exception as e:
        print(f"✗ Setup Quality Filter test failed: {str(e)}")
        return False

def test_temporal_filter():
    """Test temporal filter functionality"""
    print("Testing Temporal Filter...")
    
    try:
        from ai.filters.temporal_filter import TemporalFilter, TradingSession, MarketHoursQuality
        
        # Initialize filter
        temporal_filter = TemporalFilter()
        
        # Test current time analysis
        current_time = datetime.now(pytz.UTC)
        analysis = temporal_filter.analyze_temporal_conditions(current_time, "EURUSD")
        
        # Verify analysis structure
        assert hasattr(analysis, 'current_session')
        assert hasattr(analysis, 'session_quality')
        assert hasattr(analysis, 'optimal_for_trading')
        assert hasattr(analysis, 'hour_quality_score')
        assert hasattr(analysis, 'expected_volatility')
        assert hasattr(analysis, 'expected_volume')
        
        # Check value ranges
        assert 0 <= analysis.hour_quality_score <= 1
        assert 0 <= analysis.session_overlap_bonus <= 1
        assert 0 <= analysis.day_of_week_factor <= 1
        assert 0 <= analysis.news_risk_score <= 1
        assert analysis.recommended_pause_minutes >= 0
        assert analysis.expected_volatility > 0
        assert analysis.expected_volume > 0
        assert 0 <= analysis.success_rate_historical <= 1
        
        print(f"  Current Session: {analysis.current_session.value}")
        print(f"  Session Quality: {analysis.session_quality.value}")
        print(f"  Optimal for Trading: {analysis.optimal_for_trading}")
        print(f"  Hour Quality: {analysis.hour_quality_score:.3f}")
        print(f"  Day Factor: {analysis.day_of_week_factor:.3f}")
        print(f"  Expected Volatility: {analysis.expected_volatility:.2f}")
        
        # Test different times
        test_times = [
            datetime(2024, 1, 15, 8, 0, tzinfo=pytz.UTC),   # London open
            datetime(2024, 1, 15, 13, 0, tzinfo=pytz.UTC),  # London-NY overlap
            datetime(2024, 1, 15, 22, 0, tzinfo=pytz.UTC),  # Sydney session
            datetime(2024, 1, 13, 10, 0, tzinfo=pytz.UTC),  # Saturday
        ]
        
        print(f"  Testing different times:")
        for i, test_time in enumerate(test_times):
            test_analysis = temporal_filter.analyze_temporal_conditions(test_time, "EURUSD")
            print(f"    Time {i+1}: {test_analysis.current_session.value} - "
                  f"{test_analysis.session_quality.value} (Optimal: {test_analysis.optimal_for_trading})")
        
        # Test optimal hours
        optimal_hours = temporal_filter.get_optimal_trading_hours("EURUSD")
        print(f"  Found {len(optimal_hours)} optimal trading hours")
        
        if optimal_hours:
            print(f"  Top 3 optimal hours:")
            for i, (hour, quality) in enumerate(optimal_hours[:3]):
                print(f"    {i+1}. {hour:02d}:00 UTC (Quality: {quality:.3f})")
        
        # Test session summary
        session_summary = temporal_filter.get_session_summary()
        print(f"  Trading sessions configured: {len(session_summary)}")
        
        print("✓ Temporal Filter test passed")
        return True
        
    except Exception as e:
        print(f"✗ Temporal Filter test failed: {str(e)}")
        return False

def test_filters_integration():
    """Test integration of all filters together"""
    print("Testing Filters Integration...")
    
    try:
        from ai.filters import MarketRegimeFilter, SetupQualityFilter, TemporalFilter
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='5T')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 100)
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
        
        # Initialize all filters
        regime_filter = MarketRegimeFilter()
        setup_filter = SetupQualityFilter()
        temporal_filter = TemporalFilter()
        
        # Run comprehensive analysis
        current_time = datetime.now(pytz.UTC)
        
        regime_analysis = regime_filter.analyze_regime(sample_data, "M5")
        setup_analysis = setup_filter.analyze_setup_quality(sample_data, signal_direction=1, timeframe="M5")
        temporal_analysis = temporal_filter.analyze_temporal_conditions(current_time, "EURUSD")
        
        # Combine results for overall trading signal assessment
        regime_score = 0.8 if regime_analysis.primary_regime.value in ['trending', 'strong_trending'] else 0.4
        setup_score = setup_analysis.quality_score / 100
        temporal_score = 0.8 if temporal_analysis.optimal_for_trading else 0.3
        
        # Weighted combined score
        combined_score = (
            regime_score * 0.3 +
            setup_score * 0.4 +
            temporal_score * 0.3
        )
        
        print(f"  Integrated Analysis Results:")
        print(f"    Market Regime: {regime_analysis.primary_regime.value} (Score: {regime_score:.2f})")
        print(f"    Setup Quality: {setup_analysis.overall_quality.value} (Score: {setup_score:.2f})")
        print(f"    Temporal: {temporal_analysis.session_quality.value} (Score: {temporal_score:.2f})")
        print(f"    Combined Score: {combined_score:.3f}")
        
        # Determine overall recommendation
        if combined_score >= 0.7:
            recommendation = "STRONG TRADE"
        elif combined_score >= 0.5:
            recommendation = "TRADE"
        elif combined_score >= 0.3:
            recommendation = "WEAK TRADE"
        else:
            recommendation = "NO TRADE"
        
        print(f"    Recommendation: {recommendation}")
        
        print("✓ Filters Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Filters Integration test failed: {str(e)}")
        return False

def main():
    """Run all filters tests"""
    print("=" * 60)
    print("AI Trading System V2 - Intelligent Filters Tests")
    print("=" * 60)
    
    tests = [
        test_market_regime_filter,
        test_setup_quality_filter,
        test_temporal_filter,
        test_filters_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {str(e)}")
        print()
    
    print("=" * 60)
    print(f"Filters Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All intelligent filters tests passed!")
        print("💡 Market regime detection, setup quality analysis, and temporal filtering are working.")
        return True
    else:
        print("❌ Some filters tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)