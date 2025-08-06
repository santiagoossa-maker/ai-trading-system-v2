"""
Test script to validate the AI Trading System V2 implementation
Tests the core components without requiring MT5 connection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_asset_configuration():
    """Test asset configuration loading"""
    print("Testing asset configuration...")
    
    config_path = "config/asset_specific_strategies.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check lot sizes
        assert 'lot_sizes' in config
        assert len(config['lot_sizes']) == 15
        assert 'R_75' in config['lot_sizes']
        assert config['lot_sizes']['R_75'] == 0.01
        
        # Check asset categories
        assert 'asset_categories' in config
        assert 'volatility_indices' in config['asset_categories']
        assert 'hz_indices' in config['asset_categories']
        assert 'step_indices' in config['asset_categories']
        
        print("✓ Asset configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Asset configuration test failed: {str(e)}")
        return False

def test_feature_collector():
    """Test feature collector with sample data"""
    print("Testing feature collector...")
    
    try:
        from ai.feature_collector import FeatureCollector, IndicatorConfig
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='5T')
        np.random.seed(42)
        
        sample_data = {
            'M5': pd.DataFrame({
                'open': np.random.randn(200).cumsum() + 100,
                'high': np.random.randn(200).cumsum() + 101,
                'low': np.random.randn(200).cumsum() + 99,
                'close': np.random.randn(200).cumsum() + 100,
                'volume': np.random.randint(100, 1000, 200)
            }, index=dates)
        }
        
        # Ensure high >= low and OHLC consistency
        sample_data['M5']['high'] = np.maximum.reduce([
            sample_data['M5']['open'], 
            sample_data['M5']['high'], 
            sample_data['M5']['low'], 
            sample_data['M5']['close']
        ])
        sample_data['M5']['low'] = np.minimum.reduce([
            sample_data['M5']['open'], 
            sample_data['M5']['high'], 
            sample_data['M5']['low'], 
            sample_data['M5']['close']
        ])
        
        # Initialize feature collector
        collector = FeatureCollector()
        
        # Generate features
        features = collector.collect_all_features(sample_data, "R_75")
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert features.shape[1] > 10  # Should have many features
        
        # Generate targets
        targets = collector.generate_target_variables(sample_data, "R_75")
        assert isinstance(targets, pd.DataFrame)
        assert not targets.empty
        
        print(f"✓ Feature collector test passed - Generated {features.shape[1]} features and {targets.shape[1]} targets")
        return True
        
    except Exception as e:
        print(f"✗ Feature collector test failed: {str(e)}")
        return False

def test_multi_strategy_engine():
    """Test multi-strategy engine"""
    print("Testing multi-strategy engine...")
    
    try:
        from strategies.multi_strategy_engine import MultiStrategyEngine, SignalType
        
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
        sample_data['high'] = np.maximum.reduce([
            sample_data['open'], 
            sample_data['high'], 
            sample_data['low'], 
            sample_data['close']
        ])
        sample_data['low'] = np.minimum.reduce([
            sample_data['open'], 
            sample_data['high'], 
            sample_data['low'], 
            sample_data['close']
        ])
        
        # Initialize strategy engine
        engine = MultiStrategyEngine()
        
        # Process symbol
        aggregated_signal = engine.process_symbol(sample_data, "R_75", "M5")
        
        # Should either return a signal or None
        if aggregated_signal:
            assert aggregated_signal.final_signal in [SignalType.BUY, SignalType.SELL]
            assert 0 <= aggregated_signal.strength <= 1
            assert 0 <= aggregated_signal.confidence <= 1
            print(f"✓ Multi-strategy engine test passed - Generated {aggregated_signal.final_signal.name} signal")
        else:
            print("✓ Multi-strategy engine test passed - No signal generated (normal)")
        
        # Test performance metrics
        performance = engine.get_strategy_performance()
        assert isinstance(performance, dict)
        assert len(performance) > 0
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-strategy engine test failed: {str(e)}")
        return False

def test_data_pipeline_structure():
    """Test data pipeline structure without MT5"""
    print("Testing data pipeline structure...")
    
    try:
        from core.data_pipeline import DataPipeline, LOTES, TIMEFRAME_MAP
        
        # Check constants
        assert len(LOTES) == 15
        assert 'R_75' in LOTES
        assert len(TIMEFRAME_MAP) == 7
        assert 'M5' in TIMEFRAME_MAP
        
        # Initialize pipeline (won't connect to MT5)
        pipeline = DataPipeline()
        
        assert hasattr(pipeline, 'all_symbols')
        assert hasattr(pipeline, 'volatility_indices')
        assert hasattr(pipeline, 'hz_indices')
        assert hasattr(pipeline, 'step_indices')
        assert len(pipeline.all_symbols) == 15
        
        print("✓ Data pipeline structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline structure test failed: {str(e)}")
        return False

def test_package_imports():
    """Test that all packages can be imported"""
    print("Testing package imports...")
    
    try:
        # Test core imports
        from ai import feature_collector
        from strategies import multi_strategy_engine
        from core import data_pipeline
        
        print("✓ Package imports test passed")
        return True
        
    except Exception as e:
        print(f"✗ Package imports test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("AI Trading System V2 - Component Tests")
    print("=" * 60)
    
    tests = [
        test_package_imports,
        test_asset_configuration,
        test_data_pipeline_structure,
        test_feature_collector,
        test_multi_strategy_engine,
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
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The AI Trading System V2 is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)