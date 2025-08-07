"""
Simplified test script without TA-Lib dependency
Tests the basic structure and configuration
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
        
        # Check specific configurations
        volatility_config = config['asset_categories']['volatility_indices']
        assert 'assets' in volatility_config
        assert 'R_75' in volatility_config['assets']
        assert 'strategy_config' in volatility_config
        
        print("✓ Asset configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Asset configuration test failed: {str(e)}")
        return False

def test_package_structure():
    """Test that the package structure is correct"""
    print("Testing package structure...")
    
    try:
        # Check that source files exist
        files_to_check = [
            'src/ai/feature_collector.py',
            'src/strategies/multi_strategy_engine.py',
            'src/core/data_pipeline.py',
            'config/asset_specific_strategies.yaml',
            'requirements.txt'
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Check __init__.py files
        init_files = [
            'src/__init__.py',
            'src/ai/__init__.py',
            'src/strategies/__init__.py',
            'src/core/__init__.py'
        ]
        
        for init_file in init_files:
            if not os.path.exists(init_file):
                raise FileNotFoundError(f"Init file not found: {init_file}")
        
        print("✓ Package structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Package structure test failed: {str(e)}")
        return False

def test_requirements_file():
    """Test that requirements.txt is comprehensive"""
    print("Testing requirements file...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = [
            'pandas',
            'numpy',
            'scikit-learn',
            'PyYAML',
            'MetaTrader5',
            'redis',
            'streamlit'
        ]
        
        for package in essential_packages:
            if package not in requirements:
                raise ValueError(f"Essential package {package} not found in requirements")
        
        # Count total packages
        lines = [line.strip() for line in requirements.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        package_count = len(lines)
        
        assert package_count > 20, f"Expected more packages, found only {package_count}"
        
        print(f"✓ Requirements file test passed - Found {package_count} packages")
        return True
        
    except Exception as e:
        print(f"✗ Requirements file test failed: {str(e)}")
        return False

def test_data_constants():
    """Test that data constants are correctly defined"""
    print("Testing data constants...")
    
    try:
        # Import without initializing the full module
        from core.data_pipeline import LOTES, TIMEFRAME_MAP
        
        # Check LOTES
        assert len(LOTES) == 15
        expected_symbols = [
            "1HZ75V", "R_75", "R_100", "1HZ100V", "R_50", "1HZ50V", 
            "R_25", "R_10", "1HZ10V", "1HZ25V", "stpRNG", "stpRNG2", 
            "stpRNG3", "stpRNG4", "stpRNG5"
        ]
        
        for symbol in expected_symbols:
            assert symbol in LOTES, f"Symbol {symbol} not found in LOTES"
            assert isinstance(LOTES[symbol], (int, float)), f"Lot size for {symbol} is not numeric"
        
        # Check TIMEFRAME_MAP
        expected_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        for tf in expected_timeframes:
            assert tf in TIMEFRAME_MAP, f"Timeframe {tf} not found in TIMEFRAME_MAP"
        
        print("✓ Data constants test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data constants test failed: {str(e)}")
        return False

def test_feature_collector_structure():
    """Test feature collector class structure"""
    print("Testing feature collector structure...")
    
    try:
        from ai.feature_collector import FeatureCollector, IndicatorConfig, MarketRegime
        
        # Test enum
        assert hasattr(MarketRegime, 'TRENDING')
        assert hasattr(MarketRegime, 'RANGING')
        
        # Test config class
        config = IndicatorConfig()
        assert hasattr(config, 'sma_periods')
        assert hasattr(config, 'ema_periods')
        assert isinstance(config.sma_periods, list)
        assert len(config.sma_periods) > 5
        
        # Test collector class
        collector = FeatureCollector()
        assert hasattr(collector, 'timeframes')
        assert hasattr(collector, 'config')
        assert 'M5' in collector.timeframes
        
        print("✓ Feature collector structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Feature collector structure test failed: {str(e)}")
        return False

def test_strategy_engine_structure():
    """Test strategy engine class structure"""
    print("Testing strategy engine structure...")
    
    try:
        from strategies.multi_strategy_engine import (
            MultiStrategyEngine, SignalType, StrategyType, 
            TradingSignal, AggregatedSignal
        )
        
        # Test enums
        assert hasattr(SignalType, 'BUY')
        assert hasattr(SignalType, 'SELL')
        assert hasattr(SignalType, 'HOLD')
        
        assert hasattr(StrategyType, 'SMA_MACD')
        assert hasattr(StrategyType, 'EMA_MACD')
        assert hasattr(StrategyType, 'TRIPLE_EMA')
        
        # Test engine initialization
        engine = MultiStrategyEngine()
        assert hasattr(engine, 'strategies')
        assert hasattr(engine, 'aggregation_weights')
        assert len(engine.strategies) > 0
        
        print("✓ Strategy engine structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Strategy engine structure test failed: {str(e)}")
        return False

def main():
    """Run all basic tests"""
    print("=" * 60)
    print("AI Trading System V2 - Basic Structure Tests")
    print("=" * 60)
    
    tests = [
        test_package_structure,
        test_asset_configuration,
        test_requirements_file,
        test_data_constants,
        test_feature_collector_structure,
        test_strategy_engine_structure,
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
        print("🎉 Basic structure tests passed! The AI Trading System V2 foundation is solid.")
        print("💡 Next steps: Install remaining dependencies for full functionality.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)