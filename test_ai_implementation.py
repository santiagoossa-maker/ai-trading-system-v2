"""
Test script for AI ensemble models
Tests the 4 AI models with minimal dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ai_models_structure():
    """Test that AI models can be imported and initialized"""
    print("Testing AI models structure...")
    
    try:
        # Test imports
        from ai.ai_strategy_manager import AIStrategyManager, AIDecisionType
        
        # Test initialization
        manager = AIStrategyManager()
        
        # Check status
        status = manager.get_system_status()
        print(f"  AI models available: {status['ai_models_available']}")
        print(f"  Strategy components available: {status['strategy_components_available']}")
        print(f"  Feature collector ready: {status['feature_collector_ready']}")
        
        # Test decision types
        assert hasattr(AIDecisionType, 'STRONG_BUY')
        assert hasattr(AIDecisionType, 'BUY')
        assert hasattr(AIDecisionType, 'HOLD')
        assert hasattr(AIDecisionType, 'SELL')
        assert hasattr(AIDecisionType, 'STRONG_SELL')
        
        print("✓ AI models structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ AI models structure test failed: {str(e)}")
        return False

def test_ai_analysis_workflow():
    """Test the complete AI analysis workflow"""
    print("Testing AI analysis workflow...")
    
    try:
        from ai.ai_strategy_manager import AIStrategyManager
        
        # Initialize manager
        manager = AIStrategyManager()
        
        # Create sample market data
        dates = pd.date_range('2024-01-01', periods=100, freq='5T')
        np.random.seed(42)
        
        sample_data = {
            'M5': pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 101,
                'low': np.random.randn(100).cumsum() + 99,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(100, 1000, 100)
            }, index=dates)
        }
        
        # Ensure OHLC consistency
        for i in range(len(sample_data['M5'])):
            ohlc = [
                sample_data['M5']['open'].iloc[i],
                sample_data['M5']['high'].iloc[i],
                sample_data['M5']['low'].iloc[i],
                sample_data['M5']['close'].iloc[i]
            ]
            sample_data['M5']['high'].iloc[i] = max(ohlc)
            sample_data['M5']['low'].iloc[i] = min(ohlc)
        
        # Test AI analysis
        analysis = manager.analyze_signal(sample_data, "R_75")
        
        # Verify analysis structure
        assert hasattr(analysis, 'signal_quality_score')
        assert hasattr(analysis, 'expected_profit_pips')
        assert hasattr(analysis, 'expected_duration_minutes')
        assert hasattr(analysis, 'max_risk_percent')
        assert hasattr(analysis, 'confidence_level')
        
        # Test trading decision
        decision = manager.make_trading_decision(sample_data, "R_75")
        
        # Verify decision structure
        assert hasattr(decision, 'decision')
        assert hasattr(decision, 'ai_analysis')
        assert hasattr(decision, 'position_size_multiplier')
        assert hasattr(decision, 'confidence_score')
        
        print(f"  Analysis - Quality: {analysis.signal_quality_score:.3f}, "
              f"Profit: {analysis.expected_profit_pips:.1f} pips, "
              f"Risk: {analysis.max_risk_percent:.2f}%")
        print(f"  Decision: {decision.decision.value} (confidence: {decision.confidence_score:.3f})")
        
        print("✓ AI analysis workflow test passed")
        return True
        
    except Exception as e:
        print(f"✗ AI analysis workflow test failed: {str(e)}")
        return False

def test_feature_integration():
    """Test feature collector integration with AI models"""
    print("Testing feature integration...")
    
    try:
        from ai.feature_collector import FeatureCollector
        
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
        
        # Ensure OHLC consistency
        for i in range(len(sample_data['M5'])):
            ohlc = [
                sample_data['M5']['open'].iloc[i],
                sample_data['M5']['high'].iloc[i],
                sample_data['M5']['low'].iloc[i],
                sample_data['M5']['close'].iloc[i]
            ]
            sample_data['M5']['high'].iloc[i] = max(ohlc)
            sample_data['M5']['low'].iloc[i] = min(ohlc)
        
        # Test feature collection
        collector = FeatureCollector()
        features = collector.collect_all_features(sample_data, "R_75")
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert features.shape[1] > 10  # Should have many features
        
        # Test target generation
        targets = collector.generate_target_variables(sample_data, "R_75")
        assert isinstance(targets, pd.DataFrame)
        assert not targets.empty
        
        print(f"  Generated {features.shape[1]} features and {targets.shape[1]} targets")
        print(f"  Feature sample: {list(features.columns[:5])}")
        print(f"  Target sample: {list(targets.columns[:3])}")
        
        print("✓ Feature integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Feature integration test failed: {str(e)}")
        return False

def test_ai_requirements():
    """Test AI requirements and dependencies"""
    print("Testing AI requirements...")
    
    try:
        # Check if AI requirements file exists
        req_file = "requirements_ai.txt"
        if not os.path.exists(req_file):
            raise FileNotFoundError(f"AI requirements file not found: {req_file}")
        
        # Read and validate requirements
        with open(req_file, 'r') as f:
            requirements = f.read()
        
        # Check for essential AI packages
        essential_ai_packages = [
            'scikit-learn',
            'xgboost', 
            'lightgbm',
            'tensorflow',
            'optuna'
        ]
        
        for package in essential_ai_packages:
            if package not in requirements:
                raise ValueError(f"Essential AI package {package} not found in requirements")
        
        # Count AI packages
        lines = [line.strip() for line in requirements.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        ai_package_count = len(lines)
        
        print(f"  Found {ai_package_count} AI packages in requirements")
        print(f"  Essential packages verified: {essential_ai_packages}")
        
        print("✓ AI requirements test passed")
        return True
        
    except Exception as e:
        print(f"✗ AI requirements test failed: {str(e)}")
        return False

def main():
    """Run all AI system tests"""
    print("=" * 60)
    print("AI Trading System V2 - AI Components Tests")
    print("=" * 60)
    
    tests = [
        test_ai_requirements,
        test_ai_models_structure,
        test_feature_integration,
        test_ai_analysis_workflow,
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
    print(f"AI Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All AI tests passed! The AI ensemble system is ready.")
        print("💡 Note: Models need to be trained with real data for full functionality.")
        return True
    else:
        print("❌ Some AI tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)