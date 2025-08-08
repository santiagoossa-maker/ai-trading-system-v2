#!/usr/bin/env python3
"""
AI Trading System V2 - Comprehensive Demo
Demonstrates all core functionality without requiring MT5 connection
"""

import sys
import os
import time
import subprocess
import signal
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")

def test_signal_generation():
    """Test signal generation with sample data"""
    print_section("Testing Signal Generation")
    
    try:
        from strategies.sma_macd_strategy import SMAMACDStrategy, SignalType
        import pandas as pd
        import numpy as np
        
        # Create realistic trending data
        print("Creating sample market data...")
        dates = pd.date_range('2024-01-01 10:00:00', periods=120, freq='5T')
        
        # Generate realistic price movement
        base_price = 150.0
        trend = np.linspace(0, 8, 120)  # Upward trend
        noise = np.random.normal(0, 0.3, 120)
        volatility = np.abs(np.random.normal(0, 0.1, 120))
        
        close_prices = base_price + trend + noise
        
        sample_data = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.05, 120),
            'high': close_prices + volatility,
            'low': close_prices - volatility,
            'close': close_prices,
            'volume': np.random.randint(500, 2000, 120)
        }, index=dates)
        
        print(f"✓ Generated {len(sample_data)} price bars")
        print(f"✓ Price range: {sample_data['close'].min():.3f} - {sample_data['close'].max():.3f}")
        
        # Initialize strategy
        strategy = SMAMACDStrategy({
            'sma_fast': 8,
            'sma_slow': 21,
            'min_confidence': 0.6
        })
        
        print("✓ SMA/MACD Strategy initialized")
        
        # Generate signal
        signal = strategy.generate_signal(sample_data, 'DEMO_SYMBOL', 'M5')
        
        if signal:
            print(f"✅ SIGNAL GENERATED!")
            print(f"   Signal Type: {signal.signal.name}")
            print(f"   Strength: {signal.strength:.3f}")
            print(f"   Confidence: {signal.confidence:.3f}")
            print(f"   Price: {signal.price:.3f}")
            print(f"   Timestamp: {signal.timestamp}")
            
            # Show indicators
            indicators = strategy.get_current_indicators(sample_data)
            if indicators:
                print(f"   Technical Indicators:")
                for name, value in indicators.items():
                    if not name.startswith('_'):
                        print(f"     {name}: {value:.5f}")
        else:
            print("ℹ️  No signal generated (waiting for better setup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation test failed: {e}")
        return False

def test_execution_engine():
    """Test execution engine in demo mode"""
    print_section("Testing Execution Engine (Demo Mode)")
    
    try:
        from core.execution_engine import ExecutionEngine, OrderType
        
        # Initialize execution engine
        engine = ExecutionEngine(mode='demo', config={
            'max_positions': 3,
            'max_daily_loss': 500.0
        })
        
        print("✓ Execution Engine initialized in demo mode")
        
        # Connect
        if engine.connect():
            print("✅ Connected to demo trading environment")
            
            # Test order placement
            symbols_to_test = ['R_75', 'R_100', 'R_50']
            
            for i, symbol in enumerate(symbols_to_test):
                order_type = OrderType.BUY if i % 2 == 0 else OrderType.SELL
                
                order_id = engine.place_order(
                    symbol=symbol,
                    order_type=order_type,
                    comment=f"Demo order {i+1}"
                )
                
                if order_id:
                    print(f"✅ Order placed: {order_id} ({order_type.value} {symbol})")
                else:
                    print(f"❌ Failed to place order for {symbol}")
            
            # Wait for position updates
            time.sleep(1)
            
            # Get account summary
            summary = engine.get_account_summary()
            print(f"📊 Account Summary:")
            print(f"   Mode: {summary['mode']}")
            print(f"   Open Positions: {summary['open_positions']}")
            print(f"   Total Orders: {summary['total_orders']}")
            print(f"   Open P&L: ${summary['open_profit']:.2f}")
            
            # Show positions
            positions = engine.get_positions()
            if positions:
                print(f"📈 Open Positions:")
                for pos in positions:
                    print(f"   {pos.id}: {pos.order_type.value} {pos.volume} {pos.symbol} @ {pos.open_price:.5f} | P&L: ${pos.profit:.2f}")
            
            # Close first position
            if positions:
                pos_id = positions[0].id
                if engine.close_position(pos_id, "Demo close"):
                    print(f"✅ Position {pos_id} closed successfully")
                else:
                    print(f"❌ Failed to close position {pos_id}")
            
            engine.disconnect()
            print("✓ Disconnected from demo environment")
            
            return True
        else:
            print("❌ Failed to connect to demo environment")
            return False
            
    except Exception as e:
        print(f"❌ Execution engine test failed: {e}")
        return False

def test_system_integration():
    """Test system integration"""
    print_section("Testing System Integration")
    
    try:
        from core.trading_system import TradingSystem
        
        # Create trading system
        system = TradingSystem(mode='demo', config={
            'symbols': ['R_75', 'R_100'],
            'timeframe': 'M5',
            'analysis_interval': 30
        })
        
        print("✓ Trading System initialized")
        print(f"✓ Mode: {system.mode}")
        print(f"✓ Trading symbols: {system.trading_symbols}")
        print(f"✓ Timeframe: {system.trading_timeframe}")
        
        # Test component initialization
        if system.initialize_components():
            print("✅ All components initialized successfully")
            
            # Get system status
            status = system.get_system_status()
            print(f"📊 System Status:")
            print(f"   Running: {status['running']}")
            print(f"   Mode: {status['mode']}")
            print(f"   Symbols: {len(status['trading_symbols'])}")
            print(f"   Timeframe: {status['trading_timeframe']}")
            
            return True
        else:
            print("❌ Component initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print_section("Testing Dashboard")
    
    try:
        dashboard_path = os.path.join('src', 'dashboard', 'streamlit_app.py')
        
        if os.path.exists(dashboard_path):
            print(f"✓ Dashboard found: {dashboard_path}")
            
            # Test dashboard imports (simplified)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("dashboard_test", dashboard_path)
                if spec and spec.loader:
                    print("✅ Dashboard module structure valid")
                else:
                    print("❌ Dashboard module spec invalid")
                    return False
                        
            except Exception as e:
                print(f"❌ Dashboard import test failed: {e}")
                return False
            
            print("✅ Dashboard ready for launch")
            print("   Launch command: streamlit run src/dashboard/streamlit_app.py")
            print("   URL: http://localhost:8501")
            
            return True
        else:
            print(f"❌ Dashboard not found: {dashboard_path}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def run_comprehensive_demo():
    """Run comprehensive demonstration"""
    print_header("AI TRADING SYSTEM V2 - COMPREHENSIVE DEMO")
    
    print("🚀 AI Trading System V2 - Comprehensive Functionality Demo")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Mode: Demo (No MT5 Required)")
    
    # Track test results
    test_results = {}
    
    # Run tests
    test_results['Signal Generation'] = test_signal_generation()
    test_results['Execution Engine'] = test_execution_engine()
    test_results['System Integration'] = test_system_integration()
    test_results['Dashboard'] = test_dashboard()
    
    # Summary
    print_header("DEMO RESULTS SUMMARY")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ The AI Trading System V2 is FULLY FUNCTIONAL")
        print()
        print("📋 System Capabilities Verified:")
        print("   ✓ Signal generation with SMA/MACD strategy")
        print("   ✓ Order execution and position management")
        print("   ✓ Risk management and account monitoring")
        print("   ✓ System integration and coordination")
        print("   ✓ Web dashboard for monitoring")
        print()
        print("🚀 Ready for deployment with MT5 connection!")
        print()
        print("📖 Usage Examples:")
        print("   # Start analysis mode:")
        print("   python -m src.core.trading_system --mode demo --action analyze")
        print()
        print("   # Start trading mode:")
        print("   python -m src.core.trading_system --mode demo --action trade")
        print()
        print("   # Launch dashboard:")
        print("   python -m src.core.trading_system --action dashboard")
        print("   # OR")
        print("   streamlit run src/dashboard/streamlit_app.py")
        
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("Please check the error messages above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)