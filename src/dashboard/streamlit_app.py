"""
Streamlit Dashboard for AI Trading System V2
Basic web interface for monitoring the trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import components (with fallback handling)
try:
    from src.core.trading_system import TradingSystem
    from src.core.market_data_engine import MarketDataEngine
    from src.strategies.sma_macd_strategy import SMAMACDStrategy
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    st.error(f"Could not import trading components: {e}")

# Page configuration
st.set_page_config(
    page_title="AI Trading System V2 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sample_data():
    """Create sample data for demonstration when components are not available"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
    price = 1000 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'time': dates,
        'open': price + np.random.randn(100) * 0.1,
        'high': price + np.abs(np.random.randn(100) * 0.3),
        'low': price - np.abs(np.random.randn(100) * 0.3),
        'close': price,
        'volume': np.random.randint(100, 1000, 100)
    })
    
    return df

def plot_price_chart(df, title="Price Chart"):
    """Create candlestick chart with indicators"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['sma_fast'],
                mode='lines',
                name='SMA 8',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_slow' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['sma_slow'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volume'],
            name="Volume",
            marker_color='rgba(0,100,80,0.7)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=True
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚀 AI Trading System V2 Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("System Control")
    
    # System status
    if COMPONENTS_AVAILABLE:
        st.sidebar.success("✅ Components Loaded")
    else:
        st.sidebar.error("❌ Components Not Available")
        st.sidebar.info("Running in demo mode with sample data")
    
    # Mode selection
    mode = st.sidebar.selectbox("Trading Mode", ["Demo", "Live"])
    
    # System control buttons
    col1, col2 = st.sidebar.columns(2)
    start_analysis = col1.button("Start Analysis")
    stop_system = col2.button("Stop System")
    
    # Main content
    if COMPONENTS_AVAILABLE:
        # Real system dashboard
        render_real_dashboard(mode.lower(), start_analysis, stop_system)
    else:
        # Demo dashboard with sample data
        render_demo_dashboard()

def render_real_dashboard(mode, start_analysis, stop_system):
    """Render dashboard with real trading system components"""
    
    # Initialize session state
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = None
    
    if 'system_running' not in st.session_state:
        st.session_state.system_running = False
    
    # System control
    if start_analysis and not st.session_state.system_running:
        try:
            st.session_state.trading_system = TradingSystem(mode=mode)
            success = st.session_state.trading_system.start_analysis()
            if success:
                st.session_state.system_running = True
                st.success(f"System started in {mode} mode")
            else:
                st.error("Failed to start system")
        except Exception as e:
            st.error(f"Error starting system: {str(e)}")
    
    if stop_system and st.session_state.system_running:
        try:
            if st.session_state.trading_system:
                st.session_state.trading_system.stop()
            st.session_state.system_running = False
            st.session_state.trading_system = None
            st.info("System stopped")
        except Exception as e:
            st.error(f"Error stopping system: {str(e)}")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.system_running:
            st.metric("System Status", "🟢 Running")
        else:
            st.metric("System Status", "🔴 Stopped")
    
    with col2:
        st.metric("Mode", mode.upper())
    
    with col3:
        if st.session_state.trading_system:
            status = st.session_state.trading_system.get_status()
            st.metric("MT5 Connection", "✅ Connected" if status['mt5_connected'] else "❌ Disconnected")
        else:
            st.metric("MT5 Connection", "❌ Not Connected")
    
    with col4:
        st.metric("Active Signals", "0")  # Placeholder
    
    # Data display
    if st.session_state.system_running and st.session_state.trading_system:
        try:
            # Get market data
            market_data_engine = MarketDataEngine()
            market_data = market_data_engine.get_latest_data()
            
            if market_data:
                # Display charts for each symbol
                for symbol, data in list(market_data.items())[:2]:  # Limit to 2 symbols
                    st.subheader(f"📊 {symbol}")
                    
                    df = data['data']
                    if df is not None and not df.empty:
                        # Add indicators using SMA MACD strategy
                        strategy = SMAMACDStrategy()
                        df_with_indicators = strategy.calculate_indicators(df)
                        
                        # Create chart
                        fig = plot_price_chart(df_with_indicators, f"{symbol} - {data['timeframe']}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate and display signal
                        signal = strategy.generate_signal(df_with_indicators, symbol)
                        if signal:
                            signal_color = "🟢" if signal.signal.value == 1 else "🔴" if signal.signal.value == -1 else "🟡"
                            st.info(f"{signal_color} Signal: {signal.signal.name} | Strength: {signal.strength:.2f} | Confidence: {signal.confidence:.2f}")
                        else:
                            st.info("🟡 No active signal")
                    else:
                        st.warning(f"No data available for {symbol}")
            else:
                st.warning("No market data available")
                
        except Exception as e:
            st.error(f"Error retrieving data: {str(e)}")
    else:
        st.info("Start the system to view live data and signals")

def render_demo_dashboard():
    """Render demo dashboard with sample data"""
    
    st.info("🔄 Running in demo mode with simulated data")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Demo Mode")
    
    with col2:
        st.metric("Demo Balance", "$10,000")
    
    with col3:
        st.metric("Active Positions", "2")
    
    with col4:
        st.metric("Today's P&L", "+$150")
    
    # Sample charts
    symbols = ["R_75", "R_100", "1HZ75V"]
    
    for symbol in symbols[:2]:  # Show 2 symbols
        st.subheader(f"📊 {symbol} (Demo)")
        
        # Create sample data
        df = create_sample_data()
        
        # Add sample indicators
        df['sma_fast'] = df['close'].rolling(8).mean()
        df['sma_slow'] = df['close'].rolling(50).mean()
        
        # Create chart
        fig = plot_price_chart(df, f"{symbol} - M5 (Sample Data)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample signal
        signal_types = ["BUY", "SELL", "HOLD"]
        signal = np.random.choice(signal_types)
        signal_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        strength = np.random.uniform(0.6, 0.9)
        st.info(f"{signal_color} Sample Signal: {signal} | Strength: {strength:.2f}")
    
    # Sample positions table
    st.subheader("💼 Current Positions (Demo)")
    
    sample_positions = pd.DataFrame({
        'Symbol': ['R_75', 'R_100'],
        'Type': ['BUY', 'SELL'],
        'Volume': [0.01, 0.5],
        'Entry Price': [1000.50, 2000.25],
        'Current Price': [1005.30, 1995.80],
        'P&L': ['+$48.00', '+$102.50']
    })
    
    st.dataframe(sample_positions, use_container_width=True)

# Auto-refresh
if COMPONENTS_AVAILABLE:
    # Add auto-refresh for live data
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    if st.sidebar.button("Enable Auto-refresh"):
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()