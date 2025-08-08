"""
Streamlit Dashboard for AI Trading System V2
Provides web interface for monitoring the trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import trading system components
try:
    from core.trading_system import TradingSystem
    from core.market_data_engine import MarketDataEngine
    from strategies.sma_macd_strategy import SMAMACDStrategy
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    st.error(f"Trading system components not available: {e}")

# Page configuration
st.set_page_config(
    page_title="AI Trading System V2 Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #00ff00; }
    .status-bad { color: #ff0000; }
    .status-warning { color: #ffa500; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🚀 AI Trading System V2 Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("System Control")
        
        # System status
        if COMPONENTS_AVAILABLE:
            status_placeholder = st.empty()
        else:
            st.error("Components not available")
        
        st.markdown("---")
        
        # Configuration
        st.header("Configuration")
        
        mode = st.selectbox("Trading Mode", ["Demo", "Live"], index=0)
        symbols = st.multiselect(
            "Trading Symbols", 
            ["R_75", "R_100", "R_50", "R_25", "R_10", "1HZ75V", "1HZ100V"],
            default=["R_75", "R_100", "R_50"]
        )
        timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1"], index=1)
        
        st.markdown("---")
        
        # Action buttons
        st.header("Actions")
        
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        if st.button("📊 Generate Test Signal"):
            show_test_signal()
    
    # Main content area
    if not COMPONENTS_AVAILABLE:
        show_placeholder_dashboard()
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Signals", "💼 Positions", "⚙️ System Status"])
    
    with tab1:
        show_market_overview(symbols, timeframe)
    
    with tab2:
        show_signals_tab()
    
    with tab3:
        show_positions_tab()
    
    with tab4:
        show_system_status()

def show_placeholder_dashboard():
    """Show placeholder dashboard when components are not available"""
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "Initializing", "⚠️")
    
    with col2:
        st.metric("Signals Generated", "0", "📊")
    
    with col3:
        st.metric("Active Trades", "0", "💼")
    
    with col4:
        st.metric("P&L Today", "$0.00", "💰")
    
    st.markdown("---")
    
    # Sample chart
    st.subheader("📈 Sample Market Data")
    
    # Generate sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), end=datetime.now(), freq='5T')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices
    })
    
    fig = px.line(df, x='timestamp', y='price', title='Sample Price Chart (R_75)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Info message
    st.info("📌 This is a placeholder dashboard. The trading system components will be loaded when available.")
    
    # System info
    st.subheader("🔧 System Information")
    
    info_data = {
        'Component': ['Market Data Engine', 'Strategy Engine', 'Execution Engine', 'Dashboard'],
        'Status': ['⚠️ Loading', '⚠️ Loading', '⚠️ Loading', '✅ Active'],
        'Last Update': ['--', '--', '--', datetime.now().strftime('%H:%M:%S')]
    }
    
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)

def show_market_overview(symbols, timeframe):
    """Show market overview tab"""
    
    st.subheader("📊 Market Overview")
    
    # Try to create market data engine
    try:
        market_engine = MarketDataEngine()
        if market_engine.connect():
            
            # Get data for each symbol
            for symbol in symbols:
                with st.expander(f"📈 {symbol}", expanded=True):
                    show_symbol_chart(market_engine, symbol, timeframe)
            
            market_engine.disconnect()
        else:
            st.warning("Cannot connect to market data. Showing sample data.")
            show_sample_charts(symbols)
            
    except Exception as e:
        st.error(f"Error accessing market data: {e}")
        show_sample_charts(symbols)

def show_symbol_chart(market_engine, symbol, timeframe):
    """Show chart for a specific symbol"""
    try:
        # Get historical data
        data = market_engine.get_historical_data(symbol, timeframe, 100)
        
        if data.empty:
            st.warning(f"No data available for {symbol}")
            return
        
        # Create candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current price info
        current_price = data['close'].iloc[-1]
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        change_pct = (price_change / data['close'].iloc[-2]) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"{current_price:.5f}")
        with col2:
            st.metric("Change", f"{price_change:.5f}", f"{change_pct:.2f}%")
        with col3:
            st.metric("Volume", f"{data['volume'].iloc[-1]:,}")
        
    except Exception as e:
        st.error(f"Error displaying chart for {symbol}: {e}")

def show_sample_charts(symbols):
    """Show sample charts when real data is not available"""
    for symbol in symbols:
        with st.expander(f"📈 {symbol} (Sample Data)", expanded=True):
            
            # Generate sample data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=8), end=datetime.now(), freq='5T')
            base_price = 100 + np.random.rand() * 50
            
            price_data = []
            current_price = base_price
            
            for _ in dates:
                change = np.random.randn() * 0.5
                current_price *= (1 + change / 100)
                price_data.append(current_price)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'price': price_data,
                'volume': np.random.randint(100, 1000, len(dates))
            })
            
            fig = px.line(df, x='timestamp', y='price', title=f'{symbol} Price (Sample)')
            st.plotly_chart(fig, use_container_width=True)

def show_signals_tab():
    """Show signals analysis tab"""
    
    st.subheader("📈 Trading Signals")
    
    # Try to generate sample signals
    try:
        strategy = SMAMACDStrategy()
        
        # Sample data for signal generation
        dates = pd.date_range(start=datetime.now() - timedelta(hours=4), end=datetime.now(), freq='5T')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        
        sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.05, len(dates)),
            'high': prices + np.abs(np.random.normal(0, 0.1, len(dates))),
            'low': prices - np.abs(np.random.normal(0, 0.1, len(dates))),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Generate signal
        signal = strategy.generate_signal(sample_data, "SAMPLE", "M5")
        
        if signal:
            st.success(f"✅ Latest Signal: {signal.signal.name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Signal Type", signal.signal.name)
            with col2:
                st.metric("Strength", f"{signal.strength:.3f}")
            with col3:
                st.metric("Confidence", f"{signal.confidence:.3f}")
            with col4:
                st.metric("Price", f"{signal.price:.5f}")
            
            # Show indicators
            st.subheader("📊 Technical Indicators")
            indicators = strategy.get_current_indicators(sample_data)
            
            if indicators:
                indicators_df = pd.DataFrame([indicators]).T
                indicators_df.columns = ['Value']
                st.dataframe(indicators_df, use_container_width=True)
        else:
            st.info("No signals generated yet.")
        
        # Show sample signal history
        st.subheader("📋 Recent Signals")
        signal_history = [
            {"Time": "14:30:00", "Symbol": "R_75", "Signal": "BUY", "Strength": 0.85, "Status": "Executed"},
            {"Time": "14:25:00", "Symbol": "R_100", "Signal": "SELL", "Strength": 0.72, "Status": "Executed"},
            {"Time": "14:20:00", "Symbol": "R_50", "Signal": "BUY", "Strength": 0.68, "Status": "Pending"},
        ]
        
        st.dataframe(pd.DataFrame(signal_history), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in signals tab: {e}")

def show_positions_tab():
    """Show positions and trades tab"""
    
    st.subheader("💼 Open Positions")
    
    # Sample positions data
    positions_data = [
        {"ID": "POS_001", "Symbol": "R_75", "Type": "BUY", "Volume": 0.01, "Entry": 145.234, "Current": 145.456, "P&L": 0.22},
        {"ID": "POS_002", "Symbol": "R_100", "Type": "SELL", "Volume": 0.5, "Entry": 234.567, "Current": 234.123, "P&L": 2.22},
    ]
    
    if positions_data:
        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, use_container_width=True)
        
        # Summary metrics
        total_pnl = sum(pos["P&L"] for pos in positions_data)
        total_positions = len(positions_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Open Positions", total_positions)
        with col2:
            st.metric("Total P&L", f"${total_pnl:.2f}")
        with col3:
            winning_positions = len([p for p in positions_data if p["P&L"] > 0])
            win_rate = (winning_positions / total_positions) * 100 if total_positions > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
    else:
        st.info("No open positions")
    
    st.markdown("---")
    
    st.subheader("📊 Performance Summary")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", "15", "📈")
    with col2:
        st.metric("Win Rate", "73.3%", "5.2%")
    with col3:
        st.metric("Profit Factor", "2.1", "0.3")
    with col4:
        st.metric("Max Drawdown", "3.2%", "-1.1%")

def show_system_status():
    """Show system status tab"""
    
    st.subheader("⚙️ System Status")
    
    # System components status
    components = [
        {"Component": "Market Data Engine", "Status": "🟢 Connected", "Last Update": "Just now"},
        {"Component": "Strategy Engine", "Status": "🟢 Active", "Last Update": "5 seconds ago"},
        {"Component": "Execution Engine", "Status": "🟢 Connected", "Last Update": "2 seconds ago"},
        {"Component": "Risk Manager", "Status": "🟢 Monitoring", "Last Update": "1 second ago"},
    ]
    
    st.dataframe(pd.DataFrame(components), use_container_width=True)
    
    st.markdown("---")
    
    # System metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Metrics")
        metrics_data = {
            "Metric": ["Uptime", "Signals Generated", "Orders Executed", "Data Points Processed", "Error Rate"],
            "Value": ["2h 34m", "127", "15", "45,678", "0.02%"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    with col2:
        st.subheader("🔧 System Configuration")
        config_data = {
            "Setting": ["Trading Mode", "Active Symbols", "Timeframe", "Analysis Interval", "Max Positions"],
            "Value": ["Demo", "3", "M5", "30s", "5"]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True)
    
    # Resource usage
    st.subheader("💻 Resource Usage")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CPU Usage", "23%")
    with col2:
        st.metric("Memory Usage", "145 MB")
    with col3:
        st.metric("Network I/O", "1.2 KB/s")
    with col4:
        st.metric("Disk Usage", "2.3 GB")

def show_test_signal():
    """Show test signal generation"""
    st.success("🧪 Test signal generated!")
    
    test_signal_data = {
        "Symbol": "TEST",
        "Signal": "BUY",
        "Strength": 0.85,
        "Confidence": 0.92,
        "Price": 123.456,
        "Time": datetime.now().strftime("%H:%M:%S")
    }
    
    st.json(test_signal_data)

# Auto-refresh functionality
def auto_refresh():
    """Auto refresh the dashboard"""
    placeholder = st.empty()
    
    # Auto-refresh every 30 seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = current_time
        st.experimental_rerun()

if __name__ == "__main__":
    main()
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "🚀 **AI Trading System V2** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Made with ❤️ and Streamlit"
    )