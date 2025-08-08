"""
Real-Time Trading Dashboard
Streamlit dashboard for monitoring and controlling the AI trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
import sys
import os
import logging
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Trading System V2 - Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit-positive {
        color: #00ff88;
    }
    .profit-negative {
        color: #ff6b6b;
    }
    .signal-buy {
        background-color: #00ff8830;
        padding: 0.5rem;
        border-radius: 0.25rem;
        color: #00ff88;
    }
    .signal-sell {
        background-color: #ff6b6b30;
        padding: 0.5rem;
        border-radius: 0.25rem;
        color: #ff6b6b;
    }
    .position-card {
        border: 1px solid #333;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_pipeline = None
        self.execution_engine = None
        self.strategy = None
        self.connected = False
        self.demo_mode = True  # Default to demo mode
        
        # Initialize session state
        if 'trading_started' not in st.session_state:
            st.session_state.trading_started = False
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
    
    def initialize_components(self):
        """Initialize trading components"""
        try:
            if not self.demo_mode:
                # Real trading mode
                from core.data_pipeline import DataPipeline
                from core.mt5_execution_engine import MT5ExecutionEngine
                from strategies.sma_macd_live_strategy import SMA_MACD_Strategy
                
                # Initialize components
                self.data_pipeline = DataPipeline()
                self.execution_engine = MT5ExecutionEngine()
                
                # Connect to MT5
                if self.execution_engine.connect() and self.data_pipeline.start():
                    self.strategy = SMA_MACD_Strategy(self.data_pipeline, self.execution_engine)
                    self.connected = True
                    return True
                else:
                    st.error("Failed to connect to MT5 or start data pipeline")
                    return False
            else:
                # Demo mode with simulated data
                self.connected = True
                return True
                
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return False
    
    def generate_demo_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Generate demo OHLCV data for testing"""
        np.random.seed(42)
        
        # Generate price data with trend
        base_price = 100.0
        price_changes = np.random.normal(0, 0.002, bars)
        prices = base_price * np.cumprod(1 + price_changes)
        
        # Generate OHLCV data
        data = []
        for i in range(bars):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(100, 1000)
            
            data.append({
                'open': open_price,
                'high': max(high, price, open_price),
                'low': min(low, price, open_price),
                'close': price,
                'volume': volume,
                'timestamp': datetime.now() - timedelta(minutes=(bars-i)*5)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_demo_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate indicators for demo data"""
        close = data['close'].values
        
        # Simple moving averages
        sma_8 = pd.Series(close).rolling(window=8).mean().values
        sma_50 = pd.Series(close).rolling(window=50).mean().values
        
        # Simple MACD
        ema_12 = pd.Series(close).ewm(span=12).mean()
        ema_26 = pd.Series(close).ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'sma_8': sma_8,
            'sma_50': sma_50,
            'macd': macd.values,
            'macd_signal': macd_signal.values,
            'macd_histogram': macd_histogram.values
        }
    
    def get_account_summary(self) -> Dict:
        """Get account summary data"""
        if self.demo_mode:
            return {
                'balance': 10000.0,
                'equity': 10150.0,
                'margin': 500.0,
                'free_margin': 9650.0,
                'margin_level': 2030.0,
                'profit': 150.0,
                'currency': 'USD'
            }
        elif self.execution_engine:
            return self.execution_engine.get_account_info()
        else:
            return {}
    
    def get_positions_data(self) -> List[Dict]:
        """Get current positions"""
        if self.demo_mode:
            # Demo positions
            return [
                {
                    'ticket': 12345,
                    'symbol': 'R_75',
                    'type': 'BUY',
                    'volume': 0.01,
                    'open_price': 1.25430,
                    'current_price': 1.25480,
                    'sl': 1.25300,
                    'tp': 1.25700,
                    'profit': 5.0,
                    'time': datetime.now() - timedelta(minutes=15)
                },
                {
                    'ticket': 12346,
                    'symbol': 'R_100',
                    'type': 'SELL',
                    'volume': 0.5,
                    'open_price': 2.10250,
                    'current_price': 2.10200,
                    'sl': 2.10400,
                    'tp': 2.10000,
                    'profit': 25.0,
                    'time': datetime.now() - timedelta(minutes=8)
                }
            ]
        elif self.execution_engine:
            summary = self.execution_engine.get_positions_summary()
            return summary.get('positions', [])
        else:
            return []
    
    def get_strategy_status(self) -> Dict:
        """Get strategy status"""
        if self.demo_mode:
            return {
                'running': st.session_state.trading_started,
                'symbols': ['R_75', 'R_100', 'R_50'],
                'timeframe': 'M5',
                'active_positions': 2,
                'trades_today': 8,
                'winning_trades': 6,
                'win_rate': 75.0,
                'total_profit': 150.0,
                'last_analysis': {
                    'R_75': {
                        'timestamp': datetime.now(),
                        'signal': 'BUY',
                        'confidence': 0.82,
                        'price': 1.25480
                    },
                    'R_100': {
                        'timestamp': datetime.now(),
                        'signal': 'SELL',
                        'confidence': 0.75,
                        'price': 2.10200
                    }
                }
            }
        elif self.strategy:
            return self.strategy.get_strategy_status()
        else:
            return {'running': False}
    
    def create_price_chart(self, symbol: str) -> go.Figure:
        """Create price chart with indicators"""
        # Get data
        if self.demo_mode:
            data = self.generate_demo_data(symbol)
            indicators = self.calculate_demo_indicators(data)
        else:
            data = self.data_pipeline.get_latest_data(symbol, 'M5', count=100)
            # Calculate indicators here if needed
            indicators = {}
        
        if data.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{symbol} Price Chart', 'MACD'],
            vertical_spacing=0.1
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add SMAs if available
        if 'sma_8' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_8'],
                    mode='lines',
                    name='SMA 8',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # MACD chart
        if 'macd' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=indicators['macd_histogram'],
                    name='Histogram',
                    marker_color='gray'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - SMA8/50 + MACD Strategy',
            template='plotly_dark',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🚀 AI Trading System V2 - Live Dashboard")
        
        # Connection status
        status_color = "🟢" if self.connected else "🔴"
        mode = "DEMO MODE" if self.demo_mode else "LIVE TRADING"
        st.markdown(f"**Status:** {status_color} {mode} {'Connected' if self.connected else 'Disconnected'}")
        
        # Current time
        st.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Trading controls
        st.sidebar.subheader("Trading Controls")
        
        if st.sidebar.button("🚀 Start Trading", type="primary"):
            if self.demo_mode or (self.strategy and self.connected):
                st.session_state.trading_started = True
                if not self.demo_mode and self.strategy:
                    self.strategy.start_trading()
                st.sidebar.success("Trading started!")
            else:
                st.sidebar.error("Please connect to MT5 first")
        
        if st.sidebar.button("⏹️ Stop Trading", type="secondary"):
            st.session_state.trading_started = False
            if not self.demo_mode and self.strategy:
                self.strategy.stop_trading()
            st.sidebar.warning("Trading stopped!")
        
        # Emergency controls
        st.sidebar.subheader("🚨 Emergency Controls")
        if st.sidebar.button("❌ Close All Positions", type="secondary"):
            if not self.demo_mode and self.execution_engine:
                results = self.execution_engine.close_all_positions()
                st.sidebar.info(f"Closed {len(results)} positions")
            else:
                st.sidebar.info("Demo mode: All positions closed")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        st.session_state.refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        # Strategy parameters
        st.sidebar.subheader("📊 Strategy Parameters")
        st.sidebar.text("SMA Fast: 8")
        st.sidebar.text("SMA Slow: 50")
        st.sidebar.text("MACD: 12, 26, 9")
        st.sidebar.text("Risk per Trade: 2%")
        st.sidebar.text("Risk/Reward: 1:2")
    
    def render_account_overview(self):
        """Render account overview section"""
        st.subheader("💰 Account Overview")
        
        account = self.get_account_summary()
        
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Balance", f"${account.get('balance', 0):,.2f}")
            
            with col2:
                equity = account.get('equity', 0)
                balance = account.get('balance', 0)
                equity_change = equity - balance
                st.metric("Equity", f"${equity:,.2f}", f"{equity_change:+.2f}")
            
            with col3:
                profit = account.get('profit', 0)
                profit_color = "normal" if profit >= 0 else "inverse"
                st.metric("Unrealized P&L", f"${profit:,.2f}", delta_color=profit_color)
            
            with col4:
                margin_level = account.get('margin_level', 0)
                st.metric("Margin Level", f"{margin_level:.0f}%")
    
    def render_strategy_status(self):
        """Render strategy status section"""
        st.subheader("🤖 Strategy Status")
        
        status = self.get_strategy_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_text = "🟢 RUNNING" if status.get('running', False) else "🔴 STOPPED"
            st.markdown(f"**Status:** {status_text}")
        
        with col2:
            st.metric("Active Positions", status.get('active_positions', 0))
        
        with col3:
            win_rate = status.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            total_profit = status.get('total_profit', 0)
            st.metric("Today's P&L", f"${total_profit:.2f}")
        
        # Last analysis
        if 'last_analysis' in status:
            st.subheader("📈 Latest Signals")
            analysis = status['last_analysis']
            
            cols = st.columns(len(analysis))
            for i, (symbol, data) in enumerate(analysis.items()):
                with cols[i]:
                    signal = data.get('signal', 'HOLD')
                    confidence = data.get('confidence', 0)
                    price = data.get('price', 0)
                    
                    signal_class = "signal-buy" if signal == 'BUY' else "signal-sell" if signal == 'SELL' else ""
                    
                    st.markdown(f"""
                    <div class="{signal_class}">
                        <strong>{symbol}</strong><br>
                        Signal: {signal}<br>
                        Confidence: {confidence:.0%}<br>
                        Price: {price:.5f}
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_positions_table(self):
        """Render active positions table"""
        st.subheader("📊 Active Positions")
        
        positions = self.get_positions_data()
        
        if positions:
            df = pd.DataFrame(positions)
            
            # Format the dataframe for display
            df['Open Time'] = pd.to_datetime(df['time']).dt.strftime('%H:%M:%S')
            df['P&L'] = df['profit'].apply(lambda x: f"${x:.2f}")
            df['Volume'] = df['volume']
            df['Open Price'] = df['open_price'].apply(lambda x: f"{x:.5f}")
            df['Current Price'] = df['current_price'].apply(lambda x: f"{x:.5f}")
            df['SL'] = df['sl'].apply(lambda x: f"{x:.5f}")
            df['TP'] = df['tp'].apply(lambda x: f"{x:.5f}")
            
            # Select columns for display
            display_df = df[['ticket', 'symbol', 'type', 'Volume', 'Open Price', 'Current Price', 'SL', 'TP', 'P&L', 'Open Time']]
            display_df.columns = ['Ticket', 'Symbol', 'Type', 'Volume', 'Open Price', 'Current', 'Stop Loss', 'Take Profit', 'P&L', 'Time']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Position management
            st.subheader("Position Management")
            col1, col2 = st.columns(2)
            
            with col1:
                selected_ticket = st.selectbox("Select Position", options=[p['ticket'] for p in positions])
            
            with col2:
                if st.button("Close Selected Position"):
                    if not self.demo_mode and self.execution_engine:
                        result = self.execution_engine.close_position(selected_ticket)
                        if result.success:
                            st.success(f"Position {selected_ticket} closed successfully")
                        else:
                            st.error(f"Failed to close position: {result.comment}")
                    else:
                        st.info(f"Demo mode: Position {selected_ticket} closed")
        else:
            st.info("No active positions")
    
    def render_price_charts(self):
        """Render price charts section"""
        st.subheader("📈 Live Price Charts")
        
        # Symbol selection
        symbols = ['R_75', 'R_100', 'R_50', 'R_25', 'R_10']
        selected_symbol = st.selectbox("Select Symbol", symbols)
        
        # Create and display chart
        chart = self.create_price_chart(selected_symbol)
        st.plotly_chart(chart, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        st.subheader("📊 Performance Metrics")
        
        # Generate demo performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        equity_curve = np.cumsum(np.random.normal(10, 50, len(dates))) + 10000
        
        performance_data = pd.DataFrame({
            'Date': dates,
            'Equity': equity_curve,
            'Drawdown': np.random.uniform(-5, 0, len(dates))
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve
            fig_equity = px.line(performance_data, x='Date', y='Equity', title='Equity Curve')
            fig_equity.update_layout(template='plotly_dark')
            st.plotly_chart(fig_equity, use_container_width=True)
        
        with col2:
            # Drawdown chart
            fig_dd = px.area(performance_data, x='Date', y='Drawdown', title='Drawdown %')
            fig_dd.update_layout(template='plotly_dark')
            st.plotly_chart(fig_dd, use_container_width=True)
    
    def run(self):
        """Main dashboard loop"""
        # Initialize components if not done
        if not hasattr(self, 'initialized'):
            self.initialize_components()
            self.initialized = True
        
        # Render all sections
        self.render_header()
        self.render_sidebar()
        
        # Main content
        self.render_account_overview()
        st.divider()
        
        self.render_strategy_status()
        st.divider()
        
        self.render_positions_table()
        st.divider()
        
        self.render_price_charts()
        st.divider()
        
        self.render_performance_metrics()
        
        # Auto refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()

def main():
    """Main function"""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()