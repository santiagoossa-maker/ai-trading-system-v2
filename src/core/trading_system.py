"""
Core Trading System
Main orchestrator class that coordinates all components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import threading
import time
import signal
import sys
from datetime import datetime, timedelta
import os
import subprocess

# Import core components
from .market_data_engine import MarketDataEngine
from .execution_engine import ExecutionEngine, OrderType
from ..strategies.sma_macd_strategy import SMAMACDStrategy, SignalType

logger = logging.getLogger(__name__)

class TradingSystem:
    """
    Main trading system that coordinates all components:
    - Market data collection
    - Signal generation
    - Order execution
    - Risk management
    - Dashboard interface
    """
    
    def __init__(self, mode: str = 'demo', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trading system
        
        Args:
            mode: 'demo' or 'live'
            config: Configuration dictionary
        """
        self.mode = mode.lower()
        self.config = config or {}
        
        # System state
        self.is_running = False
        self.is_analyzing = False
        self.is_trading = False
        
        # Components
        self.market_data = None
        self.execution_engine = None
        self.strategy = None
        self.dashboard_process = None
        
        # Trading parameters
        self.trading_symbols = self.config.get('symbols', ['R_75', 'R_100', 'R_50'])
        self.trading_timeframe = self.config.get('timeframe', 'M5')
        self.analysis_interval = self.config.get('analysis_interval', 30)  # seconds
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_analysis_time = None
        self.system_start_time = None
        
        # Threading
        self._analysis_thread = None
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Trading System initialized in {mode} mode")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize market data engine
            self.market_data = MarketDataEngine(self.config.get('market_data_config'))
            
            # Initialize execution engine
            execution_config = self.config.get('execution_config', {})
            execution_config['mode'] = self.mode
            self.execution_engine = ExecutionEngine(self.mode, execution_config)
            
            # Initialize strategy
            strategy_config = self.config.get('strategy_config', {})
            self.strategy = SMAMACDStrategy(strategy_config)
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def connect(self) -> bool:
        """Connect all components"""
        try:
            if not self.initialize_components():
                return False
            
            logger.info("Connecting to trading infrastructure...")
            
            # Connect market data
            if not self.market_data.connect():
                logger.error("Failed to connect market data engine")
                return False
            
            # Connect execution engine
            if not self.execution_engine.connect():
                logger.error("Failed to connect execution engine")
                return False
            
            # Wait for initial data
            logger.info("Waiting for initial market data...")
            time.sleep(5)
            
            # Verify data availability
            test_data = self.market_data.get_historical_data('R_75', 'M5', 50)
            if test_data.empty:
                logger.warning("No market data available, but continuing...")
            else:
                logger.info(f"Market data available: {len(test_data)} bars for R_75")
            
            self.system_start_time = datetime.now()
            logger.info("All systems connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting systems: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect all components"""
        try:
            logger.info("Disconnecting trading system...")
            
            if self.market_data:
                self.market_data.disconnect()
            
            if self.execution_engine:
                self.execution_engine.disconnect()
            
            if self.dashboard_process:
                try:
                    self.dashboard_process.terminate()
                    self.dashboard_process.wait(timeout=5)
                    logger.info("Dashboard process terminated")
                except:
                    self.dashboard_process.kill()
                    logger.info("Dashboard process killed")
            
            logger.info("All systems disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
    
    def start_analysis(self):
        """Start market analysis without trading"""
        try:
            if not self.connect():
                logger.error("Failed to connect, cannot start analysis")
                return False
            
            logger.info("Starting market analysis...")
            self.is_running = True
            self.is_analyzing = True
            self.is_trading = False
            
            # Start analysis thread
            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._analysis_thread.start()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            
            logger.info("Market analysis started successfully")
            logger.info("System is now analyzing markets and generating signals")
            logger.info("Use start_trading() to enable actual trading")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting analysis: {str(e)}")
            return False
    
    def start_trading(self):
        """Start full trading (analysis + execution)"""
        try:
            if not self.is_analyzing:
                # Start analysis first
                if not self.start_analysis():
                    return False
            
            logger.info("Enabling trading mode...")
            self.is_trading = True
            
            logger.info("Full trading system active")
            logger.warning(f"TRADING LIVE IN {self.mode.upper()} MODE!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            return False
    
    def stop(self):
        """Stop the trading system"""
        try:
            logger.info("Stopping trading system...")
            
            # Set shutdown flag
            self._shutdown_event.set()
            
            # Stop trading and analysis
            self.is_trading = False
            self.is_analyzing = False
            self.is_running = False
            
            # Wait for threads to finish
            if self._analysis_thread and self._analysis_thread.is_alive():
                self._analysis_thread.join(timeout=10)
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=10)
            
            # Disconnect components
            self.disconnect()
            
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {str(e)}")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        logger.info("Analysis loop started")
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Analyze each symbol
                for symbol in self.trading_symbols:
                    if not self.is_running:
                        break
                    
                    self._analyze_symbol(symbol)
                
                self.last_analysis_time = datetime.now()
                
                # Sleep until next analysis cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - elapsed)
                
                if sleep_time > 0:
                    self._shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {str(e)}")
                time.sleep(5)
        
        logger.info("Analysis loop stopped")
    
    def _analyze_symbol(self, symbol: str):
        """Analyze a single symbol for trading signals"""
        try:
            # Get market data
            data = self.market_data.get_historical_data(symbol, self.trading_timeframe, 100)
            
            if data.empty:
                logger.debug(f"No data available for {symbol}")
                return
            
            # Generate signal
            signal = self.strategy.generate_signal(data, symbol, self.trading_timeframe)
            
            if signal:
                self.signals_generated += 1
                logger.info(f"Signal generated for {symbol}: {signal.signal.name} "
                          f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})")
                
                # Execute trade if trading is enabled
                if self.is_trading:
                    self._execute_signal(signal)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
    
    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            if signal.signal == SignalType.HOLD:
                return
            
            # Determine order type
            order_type = OrderType.BUY if signal.signal == SignalType.BUY else OrderType.SELL
            
            # Calculate position size (can be enhanced with more sophisticated sizing)
            volume = None  # Use default lot size from execution engine
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(signal)
            
            # Place order
            order_id = self.execution_engine.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                volume=volume,
                price=signal.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"SMA/MACD signal (S:{signal.strength:.2f}, C:{signal.confidence:.2f})"
            )
            
            if order_id:
                self.trades_executed += 1
                logger.info(f"Trade executed: {order_id} for {signal.symbol}")
            else:
                logger.warning(f"Failed to execute trade for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
    
    def _calculate_risk_levels(self, signal) -> tuple:
        """Calculate stop loss and take profit levels"""
        try:
            # Simple risk calculation - can be enhanced
            price = signal.price
            
            # Default risk parameters
            risk_percent = 0.02  # 2% risk
            reward_ratio = 2.0   # 1:2 risk/reward
            
            if signal.signal == SignalType.BUY:
                stop_loss = price * (1 - risk_percent)
                take_profit = price * (1 + risk_percent * reward_ratio)
            else:  # SELL
                stop_loss = price * (1 + risk_percent)
                take_profit = price * (1 - risk_percent * reward_ratio)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {str(e)}")
            return None, None
    
    def _monitoring_loop(self):
        """Monitor system health and performance"""
        logger.info("Monitoring loop started")
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Log system status periodically
                if self.signals_generated % 10 == 0 and self.signals_generated > 0:
                    self._log_system_status()
                
                # Check system health
                self._check_system_health()
                
                # Sleep
                self._shutdown_event.wait(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)
        
        logger.info("Monitoring loop stopped")
    
    def _log_system_status(self):
        """Log current system status"""
        try:
            uptime = datetime.now() - self.system_start_time if self.system_start_time else timedelta(0)
            
            status = {
                'mode': self.mode,
                'uptime': str(uptime).split('.')[0],  # Remove microseconds
                'analyzing': self.is_analyzing,
                'trading': self.is_trading,
                'signals_generated': self.signals_generated,
                'trades_executed': self.trades_executed,
                'last_analysis': self.last_analysis_time.strftime('%H:%M:%S') if self.last_analysis_time else 'Never'
            }
            
            logger.info(f"System Status: {status}")
            
        except Exception as e:
            logger.error(f"Error logging system status: {str(e)}")
    
    def _check_system_health(self):
        """Check system health and connectivity"""
        try:
            # Check market data connection
            if not self.market_data.is_connected():
                logger.warning("Market data connection lost!")
            
            # Check execution engine connection
            if not self.execution_engine.is_connected:
                logger.warning("Execution engine connection lost!")
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
    
    def launch_dashboard(self, host: str = 'localhost', port: int = 8501):
        """Launch the Streamlit dashboard"""
        try:
            dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'streamlit_app.py')
            
            if not os.path.exists(dashboard_path):
                logger.error("Dashboard not found. Creating placeholder...")
                self._create_placeholder_dashboard()
                return False
            
            # Launch Streamlit
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', 
                dashboard_path,
                '--server.address', host,
                '--server.port', str(port),
                '--server.headless', 'true'
            ]
            
            logger.info(f"Launching dashboard at http://{host}:{port}")
            self.dashboard_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(3)
            
            if self.dashboard_process.poll() is None:
                logger.info(f"Dashboard launched successfully at http://{host}:{port}")
                return True
            else:
                logger.error("Dashboard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error launching dashboard: {str(e)}")
            return False
    
    def _create_placeholder_dashboard(self):
        """Create a placeholder dashboard if it doesn't exist"""
        try:
            dashboard_dir = os.path.join(os.path.dirname(__file__), '..', 'dashboard')
            os.makedirs(dashboard_dir, exist_ok=True)
            
            dashboard_path = os.path.join(dashboard_dir, 'streamlit_app.py')
            
            placeholder_content = '''
import streamlit as st
import time
from datetime import datetime

st.set_page_config(page_title="AI Trading System V2", layout="wide")

st.title("🚀 AI Trading System V2 Dashboard")
st.write("Dashboard is being developed. This is a placeholder.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "Running", "✅")

with col2:
    st.metric("Signals Generated", "0", "📊")

with col3:
    st.metric("Active Trades", "0", "💼")

st.write("---")
st.write("Full dashboard implementation coming soon!")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
'''
            
            with open(dashboard_path, 'w') as f:
                f.write(placeholder_content)
            
            logger.info("Placeholder dashboard created")
            
        except Exception as e:
            logger.error(f"Error creating placeholder dashboard: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = datetime.now() - self.system_start_time if self.system_start_time else timedelta(0)
            
            status = {
                'mode': self.mode,
                'running': self.is_running,
                'analyzing': self.is_analyzing,
                'trading': self.is_trading,
                'uptime_seconds': uptime.total_seconds(),
                'uptime_str': str(uptime).split('.')[0],
                'signals_generated': self.signals_generated,
                'trades_executed': self.trades_executed,
                'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'trading_symbols': self.trading_symbols,
                'trading_timeframe': self.trading_timeframe,
                'analysis_interval': self.analysis_interval
            }
            
            # Add component status
            if self.market_data:
                status['market_data'] = self.market_data.get_connection_status()
            
            if self.execution_engine:
                status['execution'] = self.execution_engine.get_account_summary()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}

def main():
    """Main entry point for the trading system"""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Trading System V2')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                        help='Trading mode (default: demo)')
    parser.add_argument('--action', choices=['analyze', 'trade', 'dashboard'], default='analyze',
                        help='Action to perform (default: analyze)')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create trading system
    system = TradingSystem(mode=args.mode, config=config)
    
    try:
        logger.info(f"Starting AI Trading System V2 in {args.mode} mode")
        
        if args.action == 'analyze':
            # Start analysis only
            if system.start_analysis():
                logger.info("Analysis started. Press Ctrl+C to stop.")
                try:
                    while system.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
            else:
                logger.error("Failed to start analysis")
                
        elif args.action == 'trade':
            # Start full trading
            if system.start_trading():
                logger.info("Trading started. Press Ctrl+C to stop.")
                try:
                    while system.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
            else:
                logger.error("Failed to start trading")
                
        elif args.action == 'dashboard':
            # Launch dashboard only
            if system.launch_dashboard():
                logger.info("Dashboard launched. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
            else:
                logger.error("Failed to launch dashboard")
    
    finally:
        system.stop()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()