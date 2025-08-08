"""
Main Trading System - Core entry point for the AI Trading System V2
Manages the overall system state, connects to MT5, and coordinates all components
"""

import sys
import os
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Optional imports with graceful degradation
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

try:
    import streamlit
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    streamlit = None

# Core imports
from src.core.data_pipeline import DataPipeline
from src.strategies.multi_strategy_engine import MultiStrategyEngine
from src.core.market_data_engine import MarketDataEngine
from src.core.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)

class TradingSystemState:
    """System state management"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ANALYSIS_ONLY = "analysis_only"
    ERROR = "error"

class TradingSystem:
    """
    Main Trading System Class
    
    Coordinates all components of the AI trading system:
    - Market data collection
    - Strategy execution
    - Signal generation
    - Order execution (in live mode)
    - Dashboard interface
    """
    
    def __init__(self, mode: str = 'demo', config_path: Optional[str] = None):
        """
        Initialize the trading system
        
        Args:
            mode: 'demo' or 'live' - determines if orders are executed
            config_path: Path to configuration file
        """
        self.mode = mode
        self.state = TradingSystemState.STOPPED
        self.config_path = config_path or "config/asset_specific_strategies.yaml"
        
        # Initialize logging
        self._setup_logging()
        
        # System components
        self.market_data_engine = None
        self.strategy_engine = None
        self.execution_engine = None
        self.data_pipeline = None
        
        # Control flags
        self._running = False
        self._analysis_thread = None
        self._trading_thread = None
        
        logger.info(f"Trading System initialized in {mode} mode")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize market data engine
            self.market_data_engine = MarketDataEngine()
            
            # Initialize data pipeline
            self.data_pipeline = DataPipeline()
            
            # Initialize strategy engine
            self.strategy_engine = MultiStrategyEngine()
            
            # Initialize execution engine (only in live mode)
            if self.mode == 'live':
                self.execution_engine = ExecutionEngine(mode='live')
            else:
                self.execution_engine = ExecutionEngine(mode='demo')
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            self.state = TradingSystemState.ERROR
            return False
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available - running in simulation mode")
            return True
        
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Check connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("Failed to get terminal info")
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            logger.info(f"Connected to MT5 - Account: {account_info.login}")
            logger.info(f"Balance: {account_info.balance}, Equity: {account_info.equity}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection failed: {str(e)}")
            return False
    
    def start_analysis(self):
        """Start real-time market analysis without trading"""
        logger.info("Starting market analysis...")
        
        self.state = TradingSystemState.STARTING
        
        # Initialize components
        if not self.initialize_components():
            return False
        
        # Connect to MT5
        if not self.connect_mt5():
            logger.warning("Continuing without MT5 connection")
        
        self.state = TradingSystemState.ANALYSIS_ONLY
        self._running = True
        
        # Start analysis thread
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        
        logger.info("Market analysis started successfully")
        return True
    
    def start_trading(self):
        """Start full trading mode (analysis + execution)"""
        logger.info(f"Starting trading in {self.mode} mode...")
        
        # First start analysis
        if not self.start_analysis():
            return False
        
        # Change state to full trading
        self.state = TradingSystemState.RUNNING
        
        # Start trading thread (only in live mode)
        if self.mode == 'live':
            self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self._trading_thread.start()
            logger.info("Live trading started")
        else:
            logger.info("Demo mode - signals generated but no orders executed")
        
        return True
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        
        self._running = False
        self.state = TradingSystemState.STOPPED
        
        # Wait for threads to finish
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5)
        
        if self._trading_thread and self._trading_thread.is_alive():
            self._trading_thread.join(timeout=5)
        
        # Shutdown MT5
        if MT5_AVAILABLE and mt5:
            mt5.shutdown()
        
        logger.info("Trading system stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        logger.info("Analysis loop started")
        
        while self._running:
            try:
                # Get market data
                market_data = self.market_data_engine.get_latest_data()
                
                if market_data:
                    # Generate signals
                    signals = self.strategy_engine.generate_signals(market_data)
                    
                    if signals:
                        for signal in signals:
                            logger.info(f"Signal generated: {signal}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _trading_loop(self):
        """Main trading loop (for live mode)"""
        logger.info("Trading loop started")
        
        while self._running and self.mode == 'live':
            try:
                # Execute pending orders
                if self.execution_engine:
                    self.execution_engine.process_pending_orders()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def launch_dashboard(self):
        """Launch the web dashboard"""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit not available - cannot launch dashboard")
            return False
        
        try:
            import subprocess
            import threading
            
            def run_dashboard():
                dashboard_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 
                    'dashboard', 
                    'streamlit_app.py'
                )
                subprocess.run([
                    'streamlit', 'run', dashboard_path, 
                    '--server.port=8501',
                    '--server.headless=true'
                ])
            
            dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            dashboard_thread.start()
            
            logger.info("Dashboard launching at http://localhost:8501")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'state': self.state,
            'mode': self.mode,
            'mt5_connected': MT5_AVAILABLE and mt5 is not None,
            'running': self._running,
            'components_initialized': all([
                self.market_data_engine is not None,
                self.strategy_engine is not None,
                self.execution_engine is not None
            ])
        }

def main():
    """Main entry point for the trading system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Trading System V2')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                       help='Trading mode (default: demo)')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run analysis only without trading')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch dashboard')
    
    args = parser.parse_args()
    
    # Create trading system
    system = TradingSystem(mode=args.mode)
    
    try:
        if args.analysis_only:
            # Start analysis only
            system.start_analysis()
        else:
            # Start full trading
            system.start_trading()
        
        if args.dashboard:
            # Launch dashboard
            system.launch_dashboard()
        
        # Keep running
        print(f"Trading System running in {args.mode} mode...")
        print("Press Ctrl+C to stop")
        
        while True:
            status = system.get_status()
            print(f"Status: {status['state']} | MT5: {status['mt5_connected']} | Running: {status['running']}")
            time.sleep(30)  # Status update every 30 seconds
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        system.stop()

if __name__ == "__main__":
    main()