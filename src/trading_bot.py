"""
Automated Trading Bot - Main Trading System
Orchestrates all components for automated trading
"""

import logging
import time
import threading
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.data_pipeline import DataPipeline
from core.mt5_execution_engine import MT5ExecutionEngine
from strategies.sma_macd_live_strategy import SMA_MACD_Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """
    Main automated trading bot that orchestrates all components
    """
    
    def __init__(self, config_path: Optional[str] = None, demo_mode: bool = True):
        self.config_path = config_path
        self.demo_mode = demo_mode
        self.running = False
        
        # Components
        self.data_pipeline = None
        self.execution_engine = None
        self.strategy = None
        
        # Configuration
        self.config = self._load_config()
        
        # Performance tracking
        self.start_time = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.initial_balance = 0.0
        
        # Risk management
        self.daily_loss_limit = self.config.get('risk_management', {}).get('daily_loss_limit', 0.05)
        self.max_positions = self.config.get('risk_management', {}).get('max_positions', 10)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'demo_mode': True,
            'symbols': ['R_75', 'R_100', 'R_50'],
            'timeframe': 'M5',
            'update_interval': 5,
            'risk_management': {
                'risk_per_trade': 0.02,
                'daily_loss_limit': 0.05,
                'max_positions': 10,
                'max_spread': 20
            },
            'strategy': {
                'sma_fast': 8,
                'sma_slow': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'risk_reward_ratio': 2.0,
                'atr_multiplier': 2.0
            },
            'mt5': {
                'login': None,
                'password': None,
                'server': None,
                'path': None
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_bot.log'
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def initialize_components(self) -> bool:
        """Initialize all trading components"""
        try:
            logger.info("Initializing trading components...")
            
            # Initialize data pipeline
            self.data_pipeline = DataPipeline()
            if not self.data_pipeline.start():
                logger.error("Failed to start data pipeline")
                return False
            
            logger.info("Data pipeline started successfully")
            
            # Initialize execution engine (skip in demo mode)
            if not self.demo_mode:
                self.execution_engine = MT5ExecutionEngine(self.config_path)
                if not self.execution_engine.connect():
                    logger.error("Failed to connect to MT5")
                    return False
                
                logger.info("MT5 execution engine connected successfully")
                
                # Get initial balance
                account_info = self.execution_engine.get_account_info()
                self.initial_balance = account_info.get('balance', 0)
                logger.info(f"Initial account balance: ${self.initial_balance:,.2f}")
            else:
                logger.info("Running in demo mode - MT5 connection skipped")
                self.initial_balance = 10000.0  # Demo balance
            
            # Initialize strategy
            if not self.demo_mode:
                self.strategy = SMA_MACD_Strategy(
                    self.data_pipeline, 
                    self.execution_engine,
                    self.config.get('strategy', {})
                )
            else:
                logger.info("Demo mode - strategy initialization skipped")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def start(self) -> bool:
        """Start the trading bot"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 STARTING AI TRADING SYSTEM V2")
            logger.info("=" * 60)
            
            # Initialize components
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            self.running = True
            self.start_time = datetime.now()
            
            logger.info(f"Trading bot started at {self.start_time}")
            logger.info(f"Mode: {'DEMO' if self.demo_mode else 'LIVE TRADING'}")
            logger.info(f"Symbols: {self.config.get('symbols', [])}")
            logger.info(f"Timeframe: {self.config.get('timeframe', 'M5')}")
            logger.info(f"Update interval: {self.config.get('update_interval', 5)} seconds")
            
            # Start strategy if not in demo mode
            if not self.demo_mode and self.strategy:
                self.strategy.start_trading()
                logger.info("Strategy started - now trading live!")
            
            # Start monitoring loop
            self._start_monitoring()
            
            # Main trading loop
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            return False
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        
        self.running = False
        
        # Stop strategy
        if self.strategy:
            self.strategy.stop_trading()
            logger.info("Strategy stopped")
        
        # Disconnect execution engine
        if self.execution_engine:
            self.execution_engine.disconnect()
            logger.info("Execution engine disconnected")
        
        # Stop data pipeline
        if self.data_pipeline:
            self.data_pipeline.stop()
            logger.info("Data pipeline stopped")
        
        # Print final statistics
        self._print_final_stats()
        
        logger.info("Trading bot stopped successfully")
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        def monitor():
            while self.running:
                try:
                    self._check_risk_limits()
                    self._update_performance_metrics()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Monitoring thread started")
    
    def _check_risk_limits(self):
        """Check risk management limits"""
        try:
            if self.demo_mode or not self.execution_engine:
                return
            
            account_info = self.execution_engine.get_account_info()
            current_balance = account_info.get('balance', 0)
            
            # Check daily loss limit
            daily_loss = (self.initial_balance - current_balance) / self.initial_balance
            if daily_loss > self.daily_loss_limit:
                logger.warning(f"Daily loss limit exceeded: {daily_loss:.2%}")
                self._emergency_shutdown("Daily loss limit exceeded")
                return
            
            # Check margin level
            margin_level = account_info.get('margin_level', 1000)
            if margin_level < 200:  # 200% margin level minimum
                logger.warning(f"Low margin level: {margin_level:.0f}%")
                if margin_level < 100:
                    self._emergency_shutdown("Critical margin level")
                    return
            
            # Check maximum positions
            positions = self.execution_engine.get_positions_summary()
            if positions['total_positions'] > self.max_positions:
                logger.warning(f"Too many positions: {positions['total_positions']}")
                # Close oldest positions if needed
                
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _emergency_shutdown(self, reason: str):
        """Emergency shutdown procedure"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        try:
            # Close all positions
            if self.execution_engine:
                results = self.execution_engine.close_all_positions()
                logger.info(f"Emergency close: {len(results)} positions closed")
            
            # Stop trading
            self.stop()
            
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            if self.demo_mode:
                return
            
            if not self.execution_engine:
                return
            
            # Get current account info
            account_info = self.execution_engine.get_account_info()
            current_balance = account_info.get('balance', 0)
            current_equity = account_info.get('equity', 0)
            
            # Calculate total profit/loss
            self.total_profit = current_balance - self.initial_balance
            
            # Calculate drawdown
            drawdown = (self.initial_balance - current_equity) / self.initial_balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # Get trading stats
            if self.strategy:
                strategy_status = self.strategy.get_strategy_status()
                self.total_trades = strategy_status.get('trades_today', 0)
                self.winning_trades = strategy_status.get('winning_trades', 0)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        logger.info("Entering main trading loop...")
        
        last_status_time = datetime.now()
        status_interval = timedelta(minutes=15)  # Print status every 15 minutes
        
        while self.running:
            try:
                # Print periodic status
                if datetime.now() - last_status_time >= status_interval:
                    self._print_status()
                    last_status_time = datetime.now()
                
                # In demo mode, just sleep
                if self.demo_mode:
                    time.sleep(self.config.get('update_interval', 5))
                    continue
                
                # Check if strategy is running
                if self.strategy and not self.strategy.running:
                    logger.warning("Strategy stopped unexpectedly, restarting...")
                    self.strategy.start_trading()
                
                # Wait for next iteration
                time.sleep(self.config.get('update_interval', 5))
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _print_status(self):
        """Print current status"""
        try:
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            logger.info("-" * 50)
            logger.info("📊 TRADING BOT STATUS")
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
            
            if not self.demo_mode and self.execution_engine:
                account_info = self.execution_engine.get_account_info()
                positions = self.execution_engine.get_positions_summary()
                
                logger.info(f"Balance: ${account_info.get('balance', 0):,.2f}")
                logger.info(f"Equity: ${account_info.get('equity', 0):,.2f}")
                logger.info(f"Profit: ${self.total_profit:,.2f}")
                logger.info(f"Active Positions: {positions['total_positions']}")
                logger.info(f"Total Trades: {self.total_trades}")
                
                if self.total_trades > 0:
                    win_rate = (self.winning_trades / self.total_trades) * 100
                    logger.info(f"Win Rate: {win_rate:.1f}%")
                
                logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
            else:
                logger.info("Demo mode - no real trading data")
            
            if self.strategy:
                strategy_status = self.strategy.get_strategy_status()
                logger.info(f"Strategy Running: {strategy_status.get('running', False)}")
            
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Error printing status: {e}")
    
    def _print_final_stats(self):
        """Print final trading statistics"""
        try:
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            logger.info("=" * 60)
            logger.info("📈 FINAL TRADING STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total Runtime: {runtime}")
            logger.info(f"Trading Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
            
            if not self.demo_mode:
                logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
                logger.info(f"Final Profit/Loss: ${self.total_profit:,.2f}")
                logger.info(f"Total Trades: {self.total_trades}")
                logger.info(f"Winning Trades: {self.winning_trades}")
                
                if self.total_trades > 0:
                    win_rate = (self.winning_trades / self.total_trades) * 100
                    logger.info(f"Win Rate: {win_rate:.1f}%")
                    
                    profit_per_trade = self.total_profit / self.total_trades
                    logger.info(f"Average Profit per Trade: ${profit_per_trade:.2f}")
                
                logger.info(f"Maximum Drawdown: {self.max_drawdown:.2%}")
                
                if self.total_profit > 0:
                    roi = (self.total_profit / self.initial_balance) * 100
                    logger.info(f"Return on Investment: {roi:.2f}%")
            else:
                logger.info("Demo mode completed successfully")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error printing final stats: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        try:
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            status = {
                'running': self.running,
                'demo_mode': self.demo_mode,
                'runtime': str(runtime),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_profit': self.total_profit,
                'max_drawdown': self.max_drawdown,
                'initial_balance': self.initial_balance
            }
            
            if not self.demo_mode and self.execution_engine:
                account_info = self.execution_engine.get_account_info()
                positions = self.execution_engine.get_positions_summary()
                
                status.update({
                    'current_balance': account_info.get('balance', 0),
                    'current_equity': account_info.get('equity', 0),
                    'active_positions': positions.get('total_positions', 0),
                    'margin_level': account_info.get('margin_level', 0)
                })
            
            if self.strategy:
                strategy_status = self.strategy.get_strategy_status()
                status['strategy_running'] = strategy_status.get('running', False)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Trading System V2 - Automated Trading Bot')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--demo', action='store_true', default=True, help='Run in demo mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    
    args = parser.parse_args()
    
    # Determine mode
    demo_mode = args.demo and not args.live
    
    if not demo_mode:
        # Confirm live trading mode
        response = input("⚠️  WARNING: You are about to start LIVE TRADING with real money. Type 'CONFIRM' to proceed: ")
        if response != 'CONFIRM':
            print("Live trading cancelled. Starting in demo mode.")
            demo_mode = True
    
    # Create and start bot
    bot = TradingBot(config_path=args.config, demo_mode=demo_mode)
    
    try:
        if bot.start():
            logger.info("Trading bot started successfully")
            # Keep running until interrupted
            while bot.running:
                time.sleep(1)
        else:
            logger.error("Failed to start trading bot")
            return 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        bot.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())