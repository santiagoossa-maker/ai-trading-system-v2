"""
Market Data Engine
Simplified wrapper around the data pipeline for easy market data access
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

# Import the comprehensive data pipeline
try:
    from .data_pipeline import DataPipeline, TickData, BarData
except ImportError:
    # Fallback for direct execution
    from data_pipeline import DataPipeline, TickData, BarData

logger = logging.getLogger(__name__)

class MarketDataEngine:
    """
    Simplified market data engine that provides easy access to MT5 data
    through the comprehensive data pipeline
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the market data engine"""
        self.data_pipeline = DataPipeline(config_path)
        self.is_running = False
        
        # Supported symbols
        self.symbols = [
            "R_75", "R_100", "R_50", "R_25", "R_10",  # Volatility indices
            "1HZ75V", "1HZ100V", "1HZ50V", "1HZ10V", "1HZ25V",  # HZ indices
            "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5"  # Step indices
        ]
        
        # Supported timeframes
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
    def connect(self) -> bool:
        """Connect to MT5 and start data collection"""
        try:
            if self.data_pipeline.start():
                self.is_running = True
                logger.info("Market Data Engine connected successfully")
                return True
            else:
                logger.error("Failed to connect Market Data Engine")
                return False
        except Exception as e:
            logger.error(f"Error connecting Market Data Engine: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5 and stop data collection"""
        try:
            if self.is_running:
                self.data_pipeline.stop()
                self.is_running = False
                logger.info("Market Data Engine disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting Market Data Engine: {str(e)}")
    
    def get_historical_data(self, symbol: str, timeframe: str = 'M5', count: int = 100) -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'R_75')
            timeframe: Timeframe ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1')
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if not self.is_running:
                logger.warning("Market Data Engine not connected")
                return pd.DataFrame()
            
            if symbol not in self.symbols:
                logger.warning(f"Unsupported symbol: {symbol}")
                return pd.DataFrame()
            
            if timeframe not in self.timeframes:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return pd.DataFrame()
            
            df = self.data_pipeline.get_latest_data(symbol, timeframe, count)
            
            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in data for {symbol}")
                return pd.DataFrame()
            
            logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            if not self.is_running:
                return None
            
            tick = self.data_pipeline.get_latest_tick(symbol)
            if tick:
                return tick.last if tick.last > 0 else (tick.bid + tick.ask) / 2
            
            # Fallback to latest close price
            df = self.get_historical_data(symbol, 'M1', 1)
            if not df.empty:
                return df['close'].iloc[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_bid_ask(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get bid/ask prices for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'bid' and 'ask' prices or None
        """
        try:
            if not self.is_running:
                return None
            
            tick = self.data_pipeline.get_latest_tick(symbol)
            if tick and tick.bid > 0 and tick.ask > 0:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': tick.ask - tick.bid,
                    'timestamp': tick.timestamp
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting bid/ask for {symbol}: {str(e)}")
            return None
    
    def get_multiple_symbols_data(self, symbols: List[str], timeframe: str = 'M5', count: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for all symbols
            count: Number of bars for each symbol
            
        Returns:
            Dict mapping symbols to their DataFrames
        """
        data = {}
        
        for symbol in symbols:
            if symbol in self.symbols:
                df = self.get_historical_data(symbol, timeframe, count)
                if not df.empty:
                    data[symbol] = df
            else:
                logger.warning(f"Skipping unsupported symbol: {symbol}")
        
        return data
    
    def is_connected(self) -> bool:
        """Check if the engine is connected and running"""
        return self.is_running and self.data_pipeline.running
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        if not self.is_running:
            return {
                'connected': False,
                'mt5_connected': False,
                'redis_connected': False,
                'active_threads': 0,
                'data_available': False
            }
        
        status = self.data_pipeline.get_system_status()
        
        # Test data availability
        data_available = False
        try:
            test_df = self.get_historical_data('R_75', 'M5', 10)
            data_available = not test_df.empty
        except:
            pass
        
        return {
            'connected': self.is_running,
            'mt5_connected': status.get('mt5_connected', False),
            'redis_connected': status.get('redis_connected', False),
            'active_threads': status.get('active_threads', 0),
            'data_available': data_available,
            'symbols_count': len(self.symbols),
            'timeframes_count': len(self.timeframes)
        }
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return self.symbols.copy()
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return self.timeframes.copy()
    
    def validate_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Validate if symbol and timeframe are supported"""
        return symbol in self.symbols and timeframe in self.timeframes

# Convenience functions for quick access
def create_market_engine(config_path: Optional[str] = None) -> MarketDataEngine:
    """Create and return a market data engine instance"""
    return MarketDataEngine(config_path)

def get_quick_data(symbol: str, timeframe: str = 'M5', count: int = 100, 
                   config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Quick function to get data without managing engine lifecycle
    
    Warning: Creates a new connection each time. Use MarketDataEngine directly 
    for better performance when making multiple calls.
    """
    engine = MarketDataEngine(config_path)
    try:
        if engine.connect():
            return engine.get_historical_data(symbol, timeframe, count)
        else:
            logger.error("Failed to connect to get quick data")
            return pd.DataFrame()
    finally:
        engine.disconnect()

if __name__ == "__main__":
    # Example usage
    import time
    
    logger.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = MarketDataEngine()
    
    try:
        # Connect
        if engine.connect():
            print("✓ Connected to Market Data Engine")
            
            # Wait a moment for data to load
            time.sleep(3)
            
            # Test data retrieval
            symbols_to_test = ['R_75', 'R_100', 'R_50']
            
            for symbol in symbols_to_test:
                print(f"\nTesting {symbol}:")
                
                # Get historical data
                df = engine.get_historical_data(symbol, 'M5', 20)
                if not df.empty:
                    print(f"  ✓ Historical data: {len(df)} bars")
                    print(f"  ✓ Latest close: {df['close'].iloc[-1]:.5f}")
                else:
                    print(f"  ✗ No historical data available")
                
                # Get current price
                current_price = engine.get_current_price(symbol)
                if current_price:
                    print(f"  ✓ Current price: {current_price:.5f}")
                else:
                    print(f"  ✗ No current price available")
                
                # Get bid/ask
                bid_ask = engine.get_bid_ask(symbol)
                if bid_ask:
                    print(f"  ✓ Bid: {bid_ask['bid']:.5f}, Ask: {bid_ask['ask']:.5f}")
                    print(f"  ✓ Spread: {bid_ask['spread']:.5f}")
                else:
                    print(f"  ✗ No bid/ask data available")
            
            # Connection status
            status = engine.get_connection_status()
            print(f"\nConnection Status: {status}")
            
        else:
            print("✗ Failed to connect to Market Data Engine")
    
    finally:
        engine.disconnect()
        print("\n✓ Disconnected from Market Data Engine")