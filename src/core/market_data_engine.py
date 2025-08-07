"""
Market Data Engine - Simplified interface to market data
Wraps the existing data_pipeline for easier use by the trading system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import time
from datetime import datetime, timedelta

# Import existing data pipeline
from .data_pipeline import DataPipeline, LOTES

logger = logging.getLogger(__name__)

class MarketDataEngine:
    """
    Simplified market data interface
    Wraps the existing DataPipeline for easier integration with TradingSystem
    """
    
    def __init__(self):
        """Initialize the market data engine"""
        self.data_pipeline = DataPipeline()
        self.symbols = list(LOTES.keys())
        self.timeframes = ['M1', 'M5', 'M15']
        self.last_update = None
        
        logger.info(f"Market Data Engine initialized with {len(self.symbols)} symbols")
    
    def get_latest_data(self, symbol: Optional[str] = None, timeframe: str = 'M1') -> Optional[Dict[str, Any]]:
        """
        Get latest market data for symbol(s)
        
        Args:
            symbol: Specific symbol or None for all symbols
            timeframe: Timeframe (M1, M5, M15, etc.)
            
        Returns:
            Dictionary with market data or None if no data available
        """
        try:
            if symbol:
                symbols_to_fetch = [symbol]
            else:
                symbols_to_fetch = self.symbols[:3]  # Limit to first 3 for demo
            
            data = {}
            
            for sym in symbols_to_fetch:
                # Get data from pipeline
                df = self.data_pipeline.get_latest_data(
                    symbol=sym,
                    timeframe=timeframe,
                    count=100  # Get last 100 bars
                )
                
                if df is not None and not df.empty:
                    data[sym] = {
                        'symbol': sym,
                        'timeframe': timeframe,
                        'data': df,
                        'latest_price': df['close'].iloc[-1] if 'close' in df.columns else None,
                        'timestamp': datetime.now()
                    }
            
            self.last_update = datetime.now()
            return data if data else None
            
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = 'M1', 
                          count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get historical data for backtesting
        
        Args:
            symbol: Symbol to fetch
            timeframe: Timeframe
            count: Number of bars
            
        Returns:
            DataFrame with historical data or None
        """
        try:
            return self.data_pipeline.get_latest_data(
                symbol=symbol,
                timeframe=timeframe,
                count=count
            )
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if market is open (simplified check)
        """
        # For synthetic indices, market is always open
        return True
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return self.symbols.copy()
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information
        
        Args:
            symbol: Symbol to get info for
            
        Returns:
            Dictionary with symbol info
        """
        if symbol not in LOTES:
            return None
        
        return {
            'symbol': symbol,
            'lot_size': LOTES[symbol],
            'digits': 5,  # Assumed for synthetic indices
            'point': 0.00001,
            'trade_allowed': True
        }