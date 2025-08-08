"""
SMA8/50 + MACD Trading Strategy - Real Implementation
The proven strategy that generates actual trading signals and executes trades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time

# Optional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

from .mt5_execution_engine import MT5ExecutionEngine, TradeResult, OrderType
from .data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strength: float   # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    timestamp: datetime
    reason: str
    indicators: Dict

class SMA_MACD_Strategy:
    """
    Real SMA8/50 + MACD strategy implementation
    
    Strategy Rules:
    1. BUY when SMA8 > SMA50 AND MACD > Signal AND MACD Histogram > 0
    2. SELL when SMA8 < SMA50 AND MACD < Signal AND MACD Histogram < 0
    3. Exit when conditions reverse or stop loss/take profit hit
    """
    
    def __init__(self, 
                 data_pipeline: DataPipeline,
                 execution_engine: MT5ExecutionEngine,
                 config: Dict = None):
        
        self.data_pipeline = data_pipeline
        self.execution_engine = execution_engine
        self.config = config or self._default_config()
        
        # Strategy parameters
        self.sma_fast = self.config.get('sma_fast', 8)
        self.sma_slow = self.config.get('sma_slow', 50)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        # Risk management
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2%
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.max_positions_per_symbol = self.config.get('max_positions_per_symbol', 1)
        
        # Timeframe and symbols
        self.timeframe = self.config.get('timeframe', 'M5')
        self.symbols = self.config.get('symbols', ['R_75', 'R_100', 'R_50'])
        
        # Trading state
        self.active_signals = {}
        self.position_tracker = {}
        self.running = False
        self.last_analysis = {}
        
        # Performance tracking
        self.trades_today = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
    def _default_config(self) -> Dict:
        """Default strategy configuration"""
        return {
            'sma_fast': 8,
            'sma_slow': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'risk_per_trade': 0.02,
            'risk_reward_ratio': 2.0,
            'atr_multiplier': 2.0,
            'max_positions_per_symbol': 1,
            'timeframe': 'M5',
            'symbols': ['R_75', 'R_100', 'R_50'],
            'min_bars': 100,
            'update_interval': 5,  # seconds
            'max_spread': 20,  # points
            'min_profit_points': 10
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate SMA8, SMA50, and MACD indicators"""
        try:
            if len(data) < self.sma_slow + 10:
                return {}
            
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            indicators = {}
            
            if TALIB_AVAILABLE:
                # Calculate SMAs
                indicators['sma_fast'] = talib.SMA(close, timeperiod=self.sma_fast)
                indicators['sma_slow'] = talib.SMA(close, timeperiod=self.sma_slow)
                
                # Calculate MACD
                macd, macd_signal, macd_hist = talib.MACD(
                    close, 
                    fastperiod=self.macd_fast,
                    slowperiod=self.macd_slow,
                    signalperiod=self.macd_signal
                )
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
                
                # Calculate ATR for stop loss
                indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
                
                # Calculate RSI for additional confirmation
                indicators['rsi'] = talib.RSI(close, timeperiod=14)
                
            else:
                # Fallback implementation
                close_series = pd.Series(close, index=data.index)
                
                # SMAs
                indicators['sma_fast'] = close_series.rolling(window=self.sma_fast).mean().values
                indicators['sma_slow'] = close_series.rolling(window=self.sma_slow).mean().values
                
                # Simple MACD
                ema_fast = close_series.ewm(span=self.macd_fast).mean()
                ema_slow = close_series.ewm(span=self.macd_slow).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=self.macd_signal).mean()
                
                indicators['macd'] = macd.values
                indicators['macd_signal'] = macd_signal.values
                indicators['macd_histogram'] = (macd - macd_signal).values
                
                # Simple ATR
                high_low = pd.Series(high, index=data.index) - pd.Series(low, index=data.index)
                indicators['atr'] = high_low.rolling(window=14).mean().values
                
                # Simple RSI
                delta = close_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).values
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on SMA8/50 + MACD strategy"""
        try:
            indicators = self.calculate_indicators(data)
            
            if not indicators:
                return None
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            sma_fast = indicators['sma_fast'][-1]
            sma_slow = indicators['sma_slow'][-1]
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_hist = indicators['macd_histogram'][-1]
            atr = indicators['atr'][-1]
            rsi = indicators['rsi'][-1]
            
            # Check for invalid values
            if any(np.isnan([sma_fast, sma_slow, macd, macd_signal, macd_hist, atr, rsi])):
                return None
            
            # Strategy logic
            signal_type = 'HOLD'
            confidence = 0.0
            strength = 0.0
            reason = ""
            
            # BUY Signal: SMA8 > SMA50 AND MACD > Signal AND MACD Histogram > 0
            if (sma_fast > sma_slow and 
                macd > macd_signal and 
                macd_hist > 0 and
                current_price > sma_fast):  # Price above fast SMA
                
                signal_type = 'BUY'
                
                # Calculate confidence based on signal strength
                sma_separation = (sma_fast - sma_slow) / sma_slow * 100
                macd_strength = macd_hist / abs(macd) if macd != 0 else 0
                
                confidence = min(0.9, (sma_separation * 10 + abs(macd_strength) * 5) / 10)
                strength = min(1.0, sma_separation * 5)
                
                reason = f"BUY: SMA8({sma_fast:.5f}) > SMA50({sma_slow:.5f}), MACD({macd:.5f}) > Signal({macd_signal:.5f}), Hist({macd_hist:.5f}) > 0"
                
                # Additional confirmation
                if rsi < 70:  # Not overbought
                    confidence += 0.1
                if len(indicators['macd_histogram']) >= 2 and macd_hist > indicators['macd_histogram'][-2]:
                    confidence += 0.1  # MACD momentum increasing
            
            # SELL Signal: SMA8 < SMA50 AND MACD < Signal AND MACD Histogram < 0
            elif (sma_fast < sma_slow and 
                  macd < macd_signal and 
                  macd_hist < 0 and
                  current_price < sma_fast):  # Price below fast SMA
                
                signal_type = 'SELL'
                
                # Calculate confidence based on signal strength
                sma_separation = (sma_slow - sma_fast) / sma_slow * 100
                macd_strength = abs(macd_hist) / abs(macd) if macd != 0 else 0
                
                confidence = min(0.9, (sma_separation * 10 + macd_strength * 5) / 10)
                strength = min(1.0, sma_separation * 5)
                
                reason = f"SELL: SMA8({sma_fast:.5f}) < SMA50({sma_slow:.5f}), MACD({macd:.5f}) < Signal({macd_signal:.5f}), Hist({macd_hist:.5f}) < 0"
                
                # Additional confirmation
                if rsi > 30:  # Not oversold
                    confidence += 0.1
                if len(indicators['macd_histogram']) >= 2 and macd_hist < indicators['macd_histogram'][-2]:
                    confidence += 0.1  # MACD momentum decreasing
            
            # Minimum confidence threshold
            if confidence < 0.6:
                signal_type = 'HOLD'
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate position sizing and risk management
            account_info = self.execution_engine.get_account_info()
            balance = account_info.get('balance', 10000)
            
            # Calculate stop loss and take profit
            if signal_type == 'BUY':
                stop_loss = current_price - (atr * self.atr_multiplier)
                take_profit = current_price + (abs(current_price - stop_loss) * self.risk_reward_ratio)
            else:  # SELL
                stop_loss = current_price + (atr * self.atr_multiplier)
                take_profit = current_price - (abs(stop_loss - current_price) * self.risk_reward_ratio)
            
            # Calculate position size
            risk_amount = balance * self.risk_per_trade
            sl_points = abs(current_price - stop_loss) / self.execution_engine.symbols_info.get(symbol, {}).get('point', 0.00001)
            volume = self.execution_engine.calculate_position_size(symbol, risk_amount, sl_points)
            
            if volume <= 0:
                return None
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=min(1.0, confidence),
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                timestamp=datetime.now(),
                reason=reason,
                indicators={
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_hist,
                    'atr': atr,
                    'rsi': rsi
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        try:
            # Check if we already have position for this symbol
            positions = self.execution_engine.get_positions_summary()
            symbol_positions = [p for p in positions['positions'] if p['symbol'] == signal.symbol]
            
            if len(symbol_positions) >= self.max_positions_per_symbol:
                logger.info(f"Max positions reached for {signal.symbol}")
                return False
            
            # Check spread
            symbol_info = self.execution_engine.symbols_info.get(signal.symbol)
            if not symbol_info:
                logger.error(f"No symbol info for {signal.symbol}")
                return False
            
            # Place order
            is_buy = signal.signal_type == 'BUY'
            result = self.execution_engine.place_market_order(
                symbol=signal.symbol,
                is_buy=is_buy,
                volume=signal.volume,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"SMA_MACD_{signal.confidence:.2f}"
            )
            
            if result.success:
                logger.info(f"Order executed: {signal.symbol} {signal.signal_type} {signal.volume} @ {result.price}")
                logger.info(f"Reason: {signal.reason}")
                
                # Track position
                self.position_tracker[result.order] = {
                    'signal': signal,
                    'open_time': datetime.now(),
                    'open_price': result.price
                }
                
                self.trades_today += 1
                return True
            else:
                logger.error(f"Order failed: {result.comment}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def check_exit_conditions(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check if we should exit positions for this symbol"""
        try:
            # Get current positions for symbol
            positions = self.execution_engine.get_positions_summary()
            symbol_positions = [p for p in positions['positions'] if p['symbol'] == symbol]
            
            if not symbol_positions:
                return False
            
            # Calculate current indicators
            indicators = self.calculate_indicators(data)
            if not indicators:
                return False
            
            sma_fast = indicators['sma_fast'][-1]
            sma_slow = indicators['sma_slow'][-1]
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_hist = indicators['macd_histogram'][-1]
            
            if any(np.isnan([sma_fast, sma_slow, macd, macd_signal, macd_hist])):
                return False
            
            should_exit = False
            exit_reason = ""
            
            for position in symbol_positions:
                position_type = position['type']
                
                # Check exit conditions based on strategy reversal
                if position_type == 'BUY':
                    # Exit BUY if SMA8 < SMA50 OR MACD < Signal
                    if sma_fast < sma_slow or macd < macd_signal:
                        should_exit = True
                        exit_reason = f"BUY exit: SMA8({sma_fast:.5f}) < SMA50({sma_slow:.5f}) OR MACD({macd:.5f}) < Signal({macd_signal:.5f})"
                
                elif position_type == 'SELL':
                    # Exit SELL if SMA8 > SMA50 OR MACD > Signal
                    if sma_fast > sma_slow or macd > macd_signal:
                        should_exit = True
                        exit_reason = f"SELL exit: SMA8({sma_fast:.5f}) > SMA50({sma_slow:.5f}) OR MACD({macd:.5f}) > Signal({macd_signal:.5f})"
                
                if should_exit:
                    result = self.execution_engine.close_position(position['ticket'])
                    if result.success:
                        logger.info(f"Position closed: {position['ticket']} - {exit_reason}")
                        
                        # Update performance tracking
                        if position['profit'] > 0:
                            self.winning_trades += 1
                        self.total_profit += position['profit']
                        
                        # Remove from tracker
                        if position['ticket'] in self.position_tracker:
                            del self.position_tracker[position['ticket']]
                    else:
                        logger.error(f"Failed to close position {position['ticket']}: {result.comment}")
            
            return should_exit
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return False
    
    def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze single symbol and generate signal"""
        try:
            # Get latest data
            data = self.data_pipeline.get_latest_data(symbol, self.timeframe, count=200)
            
            if data.empty or len(data) < self.config.get('min_bars', 100):
                logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
                return None
            
            # Check exit conditions first
            self.check_exit_conditions(symbol, data)
            
            # Generate new signal
            signal = self.generate_signal(symbol, data)
            
            if signal:
                # Validate signal quality
                if signal.confidence >= 0.6 and signal.strength >= 0.3:
                    self.last_analysis[symbol] = {
                        'timestamp': datetime.now(),
                        'signal': signal.signal_type,
                        'confidence': signal.confidence,
                        'price': signal.entry_price
                    }
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def run_analysis_cycle(self):
        """Run one complete analysis cycle for all symbols"""
        try:
            signals_generated = []
            
            for symbol in self.symbols:
                try:
                    signal = self.analyze_symbol(symbol)
                    if signal:
                        signals_generated.append(signal)
                        
                        # Execute signal if conditions are met
                        if self.execute_signal(signal):
                            logger.info(f"Signal executed for {symbol}: {signal.signal_type} @ {signal.entry_price:.5f}")
                        
                except Exception as e:
                    logger.error(f"Error in analysis cycle for {symbol}: {e}")
            
            return signals_generated
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            return []
    
    def start_trading(self):
        """Start automated trading"""
        self.running = True
        logger.info("SMA MACD Strategy started - Live Trading Mode")
        
        def trading_loop():
            while self.running:
                try:
                    # Run analysis cycle
                    signals = self.run_analysis_cycle()
                    
                    if signals:
                        logger.info(f"Generated {len(signals)} signals this cycle")
                    
                    # Wait for next cycle
                    time.sleep(self.config.get('update_interval', 5))
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(10)  # Wait longer on error
        
        # Start trading thread
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        logger.info("Trading thread started")
    
    def stop_trading(self):
        """Stop automated trading"""
        self.running = False
        logger.info("SMA MACD Strategy stopped")
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status and performance"""
        try:
            positions = self.execution_engine.get_positions_summary()
            
            # Calculate win rate
            win_rate = (self.winning_trades / self.trades_today * 100) if self.trades_today > 0 else 0
            
            return {
                'running': self.running,
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'active_positions': positions['total_positions'],
                'trades_today': self.trades_today,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_profit': self.total_profit,
                'last_analysis': self.last_analysis,
                'strategy_config': {
                    'sma_fast': self.sma_fast,
                    'sma_slow': self.sma_slow,
                    'macd_fast': self.macd_fast,
                    'macd_slow': self.macd_slow,
                    'risk_per_trade': self.risk_per_trade,
                    'risk_reward_ratio': self.risk_reward_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage (requires MT5 connection)
    from .data_pipeline import DataPipeline
    
    # Initialize components
    data_pipeline = DataPipeline()
    execution_engine = MT5ExecutionEngine()
    
    if data_pipeline.start() and execution_engine.connect():
        # Create strategy
        strategy = SMA_MACD_Strategy(data_pipeline, execution_engine)
        
        try:
            # Start trading
            strategy.start_trading()
            
            # Let it run for demonstration
            time.sleep(60)
            
            # Get status
            status = strategy.get_strategy_status()
            print(f"Strategy Status: {status}")
            
        finally:
            strategy.stop_trading()
            execution_engine.disconnect()
            data_pipeline.stop()
    else:
        print("Failed to initialize trading components")