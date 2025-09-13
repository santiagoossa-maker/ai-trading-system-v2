"""
MT5 Execution Engine - Real Trading Implementation
Handles actual trade execution, position management, and risk control
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import threading
import yaml
import os

logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class TradeAction(Enum):
    DEAL = mt5.TRADE_ACTION_DEAL
    PENDING = mt5.TRADE_ACTION_PENDING
    SLTP = mt5.TRADE_ACTION_SLTP
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE

@dataclass
class TradeRequest:
    symbol: str
    action: TradeAction
    order_type: OrderType
    volume: float
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    comment: str = "AI Trading System V2"
    magic: int = 234000
    type_time: int = mt5.ORDER_TIME_GTC
    type_filling: int = mt5.ORDER_FILLING_IOC

@dataclass
class Position:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    profit: float
    comment: str
    time: datetime
    magic: int

@dataclass
class TradeResult:
    success: bool
    order: int = 0
    deal: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""
    error_code: int = 0
    error_description: str = ""

class MT5ExecutionEngine:
    """
    Real MT5 execution engine for live trading
    Handles all trade operations with proper risk management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.connected = False
        self.symbols_info = {}
        self.active_positions = {}
        self.trade_history = []
        self.risk_manager = None
        self.last_error = None
        
        # Trading parameters
        self.max_spread = self.config.get('max_spread', 20)  # points
        self.max_slippage = self.config.get('max_slippage', 10)  # points
        self.magic_number = self.config.get('magic_number', 234000)
        self.max_positions = self.config.get('max_positions', 10)
        
        # Risk management
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5%
        self.max_total_risk = self.config.get('max_total_risk', 0.10)  # 10%
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load MT5 configuration"""
        default_config = {
            'login': None,
            'password': None,
            'server': None,
            'path': None,
            'timeout': 60,
            'max_spread': 20,
            'max_slippage': 10,
            'magic_number': 234000,
            'max_positions': 10,
            'max_risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'max_total_risk': 0.10
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config.get('mt5', {}))
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5
            if not mt5.initialize(
                path=self.config.get('path'),
                login=self.config.get('login'),
                password=self.config.get('password'),
                server=self.config.get('server'),
                timeout=self.config.get('timeout', 60)
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                self.last_error = error
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False
            
            self.connected = True
            logger.info(f"Connected to MT5 - Account: {account_info.login}, Server: {account_info.server}")
            
            # Load symbols information
            self._load_symbols_info()
            
            # Start monitoring
            self._start_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            self.last_error = str(e)
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        self.connected = False
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def _load_symbols_info(self):
        """Load symbol information for all traded assets"""
        from .data_pipeline import LOTES
        
        for symbol in LOTES.keys():
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    # Ensure symbol is visible in Market Watch
                    if not symbol_info.visible:
                        if not mt5.symbol_select(symbol, True):
                            logger.warning(f"Failed to select symbol {symbol}")
                            continue
                    
                    self.symbols_info[symbol] = {
                        'point': symbol_info.point,
                        'digits': symbol_info.digits,
                        'spread': symbol_info.spread,
                        'contract_size': symbol_info.trade_contract_size,
                        'min_lot': symbol_info.volume_min,
                        'max_lot': symbol_info.volume_max,
                        'lot_step': symbol_info.volume_step,
                        'margin_required': symbol_info.margin_initial,
                        'tick_value': symbol_info.trade_tick_value,
                        'tick_size': symbol_info.trade_tick_size
                    }
                    logger.info(f"Symbol {symbol} loaded successfully")
                else:
                    logger.warning(f"Could not get info for symbol {symbol}")
            except Exception as e:
                logger.error(f"Error loading symbol {symbol}: {e}")
    
    def _start_monitoring(self):
        """Start position monitoring thread"""
        def monitor_positions():
            while self.connected:
                try:
                    self._update_positions()
                    time.sleep(1)  # Update every second
                except Exception as e:
                    logger.error(f"Error in position monitoring: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_positions, daemon=True)
        monitor_thread.start()
    
    def _update_positions(self):
        """Update active positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            current_positions = {}
            for pos in positions:
                if pos.magic == self.magic_number:
                    position = Position(
                        ticket=pos.ticket,
                        symbol=pos.symbol,
                        type=pos.type,
                        volume=pos.volume,
                        price_open=pos.price_open,
                        sl=pos.sl,
                        tp=pos.tp,
                        price_current=pos.price_current,
                        profit=pos.profit,
                        comment=pos.comment,
                        time=datetime.fromtimestamp(pos.time),
                        magic=pos.magic
                    )
                    current_positions[pos.ticket] = position
            
            self.active_positions = current_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = mt5.account_info()
            if account is None:
                return {}
            
            return {
                'login': account.login,
                'server': account.server,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit,
                'currency': account.currency,
                'leverage': account.leverage
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, risk_amount: float, sl_points: int) -> float:
        """Calculate optimal position size based on risk"""
        try:
            if symbol not in self.symbols_info:
                logger.error(f"Symbol {symbol} not found in symbols info")
                return 0.0
            
            symbol_info = self.symbols_info[symbol]
            account = mt5.account_info()
            
            if account is None:
                return 0.0
            
            # Calculate position size using risk amount and stop loss
            tick_value = symbol_info['tick_value']
            point = symbol_info['point']
            
            # Risk per point
            risk_per_point = risk_amount / (sl_points * point / point)
            
            # Position size in lots
            position_size = risk_per_point / tick_value
            
            # Apply constraints
            min_lot = symbol_info['min_lot']
            max_lot = symbol_info['max_lot']
            lot_step = symbol_info['lot_step']
            
            # Round to valid lot size
            position_size = max(min_lot, min(max_lot, position_size))
            position_size = round(position_size / lot_step) * lot_step
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, is_buy: bool, atr: float) -> float:
        """Calculate dynamic stop loss based on ATR"""
        try:
            if symbol not in self.symbols_info:
                return 0.0
            
            # Use 2x ATR for stop loss
            sl_distance = atr * 2
            
            if is_buy:
                sl_price = entry_price - sl_distance
            else:
                sl_price = entry_price + sl_distance
            
            # Round to valid price
            point = self.symbols_info[symbol]['point']
            digits = self.symbols_info[symbol]['digits']
            
            sl_price = round(sl_price / point) * point
            sl_price = round(sl_price, digits)
            
            return sl_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}")
            return 0.0
    
    def calculate_take_profit(self, symbol: str, entry_price: float, sl_price: float, is_buy: bool, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio"""
        try:
            sl_distance = abs(entry_price - sl_price)
            tp_distance = sl_distance * risk_reward_ratio
            
            if is_buy:
                tp_price = entry_price + tp_distance
            else:
                tp_price = entry_price - tp_distance
            
            # Round to valid price
            point = self.symbols_info[symbol]['point']
            digits = self.symbols_info[symbol]['digits']
            
            tp_price = round(tp_price / point) * point
            tp_price = round(tp_price, digits)
            
            return tp_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {e}")
            return 0.0
    
    def validate_trade_conditions(self, symbol: str) -> Tuple[bool, str]:
        """Validate conditions before placing trade"""
        try:
            # Check if market is open
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not available"
            
            if not symbol_info.trade_mode:
                return False, f"Trading disabled for {symbol}"
            
            # Check spread
            current_spread = symbol_info.spread
            if current_spread > self.max_spread:
                return False, f"Spread too high: {current_spread} > {self.max_spread}"
            
            # Check account status
            account = mt5.account_info()
            if account is None:
                return False, "Cannot get account information"
            
            if not account.trade_allowed:
                return False, "Trading not allowed on account"
            
            # Check maximum positions
            if len(self.active_positions) >= self.max_positions:
                return False, f"Maximum positions reached: {len(self.active_positions)}"
            
            # Check daily loss limit
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_deals = mt5.history_deals_get(today_start, datetime.now())
            
            if today_deals:
                daily_profit = sum(deal.profit for deal in today_deals if deal.magic == self.magic_number)
                daily_loss_pct = abs(daily_profit) / account.balance
                
                if daily_profit < 0 and daily_loss_pct > self.max_daily_loss:
                    return False, f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
            
            return True, "All conditions valid"
            
        except Exception as e:
            logger.error(f"Error validating trade conditions: {e}")
            return False, str(e)
    
    def place_market_order(self, symbol: str, is_buy: bool, volume: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> TradeResult:
        """Place market order"""
        try:
            # Validate conditions
            valid, message = self.validate_trade_conditions(symbol)
            if not valid:
                return TradeResult(success=False, comment=message)
            
            # Get current prices
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return TradeResult(success=False, comment="Cannot get current price")
            
            # Determine order type and price
            if is_buy:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or "AI Trading System V2",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return TradeResult(
                    success=False,
                    error_code=error[0] if error else 0,
                    error_description=error[1] if error else "Unknown error"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=False,
                    error_code=result.retcode,
                    error_description=f"Order failed: {result.comment}"
                )
            
            # Order successful
            trade_result = TradeResult(
                success=True,
                order=result.order,
                deal=result.deal,
                volume=result.volume,
                price=result.price,
                comment="Order placed successfully"
            )
            
            logger.info(f"Market order placed: {symbol} {'BUY' if is_buy else 'SELL'} {volume} @ {result.price}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return TradeResult(success=False, comment=str(e))
    
    def close_position(self, ticket: int) -> TradeResult:
        """Close position by ticket"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return TradeResult(success=False, comment="Position not found")
            
            pos = position[0]
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return TradeResult(success=False, comment="Cannot get current price")
            
            close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": "Position closed by AI System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return TradeResult(
                    success=False,
                    error_code=error[0] if error else 0,
                    error_description=error[1] if error else "Unknown error"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=False,
                    error_code=result.retcode,
                    error_description=f"Close failed: {result.comment}"
                )
            
            logger.info(f"Position closed: {ticket} at {result.price}")
            
            return TradeResult(
                success=True,
                order=result.order,
                deal=result.deal,
                volume=result.volume,
                price=result.price,
                comment="Position closed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return TradeResult(success=False, comment=str(e))
    
    def close_all_positions(self, symbol: str = None) -> List[TradeResult]:
        """Close all positions or all positions for specific symbol"""
        results = []
        
        for ticket, position in self.active_positions.items():
            if symbol is None or position.symbol == symbol:
                result = self.close_position(ticket)
                results.append(result)
        
        return results
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> TradeResult:
        """Modify position stop loss and take profit"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return TradeResult(success=False, comment="Position not found")
            
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return TradeResult(
                    success=False,
                    error_code=error[0] if error else 0,
                    error_description=error[1] if error else "Unknown error"
                )
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=False,
                    error_code=result.retcode,
                    error_description=f"Modify failed: {result.comment}"
                )
            
            logger.info(f"Position modified: {ticket} SL:{sl} TP:{tp}")
            
            return TradeResult(
                success=True,
                comment="Position modified successfully"
            )
            
        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {e}")
            return TradeResult(success=False, comment=str(e))
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all active positions"""
        try:
            if not self.active_positions:
                return {
                    'total_positions': 0,
                    'total_profit': 0.0,
                    'total_volume': 0.0,
                    'buy_positions': 0,
                    'sell_positions': 0,
                    'positions': []
                }
            
            total_profit = sum(pos.profit for pos in self.active_positions.values())
            total_volume = sum(pos.volume for pos in self.active_positions.values())
            buy_positions = sum(1 for pos in self.active_positions.values() if pos.type == 0)
            sell_positions = sum(1 for pos in self.active_positions.values() if pos.type == 1)
            
            positions_list = []
            for pos in self.active_positions.values():
                positions_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': pos.time.isoformat()
                })
            
            return {
                'total_positions': len(self.active_positions),
                'total_profit': total_profit,
                'total_volume': total_volume,
                'buy_positions': buy_positions,
                'sell_positions': sell_positions,
                'positions': positions_list
            }
            
        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return {}
    
    def get_trading_stats(self) -> Dict:
        """Get trading statistics"""
        try:
            account = mt5.account_info()
            if account is None:
                return {}
            
            # Get today's deals
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_deals = mt5.history_deals_get(today_start, datetime.now())
            
            today_profit = 0.0
            today_trades = 0
            winning_trades = 0
            
            if today_deals:
                for deal in today_deals:
                    if deal.magic == self.magic_number and deal.entry == 1:  # Exit deals
                        today_profit += deal.profit
                        today_trades += 1
                        if deal.profit > 0:
                            winning_trades += 1
            
            win_rate = (winning_trades / today_trades * 100) if today_trades > 0 else 0
            
            return {
                'account_balance': account.balance,
                'account_equity': account.equity,
                'account_profit': account.profit,
                'margin_level': account.margin_level,
                'free_margin': account.margin_free,
                'today_profit': today_profit,
                'today_trades': today_trades,
                'win_rate': win_rate,
                'active_positions': len(self.active_positions)
            }
            
        except Exception as e:
            logger.error(f"Error getting trading stats: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    engine = MT5ExecutionEngine()
    
    if engine.connect():
        print("Connected to MT5 successfully")
        
        # Get account info
        account = engine.get_account_info()
        print(f"Account Balance: {account.get('balance', 0)}")
        
        # Get positions summary
        positions = engine.get_positions_summary()
        print(f"Active Positions: {positions['total_positions']}")
        
        engine.disconnect()
    else:
        print("Failed to connect to MT5")