"""
Execution Engine - Handles order execution, position management, and risk control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime
import threading

# Optional MT5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure"""
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 123456
    timestamp: datetime = None
    order_id: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    position_type: OrderType
    volume: float
    open_price: float
    current_price: float
    profit: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic: int = 123456

class ExecutionEngine:
    """
    Execution Engine for order management and position tracking
    
    Handles:
    - Order execution (demo/live)
    - Position management
    - Risk management (stop loss, take profit)
    - Portfolio tracking
    """
    
    def __init__(self, mode: str = 'demo'):
        """
        Initialize execution engine
        
        Args:
            mode: 'demo' or 'live' trading mode
        """
        self.mode = mode
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.positions: List[Position] = []
        
        # Risk management parameters
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_total_risk = 0.10      # 10% total portfolio risk
        self.max_positions = 5          # Maximum simultaneous positions
        
        # Demo mode simulation
        self.demo_balance = 10000.0
        self.demo_equity = 10000.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Execution Engine initialized in {mode} mode")
    
    def place_order(self, symbol: str, order_type: OrderType, volume: float, 
                   price: Optional[float] = None, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None, comment: str = "") -> bool:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            volume: Order volume (lot size)
            price: Entry price (None for market order)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            True if order placed successfully
        """
        with self._lock:
            try:
                # Risk checks
                if not self._risk_check(symbol, volume):
                    logger.warning(f"Risk check failed for {symbol} {volume} lot order")
                    return False
                
                # Create order
                order = Order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    price=price or self._get_current_price(symbol),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment=comment,
                    timestamp=datetime.now()
                )
                
                if self.mode == 'live' and MT5_AVAILABLE:
                    # Execute live order
                    success = self._execute_live_order(order)
                else:
                    # Execute demo order
                    success = self._execute_demo_order(order)
                
                if success:
                    self.pending_orders.append(order)
                    logger.info(f"Order placed: {order_type.value} {volume} {symbol} at {order.price}")
                    return True
                else:
                    logger.error(f"Failed to place order: {order_type.value} {volume} {symbol}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error placing order: {str(e)}")
                return False
    
    def _execute_live_order(self, order: Order) -> bool:
        """Execute order on live MT5 account"""
        if not MT5_AVAILABLE:
            return False
        
        try:
            # Prepare order request
            order_type_mt5 = mt5.ORDER_TYPE_BUY if order.order_type == OrderType.BUY else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.volume,
                "type": order_type_mt5,
                "price": order.price,
                "deviation": 20,
                "magic": order.magic,
                "comment": order.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if order.stop_loss:
                request["sl"] = order.stop_loss
            if order.take_profit:
                request["tp"] = order.take_profit
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("Order send failed - no result")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
            
            # Update order with MT5 details
            order.order_id = result.order
            order.status = OrderStatus.FILLED
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing live order: {str(e)}")
            return False
    
    def _execute_demo_order(self, order: Order) -> bool:
        """Execute order in demo mode (simulation)"""
        try:
            # Simulate order execution
            order.status = OrderStatus.FILLED
            order.order_id = len(self.filled_orders) + 1
            
            # Create position
            position = Position(
                symbol=order.symbol,
                position_type=order.order_type,
                volume=order.volume,
                open_price=order.price,
                current_price=order.price,
                profit=0.0,
                open_time=order.timestamp,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                magic=order.magic
            )
            
            self.positions.append(position)
            self.filled_orders.append(order)
            
            logger.info(f"Demo order executed: {order.order_type.value} {order.volume} {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing demo order: {str(e)}")
            return False
    
    def _risk_check(self, symbol: str, volume: float) -> bool:
        """
        Perform risk management checks
        
        Args:
            symbol: Trading symbol
            volume: Order volume
            
        Returns:
            True if risk checks pass
        """
        try:
            # Check maximum positions
            if len(self.positions) >= self.max_positions:
                return False
            
            # Check volume is reasonable
            if volume <= 0 or volume > 10:  # Max 10 lots
                return False
            
            # Check total exposure
            total_risk = sum(pos.volume for pos in self.positions)
            if total_risk + volume > self.max_total_risk * 100:  # Rough calculation
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {str(e)}")
            return False
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        if MT5_AVAILABLE and self.mode == 'live':
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return (tick.bid + tick.ask) / 2
            except:
                pass
        
        # Fallback: simulate price
        return 1000.0 + np.random.normal(0, 10)  # Simulated price around 1000
    
    def update_positions(self, market_data: Dict[str, Any]):
        """
        Update position profits and check for stop loss/take profit
        
        Args:
            market_data: Current market data
        """
        with self._lock:
            try:
                for position in self.positions[:]:  # Copy list to avoid modification issues
                    symbol = position.symbol
                    
                    if symbol in market_data:
                        current_price = market_data[symbol].get('latest_price')
                        if current_price:
                            position.current_price = current_price
                            
                            # Calculate profit
                            if position.position_type == OrderType.BUY:
                                position.profit = (current_price - position.open_price) * position.volume * 100000  # Rough calculation
                            else:
                                position.profit = (position.open_price - current_price) * position.volume * 100000
                            
                            # Check stop loss
                            if position.stop_loss:
                                if ((position.position_type == OrderType.BUY and current_price <= position.stop_loss) or
                                    (position.position_type == OrderType.SELL and current_price >= position.stop_loss)):
                                    self._close_position(position, "Stop Loss Hit")
                            
                            # Check take profit
                            if position.take_profit:
                                if ((position.position_type == OrderType.BUY and current_price >= position.take_profit) or
                                    (position.position_type == OrderType.SELL and current_price <= position.take_profit)):
                                    self._close_position(position, "Take Profit Hit")
                
            except Exception as e:
                logger.error(f"Error updating positions: {str(e)}")
    
    def _close_position(self, position: Position, reason: str = "Manual Close"):
        """Close a position"""
        try:
            logger.info(f"Closing position: {position.symbol} {position.position_type.value} - {reason}")
            
            if self.mode == 'live' and MT5_AVAILABLE:
                # Close live position
                close_type = mt5.ORDER_TYPE_SELL if position.position_type == OrderType.BUY else mt5.ORDER_TYPE_BUY
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "magic": position.magic,
                    "comment": f"Close: {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.positions.remove(position)
            else:
                # Close demo position
                self.demo_equity += position.profit
                self.positions.remove(position)
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
    
    def process_pending_orders(self):
        """Process any pending orders"""
        # In this simplified implementation, orders are executed immediately
        # In a real system, this would handle order management
        pass
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        with self._lock:
            total_profit = sum(pos.profit for pos in self.positions)
            
            return {
                'mode': self.mode,
                'balance': self.demo_balance if self.mode == 'demo' else self._get_account_balance(),
                'equity': self.demo_equity + total_profit if self.mode == 'demo' else self._get_account_equity(),
                'total_profit': total_profit,
                'open_positions': len(self.positions),
                'pending_orders': len(self.pending_orders),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'type': pos.position_type.value,
                        'volume': pos.volume,
                        'profit': pos.profit
                    } for pos in self.positions
                ]
            }
    
    def _get_account_balance(self) -> float:
        """Get account balance from MT5"""
        if MT5_AVAILABLE and self.mode == 'live':
            try:
                account_info = mt5.account_info()
                return account_info.balance if account_info else 0.0
            except:
                pass
        return self.demo_balance
    
    def _get_account_equity(self) -> float:
        """Get account equity from MT5"""
        if MT5_AVAILABLE and self.mode == 'live':
            try:
                account_info = mt5.account_info()
                return account_info.equity if account_info else 0.0
            except:
                pass
        return self.demo_equity