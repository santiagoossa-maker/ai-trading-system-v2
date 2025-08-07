"""
Execution Engine
Handles order execution, position management, and risk controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time

# Import for MT5 integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

class PositionStatus(Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"

@dataclass
class Order:
    """Trading order"""
    id: str
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    filled_time: Optional[datetime] = None
    filled_price: Optional[float] = None
    comment: str = ""
    magic_number: int = 12345

@dataclass
class Position:
    """Trading position"""
    id: str
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    profit: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    comment: str = ""

class ExecutionEngine:
    """
    Execution engine for managing orders and positions
    Supports both demo and live trading modes
    """
    
    def __init__(self, mode: str = 'demo', config: Optional[Dict[str, Any]] = None):
        """
        Initialize execution engine
        
        Args:
            mode: 'demo' or 'live'
            config: Configuration dictionary
        """
        self.mode = mode.lower()
        self.config = config or {}
        
        # Trading state
        self.is_connected = False
        self.orders = {}  # Dict[str, Order]
        self.positions = {}  # Dict[str, Position]
        self.order_counter = 0
        
        # Risk management
        self.max_positions = self.config.get('max_positions', 5)
        self.max_daily_loss = self.config.get('max_daily_loss', 1000.0)
        self.max_position_size = self.config.get('max_position_size', 1.0)
        self.daily_pnl = 0.0
        self.daily_pnl_date = datetime.now().date()
        
        # Asset lot sizes (from config)
        self.lot_sizes = {
            "1HZ75V": 0.05, "R_75": 0.01, "R_100": 0.5, "1HZ100V": 0.2,
            "R_50": 4.0, "1HZ50V": 0.01, "R_25": 0.5, "R_10": 0.5,
            "1HZ10V": 0.5, "1HZ25V": 0.01, "stpRNG": 0.1, "stpRNG2": 0.1,
            "stpRNG3": 0.1, "stpRNG4": 0.1, "stpRNG5": 0.1
        }
        
        # Update with config if provided
        if 'lot_sizes' in self.config:
            self.lot_sizes.update(self.config['lot_sizes'])
        
        # Threading for position monitoring
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info(f"Execution Engine initialized in {mode} mode")
    
    def connect(self) -> bool:
        """Connect to trading platform"""
        try:
            if self.mode == 'demo':
                # For demo mode, we simulate connection
                self.is_connected = True
                logger.info("Connected to demo trading environment")
                
                # Start position monitoring
                self._start_monitoring()
                return True
                
            elif self.mode == 'live':
                if not MT5_AVAILABLE:
                    logger.error("MT5 not available for live trading")
                    return False
                
                # Initialize MT5
                if not mt5.initialize():
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
                
                # Login if credentials provided
                if all(key in self.config for key in ['login', 'password', 'server']):
                    if not mt5.login(self.config['login'], self.config['password'], self.config['server']):
                        logger.error(f"MT5 login failed: {mt5.last_error()}")
                        return False
                
                self.is_connected = True
                logger.info("Connected to live trading environment")
                
                # Start position monitoring
                self._start_monitoring()
                return True
            
            else:
                logger.error(f"Unknown trading mode: {self.mode}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to trading platform: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from trading platform"""
        try:
            self._stop_monitoring()
            
            if self.mode == 'live' and MT5_AVAILABLE:
                mt5.shutdown()
            
            self.is_connected = False
            logger.info("Disconnected from trading platform")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
    
    def _start_monitoring(self):
        """Start position monitoring thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self._monitor_thread.start()
        logger.info("Position monitoring started")
    
    def _stop_monitoring(self):
        """Stop position monitoring thread"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Position monitoring stopped")
    
    def _monitor_positions(self):
        """Monitor open positions and update their status"""
        while self._monitoring:
            try:
                # Update daily PnL tracking
                current_date = datetime.now().date()
                if current_date != self.daily_pnl_date:
                    self.daily_pnl = 0.0
                    self.daily_pnl_date = current_date
                
                # Update position profits
                for position in list(self.positions.values()):
                    if position.status == PositionStatus.OPEN:
                        self._update_position_profit(position)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}")
                time.sleep(5)
    
    def _update_position_profit(self, position: Position):
        """Update position profit based on current market price"""
        try:
            # For demo mode, simulate price updates
            if self.mode == 'demo':
                # Simple random walk for simulation
                price_change = np.random.normal(0, 0.001)
                if position.current_price is None:
                    position.current_price = position.open_price
                position.current_price *= (1 + price_change)
            
            # Calculate profit
            if position.current_price:
                if position.order_type == OrderType.BUY:
                    position.profit = (position.current_price - position.open_price) * position.volume
                else:  # SELL
                    position.profit = (position.open_price - position.current_price) * position.volume
                
                # Check stop loss and take profit
                self._check_stop_loss_take_profit(position)
                
        except Exception as e:
            logger.error(f"Error updating position profit: {str(e)}")
    
    def _check_stop_loss_take_profit(self, position: Position):
        """Check if position should be closed due to SL/TP"""
        if not position.current_price or position.status != PositionStatus.OPEN:
            return
        
        close_position = False
        close_reason = ""
        
        if position.order_type == OrderType.BUY:
            if position.stop_loss and position.current_price <= position.stop_loss:
                close_position = True
                close_reason = "Stop Loss"
            elif position.take_profit and position.current_price >= position.take_profit:
                close_position = True
                close_reason = "Take Profit"
        
        else:  # SELL
            if position.stop_loss and position.current_price >= position.stop_loss:
                close_position = True
                close_reason = "Stop Loss"
            elif position.take_profit and position.current_price <= position.take_profit:
                close_position = True
                close_reason = "Take Profit"
        
        if close_position:
            self.close_position(position.id, close_reason)
    
    def place_order(self, symbol: str, order_type: OrderType, volume: Optional[float] = None,
                   price: Optional[float] = None, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None, comment: str = "") -> Optional[str]:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            volume: Order volume (if None, uses default lot size)
            price: Entry price (current market price if None)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if not self.is_connected:
                logger.error("Not connected to trading platform")
                return None
            
            # Risk checks
            if not self._risk_checks(symbol, volume):
                return None
            
            # Use default lot size if volume not specified
            if volume is None:
                volume = self.lot_sizes.get(symbol, 0.1)
            
            # Generate order ID
            self.order_counter += 1
            order_id = f"ORD_{self.order_counter:06d}"
            
            # Create order
            order = Order(
                id=order_id,
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )
            
            # Execute order based on mode
            if self.mode == 'demo':
                success = self._execute_demo_order(order)
            else:  # live
                success = self._execute_live_order(order)
            
            if success:
                self.orders[order_id] = order
                logger.info(f"Order placed successfully: {order_id} {order_type.value} {volume} {symbol}")
                return order_id
            else:
                logger.error(f"Failed to place order: {order_type.value} {volume} {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def _risk_checks(self, symbol: str, volume: Optional[float]) -> bool:
        """Perform risk management checks"""
        try:
            # Check max positions
            open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
            if open_positions >= self.max_positions:
                logger.warning(f"Maximum positions reached: {open_positions}")
                return False
            
            # Check daily loss limit
            if abs(self.daily_pnl) >= self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
                return False
            
            # Check position size
            if volume and volume > self.max_position_size:
                logger.warning(f"Position size too large: {volume}")
                return False
            
            # Check if symbol is supported
            if symbol not in self.lot_sizes:
                logger.warning(f"Unsupported symbol: {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk checks: {str(e)}")
            return False
    
    def _execute_demo_order(self, order: Order) -> bool:
        """Execute order in demo mode"""
        try:
            # Simulate order execution
            time.sleep(0.1)  # Simulate execution delay
            
            # Simulate market price
            if order.price is None:
                order.price = 100.0 + np.random.normal(0, 1)  # Simulate market price
            
            # Mark order as filled
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.now()
            order.filled_price = order.price
            
            # Create position
            position_id = f"POS_{len(self.positions) + 1:06d}"
            position = Position(
                id=position_id,
                symbol=order.symbol,
                order_type=order.order_type,
                volume=order.volume,
                open_price=order.filled_price,
                open_time=order.filled_time,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                current_price=order.filled_price,
                comment=order.comment
            )
            
            self.positions[position_id] = position
            logger.info(f"Demo position opened: {position_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing demo order: {str(e)}")
            order.status = OrderStatus.FAILED
            return False
    
    def _execute_live_order(self, order: Order) -> bool:
        """Execute order in live mode using MT5"""
        try:
            if not MT5_AVAILABLE:
                logger.error("MT5 not available for live trading")
                return False
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.volume,
                "type": mt5.ORDER_TYPE_BUY if order.order_type == OrderType.BUY else mt5.ORDER_TYPE_SELL,
                "price": order.price or mt5.symbol_info_tick(order.symbol).ask,
                "sl": order.stop_loss or 0,
                "tp": order.take_profit or 0,
                "deviation": 20,
                "magic": order.magic_number,
                "comment": order.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.now()
                order.filled_price = result.price
                
                # Create position tracking
                position_id = f"POS_{result.order}"
                position = Position(
                    id=position_id,
                    symbol=order.symbol,
                    order_type=order.order_type,
                    volume=order.volume,
                    open_price=result.price,
                    open_time=datetime.now(),
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    current_price=result.price,
                    comment=order.comment
                )
                
                self.positions[position_id] = position
                logger.info(f"Live position opened: {position_id}")
                
                return True
            else:
                logger.error(f"Live order failed: {result.retcode} - {result.comment}")
                order.status = OrderStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"Error executing live order: {str(e)}")
            order.status = OrderStatus.FAILED
            return False
    
    def close_position(self, position_id: str, reason: str = "") -> bool:
        """Close an open position"""
        try:
            if position_id not in self.positions:
                logger.error(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
            if position.status != PositionStatus.OPEN:
                logger.warning(f"Position already closed: {position_id}")
                return True
            
            # Execute close based on mode
            if self.mode == 'demo':
                success = self._close_demo_position(position, reason)
            else:  # live
                success = self._close_live_position(position, reason)
            
            if success:
                # Update daily PnL
                self.daily_pnl += position.profit
                logger.info(f"Position closed: {position_id}, Profit: {position.profit:.2f}, Reason: {reason}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {str(e)}")
            return False
    
    def _close_demo_position(self, position: Position, reason: str) -> bool:
        """Close position in demo mode"""
        try:
            position.status = PositionStatus.CLOSED
            position.close_time = datetime.now()
            position.close_price = position.current_price or position.open_price
            position.comment += f" | Closed: {reason}"
            
            # Final profit calculation
            if position.order_type == OrderType.BUY:
                position.profit = (position.close_price - position.open_price) * position.volume
            else:
                position.profit = (position.open_price - position.close_price) * position.volume
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing demo position: {str(e)}")
            return False
    
    def _close_live_position(self, position: Position, reason: str) -> bool:
        """Close position in live mode"""
        try:
            if not MT5_AVAILABLE:
                return False
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.order_type == OrderType.BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).bid if position.order_type == OrderType.BUY else mt5.symbol_info_tick(position.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": 12345,
                "comment": f"Close {position.id}: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                position.status = PositionStatus.CLOSED
                position.close_time = datetime.now()
                position.close_price = result.price
                position.comment += f" | Closed: {reason}"
                
                # Final profit calculation
                if position.order_type == OrderType.BUY:
                    position.profit = (position.close_price - position.open_price) * position.volume
                else:
                    position.profit = (position.open_price - position.close_price) * position.volume
                
                return True
            else:
                logger.error(f"Failed to close live position: {result.retcode}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing live position: {str(e)}")
            return False
    
    def get_positions(self, status: Optional[PositionStatus] = None) -> List[Position]:
        """Get positions, optionally filtered by status"""
        positions = list(self.positions.values())
        if status:
            positions = [p for p in positions if p.status == status]
        return positions
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status"""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        open_positions = self.get_positions(PositionStatus.OPEN)
        closed_positions = self.get_positions(PositionStatus.CLOSED)
        
        total_profit = sum(p.profit for p in self.positions.values())
        open_profit = sum(p.profit for p in open_positions)
        realized_profit = sum(p.profit for p in closed_positions)
        
        return {
            'mode': self.mode,
            'connected': self.is_connected,
            'total_orders': len(self.orders),
            'total_positions': len(self.positions),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_profit': total_profit,
            'open_profit': open_profit,
            'realized_profit': realized_profit,
            'daily_pnl': self.daily_pnl,
            'max_positions': self.max_positions,
            'max_daily_loss': self.max_daily_loss
        }

# Convenience functions
def create_execution_engine(mode: str = 'demo', config: Optional[Dict[str, Any]] = None) -> ExecutionEngine:
    """Create and return an execution engine instance"""
    return ExecutionEngine(mode, config)

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Execution Engine")
    print("=" * 40)
    
    # Create demo engine
    engine = ExecutionEngine(mode='demo')
    
    try:
        # Connect
        if engine.connect():
            print("✓ Connected to demo trading")
            
            # Place some test orders
            symbols = ['R_75', 'R_100']
            
            for i, symbol in enumerate(symbols):
                order_type = OrderType.BUY if i % 2 == 0 else OrderType.SELL
                
                order_id = engine.place_order(
                    symbol=symbol,
                    order_type=order_type,
                    comment=f"Test order {i+1}"
                )
                
                if order_id:
                    print(f"✓ Order placed: {order_id}")
                else:
                    print(f"✗ Failed to place order for {symbol}")
            
            # Wait a moment for monitoring
            time.sleep(2)
            
            # Get account summary
            summary = engine.get_account_summary()
            print(f"\nAccount Summary: {summary}")
            
            # Show positions
            positions = engine.get_positions()
            print(f"\nPositions ({len(positions)}):")
            for pos in positions:
                print(f"  {pos.id}: {pos.order_type.value} {pos.volume} {pos.symbol} @ {pos.open_price:.5f}, Profit: {pos.profit:.2f}")
            
            # Close first position if any
            if positions:
                if engine.close_position(positions[0].id, "Test close"):
                    print(f"✓ Position {positions[0].id} closed")
                else:
                    print(f"✗ Failed to close position {positions[0].id}")
            
            # Final summary
            final_summary = engine.get_account_summary()
            print(f"\nFinal Summary: {final_summary}")
            
        else:
            print("✗ Failed to connect to demo trading")
    
    finally:
        engine.disconnect()
        print("\n✓ Disconnected from trading platform")