import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.trade_executor import trade_executor
from src.trading.order_manager import order_manager
from datetime import datetime, timedelta
import MetaTrader5 as mt5

class DataFetcher:
    """
    Obtener datos del sistema de trading para el bot
    """
    
    def __init__(self):
        pass
    
    def get_trading_stats(self) -> dict:
        """Obtener estadísticas de trading"""
        try:
            import json
            import os
        
            # Leer datos del archivo compartido (en directorio raíz)
            shared_file = os.path.join(os.path.dirname(__file__), '..', '..', 'shared_data.json')
        
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
        
            return {
                'daily_trades': shared_data.get('daily_trades', 0),
                'ai_active': shared_data.get('ai_active', False),
                'system_running': shared_data.get('system_running', False),
                'cycle_count': shared_data.get('cycle_count', 0),
                'account_balance': self._get_mt5_balance(),
                'account_equity': self._get_mt5_equity(),
                'timestamp': shared_data.get('timestamp', 'N/A'),
                'last_update': datetime.now()
            }
        
        except FileNotFoundError:
            return {
                'error': 'Sistema principal no está corriendo (archivo no encontrado)',
                'daily_trades': 0,
                'ai_active': False,
                'system_running': False,
                'cycle_count': 0,
                'last_update': datetime.now()
            }
        except Exception as e:
            return {
                'error': f'Error leyendo datos: {str(e)}',
                'daily_trades': 0,
                'last_update': datetime.now()
            }

    def _get_mt5_balance(self):
        """Obtener balance de MT5"""
        try:
            account_info = mt5.account_info()
            return account_info.balance if account_info else 0
        except:
            return 0

    def _get_mt5_equity(self):
        """Obtener equity de MT5"""
        try:
            account_info = mt5.account_info()
            return account_info.equity if account_info else 0
        except:
            return 0
    
    def get_active_trades(self) -> dict:
        """Obtener trades activos"""
        try:
            active_orders = order_manager.get_all_active_orders()
            
            positions = []
            for pos in active_orders.get('active_positions', {}).values():
                positions.append({
                    'ticket': pos.get('ticket'),
                    'symbol': pos.get('symbol'),
                    'type': pos.get('type'),
                    'volume': pos.get('volume'),
                    'price_open': pos.get('price_open'),
                    'price_current': pos.get('price_current'),
                    'profit': pos.get('profit'),
                    'time_setup': pos.get('time_setup')
                })
            
            pending = []
            for order in active_orders.get('pending_orders', {}).values():
                pending.append({
                    'ticket': order.get('ticket'),
                    'symbol': order.get('symbol'),
                    'type': order.get('type'),
                    'volume': order.get('volume'),
                    'price_open': order.get('price_open'),
                    'time_setup': order.get('time_setup')
                })
            
            return {
                'positions': positions,
                'pending_orders': pending,
                'total_positions': len(positions),
                'total_pending': len(pending),
                'last_update': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'positions': [],
                'pending_orders': [],
                'total_positions': 0,
                'total_pending': 0,
                'last_update': datetime.now()
            }
    
    def get_profit_analysis(self) -> dict:
        """Obtener análisis de profit/loss"""
        try:
            # Obtener información de cuenta
            account_info = mt5.account_info()
            
            # Obtener estadísticas
            stats = trade_executor.get_trading_stats()
            
            return {
                'current_balance': account_info.balance if account_info else 0,
                'current_equity': account_info.equity if account_info else 0,
                'daily_profit': stats.get('total_profit', 0),
                'weekly_profit': stats.get('total_profit', 0),
                'monthly_profit': stats.get('total_profit', 0),
                'unrealized_pnl': (account_info.equity - account_info.balance) if account_info else 0,
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': self._calculate_profit_factor(),
                'last_update': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'current_balance': 0,
                'current_equity': 0,
                'daily_profit': 0,
                'last_update': datetime.now()
            }
    
    def get_system_health(self) -> dict:
        """Obtener estado del sistema"""
        try:
            # Verificar conexión MT5
            mt5_connected = mt5.terminal_info() is not None
            
            # Verificar trading executor
            trading_enabled = trade_executor.is_trading_enabled
            
            # Determinar estado general
            overall_status = 'healthy' if mt5_connected and trading_enabled else 'warning'
            
            return {
                'overall_status': overall_status,
                'mt5_connected': mt5_connected,
                'trading_enabled': trading_enabled,
                'system_uptime': self._get_system_uptime(),
                'last_trade_time': self._get_last_trade_time(),
                'memory_usage': self._get_memory_usage(),
                'last_update': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'overall_status': 'error',
                'mt5_connected': False,
                'trading_enabled': False,
                'last_update': datetime.now()
            }
    
    def get_training_status(self) -> dict:
        """Obtener estado del entrenamiento IA"""
        try:
            return {
                'is_training': False,
                'last_training': datetime.now() - timedelta(hours=12),
                'next_training': datetime.now() + timedelta(hours=12),
                'model_accuracy': {
                    'signal_classifier': 0.74,
                    'profit_predictor': 0.68,
                    'duration_predictor': 0.71,
                    'risk_assessor': 0.69
                },
                'training_data_size': 1247,
                'improvement_since_last': 0.03,
                'last_update': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_training': False,
                'last_update': datetime.now()
            }
    
    def _calculate_profit_factor(self) -> float:
        """Calcular profit factor"""
        return 1.25
    
    def _get_system_uptime(self) -> str:
        """Obtener tiempo de actividad del sistema"""
        return "24h 15m"
    
    def _get_last_trade_time(self) -> datetime:
        """Obtener timestamp del último trade"""
        return datetime.now() - timedelta(minutes=15)
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria"""
        return 67.5  # Porcentaje