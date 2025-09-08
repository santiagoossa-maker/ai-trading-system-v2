import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import json
import os

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Gestor completo de órdenes MT5
    Maneja todo el ciclo de vida de las órdenes
    """
    
    def __init__(self):
        self.active_orders = {}  # Órdenes pendientes
        self.active_positions = {}  # Posiciones abiertas
        self.order_history = []  # Historial completo
        self.magic_number = 777888
        self.monitoring = False
        self.monitor_thread = None
        
        # Configuración
        self.max_slippage = 20
        self.max_orders_per_symbol = 3
        self.max_total_orders = 15
        
        # Estadísticas
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_profit': 0.0,
            'total_trades': 0
        }
        
        logger.info("📊 OrderManager inicializado")
    
    def start_monitoring(self):
        """Iniciar monitoreo de órdenes"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("👁️ Monitoreo de órdenes iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de órdenes"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ Monitoreo de órdenes detenido")
    
    def place_market_order(self, symbol: str, order_type: str, volume: float, 
                          sl: float = 0, tp: float = 0, comment: str = "") -> Dict:
        """
        Colocar orden de mercado
        """
        try:
            # Validaciones
            if not self._validate_order_params(symbol, volume):
                return {"success": False, "error": "Parámetros inválidos"}
            
            # Verificar límites
            if not self._check_order_limits(symbol):
                return {"success": False, "error": "Límites de órdenes excedidos"}
            
            # Obtener precios
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {"success": False, "error": f"No se pudo obtener precio para {symbol}"}
            
            # Determinar precio y tipo MT5
            if order_type.upper() == "BUY":
                price = tick.ask
                mt5_type = mt5.ORDER_TYPE_BUY
            elif order_type.upper() == "SELL":
                price = tick.bid
                mt5_type = mt5.ORDER_TYPE_SELL
            else:
                return {"success": False, "error": "Tipo de orden inválido"}

            # 🔍 DEBUG: Información detallada de la orden
            logger.error(f"[DEBUG_ORDER] {symbol}: price_current={price}, sl={sl}, tp={tp}")
            logger.error(f"[DEBUG_ORDER] {symbol}: sl_distance={abs(price-sl):.6f}, tp_distance={abs(price-tp):.6f}")

            # Verificar si el símbolo está disponible para trading
            symbol_info = mt5.symbol_info(symbol)
            logger.error(f"[DEBUG_ORDER] {symbol}: trade_mode={symbol_info.trade_mode if symbol_info else 'None'}")
            
            # Preparar request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or f"AI_{order_type}_{datetime.now().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            # 🎯 FILLING MODE LOGIC:
            logger.error(f"[PRE_SEND] {symbol}:")
            logger.error(f"  action={request['action']}")
            logger.error(f"  type={request['type']}")  
            logger.error(f"  price={request.get('price', 'NO_PRICE')}")
            logger.error(f"  sl={request.get('sl', 'NO_SL')}")
            logger.error(f"  tp={request.get('tp', 'NO_TP')}")
            logger.error(f"  volume={request['volume']}")

            filling_modes = []
            if hasattr(symbol_info, "trade_filling_modes"):
                filling_modes = getattr(symbol_info, "trade_filling_modes")
            if not filling_modes:
                filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]

            logger.error(f"[FILLING_MODES] {symbol}: Probando modos: {filling_modes}")

            # PROBAR CADA FILLING MODE:
            result = None
            for filling_mode in filling_modes:
                request["type_filling"] = filling_mode
                logger.error(f"[TRYING_FILLING] {symbol}: Modo {filling_mode}")
    
                result = mt5.order_send(request)
    
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.error(f"[SUCCESS_FILLING] {symbol}: Modo {filling_mode} FUNCIONÓ!")
                    break
                elif result and result.retcode == 10030:
                    logger.error(f"[FAILED_FILLING] {symbol}: Modo {filling_mode} no soportado, probando siguiente...")
                    continue
                else:
                    logger.error(f"[ERROR_FILLING] {symbol}: Modo {filling_mode} - Error {result.retcode if result else 'None'}")
                    break

            # Si llegamos aquí sin éxito, todos los modos fallaron
            if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"[ALL_FAILED] {symbol}: Todos los filling modes fallaron")

            
            # Procesar resultado
            return self._process_order_result(result, request)
            
        except Exception as e:
            logger.error(f"Error colocando orden de mercado: {e}")
            return {"success": False, "error": str(e)}
    
    def place_pending_order(self, symbol: str, order_type: str, volume: float, 
                           price: float, sl: float = 0, tp: float = 0, 
                           comment: str = "") -> Dict:
        """
        Colocar orden pendiente (Buy Limit, Sell Limit, Buy Stop, Sell Stop)
        """
        try:
            # Validaciones
            if not self._validate_order_params(symbol, volume, price):
                return {"success": False, "error": "Parámetros inválidos"}
            
            if not self._check_order_limits(symbol):
                return {"success": False, "error": "Límites de órdenes excedidos"}
            
            # Mapear tipos de orden
            order_types = {
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
            }
            
            mt5_type = order_types.get(order_type.upper())
            if mt5_type is None:
                return {"success": False, "error": "Tipo de orden pendiente inválido"}
            
            # Preparar request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": self.magic_number,
                "comment": comment or f"AI_{order_type}_{datetime.now().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            # Ejecutar orden
            result = mt5.order_send(request)
            
            # Procesar resultado
            return self._process_order_result(result, request, pending=True)
            
        except Exception as e:
            logger.error(f"Error colocando orden pendiente: {e}")
            return {"success": False, "error": str(e)}
    
    def modify_order(self, ticket: int, sl: float = None, tp: float = None, price: float = None) -> Dict:
        """
        Modificar orden existente
        """
        try:
            # Obtener información de la orden/posición
            position = self._get_position_by_ticket(ticket)
            order = self._get_order_by_ticket(ticket)
            
            if not position and not order:
                return {"success": False, "error": "Orden/posición no encontrada"}
            
            # Preparar request de modificación
            if position:
                # Modificar posición
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    "position": ticket,
                    "sl": sl if sl is not None else position.sl,
                    "tp": tp if tp is not None else position.tp,
                }
            else:
                # Modificar orden pendiente
                request = {
                    "action": mt5.TRADE_ACTION_MODIFY,
                    "order": ticket,
                    "price": price if price is not None else order.price_open,
                    "sl": sl if sl is not None else order.sl,
                    "tp": tp if tp is not None else order.tp,
                }
            
            # Ejecutar modificación
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Orden {ticket} modificada exitosamente")
                return {"success": True, "ticket": ticket, "result": result}
            else:
                error = result.retcode if result else "Sin respuesta"
                logger.error(f"❌ Error modificando orden {ticket}: {error}")
                return {"success": False, "error": str(error)}
                
        except Exception as e:
            logger.error(f"Error modificando orden {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    def close_position(self, ticket: int, volume: float = None) -> Dict:
        """
        Cerrar posición (total o parcial)
        """
        try:
            # Obtener información de la posición
            position = self._get_position_by_ticket(ticket)
            if not position:
                return {"success": False, "error": "Posición no encontrada"}
            
            # Volumen a cerrar
            close_volume = volume if volume is not None else position.volume
            
            # Determinar tipo de orden de cierre
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            # Preparar request de cierre
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": f"Close_{ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Ejecutar cierre
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Posición {ticket} cerrada: {close_volume} lotes")
                return {"success": True, "ticket": ticket, "volume": close_volume, "result": result}
            else:
                error = result.retcode if result else "Sin respuesta"
                logger.error(f"❌ Error cerrando posición {ticket}: {error}")
                return {"success": False, "error": str(error)}
                
        except Exception as e:
            logger.error(f"Error cerrando posición {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_order(self, ticket: int) -> Dict:
        """
        Cancelar orden pendiente
        """
        try:
            # Verificar que la orden existe
            order = self._get_order_by_ticket(ticket)
            if not order:
                return {"success": False, "error": "Orden no encontrada"}
            
            # Preparar request de cancelación
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            # Ejecutar cancelación
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Orden {ticket} cancelada")
                return {"success": True, "ticket": ticket, "result": result}
            else:
                error = result.retcode if result else "Sin respuesta"
                logger.error(f"❌ Error cancelando orden {ticket}: {error}")
                return {"success": False, "error": str(error)}
                
        except Exception as e:
            logger.error(f"Error cancelando orden {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_order_params(self, symbol: str, volume: float, price: float = None) -> bool:
        """Validar parámetros de orden"""
        try:
            # Verificar símbolo
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Símbolo {symbol} no encontrado")
                return False
            
            if not symbol_info.visible:
                logger.error(f"❌ Símbolo {symbol} no está visible")
                return False
            
            # Verificar volumen
            if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
                logger.error(f"❌ Volumen inválido para {symbol}: {volume}")
                return False
            
            # Verificar step de volumen
            if (volume / symbol_info.volume_step) != int(volume / symbol_info.volume_step):
                logger.error(f"❌ Volumen no alineado al step para {symbol}: {volume}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando parámetros: {e}")
            return False
    
    def _check_order_limits(self, symbol: str) -> bool:
        """Verificar límites de órdenes"""
        try:
            # Contar órdenes por símbolo
            symbol_orders = len([o for o in self.active_orders.values() if o.get('symbol') == symbol])
            if symbol_orders >= self.max_orders_per_symbol:
                logger.warning(f"❌ Máximo de órdenes para {symbol} alcanzado: {symbol_orders}")
                return False
            
            # Contar órdenes totales
            total_orders = len(self.active_orders) + len(self.active_positions)
            if total_orders >= self.max_total_orders:
                logger.warning(f"❌ Máximo de órdenes totales alcanzado: {total_orders}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando límites: {e}")
            return False
    
    def _process_order_result(self, result, request: Dict, pending: bool = False) -> Dict:
        """Procesar resultado de orden"""
        try:
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Orden exitosa
                order_info = {
                    'ticket': result.order,
                    'symbol': request['symbol'],
                    'volume': request['volume'],
                    'type': request['type'],
                    'price': request.get('price', 0),
                    'sl': request.get('sl', 0),
                    'tp': request.get('tp', 0),
                    'time': datetime.now(),
                    'comment': request.get('comment', ''),
                    'pending': pending
                }
                
                # Guardar en tracking apropiado
                if pending:
                    self.active_orders[result.order] = order_info
                    logger.info(f"✅ Orden pendiente colocada: {request['symbol']} #{result.order}")
                else:
                    self.active_positions[result.order] = order_info
                    logger.info(f"✅ Posición abierta: {request['symbol']} #{result.order}")
                
                # Actualizar estadísticas
                self.stats['total_orders'] += 1
                self.stats['successful_orders'] += 1
                
                return {
                    "success": True,
                    "ticket": result.order,
                    "order_info": order_info,
                    "result": result
                }
            else:
                # Orden fallida
                error_code = result.retcode if result else "Sin respuesta"
                error_msg = f"Error {error_code}"
                
                logger.error(f"❌ Orden fallida para {request['symbol']}: {error_msg}")
                
                # Actualizar estadísticas
                self.stats['total_orders'] += 1
                self.stats['failed_orders'] += 1
                
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": error_code
                }
                
        except Exception as e:
            logger.error(f"Error procesando resultado de orden: {e}")
            return {"success": False, "error": str(e)}
    
    def _monitoring_loop(self):
        """Loop de monitoreo de órdenes"""
        logger.info("👁️ Iniciando monitoreo de órdenes...")
        
        while self.monitoring:
            try:
                # Actualizar órdenes pendientes
                self._update_pending_orders()
                
                # Actualizar posiciones activas
                self._update_active_positions()
                
                # Verificar órdenes ejecutadas/canceladas
                self._check_order_changes()
                
                # Esperar antes del siguiente ciclo
                time.sleep(5)  # Verificar cada 5 segundos
                
            except Exception as e:
                logger.error(f"Error en monitoreo de órdenes: {e}")
                time.sleep(10)
    
    def _update_pending_orders(self):
        """Actualizar estado de órdenes pendientes"""
        try:
            # Obtener órdenes pendientes de MT5
            orders = mt5.orders_get()
            if orders is None:
                return
            
            # Filtrar nuestras órdenes
            our_orders = [order for order in orders if order.magic == self.magic_number]
            current_tickets = [order.ticket for order in our_orders]
            
            # Actualizar tracking
            for ticket in list(self.active_orders.keys()):
                if ticket not in current_tickets:
                    # Orden ya no está pendiente (ejecutada o cancelada)
                    self._handle_order_removed(ticket, 'pending')
            
        except Exception as e:
            logger.error(f"Error actualizando órdenes pendientes: {e}")
    
    def _update_active_positions(self):
        """Actualizar estado de posiciones activas"""
        try:
            # Obtener posiciones activas de MT5
            positions = mt5.positions_get()
            if positions is None:
                return
            
            # Filtrar nuestras posiciones
            our_positions = [pos for pos in positions if pos.magic == self.magic_number]
            current_tickets = [pos.ticket for pos in our_positions]
            
            # Actualizar tracking
            for ticket in list(self.active_positions.keys()):
                if ticket not in current_tickets:
                    # Posición cerrada
                    self._handle_position_closed(ticket)
            
        except Exception as e:
            logger.error(f"Error actualizando posiciones activas: {e}")
    
    def _check_order_changes(self):
        """Verificar cambios en órdenes"""
        try:
            # Verificar nuevas posiciones (órdenes pendientes ejecutadas)
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    if pos.magic == self.magic_number and pos.ticket not in self.active_positions:
                        # Nueva posición detectada
                        self._handle_new_position(pos.ticket)
            
        except Exception as e:
            logger.error(f"Error verificando cambios de órdenes: {e}")
    
    def _handle_order_removed(self, ticket: int, order_type: str):
        """Manejar orden removida"""
        try:
            if order_type == 'pending' and ticket in self.active_orders:
                order_info = self.active_orders[ticket]
                logger.info(f"📝 Orden pendiente removida: {order_info['symbol']} #{ticket}")
                del self.active_orders[ticket]
                
        except Exception as e:
            logger.error(f"Error manejando orden removida {ticket}: {e}")
    
    def _handle_position_closed(self, ticket: int):
        """Manejar posición cerrada"""
        try:
            if ticket in self.active_positions:
                position_info = self.active_positions[ticket]
                
                # Obtener resultado del trade
                deals = mt5.history_deals_get(position=ticket)
                if deals and len(deals) >= 2:
                    profit = sum([deal.profit for deal in deals])
                    self.stats['total_profit'] += profit
                    self.stats['total_trades'] += 1
                    
                    logger.info(f"🔚 Posición cerrada: {position_info['symbol']} #{ticket} "
                               f"Profit: ${profit:.2f}")
                
                del self.active_positions[ticket]
                
        except Exception as e:
            logger.error(f"Error manejando posición cerrada {ticket}: {e}")
    
    def _handle_new_position(self, ticket: int):
        """Manejar nueva posición detectada"""
        try:
            # Obtener información de la posición
            positions = mt5.positions_get(ticket=ticket)
            if positions and len(positions) > 0:
                pos = positions[0]
                
                position_info = {
                    'ticket': ticket,
                    'symbol': pos.symbol,
                    'volume': pos.volume,
                    'type': pos.type,
                    'price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'time': datetime.fromtimestamp(pos.time),
                    'comment': pos.comment,
                    'pending': False
                }
                
                self.active_positions[ticket] = position_info
                logger.info(f"🆕 Nueva posición detectada: {pos.symbol} #{ticket}")
                
        except Exception as e:
            logger.error(f"Error manejando nueva posición {ticket}: {e}")
    
    def _get_position_by_ticket(self, ticket: int):
        """Obtener posición por ticket"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            return positions[0] if positions else None
        except:
            return None
    
    def _get_order_by_ticket(self, ticket: int):
        """Obtener orden por ticket"""
        try:
            orders = mt5.orders_get(ticket=ticket)
            return orders[0] if orders else None
        except:
            return None
    
    def get_all_active_orders(self) -> Dict:
        """Obtener todas las órdenes activas"""
        return {
            'pending_orders': self.active_orders.copy(),
            'active_positions': self.active_positions.copy()
        }
    
    def get_trading_statistics(self) -> Dict:
        """Obtener estadísticas de trading"""
        return self.stats.copy()
    
    def emergency_close_all(self) -> Dict:
        """Cerrar todas las posiciones (emergencia)"""
        try:
            results = []
            
            # Cerrar todas las posiciones activas
            for ticket in list(self.active_positions.keys()):
                result = self.close_position(ticket)
                results.append(result)
            
            # Cancelar todas las órdenes pendientes
            for ticket in list(self.active_orders.keys()):
                result = self.cancel_order(ticket)
                results.append(result)
            
            logger.warning("🚨 EMERGENCIA: Todas las órdenes cerradas/canceladas")
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error(f"Error en cierre de emergencia: {e}")
            return {"success": False, "error": str(e)}

# Instancia global
order_manager = OrderManager()