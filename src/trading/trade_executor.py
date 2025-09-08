import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from .order_manager import order_manager

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Ejecutor de trades mejorado que usa OrderManager
    """
    
    def __init__(self):
        self.is_trading_enabled = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
        self.max_risk_per_trade = float(os.getenv('MAX_RISK_PER_TRADE', '0.01'))
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '50'))
        self.min_account_balance = float(os.getenv('MIN_ACCOUNT_BALANCE', '1000'))
        
        # 🎯 CONFIGURACIÓN BALANCEADA
        self.max_positions_per_symbol = 2      # Máximo 2 posiciones por símbolo
        self.min_confidence_additional = 0.80  # 80% para segunda posición
        self.cooldown_seconds = 120            # 2 minutos entre trades
        self.max_total_positions = 8           # Máximo 8 posiciones totales
        self.symbol_last_trade = {}            # Tracking último trade por símbolo

        # Contadores
        self.daily_trades = 0
        self.trade_history = []
        
        # Iniciar order manager
        order_manager.start_monitoring()
        
        logger.info(f"🎯 TradeExecutor mejorado - Trading: {'HABILITADO' if self.is_trading_enabled else 'DESHABILITADO'}")

    def _get_real_symbol_name(self, symbol: str) -> str:
        """Convertir símbolo interno a nombre real de MT5"""
        SYMBOL_MAPPING = {
            "1HZ75V": "Volatility 75 (1s) Index",
            "R_75": "Volatility 75 Index",
            "R_100": "Volatility 100 Index",
            "1HZ100V": "Volatility 100 (1s) Index",
            "R_50": "Volatility 50 Index",
            "1HZ50V": "Volatility 50 (1s) Index",
            "R_25": "Volatility 25 Index",
            "1HZ25V": "Volatility 25 (1s) Index",
            "R_10": "Volatility 10 Index",
            "1HZ10V": "Volatility 10 (1s) Index",
            "stpRNG": "Step Index",
            "stpRNG2": "Step Index 200",
            "stpRNG3": "Step Index 300",
            "stpRNG4": "Step Index 400",
            "stpRNG5": "Step Index 500"
        }
        return SYMBOL_MAPPING.get(symbol, symbol)

    def debug_broker_requirements(self, symbol: str):
        """Diagnosticar requisitos del broker"""
        real_symbol = self._get_real_symbol_name(symbol)
        symbol_info = mt5.symbol_info(real_symbol)
    
        if symbol_info:
            logger.error(f"[BROKER_INFO] {symbol}:")
            logger.error(f"  stops_level: {symbol_info.trade_stops_level}")
            logger.error(f"  freeze_level: {symbol_info.trade_freeze_level}")
            logger.error(f"  point: {symbol_info.point}")
            logger.error(f"  digits: {symbol_info.digits}")
            logger.error(f"  spread: {getattr(symbol_info, 'spread', 'N/A')}")
        else:
            logger.error(f"[BROKER_INFO] {symbol}: No se pudo obtener symbol_info")
    
    def execute_trade_decision(self, symbol: str, decision: Dict) -> bool:
        """Ejecutar decisión de trading usando OrderManager"""

        # 🔍 LOG TEMPORAL - ESTADO DE TRADING
        logger.info(f"🎯 TRADE EXECUTOR: Trading={'✅ HABILITADO' if self.is_trading_enabled else '❌ DESHABILITADO'}")

        if not self.is_trading_enabled:
            logger.warning(f"[FAILED] {symbol}: Trading deshabilitado por configuración (MODO DEMO). Decisión: {decision}")
            return False

        try:
            # Extraer información PRIMERO
            signal = decision.get('signal', 0)
            confidence = decision.get('confidence', 0)
            predicted_profit = decision.get('predicted_profit', 0)
            risk_level = decision.get('risk_level', 1)
            logger.error(f"[DEBUG] {symbol}: PASO 1 - signal={signal}, confidence={confidence}, profit={predicted_profit}, risk={risk_level}")

            # Verificar límites
            logger.error(f"[DEBUG] {symbol}: PASO 2 - Verificando trading limits...")
            if not self._check_trading_limits(symbol, confidence):
                logger.error(f"[FAILED] {symbol}: Fallo en _check_trading_limits(). Revisa logs anteriores para detalles.")
                return False
            logger.error(f"[DEBUG] {symbol}: PASO 2 - Trading limits OK")
    
            # Validar decisión
            logger.error(f"[DEBUG] {symbol}: PASO 3 - Validando trade decision...")
            if not self._validate_trade_decision(decision):
                logger.error(f"[FAILED] {symbol}: Fallo en _validate_trade_decision(). Decisión no válida/confianza/riesgo/profit.")
                return False
            logger.error(f"[DEBUG] {symbol}: PASO 3 - Trade decision válida")
    
            # Calcular parámetros
            logger.error(f"[DEBUG] {symbol}: PASO 4 - Calculando position size...")
            lot_size = self._calculate_position_size(symbol, risk_level, predicted_profit)
            logger.error(f"[DEBUG] {symbol}: PASO 4 - lot_size resultado = {lot_size}")
        
            if lot_size <= 0:
                logger.error(f"[FAILED] {symbol}: lot_size calculado <= 0. No se puede abrir posición.")
                return False
        
            # Usar símbolo real para MT5
            real_symbol = self._get_real_symbol_name(symbol)
            logger.error(f"[DEBUG] {symbol}: PASO 5 - Usando símbolo real {real_symbol} para MT5")

            # 🔍 AGREGAR ESTA LÍNEA:
            self.debug_broker_requirements(symbol)
        
            sl, tp = self._calculate_sl_tp(symbol, decision, signal)
            if sl == 0 or tp == 0:
                logger.error(f"[FAILED] {symbol}: SL/TP calculados inválidos (sl={sl}, tp={tp}).")
                return False

            # Ejecutar usando OrderManager
            if signal == 1:  # BUY
                result = order_manager.place_market_order(
                    symbol=real_symbol,  # ← USAR real_symbol
                    order_type="BUY",
                    volume=lot_size,
                    sl=sl,
                    tp=tp,
                    comment=f"AI_BUY_C{confidence:.2f}_P{predicted_profit:.3f}"
                )
            elif signal == 2:  # SELL
                result = order_manager.place_market_order(
                    symbol=real_symbol,  # ← USAR real_symbol
                    order_type="SELL",
                    volume=lot_size,
                    sl=sl,
                    tp=tp,
                    comment=f"AI_SELL_C{confidence:.2f}_P{predicted_profit:.3f}"
                )
            else:
                logger.error(f"[FAILED] {symbol}: Signal inválido ({signal}). Debe ser 1 (BUY) o 2 (SELL).")
                return False
        
            if result.get('success'):
                self.daily_trades += 1
            
                # 🎯 Registrar último trade para cooldown
                import time
                self.symbol_last_trade[symbol] = time.time()
            
                logger.info(f"✅ Trade ejecutado: {symbol} {signal} - Ticket: {result.get('ticket')}")
                return True
            else:
                logger.error(f"[FAILED] {symbol}: Error ejecutando trade con OrderManager: {result.get('error')}")
                return False
            
        except Exception as e:
            logger.error(f"[FAILED] {symbol}: Excepción en execute_trade_decision: {e}")
            return False
    
    def _check_trading_limits(self, symbol: str = None, confidence: float = 0.0) -> bool:
        """Verificar límites de trading"""
        try:
            # Límite diario
            if self.daily_trades >= self.max_daily_trades:
                logger.warning(f"❌ Límite diario alcanzado: {self.daily_trades}/{self.max_daily_trades}")
                return False
            
            # Balance mínimo
            account_info = mt5.account_info()
            if not account_info or account_info.balance < self.min_account_balance:
                logger.warning(f"❌ Balance insuficiente: {account_info.balance if account_info else 'N/A'}")
                return False
            
            # 🎯 LÍMITES BALANCEADOS
            active_orders = order_manager.get_all_active_orders()
            total_active = len(active_orders['pending_orders']) + len(active_orders['active_positions'])
            
            # Límite de posiciones totales
            if total_active >= self.max_total_positions:
                logger.warning(f"❌ Demasiadas posiciones activas: {total_active}/{self.max_total_positions}")
                return False
            
            # Límites específicos por símbolo
            if symbol:
                # Mapear símbolo interno a nombre real para búsqueda correcta
                real_symbol = self._get_real_symbol_name(symbol)
                symbol_positions = len([pos for pos in active_orders['active_positions'].values() 
                                      if real_symbol.lower() in pos.get('symbol', '').lower()])
                
                logger.error(f"[DEBUG_LIMITS] {symbol}: real_symbol='{real_symbol}', positions_found={symbol_positions}")
                
                # Primera posición: siempre permitida (si cumple otros límites)
                if symbol_positions == 0:
                    logger.error(f"[DEBUG] {symbol}: Primera posición - OK")
                    return True
                
                # Segunda posición: solo si confidence > 80%
                elif symbol_positions == 1:
                    if confidence >= self.min_confidence_additional:
                        logger.error(f"[DEBUG] {symbol}: Segunda posición permitida (conf: {confidence:.2f})")
                        return True
                    else:
                        logger.warning(f"❌ {symbol}: Segunda posición requiere conf > {self.min_confidence_additional:.2f} (actual: {confidence:.2f})")
                        return False
                
                # Tercera+ posición: no permitida
                else:
                    logger.warning(f"❌ {symbol}: Máximo posiciones alcanzado: {symbol_positions}/{self.max_positions_per_symbol}")
                    return False
                
                # Cooldown entre trades del mismo símbolo
                import time
                current_time = time.time()
                last_trade_time = self.symbol_last_trade.get(symbol, 0)
                
                if current_time - last_trade_time < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - (current_time - last_trade_time)
                    logger.warning(f"❌ {symbol}: Cooldown activo - {remaining:.0f}s restantes")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando límites de trading: {e}")
            return False
    
    def _validate_trade_decision(self, decision: Dict) -> bool:
        """Validar calidad de decisión"""
        confidence = decision.get('confidence', 0)
        risk_level = decision.get('risk_level', 2)
        predicted_profit = decision.get('predicted_profit', 0)
        
        return (
            confidence >= 0.6 and
            risk_level <= 1 and
            abs(predicted_profit) >= 0.005
        )
    
    def _calculate_position_size(self, symbol: str, risk_level: int, predicted_profit: float) -> float:
        """Calcular tamaño de posición basado en riesgo y capital disponible"""
    
        # 🔍 HOMOLOGACIÓN DE SÍMBOLO
        real_symbol = self._get_real_symbol_name(symbol)
        logger.error(f"[DEBUG] {symbol}: Mapeando a símbolo real: {real_symbol}")
    
        try:
            # Obtener información del balance
            logger.error(f"[DEBUG] {symbol}: Obteniendo account_info...")
            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"[DEBUG] {symbol}: account_info es None - PROBLEMA MT5")
                return 0
        
            balance = account_info.balance
            logger.error(f"[DEBUG] {symbol}: Balance obtenido = {balance}")
        
            # Obtener información del símbolo REAL
            logger.error(f"[DEBUG] {symbol}: Obteniendo symbol_info para {real_symbol}...")
            symbol_info = mt5.symbol_info(real_symbol)
            if symbol_info is None:
                logger.error(f"[DEBUG] {symbol}: symbol_info es None para {real_symbol} - SÍMBOLO NO ENCONTRADO")
                return 0
        
            logger.error(f"[DEBUG] {symbol}: symbol_info OK - volume_min={symbol_info.volume_min}, volume_step={symbol_info.volume_step}")
        
            # TU CÓDIGO ORIGINAL AQUÍ (el resto de la función)
            fixed_balance = 100.0  # Usar siempre $100 independiente del balance real
            risk_percentage = 0.005  # 0.5% del balance fijo
            if risk_level == 2:  # ALTO
                risk_percentage = 0.005  # 0.5%
            elif risk_level == 0:  # BAJO  
                risk_percentage = 0.015  # 1.5%
        
            risk_amount = fixed_balance * risk_percentage  # Siempre basado en $100
            logger.error(f"[DEBUG] {symbol}: BALANCE_REAL={balance}, BALANCE_USADO={fixed_balance}, risk_amount={risk_amount:.6f}")
        
            # Usar precio actual para calcular lot size
            tick_info = mt5.symbol_info_tick(real_symbol)  # ← USAR real_symbol
            if tick_info is None:
                logger.error(f"[DEBUG] {symbol}: No se pudo obtener tick info para {real_symbol}")
                return 0
        
            current_price = tick_info.bid
        
            # ✅ SOLUCIÓN GENÉRICA CON DATOS REALES MT5
            contract_size = symbol_info.trade_contract_size
            tick_size = symbol_info.point
            tick_value = symbol_info.trade_tick_value

            # SL distance conservador (2% del precio)
            sl_distance_price = current_price * 0.02
            sl_distance_ticks = sl_distance_price / tick_size if tick_size > 0 else 0

            # Position size usando valores REALES
            if tick_value > 0 and sl_distance_ticks > 0:
                lot_size_antes_redondeo = risk_amount / (sl_distance_ticks * tick_value)
            else:
                logger.error(f"[DEBUG] {symbol}: tick_value={tick_value}, sl_ticks={sl_distance_ticks} - valores inválidos")
                return 0

            # Límites específicos por símbolo (LOTAJES MÍNIMOS REALES)
            MIN_LOTS = {
                "1HZ75V": 0.05, "R_75": 0.01, "R_100": 0.5, "1HZ100V": 0.2,
                "R_50": 4.0, "1HZ50V": 0.01, "R_25": 0.5, "R_10": 0.5,
                "1HZ10V": 0.5, "1HZ25V": 0.01, "stpRNG": 0.1, "stpRNG2": 0.1,
                "stpRNG3": 0.1, "stpRNG4": 0.1, "stpRNG5": 0.1
            }

            # Aplicar límites específicos
            min_lot_required = MIN_LOTS.get(symbol, symbol_info.volume_min)
            max_lot_allowed = symbol_info.volume_max

            # Redondear según volume_step
            volume_step = symbol_info.volume_step
            lot_size = round(lot_size_antes_redondeo / volume_step) * volume_step

            logger.error(f"[CALC_COMPLETE] {symbol}:")
            logger.error(f"  risk_amount={risk_amount}")
            logger.error(f"  sl_distance_ticks={sl_distance_ticks}")
            logger.error(f"  tick_value={tick_value}")
            logger.error(f"  lot_formula = {risk_amount} / ({sl_distance_ticks} * {tick_value}) = {lot_size_antes_redondeo}")
            logger.error(f"  volume_step={volume_step}")
            logger.error(f"  min_lot_required={min_lot_required}")
            logger.error(f"  max_lot_allowed={max_lot_allowed}")

            # Validar si podemos cumplir el lotaje mínimo
            # 🎯 AJUSTE MATEMÁTICO PROPORCIONAL DE RISK
            if lot_size < min_lot_required:
                # Calcular ratio de ajuste
                ratio_ajuste = min_lot_required / lot_size
                nuevo_risk_amount = risk_amount * ratio_ajuste
                
                logger.warning(f"[MATH_ADJUST] {symbol}:")
                logger.warning(f"  lot_calculado={lot_size:.6f} < lot_mínimo={min_lot_required}")
                logger.warning(f"  ratio_ajuste={ratio_ajuste:.3f}")
                logger.warning(f"  risk_original=${risk_amount:.6f} → risk_ajustado=${nuevo_risk_amount:.6f}")
                logger.warning(f"  lot_ajustado={min_lot_required}")
                
                # Actualizar valores
                lot_size = min_lot_required
                risk_amount = nuevo_risk_amount  # ← ACTUALIZAR RISK PARA LOGS
                
                # 🔬 VERIFICACIÓN MATEMÁTICA
                verification_risk = lot_size * sl_distance_ticks * tick_value
                logger.warning(f"  verificación_risk=${verification_risk:.6f} (debe ≈ ${nuevo_risk_amount:.6f})")
            else:
                logger.error(f"[NO_ADJUST] {symbol}: lot_calculado={lot_size:.6f} >= lot_mínimo={min_lot_required} ✅")

            # Aplicar límite máximo
            lot_size = min(lot_size, max_lot_allowed)
        
            logger.error(f"[DEBUG] {symbol}: balance={balance}, risk_amount_final={risk_amount:.6f}, lot_calculado={lot_size_antes_redondeo:.6f}, lot_final={lot_size}")
        
            return lot_size
        
        except Exception as e:
            logger.error(f"[DEBUG] {symbol}: Exception en _calculate_position_size: {e}")
            return 0
    
    def _calculate_sl_tp(self, symbol: str, decision: Dict, signal: int) -> Tuple[float, float]:
        """Calcular Stop Loss y Take Profit - CON DEBUG DETALLADO"""
        try:
            real_symbol = self._get_real_symbol_name(symbol)
            tick = mt5.symbol_info_tick(real_symbol)
            symbol_info = mt5.symbol_info(real_symbol)
        
            if not tick or not symbol_info:
                return 0, 0
    
            risk_level = decision.get('risk_level', 1)
    
            # 🔧 SL Y TP CONSERVADORES Y REALISTAS
            sl_distance_pct = {0: 0.005, 1: 0.01, 2: 0.015}[risk_level]  # 0.5%, 1%, 1.5%
            tp_distance_pct = {0: 0.01, 1: 0.015, 2: 0.02}[risk_level]   # 1%, 1.5%, 2%
    
            # Convert to price
            if signal == 1:  # BUY
                price = tick.ask
                sl = price * (1 - sl_distance_pct)
                tp = price * (1 + tp_distance_pct)
            else:  # SELL
                price = tick.bid
                sl = price * (1 + sl_distance_pct)
                tp = price * (1 - tp_distance_pct)
    
            # 🔍 VERIFICACIÓN DETALLADA DE PUNTOS
            sl_distance_price = abs(price - sl)
            tp_distance_price = abs(tp - price)
            point = symbol_info.point
            sl_distance_points = sl_distance_price / point if point > 0 else 0
            tp_distance_points = tp_distance_price / point if point > 0 else 0

            logger.error(f"[POINTS_DEBUG] {symbol}:")
            logger.error(f"  price={price}, point={point}")
            logger.error(f"  sl={sl}, tp={tp}")
            logger.error(f"  sl_distance_price={sl_distance_price:.6f}")
            logger.error(f"  sl_distance_points={sl_distance_points:.1f}")
            logger.error(f"  tp_distance_points={tp_distance_points:.1f}")
            logger.error(f"  broker_min_required={symbol_info.trade_stops_level}")
            logger.error(f"  ¿CUMPLE SL? {sl_distance_points >= symbol_info.trade_stops_level}")
            logger.error(f"  ¿CUMPLE TP? {tp_distance_points >= symbol_info.trade_stops_level}")
    
            # 🔧 REDONDEAR A DIGITS CORRECTOS
            digits = symbol_info.digits
            sl = round(sl, digits)
            tp = round(tp, digits)

            logger.error(f"[ROUNDED] {symbol}: sl={sl}, tp={tp} (digits={digits})")

            return sl, tp
    
        except Exception as e:
            logger.error(f"Error calculando SL/TP: {e}")
            return 0, 0

    def get_trading_stats(self) -> Dict:
        """Obtener estadísticas usando OrderManager"""
        try:
            order_stats = order_manager.get_trading_statistics()
            
            return {
                'daily_trades': self.daily_trades,
                'total_orders': order_stats.get('total_orders', 0),
                'successful_orders': order_stats.get('successful_orders', 0),
                'failed_orders': order_stats.get('failed_orders', 0),
                'total_profit': order_stats.get('total_profit', 0),
                'total_trades': order_stats.get('total_trades', 0),
                'win_rate': (order_stats.get('successful_orders', 0) / max(1, order_stats.get('total_orders', 1))) * 100
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo stats: {e}")
            return {}
    
    def emergency_stop(self):
        """Parada de emergencia"""
        logger.warning("🚨 PARADA DE EMERGENCIA ACTIVADA")
        return order_manager.emergency_close_all()

# Instancia global
trade_executor = TradeExecutor()