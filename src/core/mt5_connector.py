#!/usr/bin/env python3
"""
MT5 Connector - AI Trading System V2
Conexión real con MetaTrader 5 usando mapeo exacto de símbolos
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class MT5Connector:
    """Conector para MetaTrader 5 con mapeo real de símbolos"""
    
    # MAPEO EXACTO DE SÍMBOLOS - NOMBRES REALES MT5
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
    
    def __init__(self):
        """Inicializar conector MT5"""
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.account_info = None
        
        # Credenciales desde .env
        self.login = int(os.getenv('MT5_LOGIN', '28758653'))
        self.password = os.getenv('MT5_PASSWORD', 'Demo@1234')
        self.server = os.getenv('MT5_SERVER', 'Deriv-Demo')
        
        # Intentar conexión automática
        self.connect()
    
    def connect(self) -> bool:
        """Conectar a MT5"""
        try:
            # Inicializar MT5
            if not mt5.initialize():
                self.logger.error("❌ No se pudo inicializar MT5")
                return False
            
            # Login
            if not mt5.login(self.login, password=self.password, server=self.server):
                error_code, error_desc = mt5.last_error()
                self.logger.error(f"❌ Error login MT5: {error_desc}")
                return False
            
            # Obtener info de cuenta
            self.account_info = mt5.account_info()
            self.connected = True
            
            self.logger.info(f"✅ MT5 conectado - Servidor: {self.server}")
            self.logger.info(f"💰 Balance: ${self.account_info.balance:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error conectando MT5: {e}")
            return False
    
    def disconnect(self):
        """Desconectar MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("🔴 MT5 desconectado")
    
    def get_real_symbol_name(self, internal_symbol: str) -> str:
        """Convertir nombre interno a nombre real MT5"""
        return self.SYMBOL_MAPPING.get(internal_symbol, internal_symbol)
    
    def get_market_data(self, symbol: str, timeframe: str = "M1", count: int = 100) -> Optional[pd.DataFrame]:
        """Obtener datos de mercado reales"""
        if not self.connected:
            self.logger.warning("⚠️ MT5 no conectado")
            return None
        
        try:
            # Convertir símbolo a nombre real MT5
            mt5_symbol = self.get_real_symbol_name(symbol)
            
            # Convertir timeframe
            mt5_timeframe = self._convert_timeframe(timeframe)
            
            # Obtener datos
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                error_code, error_desc = mt5.last_error()
                self.logger.error(f"❌ Error obteniendo datos {mt5_symbol}: {error_desc}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Renombrar columnas para consistencia
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            self.logger.debug(f"📊 Datos obtenidos {symbol}: {len(df)} velas")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo datos {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtener precio actual"""
        if not self.connected:
            return None
        
        try:
            mt5_symbol = self.get_real_symbol_name(symbol)
            tick = mt5.symbol_info_tick(mt5_symbol)
            
            if tick is None:
                return None
            
            return float(tick.bid)
            
        except Exception as e:
            self.logger.error(f"❌ Error precio actual {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Obtener precios múltiples símbolos"""
        prices = {}
        
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def get_account_balance(self) -> float:
        """Obtener balance de cuenta"""
        if self.account_info:
            return float(self.account_info.balance)
        return 0.0
    
    def get_account_equity(self) -> float:
        """Obtener equity de cuenta"""
        if self.account_info:
            return float(self.account_info.equity)
        return 0.0
    
    def is_market_open(self, symbol: str) -> bool:
        """Verificar si mercado está abierto"""
        try:
            mt5_symbol = self.get_real_symbol_name(symbol)
            symbol_info = mt5.symbol_info(mt5_symbol)
            
            if symbol_info is None:
                return False
            
            # Para índices sintéticos, suelen estar abiertos 24/7
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verificando mercado {symbol}: {e}")
            return False
    
    def _convert_timeframe(self, timeframe: str) -> int:
        """Convertir timeframe a formato MT5"""
        timeframe_mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        return timeframe_mapping.get(timeframe, mt5.TIMEFRAME_M1)
    
    def test_connection(self) -> Dict:
        """Test completo de conexión"""
        results = {
            "connected": self.connected,
            "account_info": None,
            "working_symbols": [],
            "failed_symbols": []
        }
        
        if not self.connected:
            return results
        
        # Info de cuenta
        if self.account_info:
            results["account_info"] = {
                "balance": self.account_info.balance,
                "equity": self.account_info.equity,
                "server": self.account_info.server,
                "currency": self.account_info.currency
            }
        
        # Probar símbolos
        test_symbols = ["R_75", "R_100", "stpRNG", "1HZ75V"]
        
        for symbol in test_symbols:
            try:
                data = self.get_market_data(symbol, count=5)
                if data is not None and len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    results["working_symbols"].append({
                        "symbol": symbol,
                        "mt5_name": self.get_real_symbol_name(symbol),
                        "price": float(current_price),
                        "data_points": len(data)
                    })
                else:
                    results["failed_symbols"].append(symbol)
            except Exception as e:
                results["failed_symbols"].append(f"{symbol}: {str(e)}")
        
        return results

# Instancia global del conector
mt5_connector = MT5Connector()

def get_mt5_connector() -> MT5Connector:
    """Obtener instancia del conector MT5"""
    return mt5_connector

# Funciones de conveniencia
def get_market_data(symbol: str, timeframe: str = "M1", count: int = 100) -> Optional[pd.DataFrame]:
    """Función de conveniencia para obtener datos"""
    return mt5_connector.get_market_data(symbol, timeframe, count)

def get_current_price(symbol: str) -> Optional[float]:
    """Función de conveniencia para precio actual"""
    return mt5_connector.get_current_price(symbol)

def get_account_info() -> Dict:
    """Función de conveniencia para info de cuenta"""
    return {
        "balance": mt5_connector.get_account_balance(),
        "equity": mt5_connector.get_account_equity(),
        "connected": mt5_connector.connected
    }

if __name__ == "__main__":
    # Test del conector
    print("🧪 PROBANDO CONECTOR MT5...")
    
    connector = get_mt5_connector()
    test_results = connector.test_connection()
    
    print(f"✅ Conectado: {test_results['connected']}")
    if test_results['account_info']:
        print(f"💰 Balance: ${test_results['account_info']['balance']:,.2f}")
    
    print(f"📊 Símbolos funcionando: {len(test_results['working_symbols'])}")
    for symbol_info in test_results['working_symbols']:
        print(f"   {symbol_info['symbol']} → ${symbol_info['price']}")