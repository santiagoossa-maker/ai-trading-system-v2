#!/usr/bin/env python3
"""
AI Trading System V2 + Dannis Bot - Sistema Integrado
Sistema completo de trading con IA + Bot romántico para Dannis
"""

import os
import sys
import time
import logging
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Core imports
try:
    from src.core.data_pipeline import DataPipeline
    from src.strategies.multi_strategy_engine import MultiStrategyEngine
    from src.ai.continuous_learning import continuous_learning
    from src.ai.feature_collector import FeatureCollector
    
    # AI IMPORTS
    from src.ai.signal_classifier import signal_classifier
    from src.ai.profit_predictor import profit_predictor
    from src.ai.duration_predictor import duration_predictor
    from src.ai.risk_assessor import risk_assessor
    
    # TRADING IMPORTS
    from src.trading.trade_executor import trade_executor
    from src.trading.order_manager import order_manager
    
    # ✅ TELEGRAM BOT IMPORT
    from telegram_bot.dannis_bot import DannisBot
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configuración logging con UTF-8
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            self.stream.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AITradingSystem:
    """Sistema principal de trading con IA + Bot integrado"""
    
    def __init__(self):
        self.running = False
        self.data_pipeline = None
        self.strategy_engine = None
        self.feature_collector = None
        self.ai_models_trained = False
        self.cycle_count = 0
        
        # ✅ BOT INTEGRATION
        self.dannis_bot = None
        self.bot_thread = None
        
        AITradingSystem._instance = self

    @classmethod
    def get_instance(cls):
        """Obtener la instancia actual del sistema"""
        return getattr(cls, '_instance', None)
    
    def _start_telegram_bot(self):
        """Iniciar bot de Telegram en thread separado"""
        try:
            # Verificar token del bot
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if not bot_token:
                logger.warning("⚠️ No se encontró TELEGRAM_BOT_TOKEN - Bot no iniciado")
                return
            
            logger.info("💕 Iniciando Dannis Bot...")
            
            # Crear nueva instancia del bot
            self.dannis_bot = DannisBot()
            
            # Crear nuevo loop asyncio para el bot
            def run_bot():
                try:
                    # Crear nuevo event loop para este thread
                    bot_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(bot_loop)
                    
                    # Ejecutar bot en este loop
                    bot_loop.run_until_complete(self.dannis_bot.run())
                    
                except Exception as e:
                    logger.error(f"❌ Error ejecutando bot: {e}")
                finally:
                    bot_loop.close()
            
            # Iniciar bot en thread daemon
            self.bot_thread = threading.Thread(target=run_bot, daemon=True, name="DannisBot")
            self.bot_thread.start()
            
            logger.info("✅ Dannis Bot iniciado exitosamente en thread separado")
            
        except Exception as e:
            logger.error(f"❌ Error iniciando Dannis Bot: {e}")
        
    def initialize(self):
        """Inicializar todos los componentes"""
        try:
            logger.info("🚀 Iniciando Sistema de Trading con IA V2 + Dannis Bot...")
            
            # ✅ INICIAR BOT PRIMERO
            self._start_telegram_bot()
            
            # Inicializar pipeline de datos
            logger.info("📊 Inicializando Data Pipeline...")
            self.data_pipeline = DataPipeline()
            if not self.data_pipeline.start():
                logger.error("❌ Error iniciando Data Pipeline")
                return False
            
            # Esperar un poco para que se carguen datos
            logger.info("⏳ Esperando carga de datos...")
            time.sleep(5)
            
            # Inicializar otros componentes
            logger.info("🎯 Inicializando Strategy Engine...")
            self.strategy_engine = MultiStrategyEngine()
            
            logger.info("🔬 Inicializando Feature Collector...")
            self.feature_collector = FeatureCollector()
            
            # Inicializar aprendizaje continuo
            logger.info("🧠 Iniciando Aprendizaje Continuo...")
            continuous_learning.start()
            
            # Entrenar modelos IA si hay datos
            self._train_ai_models()
            
            logger.info("✅ Todos los componentes inicializados correctamente")
            logger.info("💕 Sistema completo funcionando: Trading IA + Dannis Bot")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            return False
    
    def _train_ai_models(self):
        """Entrenar modelos de IA con datos disponibles"""
        try:
            logger.info("🤖 Entrenando modelos de IA...")
            
            # Obtener datos del pipeline
            if self.data_pipeline:
                data_dict = self.data_pipeline.get_all_symbols_data('M5')
                
                if not data_dict:
                    logger.warning("⚠️ No hay datos disponibles para entrenar IA")
                    return
                
                # Entrenar cada modelo
                models_trained = 0
                
                # 1. Signal Classifier
                try:
                    if signal_classifier.train(data_dict):
                        models_trained += 1
                        logger.info("✅ Signal Classifier entrenado")
                except Exception as e:
                    logger.error(f"❌ Error entrenando Signal Classifier: {e}")
                
                # 2. Profit Predictor
                try:
                    if profit_predictor.train(data_dict):
                        models_trained += 1
                        logger.info("✅ Profit Predictor entrenado")
                except Exception as e:
                    logger.error(f"❌ Error entrenando Profit Predictor: {e}")
                
                # 3. Duration Predictor
                try:
                    if duration_predictor.train(data_dict):
                        models_trained += 1
                        logger.info("✅ Duration Predictor entrenado")
                except Exception as e:
                    logger.error(f"❌ Error entrenando Duration Predictor: {e}")
                
                # 4. Risk Assessor
                try:
                    if risk_assessor.train(data_dict):
                        models_trained += 1
                        logger.info("✅ Risk Assessor entrenado")
                except Exception as e:
                    logger.error(f"❌ Error entrenando Risk Assessor: {e}")
                
                if models_trained > 0:
                    self.ai_models_trained = True
                    logger.info(f"🎉 {models_trained}/4 modelos de IA entrenados exitosamente")
                else:
                    logger.warning("⚠️ No se pudo entrenar ningún modelo de IA")
            
        except Exception as e:
            logger.error(f"Error entrenando modelos de IA: {e}")
    
    def start(self):
        """Iniciar el sistema de trading"""
        if not self.initialize():
            return False
            
        self.running = True
        logger.info("▶️ Sistema de trading + Bot iniciado")
        
        try:
            while self.running:
                # Ciclo principal del trading
                self._trading_cycle()
                time.sleep(1)  # Pausa de 1 segundo
                
        except KeyboardInterrupt:
            logger.info("🛑 Sistema detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error en ciclo principal: {e}")
        finally:
            self.stop()
    
    def _trading_cycle(self):
        """Ciclo principal de trading"""
        try:
            self.cycle_count += 1
            current_time = datetime.now()
            
            # Log principal cada 30 segundos
            if current_time.second % 30 == 0:
                status_msg = f"💹 Sistema funcionando - {current_time.strftime('%H:%M:%S')}"
                if self.ai_models_trained:
                    status_msg += " | 🤖 IA ACTIVA"
                else:
                    status_msg += " | ⚠️ IA EN ENTRENAMIENTO"
                
                # ✅ ESTADO DEL BOT
                bot_status = "💕 BOT ACTIVO" if self.bot_thread and self.bot_thread.is_alive() else "❌ BOT INACTIVO"
                status_msg += f" | {bot_status}"
                
                logger.info(status_msg)
                self._update_shared_data()
            
            # Mostrar estado de componentes cada 2 minutos
            if self.cycle_count % 120 == 0:  # 120 segundos = 2 minutos
                self._show_system_status()
            
            # Verificar oportunidades de trading cada 10 segundos
            if current_time.second % 10 == 0 and self.ai_models_trained:
                self._check_trading_opportunities()
                
        except Exception as e:
            logger.error(f"Error en ciclo de trading: {e}")
    
    def _show_system_status(self):
        """Mostrar estado detallado del sistema"""
        try:
            logger.info("📊 === ESTADO DEL SISTEMA INTEGRADO ===")
            
            # Estado del pipeline
            if self.data_pipeline:
                status = self.data_pipeline.get_system_status()
                logger.info(f"🔗 MT5 Conectado: {status.get('mt5_connected', False)}")
                logger.info(f"📈 Símbolos activos: {status.get('symbols_count', 0)}")
                logger.info(f"🧵 Hilos activos: {status.get('active_threads', 0)}")
            
            # Estado de los modelos IA
            if self.ai_models_trained:
                logger.info("🤖 Modelos IA: ENTRENADOS Y ACTIVOS")
            else:
                logger.info("🤖 Modelos IA: EN PROCESO DE ENTRENAMIENTO")
            
            # ✅ ESTADO DEL BOT
            if self.bot_thread and self.bot_thread.is_alive():
                logger.info("💕 Dannis Bot: ACTIVO Y OPERACIONAL")
            else:
                logger.info("❌ Dannis Bot: INACTIVO")
            
            # Estado del aprendizaje continuo
            learning_status = continuous_learning.get_system_status()
            logger.info(f"🧠 Aprendizaje Continuo: {'ACTIVO' if learning_status['running'] else 'INACTIVO'}")
            logger.info(f"📚 Trades registrados: {learning_status['total_trades']}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error mostrando estado del sistema: {e}")
    
    def _check_trading_opportunities(self):
        """Verificar oportunidades de trading con IA"""
        try:
            if not self.ai_models_trained:
                return
            
            # Obtener datos actuales del pipeline
            if not self.data_pipeline:
                return
            
            # USAR LOS 15 SÍMBOLOS REALES DEL SISTEMA
            available_symbols = []
            
            # Obtener símbolos disponibles del data pipeline
            if hasattr(self.data_pipeline, 'all_symbols'):
                available_symbols = self.data_pipeline.all_symbols
            else:
                # Fallback - usar símbolos del buffer de datos
                if hasattr(self.data_pipeline, 'data_buffer'):
                    symbol_keys = set()
                    for key in self.data_pipeline.data_buffer.keys():
                        if '_' in key:
                            symbol = key.split('_')[0]
                            symbol_keys.add(symbol)
                    available_symbols = list(symbol_keys)
            
            if not available_symbols:
                logger.debug("⚠️ No hay símbolos disponibles para trading")
                return
            
            logger.debug(f"📊 Analizando {len(available_symbols)} símbolos: {available_symbols}")
            
            for symbol in available_symbols:
                try:
                    # Obtener datos del símbolo (M5 por defecto)
                    data = self.data_pipeline.get_latest_data(symbol, 'M5', 100)
                    if data is None or len(data) < 50:
                        continue
                    
                    # Usar modelos IA para tomar decisión
                    decision = self._get_ai_decision(symbol, data)
                    if decision and decision.get('should_trade', False):
                        logger.info(f"⚡ SEÑAL FINAL→EXECUTOR: {symbol} {decision['signal_name']} (Conf: {decision['confidence']:.2f})")
                    
                    if decision and decision.get('should_trade', False):
                        # Ejecutar trade usando TradeExecutor
                        success = trade_executor.execute_trade_decision(symbol, decision)
                        execution_result = "SUCCESS" if success else "FAILED"
                        logger.info(f"💹 RESULTADO EJECUCIÓN: {symbol} - {execution_result}")
                        if success:
                            # Registrar para aprendizaje continuo
                            self._record_trade_for_learning(symbol, decision)
                            logger.info(f"🎯 Trade ejecutado: {symbol} - {decision}")
                    
                except Exception as e:
                    logger.error(f"Error analizando {symbol}: {e}")
            
            # Log estadísticas cada minuto
            if datetime.now().second == 0:
                self._log_trading_stats()
                
        except Exception as e:
            logger.error(f"Error verificando oportunidades de trading: {e}")

    def _get_ai_decision(self, symbol: str, data) -> Dict:
        """Obtener decisión de trading usando modelos IA"""
        try:
            # 1. Obtener señal (BUY/SELL/HOLD)
            signal, signal_confidence = signal_classifier.predict(data)
            
            # Solo proceder si hay señal de BUY o SELL
            if signal == 0:  # HOLD
                return None

            # 2. Predecir profit potencial
            predicted_profit, profit_confidence = profit_predictor.predict(data)

            # 3. Predecir duración óptima
            predicted_duration, duration_confidence = duration_predictor.predict(data)

            # 4. Evaluar riesgo
            risk_level, risk_confidence = risk_assessor.predict(data)

            # 5. Calcular confianza general
            overall_confidence = (signal_confidence + profit_confidence + 
                                  duration_confidence + risk_confidence) / 4

            # 6. Filtros de calidad
            should_trade = (
                overall_confidence > 0.65 and  # Confianza mínima 65%
                abs(predicted_profit) > 0.008 and  # Profit mínimo 0.8%
                risk_level <= 1 and  # Solo riesgo BAJO o MEDIO
                signal_confidence > 0.6 and  # Confianza en señal > 60%
                predicted_duration <= 15  # Duración máxima 15 barras
            )

            decision = {
                'signal': signal,
                'confidence': overall_confidence,
                'predicted_profit': predicted_profit,
                'predicted_duration': predicted_duration,
                'risk_level': risk_level,
                'should_trade': should_trade,
                'signal_name': signal_classifier.get_signal_name(signal),
                'risk_name': risk_assessor.get_risk_name(risk_level),
                'timestamp': datetime.now(),
                'individual_confidences': {
                    'signal': signal_confidence,
                    'profit': profit_confidence,
                    'duration': duration_confidence,
                    'risk': risk_confidence
                }
            }
            
            # Log de decisión
            if should_trade:
                logger.info(f"🎯 DECISIÓN IA {symbol}: {decision['signal_name']} "
                            f"(Conf: {overall_confidence:.2f}, "
                            f"Profit: {predicted_profit:.3f}, "
                            f"Riesgo: {decision['risk_name']})")
            
            return decision

        except Exception as e:
            logger.error(f"Error obteniendo decisión IA para {symbol}: {e}")
            return None

    def _record_trade_for_learning(self, symbol: str, decision: Dict):
        """Registrar trade para aprendizaje continuo"""
        try:
            trade_data = {
                'symbol': symbol,
                'action': decision['signal_name'],
                'predicted_signal': decision['signal'],
                'predicted_profit': decision['predicted_profit'],
                'predicted_duration': decision['predicted_duration'],
                'predicted_risk': decision['risk_level'],
                'signal_confidence': decision['confidence'],
                'timestamp': decision['timestamp']
            }
            logger.debug(f"📚 Trade registrado para aprendizaje: {symbol}")
            
        except Exception as e:
            logger.error(f"Error registrando trade para aprendizaje: {e}")

    def _log_trading_stats(self):
        """Log estadísticas de trading"""
        try:
            # Estadísticas del trade executor
            executor_stats = trade_executor.get_trading_stats()
            
            # Estadísticas del order manager
            order_stats = order_manager.get_trading_statistics()
            active_orders = order_manager.get_all_active_orders()
            
            logger.info(f"📊 === ESTADÍSTICAS DE TRADING ===")
            logger.info(f"💰 Trades hoy: {executor_stats.get('daily_trades', 0)}")
            logger.info(f"📈 Total órdenes: {order_stats.get('total_orders', 0)}")
            logger.info(f"✅ Exitosas: {order_stats.get('successful_orders', 0)}")
            logger.info(f"❌ Fallidas: {order_stats.get('failed_orders', 0)}")
            logger.info(f"🏆 Win Rate: {executor_stats.get('win_rate', 0):.1f}%")
            logger.info(f"💵 Profit Total: ${order_stats.get('total_profit', 0):.2f}")
            logger.info(f"🔄 Posiciones Activas: {len(active_orders.get('active_positions', {}))}")
            logger.info(f"⏳ Órdenes Pendientes: {len(active_orders.get('pending_orders', {}))}")
            logger.info("=" * 45)
            
        except Exception as e:
            logger.error(f"Error loggeando estadísticas: {e}")
    
    def stop(self):
        """Detener el sistema"""
        self.running = False
        
        try:
            # ✅ DETENER BOT PRIMERO
            if self.bot_thread and self.bot_thread.is_alive():
                logger.info("💕 Deteniendo Dannis Bot...")
                # El bot se detendrá automáticamente cuando termine el programa principal
            
            # Detener components de trading
            logger.info("💰 Deteniendo Trading System...")
            try:
                order_manager.stop_monitoring()
            except:
                pass
            
            # Detener aprendizaje continuo
            logger.info("🧠 Deteniendo Aprendizaje Continuo...")
            continuous_learning.stop()
            
            # Detener data pipeline
            if self.data_pipeline:
                logger.info("📊 Deteniendo Data Pipeline...")
                self.data_pipeline.stop()
            
            # Guardar estados finales
            logger.info("💾 Guardando estados del sistema...")
            continuous_learning.save_trade_history()
            continuous_learning.save_model_performance()
            
        except Exception as e:
            logger.error(f"Error deteniendo componentes: {e}")

        logger.info("🔴 Sistema completo detenido (Trading + Bot)")

    def _update_shared_data(self):
        """Actualizar datos compartidos para el bot"""
        try:
            import json
            from datetime import datetime
        
            shared_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'ai_active': self.ai_models_trained,
                'system_running': self.running,
                'cycle_count': self.cycle_count,
                'bot_active': self.bot_thread and self.bot_thread.is_alive(),
                'daily_trades': 0,  # TODO: implementar contador real
                'last_update': datetime.now().isoformat()
            }
        
            with open('shared_data.json', 'w') as f:
                json.dump(shared_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error actualizando datos compartidos: {e}")    
 
def main():
    """Función principal del sistema integrado"""
    print("=" * 70)
    print("🤖💕 AI TRADING SYSTEM V2 + DANNIS BOT - INICIANDO...")
    print("=" * 70)
    
    # Verificar credenciales básicas
    mt5_login = os.getenv('MT5_LOGIN')
    if not mt5_login:
        logger.error("❌ No se encontraron credenciales MT5 en .env")
        return
    
    # ✅ VERIFICAR BOT TOKEN
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if bot_token:
        logger.info(f"💕 Bot Token configurado: {bot_token[:10]}...")
    else:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN no configurado - Bot no iniciará")
    
    logger.info(f"📊 MT5 Login: {mt5_login}")
    logger.info(f"🔧 Trading habilitado: {os.getenv('TRADING_ENABLED', 'false')}")
    
    # Mostrar configuración del sistema
    logger.info("🎯 Configuración del sistema:")
    logger.info(f"   - Trading: {'✅ HABILITADO' if os.getenv('TRADING_ENABLED') == 'true' else '❌ DESHABILITADO'}")
    logger.info(f"   - Riesgo máximo: {os.getenv('MAX_RISK_PER_TRADE', '0.01')}")
    logger.info(f"   - Trades diarios: {os.getenv('MAX_DAILY_TRADES', '50')}")
    logger.info(f"   - Bot Telegram: {'✅ CONFIGURADO' if bot_token else '❌ NO CONFIGURADO'}")
    
    # Crear e iniciar sistema integrado
    system = AITradingSystem()
    system.start()

if __name__ == "__main__":
    main()