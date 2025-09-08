import numpy as np
import pandas as pd
import logging
import threading
import time
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional

# Importar nuestros modelos IA
from .signal_classifier import signal_classifier
from .profit_predictor import profit_predictor
from .duration_predictor import duration_predictor
from .risk_assessor import risk_assessor

logger = logging.getLogger(__name__)

class ContinuousLearning:
    """
    Sistema de Aprendizaje Continuo para mejorar los modelos de IA automáticamente
    """
    
    def __init__(self):
        self.is_running = False
        self.learning_thread = None
        self.trade_history = []
        self.model_performance = {
            'signal_classifier': {'accuracy': 0, 'last_updated': None, 'predictions': 0},
            'profit_predictor': {'mse': float('inf'), 'last_updated': None, 'predictions': 0},
            'duration_predictor': {'mse': float('inf'), 'last_updated': None, 'predictions': 0},
            'risk_assessor': {'accuracy': 0, 'last_updated': None, 'predictions': 0}
        }
        
        # Configuración de aprendizaje
        self.min_trades_for_retrain = 50  # Mínimo trades para re-entrenar
        self.retrain_interval_hours = 24   # Re-entrenar cada 24 horas
        self.performance_threshold = 0.6   # Umbral mínimo de performance
        
        # Almacenamiento de datos
        self.data_storage_path = 'data/learning'
        os.makedirs(self.data_storage_path, exist_ok=True)
        
        # Cargar historial previo
        self.load_trade_history()
        self.load_model_performance()
    
    def start(self):
        """Iniciar sistema de aprendizaje continuo"""
        if self.is_running:
            logger.warning("Sistema de aprendizaje ya está ejecutándose")
            return
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("🧠 Sistema de Aprendizaje Continuo iniciado")
    
    def stop(self):
        """Detener sistema de aprendizaje"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("🧠 Sistema de Aprendizaje Continuo detenido")
    
    def record_trade(self, trade_data: Dict):
        """
        Registrar un trade para aprendizaje futuro
        """
        try:
            # Estructura del trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_data.get('symbol'),
                'action': trade_data.get('action'),  # BUY/SELL
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'actual_profit': trade_data.get('actual_profit'),  # % real
                'actual_duration': trade_data.get('actual_duration'),  # barras reales
                'actual_risk': trade_data.get('actual_risk'),  # drawdown máximo real
                
                # Predicciones IA (para evaluar performance)
                'predicted_signal': trade_data.get('predicted_signal'),
                'predicted_profit': trade_data.get('predicted_profit'),
                'predicted_duration': trade_data.get('predicted_duration'),
                'predicted_risk': trade_data.get('predicted_risk'),
                
                # Confianzas
                'signal_confidence': trade_data.get('signal_confidence', 0),
                'profit_confidence': trade_data.get('profit_confidence', 0),
                'duration_confidence': trade_data.get('duration_confidence', 0),
                'risk_confidence': trade_data.get('risk_confidence', 0),
                
                # Features utilizadas (para re-entrenamiento)
                'features': trade_data.get('features', {}),
                'market_data': trade_data.get('market_data', {}),
                
                # Resultado del trade
                'trade_result': 'WIN' if trade_data.get('actual_profit', 0) > 0 else 'LOSS',
                'strategy_used': trade_data.get('strategy', 'unknown')
            }
            
            self.trade_history.append(trade_record)
            
            # Guardar cada 10 trades
            if len(self.trade_history) % 10 == 0:
                self.save_trade_history()
            
            # Evaluar performance inmediatamente
            self._evaluate_predictions(trade_record)
            
            logger.info(f"📊 Trade registrado para aprendizaje: {trade_data.get('symbol')} {trade_data.get('action')}")
            
        except Exception as e:
            logger.error(f"Error registrando trade para aprendizaje: {e}")
    
    def _evaluate_predictions(self, trade_record: Dict):
        """
        Evaluar la precisión de las predicciones vs realidad
        """
        try:
            # Evaluar Signal Classifier
            if trade_record.get('predicted_signal') is not None:
                predicted_signal = trade_record['predicted_signal']
                actual_result = 1 if trade_record['trade_result'] == 'WIN' else 0
                
                # Convertir señal a resultado esperado
                if predicted_signal == 1:  # BUY prediction
                    expected_result = 1 if trade_record.get('actual_profit', 0) > 0 else 0
                elif predicted_signal == 2:  # SELL prediction
                    expected_result = 1 if trade_record.get('actual_profit', 0) > 0 else 0
                else:  # HOLD
                    expected_result = 0
                
                # Actualizar accuracy
                self._update_model_accuracy('signal_classifier', expected_result == actual_result)
            
            # Evaluar Profit Predictor
            if trade_record.get('predicted_profit') is not None and trade_record.get('actual_profit') is not None:
                predicted = trade_record['predicted_profit']
                actual = trade_record['actual_profit']
                error = (predicted - actual) ** 2
                self._update_model_mse('profit_predictor', error)
            
            # Evaluar Duration Predictor
            if trade_record.get('predicted_duration') is not None and trade_record.get('actual_duration') is not None:
                predicted = trade_record['predicted_duration']
                actual = trade_record['actual_duration']
                error = (predicted - actual) ** 2
                self._update_model_mse('duration_predictor', error)
            
            # Evaluar Risk Assessor
            if trade_record.get('predicted_risk') is not None and trade_record.get('actual_risk') is not None:
                predicted_risk = trade_record['predicted_risk']
                actual_risk = trade_record['actual_risk']
                
                # Convertir riesgo real a categoría
                if actual_risk < 0.005:
                    actual_risk_category = 0  # BAJO
                elif actual_risk < 0.015:
                    actual_risk_category = 1  # MEDIO
                else:
                    actual_risk_category = 2  # ALTO
                
                correct = (predicted_risk == actual_risk_category)
                self._update_model_accuracy('risk_assessor', correct)
            
        except Exception as e:
            logger.error(f"Error evaluando predicciones: {e}")
    
    def _update_model_accuracy(self, model_name: str, correct: bool):
        """Actualizar accuracy de un modelo"""
        try:
            current_accuracy = self.model_performance[model_name]['accuracy']
            predictions = self.model_performance[model_name]['predictions']
            
            # Promedio móvil de accuracy
            new_accuracy = (current_accuracy * predictions + (1 if correct else 0)) / (predictions + 1)
            
            self.model_performance[model_name]['accuracy'] = new_accuracy
            self.model_performance[model_name]['predictions'] += 1
            self.model_performance[model_name]['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error actualizando accuracy de {model_name}: {e}")
    
    def _update_model_mse(self, model_name: str, error: float):
        """Actualizar MSE de un modelo"""
        try:
            current_mse = self.model_performance[model_name]['mse']
            predictions = self.model_performance[model_name]['predictions']
            
            # Promedio móvil de MSE
            if predictions == 0:
                new_mse = error
            else:
                new_mse = (current_mse * predictions + error) / (predictions + 1)
            
            self.model_performance[model_name]['mse'] = new_mse
            self.model_performance[model_name]['predictions'] += 1
            self.model_performance[model_name]['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error actualizando MSE de {model_name}: {e}")
    
    def _learning_loop(self):
        """Loop principal de aprendizaje continuo"""
        logger.info("🔄 Iniciando loop de aprendizaje continuo")
        
        while self.is_running:
            try:
                # Verificar si es momento de re-entrenar
                self._check_retrain_conditions()
                
                # Limpiar datos antiguos
                self._cleanup_old_data()
                
                # Generar reporte de performance
                self._generate_performance_report()
                
                # Esperar antes del siguiente ciclo
                time.sleep(3600)  # Revisar cada hora
                
            except Exception as e:
                logger.error(f"Error en loop de aprendizaje: {e}")
                time.sleep(60)  # Esperar 1 minuto antes de reintentar
    
    def _check_retrain_conditions(self):
        """Verificar si es necesario re-entrenar modelos"""
        try:
            current_time = datetime.now()
            
            # Verificar si tenemos suficientes trades
            if len(self.trade_history) < self.min_trades_for_retrain:
                return
            
            for model_name, performance in self.model_performance.items():
                should_retrain = False
                
                # Verificar tiempo desde último entrenamiento
                if performance['last_updated']:
                    last_update = datetime.fromisoformat(performance['last_updated'])
                    hours_since_update = (current_time - last_update).total_seconds() / 3600
                    
                    if hours_since_update > self.retrain_interval_hours:
                        should_retrain = True
                        logger.info(f"⏰ {model_name}: {hours_since_update:.1f}h desde última actualización")
                
                # Verificar performance pobre
                if model_name in ['signal_classifier', 'risk_assessor']:
                    if performance['accuracy'] < self.performance_threshold:
                        should_retrain = True
                        logger.info(f"📉 {model_name}: Accuracy baja {performance['accuracy']:.3f}")
                
                # Re-entrenar si es necesario
                if should_retrain:
                    self._retrain_model(model_name)
            
        except Exception as e:
            logger.error(f"Error verificando condiciones de re-entrenamiento: {e}")
    
    def _retrain_model(self, model_name: str):
        """Re-entrenar un modelo específico"""
        try:
            logger.info(f"🔄 Re-entrenando {model_name}...")
            
            # Preparar datos de entrenamiento desde historial
            training_data = self._prepare_training_data()
            
            if not training_data:
                logger.warning(f"No hay datos suficientes para re-entrenar {model_name}")
                return
            
            # Re-entrenar modelo específico
            success = False
            
            if model_name == 'signal_classifier':
                success = signal_classifier.retrain_incremental(training_data)
            elif model_name == 'profit_predictor':
                success = profit_predictor.train(training_data)
            elif model_name == 'duration_predictor':
                success = duration_predictor.train(training_data)
            elif model_name == 'risk_assessor':
                success = risk_assessor.train(training_data)
            
            if success:
                self.model_performance[model_name]['last_updated'] = datetime.now().isoformat()
                logger.info(f"✅ {model_name} re-entrenado exitosamente")
            else:
                logger.error(f"❌ Error re-entrenando {model_name}")
            
        except Exception as e:
            logger.error(f"Error re-entrenando {model_name}: {e}")
    
    def _prepare_training_data(self) -> Dict:
        """Preparar datos de entrenamiento desde el historial"""
        try:
            # Obtener trades recientes (últimos 1000)
            recent_trades = self.trade_history[-1000:] if len(self.trade_history) > 1000 else self.trade_history
            
            # Agrupar por símbolo
            training_data = {}
            
            for trade in recent_trades:
                symbol = trade.get('symbol')
                if not symbol:
                    continue
                
                # Reconstruir datos de mercado si están disponibles
                market_data = trade.get('market_data', {})
                if market_data:
                    if symbol not in training_data:
                        training_data[symbol] = []
                    training_data[symbol].append(market_data)
            
            # Convertir a DataFrames
            symbol_dataframes = {}
            for symbol, data_list in training_data.items():
                if len(data_list) > 10:  # Mínimo 10 puntos de datos
                    try:
                        df = pd.DataFrame(data_list)
                        if not df.empty:
                            symbol_dataframes[symbol] = df
                    except Exception as e:
                        logger.error(f"Error creando DataFrame para {symbol}: {e}")
            
            return symbol_dataframes
            
        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {e}")
            return {}
    
    def _cleanup_old_data(self):
        """Limpiar datos antiguos para evitar sobreuso de memoria"""
        try:
            # Mantener solo los últimos 5000 trades
            if len(self.trade_history) > 5000:
                self.trade_history = self.trade_history[-5000:]
                logger.info("🧹 Limpieza de datos antiguos completada")
            
        except Exception as e:
            logger.error(f"Error limpiando datos antiguos: {e}")
    
    def _generate_performance_report(self):
        """Generar reporte de performance de modelos"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_trades': len(self.trade_history),
                'models_performance': self.model_performance.copy()
            }
            
            # Guardar reporte
            report_path = os.path.join(self.data_storage_path, 'performance_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log cada 6 horas
            current_hour = datetime.now().hour
            if current_hour % 6 == 0:
                logger.info(f"📊 Reporte de Performance - Trades: {len(self.trade_history)}")
                for model, perf in self.model_performance.items():
                    if 'accuracy' in perf:
                        logger.info(f"  {model}: Accuracy {perf['accuracy']:.3f} ({perf['predictions']} predicciones)")
                    else:
                        logger.info(f"  {model}: MSE {perf['mse']:.3f} ({perf['predictions']} predicciones)")
            
        except Exception as e:
            logger.error(f"Error generando reporte de performance: {e}")
    
    def save_trade_history(self):
        """Guardar historial de trades"""
        try:
            history_path = os.path.join(self.data_storage_path, 'trade_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")
    
    def load_trade_history(self):
        """Cargar historial de trades previo"""
        try:
            history_path = os.path.join(self.data_storage_path, 'trade_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"📚 Cargado historial de {len(self.trade_history)} trades")
        except Exception as e:
            logger.error(f"Error cargando historial: {e}")
    
    def save_model_performance(self):
        """Guardar performance de modelos"""
        try:
            perf_path = os.path.join(self.data_storage_path, 'model_performance.json')
            with open(perf_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando performance: {e}")
    
    def load_model_performance(self):
        """Cargar performance de modelos previo"""
        try:
            perf_path = os.path.join(self.data_storage_path, 'model_performance.json')
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    self.model_performance = json.load(f)
                logger.info("📈 Performance de modelos cargada")
        except Exception as e:
            logger.error(f"Error cargando performance: {e}")
    
    def get_system_status(self) -> Dict:
        """Obtener estado del sistema de aprendizaje"""
        return {
            'running': self.is_running,
            'total_trades': len(self.trade_history),
            'models_performance': self.model_performance,
            'last_cleanup': datetime.now().isoformat()
        }

# Instancia global
continuous_learning = ContinuousLearning()