import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class RiskAssessor:
    """
    Modelo de IA para evaluar el riesgo de un trade
    Predice: BAJO (0), MEDIO (1), ALTO (2) riesgo
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = 'models/risk_assessor.pkl'
        
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Intentar cargar modelo pre-entrenado
        self.load_model()
    
    def prepare_features(self, data):
        """
        Preparar características específicas para evaluación de riesgo
        """
        features = {}
        
        # Indicadores técnicos básicos
        features['sma_8'] = data['sma_8'].fillna(0)
        features['sma_21'] = data['sma_21'].fillna(0)
        features['sma_50'] = data['sma_50'].fillna(0)
        features['ema_8'] = data['ema_8'].fillna(0)
        features['ema_21'] = data['ema_21'].fillna(0)
        
        # MACD
        features['macd'] = data['macd'].fillna(0)
        features['macd_signal'] = data['macd_signal'].fillna(0)
        features['macd_histogram'] = data['macd_histogram'].fillna(0)
        
        # RSI
        features['rsi'] = data['rsi'].fillna(50)
        
        # Bollinger Bands
        features['bb_upper'] = data['bb_upper'].fillna(data['Close'])
        features['bb_lower'] = data['bb_lower'].fillna(data['Close'])
        features['bb_middle'] = data['bb_middle'].fillna(data['Close'])
        
        # ADX
        features['adx'] = data['adx'].fillna(25)
        features['di_plus'] = data['di_plus'].fillna(25)
        features['di_minus'] = data['di_minus'].fillna(25)
        
        # Price action
        features['price'] = data['Close']
        features['volume'] = data['Volume'].fillna(0)
        features['high'] = data['High']
        features['low'] = data['Low']
        
        # CARACTERÍSTICAS ESPECÍFICAS PARA RIESGO:
        
        # 1. VOLATILIDAD (Principal factor de riesgo)
        features['atr'] = data.get('atr', (data['High'] - data['Low']).rolling(14).mean()).fillna(0)
        features['atr_normalized'] = features['atr'] / features['price']
        features['volatility_5'] = data['Close'].rolling(5).std().fillna(0)
        features['volatility_20'] = data['Close'].rolling(20).std().fillna(0)
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20'].replace(0, 1)
        
        # 2. BOLLINGER BANDS WIDTH (Expansión = Mayor riesgo)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_expansion'] = features['bb_width'] / features['bb_width'].rolling(20).mean().replace(0, 1)
        
        # 3. POSICIÓN EN BANDAS (Extremos = Mayor riesgo)
        features['bb_position'] = (features['price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_extreme'] = np.maximum(
            np.maximum(features['bb_position'] - 0.9, 0),  # Cerca del upper
            np.maximum(0.1 - features['bb_position'], 0)   # Cerca del lower
        )
        
        # 4. RSI EXTREMOS (Sobrecompra/sobreventa = Riesgo de reversión)
        features['rsi_overbought'] = np.maximum(features['rsi'] - 70, 0)
        features['rsi_oversold'] = np.maximum(30 - features['rsi'], 0)
        features['rsi_extreme'] = features['rsi_overbought'] + features['rsi_oversold']
        features['rsi_momentum'] = features['rsi'].diff(5).abs().fillna(0)
        
        # 5. DIVERGENCIAS MACD (Divergencias = Mayor riesgo)
        features['macd_strength'] = np.abs(features['macd'] - features['macd_signal'])
        features['macd_acceleration'] = features['macd_histogram'].diff(3).abs().fillna(0)
        features['macd_divergence'] = np.abs(features['macd'].diff(10) - (features['price'].pct_change(10) * 100))
        
        # 6. TENDENCIA Y CONSISTENCIA (Falta de tendencia = Mayor riesgo)
        features['trend_strength'] = features['adx']
        features['trend_direction'] = features['di_plus'] - features['di_minus']
        features['trend_consistency'] = features['adx'].rolling(10).std().fillna(0)
        
        # 7. VOLUMEN ANÓMALO (Volumen inusual = Mayor riesgo)
        features['volume_sma'] = features['volume'].rolling(20).mean().fillna(0)
        features['volume_ratio'] = features['volume'] / features['volume_sma'].replace(0, 1)
        features['volume_spike'] = np.maximum(features['volume_ratio'] - 2, 0)  # Spikes > 2x promedio
        features['volume_dryup'] = np.maximum(0.5 - features['volume_ratio'], 0)  # Volumen < 50% promedio
        
        # 8. GAPS Y MOVIMIENTOS BRUSCOS
        features['price_gap'] = np.abs(data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['intraday_range'] = (features['high'] - features['low']) / features['price']
        features['price_acceleration'] = features['price'].pct_change(3).abs().fillna(0)
        
        # 9. SOPORTE Y RESISTENCIA (Cerca de niveles = Mayor riesgo de reversión)
        features['support_20'] = features['low'].rolling(20).min()
        features['resistance_20'] = features['high'].rolling(20).max()
        features['support_distance'] = (features['price'] - features['support_20']) / features['price']
        features['resistance_distance'] = (features['resistance_20'] - features['price']) / features['price']
        features['sr_proximity'] = np.minimum(features['support_distance'], features['resistance_distance'])
        features['near_sr'] = (features['sr_proximity'] < 0.02).astype(int)  # Dentro del 2%
        
        # 10. CORRELACIÓN Y MOMENTUM
        features['price_volume_corr'] = features['price'].pct_change(5).rolling(10).corr(
            features['volume'].pct_change(5)
        ).fillna(0)
        features['momentum_5'] = features['price'].pct_change(5).abs()
        features['momentum_consistency'] = features['price'].pct_change(3).rolling(5).std().fillna(0)
        
        # 11. MARKET MICROSTRUCTURE RISK
        features['bid_ask_spread_proxy'] = features['intraday_range']
        features['liquidity_risk'] = features['volume_dryup'] * features['bid_ask_spread_proxy']
        
        # 12. BREAKOUT RISK (Breakouts falsos = Alto riesgo)
        features['bb_breakout_upper'] = (features['price'] > features['bb_upper']).astype(int)
        features['bb_breakout_lower'] = (features['price'] < features['bb_lower']).astype(int)
        features['volume_confirmation'] = (features['volume_ratio'] > 1.5).astype(int)
        features['false_breakout_risk'] = (
            (features['bb_breakout_upper'] | features['bb_breakout_lower']) & 
            (features['volume_confirmation'] == 0)
        ).astype(int)
        
        # 13. REGIME CHANGE RISK
        features['volatility_regime_change'] = (features['volatility_ratio'] > 1.5).astype(int)
        features['trend_change'] = np.abs(features['trend_direction'].diff(5)).fillna(0)
        features['regime_instability'] = features['volatility_regime_change'] + (features['trend_change'] > 10).astype(int)
        
        # 14. TIME-BASED RISK
        if 'timestamp' in data.columns or hasattr(data.index, 'hour'):
            try:
                hour = pd.to_datetime(data.index).hour if hasattr(data.index, 'hour') else data['timestamp'].dt.hour
                # Horas de bajo volumen = Mayor riesgo de gaps
                features['low_liquidity_hours'] = ((hour >= 22) | (hour <= 2)).astype(int)
                # Overlap de sesiones = Menor riesgo
                features['session_overlap'] = ((hour >= 8) & (hour <= 10)).astype(int) | ((hour >= 15) & (hour <= 17)).astype(int)
                features['weekend_proximity'] = 0  # Simplificado
            except:
                features['low_liquidity_hours'] = 0
                features['session_overlap'] = 1
                features['weekend_proximity'] = 0
        else:
            features['low_liquidity_hours'] = 0
            features['session_overlap'] = 1
            features['weekend_proximity'] = 0
        
        # 15. RISK SCORE COMPOSITE
        features['volatility_score'] = (
            features['atr_normalized'] * 100 + 
            features['bb_width'] * 50 + 
            features['volatility_ratio'] * 10
        )
        
        features['momentum_risk_score'] = (
            features['rsi_extreme'] + 
            features['price_acceleration'] * 100 + 
            features['momentum_consistency'] * 100
        )
        
        features['liquidity_risk_score'] = (
            features['volume_spike'] + 
            features['volume_dryup'] * 10 + 
            features['liquidity_risk'] * 100
        )
        
        # Convertir a DataFrame
        df_features = pd.DataFrame(features)
        
        # Limpiar valores infinitos y NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features
    
    def calculate_risk_labels(self, data, entry_points, risk_periods=[3, 5, 10]):
        """
        Calcular etiquetas de riesgo basadas en drawdown máximo
        """
        risk_labels = []
        close_prices = data['Close'].values
        
        for entry_idx in entry_points:
            if entry_idx >= len(close_prices) - max(risk_periods):
                risk_labels.append(1)  # Riesgo medio por defecto
                continue
            
            entry_price = close_prices[entry_idx]
            max_drawdown = 0
            
            # Calcular máximo drawdown en diferentes períodos
            for period in risk_periods:
                for i in range(1, period + 1):
                    if entry_idx + i >= len(close_prices):
                        break
                    
                    current_price = close_prices[entry_idx + i]
                    drawdown = abs((current_price - entry_price) / entry_price)
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Clasificar riesgo basado en drawdown máximo
            if max_drawdown < 0.005:      # < 0.5%
                risk_label = 0  # BAJO
            elif max_drawdown < 0.015:    # < 1.5%
                risk_label = 1  # MEDIO
            else:                         # > 1.5%
                risk_label = 2  # ALTO
            
            risk_labels.append(risk_label)
        
        return np.array(risk_labels)
    
    def train(self, data_dict):
        """
        Entrenar el modelo con datos históricos
        """
        logger.info("🛡️ Entrenando Risk Assessor...")
        
        all_features = []
        all_risk_labels = []
        
        for symbol, data in data_dict.items():
            try:
                # Preparar características
                features = self.prepare_features(data)
                
                # Crear puntos de entrada
                entry_points = list(range(0, len(data) - 15, 5))
                
                # Calcular etiquetas de riesgo
                risk_labels = self.calculate_risk_labels(data, entry_points)
                
                # Seleccionar características correspondientes
                feature_subset = features.iloc[entry_points]
                
                # Asegurar misma longitud
                min_len = min(len(feature_subset), len(risk_labels))
                feature_subset = feature_subset.iloc[:min_len]
                risk_labels = risk_labels[:min_len]
                
                all_features.append(feature_subset)
                all_risk_labels.extend(risk_labels)
                
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {e}")
                continue
        
        if not all_features:
            logger.error("No se pudieron preparar características")
            return False
        
        # Combinar todas las características
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_risk_labels)
        
        # Verificar distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Distribución de riesgo - Bajo: {counts[0] if 0 in unique else 0}, "
                   f"Medio: {counts[1] if 1 in unique else 0}, "
                   f"Alto: {counts[2] if 2 in unique else 0}")
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelos
        scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                score = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy').mean()
                scores[name] = score
                logger.info(f"Modelo {name}: Accuracy = {score:.3f}")
            except Exception as e:
                logger.error(f"Error entrenando {name}: {e}")
                scores[name] = 0
        
        self.is_trained = True
        
        # Guardar modelo
        self.save_model()
        
        best_score = max(scores.values())
        logger.info(f"✅ Risk Assessor entrenado. Mejor accuracy: {best_score:.3f}")
        return True
    
    def predict(self, features):
        """
        Predecir nivel de riesgo
        """
        if not self.is_trained:
            return 1, 0.5  # Riesgo medio por defecto
        
        try:
            # Preparar características
            feature_df = self.prepare_features(features)
            
            # Asegurar que tenemos todas las características necesarias
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            # Ordenar columnas
            feature_df = feature_df[self.feature_names]
            
            # Escalar
            X_scaled = self.scaler.transform(feature_df.iloc[-1:])
            
            # Predicciones de todos los modelos
            predictions = []
            confidences = []
            
            for model in self.models.values():
                try:
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    confidence = np.max(prob)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                except:
                    predictions.append(1)  # Riesgo medio
                    confidences.append(0.5)
            
            # Predicción final (voting conservador - elegir el riesgo más alto)
            final_risk = max(predictions)  # Conservador
            final_confidence = np.mean(confidences)
            
            return int(final_risk), float(final_confidence)
            
        except Exception as e:
            logger.error(f"Error en evaluación de riesgo: {e}")
            return 1, 0.5
    
    def get_risk_name(self, risk_level):
        """Convertir número a nombre de riesgo"""
        risk_names = {0: 'BAJO', 1: 'MEDIO', 2: 'ALTO'}
        return risk_names.get(risk_level, 'DESCONOCIDO')
    
    def save_model(self):
        """Guardar modelo entrenado"""
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Risk Assessor guardado en {self.model_path}")
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    def load_model(self):
        """Cargar modelo pre-entrenado"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = model_data['is_trained']
                logger.info("Risk Assessor cargado exitosamente")
                return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
        
        return False

# Instancia global
risk_assessor = RiskAssessor()