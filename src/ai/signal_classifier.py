import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class SignalClassifier:
    """
    Modelo de IA para clasificar señales de trading
    Predice: BUY, SELL, HOLD basado en indicadores técnicos
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = 'models/signal_classifier.pkl'
        
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Intentar cargar modelo pre-entrenado
        self.load_model()
    
    def prepare_features(self, data):
        """
        Preparar características para el modelo
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
        
        # Stochastic
        features['stoch_k'] = data['stoch_k'].fillna(50)
        features['stoch_d'] = data['stoch_d'].fillna(50)
        
        # Price action features
        features['price'] = data['Close']
        features['volume'] = data['Volume'].fillna(0)
        features['high'] = data['High']
        features['low'] = data['Low']
        features['open'] = data['Open']
        
        # Ratios y diferencias
        features['price_sma8_ratio'] = features['price'] / features['sma_8'].replace(0, 1)
        features['price_sma21_ratio'] = features['price'] / features['sma_21'].replace(0, 1)
        features['price_sma50_ratio'] = features['price'] / features['sma_50'].replace(0, 1)
        
        # Volatilidad
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['price_bb_position'] = (features['price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Momentum
        features['rsi_momentum'] = features['rsi'] - 50
        features['macd_momentum'] = features['macd'] - features['macd_signal']
        
        # Convertir a DataFrame
        df_features = pd.DataFrame(features)
        
        # Limpiar valores infinitos y NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        return df_features
    
    def generate_labels(self, data, lookahead_periods=5):
        """
        Generar etiquetas para entrenamiento
        0 = HOLD, 1 = BUY, 2 = SELL
        """
        labels = []
        close_prices = data['Close'].values
        
        for i in range(len(close_prices)):
            if i + lookahead_periods >= len(close_prices):
                labels.append(0)  # HOLD para los últimos datos
                continue
            
            current_price = close_prices[i]
            future_price = close_prices[i + lookahead_periods]
            
            price_change = (future_price - current_price) / current_price
            
            # Thresholds para clasificación
            buy_threshold = 0.001   # 0.1%
            sell_threshold = -0.001 # -0.1%
            
            if price_change > buy_threshold:
                labels.append(1)  # BUY
            elif price_change < sell_threshold:
                labels.append(2)  # SELL
            else:
                labels.append(0)  # HOLD
        
        return np.array(labels)
    
    def train(self, data_dict):
        """
        Entrenar el modelo con datos históricos
        """
        logger.info("🧠 Entrenando Signal Classifier...")
        
        all_features = []
        all_labels = []
        
        for symbol, data in data_dict.items():
            try:
                # Preparar características
                features = self.prepare_features(data)
                labels = self.generate_labels(data)
                
                # Asegurar que tengan la misma longitud
                min_len = min(len(features), len(labels))
                features = features.iloc[:min_len]
                labels = labels[:min_len]
                
                all_features.append(features)
                all_labels.extend(labels)
                
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {e}")
                continue
        
        if not all_features:
            logger.error("No se pudieron preparar características")
            return False
        
        # Combinar todas las características
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        
        logger.info(f"Distribución de señales en entrenamiento: {pd.Series(y).value_counts().to_dict()}")
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()

        # Escalar características
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar modelos
        scores = {}
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            score = cross_val_score(model, X_scaled, y, cv=3).mean()
            scores[name] = score
            logger.info(f"Modelo {name}: Score = {score:.3f}")

        # <---- AQUI:
        y_pred_rf = self.models['rf'].predict(X_scaled)
        y_pred_gb = self.models['gb'].predict(X_scaled)
        from collections import Counter
        logger.info(f"PRED RF: {Counter(y_pred_rf)}")
        logger.info(f"PRED GB: {Counter(y_pred_gb)}")
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelos
        scores = {}
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            score = cross_val_score(model, X_scaled, y, cv=3).mean()
            scores[name] = score
            logger.info(f"Modelo {name}: Score = {score:.3f}")
        
        for i in range(1, 11):
            idx = -i
            feats = X_scaled[idx:idx+1]
            pred_rf = self.models['rf'].predict(feats)[0]
            pred_gb = self.models['gb'].predict(feats)[0]
            real = y[idx]
            logger.info(f"ULTIMA[{i}]: TARGET={real}, RF={pred_rf}, GB={pred_gb}")
        
        self.is_trained = True
        
        # Guardar modelo
        self.save_model()
        
        logger.info(f"✅ Signal Classifier entrenado. Mejor score: {max(scores.values()):.3f}")
        return True
    
    def predict(self, features):
        """
        Predecir señal para nuevos datos
        """
        if not self.is_trained:
            return 0, 0.0  # HOLD con confianza 0
        
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
            
            # Predicciones de ambos modelos
            predictions = []
            confidences = []
            
            for model in self.models.values():
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                confidence = np.max(prob)
                
                predictions.append(pred)
                confidences.append(confidence)
            
            # Predicción final (voting)
            final_prediction = max(set(predictions), key=predictions.count)
            final_confidence = np.mean(confidences)
            
            return int(final_prediction), float(final_confidence)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return 0, 0.0
    
    def get_signal_name(self, signal):
        """Convertir número a nombre de señal"""
        signals = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return signals.get(signal, 'UNKNOWN')
    
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
            logger.info(f"Modelo guardado en {self.model_path}")
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
                logger.info("Modelo cargado exitosamente")
                return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
        
        return False
    
    def retrain_incremental(self, new_data):
        """
        Re-entrenar con nuevos datos (aprendizaje continuo)
        """
        if not self.is_trained:
            return self.train(new_data)
        
        try:
            # Preparar nuevos datos
            features = self.prepare_features(new_data)
            labels = self.generate_labels(new_data)
            
            # Actualizar modelos (simplificado - en producción usar partial_fit)
            X_scaled = self.scaler.transform(features)
            
            for model in self.models.values():
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_scaled, labels)
            
            # Guardar modelo actualizado
            self.save_model()
            
            logger.info("✅ Modelo actualizado con nuevos datos")
            return True
            
        except Exception as e:
            logger.error(f"Error en re-entrenamiento: {e}")
            return False

# Instancia global
signal_classifier = SignalClassifier()