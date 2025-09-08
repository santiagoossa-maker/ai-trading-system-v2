import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DurationPredictor:
    """
    Modelo de IA para predecir la duración óptima de un trade
    Predice: Número de barras/períodos que debe durar el trade
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = 'models/duration_predictor.pkl'
        
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Intentar cargar modelo pre-entrenado
        self.load_model()
    
    def prepare_features(self, data):
        """
        Preparar características específicas para predicción de duración
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
        
        # CARACTERÍSTICAS ESPECÍFICAS PARA DURACIÓN:
        
        # Volatilidad (alta volatilidad = trades más cortos)
        features['atr'] = data.get('atr', (data['High'] - data['Low']).rolling(14).mean()).fillna(0)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['volatility_20'] = data['Close'].rolling(20).std().fillna(0)
        features['volatility_5'] = data['Close'].rolling(5).std().fillna(0)
        
        # Momentum strength (momentum fuerte = duración más larga)
        features['roc_5'] = data['Close'].pct_change(5).fillna(0) * 100
        features['roc_10'] = data['Close'].pct_change(10).fillna(0) * 100
        features['roc_20'] = data['Close'].pct_change(20).fillna(0) * 100
        
        # Trend strength indicators
        features['sma_slope_8'] = (features['sma_8'] - features['sma_8'].shift(5)) / features['sma_8'].shift(5)
        features['sma_slope_21'] = (features['sma_21'] - features['sma_21'].shift(5)) / features['sma_21'].shift(5)
        features['ema_slope_8'] = (features['ema_8'] - features['ema_8'].shift(5)) / features['ema_8'].shift(5)
        
        # Price position (posición en canal = duración)
        features['price_bb_position'] = (features['price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['price_channel_position'] = (features['price'] - features['low'].rolling(20).min()) / (features['high'].rolling(20).max() - features['low'].rolling(20).min())
        
        # Volume analysis (volumen alto = cambios rápidos)
        features['volume_sma_20'] = features['volume'].rolling(20).mean().fillna(0)
        features['volume_ratio'] = features['volume'] / features['volume_sma_20'].replace(0, 1)
        features['volume_momentum'] = features['volume'].pct_change(5).fillna(0)
        
        # Market regime indicators
        features['trend_consistency'] = np.abs(features['sma_slope_8'] - features['sma_slope_21'])
        features['momentum_consistency'] = np.abs(features['roc_5'] - features['roc_10'])
        
        # RSI patterns (RSI extremos = reversiones rápidas)
        features['rsi_extreme'] = np.maximum(np.maximum(features['rsi'] - 70, 0), np.maximum(30 - features['rsi'], 0))
        features['rsi_momentum'] = features['rsi'].diff(5).fillna(0)
        
        # MACD patterns
        features['macd_strength'] = np.abs(features['macd'] - features['macd_signal'])
        features['macd_acceleration'] = features['macd_histogram'].diff(3).fillna(0)
        
        # Support/Resistance proximity (cerca de S/R = reversiones rápidas)
        features['support_20'] = features['low'].rolling(20).min()
        features['resistance_20'] = features['high'].rolling(20).max()
        features['support_distance'] = (features['price'] - features['support_20']) / features['price']
        features['resistance_distance'] = (features['resistance_20'] - features['price']) / features['price']
        features['sr_proximity'] = np.minimum(features['support_distance'], features['resistance_distance'])
        
        # Breakout indicators (breakouts = duración más larga)
        features['price_above_bb_upper'] = (features['price'] > features['bb_upper']).astype(int)
        features['price_below_bb_lower'] = (features['price'] < features['bb_lower']).astype(int)
        features['volume_breakout'] = (features['volume'] > features['volume_sma_20'] * 1.5).astype(int)
        
        # Time decay factors
        features['adx_strength'] = np.where(features['adx'] > 25, features['adx'] - 25, 0)
        features['trend_duration_estimate'] = features['adx_strength'] * features['trend_consistency']
        
        # Correlation patterns (divergencias = cambios rápidos)
        price_change = features['price'].pct_change(5)
        volume_change = features['volume'].pct_change(5)
        features['price_volume_correlation'] = price_change.rolling(10).corr(volume_change).fillna(0)
        
        # Market microstructure
        features['bid_ask_proxy'] = (features['high'] - features['low']) / features['price']  # Proxy para spread
        features['price_efficiency'] = features['price'].diff().abs().rolling(10).mean().fillna(0)
        
        # Seasonal patterns (hora del día afecta duración)
        if 'timestamp' in data.columns or hasattr(data.index, 'hour'):
            try:
                hour = pd.to_datetime(data.index).hour if hasattr(data.index, 'hour') else data['timestamp'].dt.hour
                features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                # Sesiones de trading (algunas son más activas)
                features['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
                features['european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
                features['us_session'] = ((hour >= 16) & (hour < 24)).astype(int)
            except:
                features['hour_sin'] = 0
                features['hour_cos'] = 1
                features['asian_session'] = 0
                features['european_session'] = 0
                features['us_session'] = 1
        else:
            features['hour_sin'] = 0
            features['hour_cos'] = 1
            features['asian_session'] = 0
            features['european_session'] = 0
            features['us_session'] = 1
        
        # Convertir a DataFrame
        df_features = pd.DataFrame(features)
        
        # Limpiar valores infinitos y NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features
    
    def calculate_optimal_duration(self, data, entry_points, max_duration=30):
        """
        Calcular la duración óptima para cada trade
        """
        durations = []
        close_prices = data['Close'].values
        
        for entry_idx in entry_points:
            if entry_idx >= len(close_prices) - max_duration:
                durations.append(5)  # Duración por defecto
                continue
            
            entry_price = close_prices[entry_idx]
            best_duration = 5
            best_profit = 0
            
            # Buscar la duración que maximiza el profit
            for duration in range(1, max_duration + 1):
                exit_idx = entry_idx + duration
                if exit_idx >= len(close_prices):
                    break
                
                exit_price = close_prices[exit_idx]
                profit = abs((exit_price - entry_price) / entry_price)
                
                # Penalizar duraciones muy largas
                adjusted_profit = profit - (duration * 0.001)  # Penalty por tiempo
                
                if adjusted_profit > best_profit:
                    best_profit = adjusted_profit
                    best_duration = duration
            
            # Limitar duración entre 1 y 20 barras
            best_duration = max(1, min(20, best_duration))
            durations.append(best_duration)
        
        return np.array(durations)
    
    def train(self, data_dict):
        """
        Entrenar el modelo con datos históricos
        """
        logger.info("⏱️ Entrenando Duration Predictor...")
        
        all_features = []
        all_durations = []
        
        for symbol, data in data_dict.items():
            try:
                # Preparar características
                features = self.prepare_features(data)
                
                # Crear puntos de entrada
                entry_points = list(range(0, len(data) - 35, 7))  # Cada 7 barras
                
                # Calcular duraciones óptimas
                durations = self.calculate_optimal_duration(data, entry_points)
                
                # Seleccionar características correspondientes
                feature_subset = features.iloc[entry_points]
                
                # Asegurar misma longitud
                min_len = min(len(feature_subset), len(durations))
                feature_subset = feature_subset.iloc[:min_len]
                durations = durations[:min_len]
                
                all_features.append(feature_subset)
                all_durations.extend(durations)
                
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {e}")
                continue
        
        if not all_features:
            logger.error("No se pudieron preparar características")
            return False
        
        # Combinar todas las características
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_durations)
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelos
        scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                score = cross_val_score(model, X_scaled, y, cv=3, scoring='r2').mean()
                scores[name] = score
                logger.info(f"Modelo {name}: R² Score = {score:.3f}")
            except Exception as e:
                logger.error(f"Error entrenando {name}: {e}")
                scores[name] = -1
        
        self.is_trained = True
        
        # Guardar modelo
        self.save_model()
        
        best_score = max(scores.values())
        logger.info(f"✅ Duration Predictor entrenado. Mejor R² score: {best_score:.3f}")
        return True
    
    def predict(self, features):
        """
        Predecir duración óptima del trade
        """
        if not self.is_trained:
            return 5, 0.5  # Duración por defecto
        
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
            for model in self.models.values():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions.append(max(1, min(20, pred)))  # Limitar entre 1 y 20
                except:
                    predictions.append(5)
            
            # Predicción final (mediana para robustez)
            final_duration = int(np.median(predictions))
            
            # Confianza basada en concordancia
            std_dev = np.std(predictions)
            confidence = max(0.1, 1 - (std_dev / 5))  # Normalizar por duración típica
            
            return final_duration, float(confidence)
            
        except Exception as e:
            logger.error(f"Error en predicción de duración: {e}")
            return 5, 0.5
    
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
            logger.info(f"Duration Predictor guardado en {self.model_path}")
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
                logger.info("Duration Predictor cargado exitosamente")
                return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
        
        return False

# Instancia global
duration_predictor = DurationPredictor()