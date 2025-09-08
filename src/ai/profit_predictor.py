import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ProfitPredictor:
    """
    Modelo de IA para predecir el profit potencial de un trade
    Predice: Ganancia/pérdida esperada en porcentaje
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = 'models/profit_predictor.pkl'
        
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Intentar cargar modelo pre-entrenado
        self.load_model()
    
    def prepare_features(self, data):
        """
        Preparar características para predecir profit
        """
        features = {}
        
        # Indicadores técnicos
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
        
        # ADX - Strength of trend
        features['adx'] = data['adx'].fillna(25)
        features['di_plus'] = data['di_plus'].fillna(25)
        features['di_minus'] = data['di_minus'].fillna(25)
        
        # Stochastic
        features['stoch_k'] = data['stoch_k'].fillna(50)
        features['stoch_d'] = data['stoch_d'].fillna(50)
        
        # Price action
        features['price'] = data['Close']
        features['volume'] = data['Volume'].fillna(0)
        features['high'] = data['High']
        features['low'] = data['Low']
        features['open'] = data['Open']
        
        # CARACTERÍSTICAS ESPECÍFICAS PARA PROFIT:
        
        # Volatilidad (crucial para profit prediction)
        features['atr'] = data.get('atr', (data['High'] - data['Low']).rolling(14).mean()).fillna(0)
        features['volatility'] = data['Close'].rolling(20).std().fillna(0)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # Momentum indicators
        features['roc'] = data['Close'].pct_change(10).fillna(0) * 100  # Rate of Change
        features['momentum'] = data['Close'] / data['Close'].shift(10).fillna(data['Close']) - 1
        
        # Price position indicators
        features['price_bb_position'] = (features['price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['price_sma8_ratio'] = features['price'] / features['sma_8'].replace(0, 1)
        features['price_sma21_ratio'] = features['price'] / features['sma_21'].replace(0, 1)
        features['price_sma50_ratio'] = features['price'] / features['sma_50'].replace(0, 1)
        
        # Trend strength
        features['sma8_sma21_ratio'] = features['sma_8'] / features['sma_21'].replace(0, 1)
        features['sma21_sma50_ratio'] = features['sma_21'] / features['sma_50'].replace(0, 1)
        
        # Volume analysis
        features['volume_sma'] = features['volume'].rolling(20).mean().fillna(0)
        features['volume_ratio'] = features['volume'] / features['volume_sma'].replace(0, 1)
        
        # Market regime indicators
        features['trend_strength'] = np.abs(features['sma8_sma21_ratio'] - 1) * 100
        features['market_volatility'] = features['bb_width'] * 100
        
        # Oscillator divergences
        features['rsi_oversold'] = np.where(features['rsi'] < 30, 30 - features['rsi'], 0)
        features['rsi_overbought'] = np.where(features['rsi'] > 70, features['rsi'] - 70, 0)
        
        # MACD signals
        features['macd_bullish'] = np.where(features['macd'] > features['macd_signal'], 
                                           features['macd'] - features['macd_signal'], 0)
        features['macd_bearish'] = np.where(features['macd'] < features['macd_signal'], 
                                           features['macd_signal'] - features['macd'], 0)
        
        # Support/Resistance levels (simplified)
        features['high_20'] = data['High'].rolling(20).max().fillna(data['High'])
        features['low_20'] = data['Low'].rolling(20).min().fillna(data['Low'])
        features['price_support_distance'] = (features['price'] - features['low_20']) / features['price']
        features['price_resistance_distance'] = (features['high_20'] - features['price']) / features['price']
        
        # Time-based features
        if 'timestamp' in data.columns or hasattr(data.index, 'hour'):
            try:
                hour = pd.to_datetime(data.index).hour if hasattr(data.index, 'hour') else data['timestamp'].dt.hour
                features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            except:
                features['hour_sin'] = 0
                features['hour_cos'] = 1
        else:
            features['hour_sin'] = 0
            features['hour_cos'] = 1
        
        # Convertir a DataFrame
        df_features = pd.DataFrame(features)
        
        # Limpiar valores infinitos y NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features
    
    def calculate_future_profit(self, data, entry_points, exit_periods=[5, 10, 15]):
        """
        Calcular profit real para cada entrada
        """
        profits = []
        close_prices = data['Close'].values
        
        for i, entry_idx in enumerate(entry_points):
            if entry_idx >= len(close_prices):
                profits.append(0)
                continue
            
            entry_price = close_prices[entry_idx]
            max_profit = 0
            
            # Buscar el mejor profit en diferentes períodos
            for period in exit_periods:
                exit_idx = min(entry_idx + period, len(close_prices) - 1)
                exit_price = close_prices[exit_idx]
                
                # Calcular profit porcentual
                profit = (exit_price - entry_price) / entry_price * 100
                max_profit = max(max_profit, abs(profit))  # Tomar el mayor movimiento
            
            profits.append(max_profit)
        
        return np.array(profits)
    
    def train(self, data_dict):
        """
        Entrenar el modelo con datos históricos
        """
        logger.info("💰 Entrenando Profit Predictor...")
        
        all_features = []
        all_profits = []
        
        for symbol, data in data_dict.items():
            try:
                # Preparar características
                features = self.prepare_features(data)
                
                # Crear puntos de entrada (cada 5 barras para tener suficientes datos)
                entry_points = list(range(0, len(data) - 20, 5))
                
                # Calcular profits reales
                profits = self.calculate_future_profit(data, entry_points)
                
                # Seleccionar características correspondientes
                feature_subset = features.iloc[entry_points]
                
                # Asegurar misma longitud
                min_len = min(len(feature_subset), len(profits))
                feature_subset = feature_subset.iloc[:min_len]
                profits = profits[:min_len]
                
                all_features.append(feature_subset)
                all_profits.extend(profits)
                
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {e}")
                continue
        
        if not all_features:
            logger.error("No se pudieron preparar características")
            return False
        
        # Combinar todas las características
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_profits)
        
        # Filtrar outliers extremos
        y_clean = np.clip(y, -10, 10)  # Limitar profits a ±10%
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelos
        scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y_clean)
                score = cross_val_score(model, X_scaled, y_clean, cv=3, scoring='r2').mean()
                scores[name] = score
                logger.info(f"Modelo {name}: R² Score = {score:.3f}")
            except Exception as e:
                logger.error(f"Error entrenando {name}: {e}")
                scores[name] = -1
        
        self.is_trained = True
        
        # Guardar modelo
        self.save_model()
        
        best_score = max(scores.values())
        logger.info(f"✅ Profit Predictor entrenado. Mejor R² score: {best_score:.3f}")
        return True
    
    def predict(self, features):
        """
        Predecir profit potencial
        """
        if not self.is_trained:
            return 0.0, 0.0
        
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
                    predictions.append(pred)
                except:
                    predictions.append(0)
            
            # Predicción final (promedio ponderado)
            weights = [0.4, 0.4, 0.2]  # RF, GB, LR
            final_prediction = np.average(predictions, weights=weights)
            
            # Confianza basada en concordancia de modelos
            std_dev = np.std(predictions)
            confidence = max(0, 1 - (std_dev / max(abs(final_prediction), 0.1)))
            
            return float(final_prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error en predicción de profit: {e}")
            return 0.0, 0.0
    
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
            logger.info(f"Profit Predictor guardado en {self.model_path}")
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
                logger.info("Profit Predictor cargado exitosamente")
                return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
        
        return False

# Instancia global
profit_predictor = ProfitPredictor()