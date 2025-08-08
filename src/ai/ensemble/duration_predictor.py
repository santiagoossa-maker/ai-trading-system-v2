"""
Duration Predictor  
LSTM + XGBoost ensemble for predicting movement duration in minutes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Model, Sequential
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    xgb = None
    tf = None

logger = logging.getLogger(__name__)

@dataclass
class DurationPrediction:
    """Container for duration prediction results"""
    expected_duration_minutes: float
    confidence_interval: Tuple[float, float]
    model_agreement: float
    volatility_adjusted_duration: float
    prediction_metadata: Dict[str, Any]

@dataclass
class DurationTrainingResult:
    """Container for training results"""
    mse: float
    mae: float
    r2_score: float
    rmse: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]

class LSTMNetwork:
    """LSTM network for time series duration prediction"""
    
    def __init__(self, sequence_length: int, n_features: int, config: Dict[str, Any]):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def build_model(self) -> Optional["tf.keras.Model"]:
        """Build the LSTM architecture"""
        try:
            if tf is None:
                logger.warning("TensorFlow not available, cannot build LSTM")
                return None
                
            model = tf.keras.Sequential([
                # First LSTM layer with return sequences
                layers.LSTM(
                    units=50,
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.n_features)
                ),
                layers.Dropout(0.2),
                
                # Second LSTM layer
                layers.LSTM(units=50, return_sequences=True),
                layers.Dropout(0.2),
                
                # Third LSTM layer
                layers.LSTM(units=50),
                layers.Dropout(0.2),
                
                # Dense layers
                layers.Dense(25, activation='relu'),
                layers.Dense(1, activation='linear')  # Linear for regression
            ])
            
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            return None
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            X, y = [], []
            
            for i in range(self.sequence_length, len(data)):
                X.append(data[i-self.sequence_length:i])
                if targets is not None:
                    y.append(targets[i])
            
            X = np.array(X)
            y = np.array(y) if targets is not None else None
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the LSTM network"""
        try:
            if tf is None:
                logger.warning("TensorFlow not available, skipping LSTM training")
                return {}
                
            # Scale the data
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Build model
            self.model = self.build_model()
            if self.model is None:
                raise ValueError("Failed to build LSTM model")
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
                validation_data = (X_val_scaled, y_val)
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train
            history = self.model.fit(
                X_scaled, y,
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                validation_data=validation_data,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            return {'history': history.history}
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None or tf is None:
                logger.warning("LSTM not available, returning default predictions")
                return np.full(len(X), 60.0)  # Default 1 hour
            
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            predictions = self.model.predict(X_scaled, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {str(e)}")
            return np.zeros(len(X))

class DurationPredictor:
    """
    Ensemble predictor combining LSTM and XGBoost
    to predict movement duration in minutes
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the duration predictor
        
        Args:
            model_config: Configuration for the models
        """
        self.config = model_config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.ensemble_weights = {'lstm': 0.6, 'xgboost': 0.4}
        self.sequence_length = self.config.get('sequence_length', 20)
        self.target_scaler = StandardScaler()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the models"""
        return {
            'sequence_length': 20,
            'lstm': {
                'epochs': 100,
                'batch_size': 32,
                'lstm_units': [50, 50, 50],
                'dropout_rate': 0.2
            },
            'xgboost': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 30,
                'verbose': False
            },
            'training': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scale_targets': True,
                'min_samples': 1000
            }
        }
    
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        try:
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            for col in features.select_dtypes(include=[np.number]).columns:
                if features[col].isnull().sum() > 0:
                    features[col].fillna(features[col].median(), inplace=True)
            
            # Remove constant columns
            constant_cols = features.columns[features.nunique() <= 1]
            if len(constant_cols) > 0:
                features = features.drop(columns=constant_cols)
            
            # Add time-based features for duration prediction
            if 'timestamp' in features.columns or features.index.name == 'timestamp':
                if features.index.name == 'timestamp':
                    timestamps = features.index
                else:
                    timestamps = pd.to_datetime(features['timestamp'])
                
                features['hour'] = timestamps.hour
                features['day_of_week'] = timestamps.dayofweek
                features['minute'] = timestamps.minute
            
            # Add volatility-based features
            for lookback in [5, 10, 20]:
                col_name = f'volatility_{lookback}'
                if col_name not in features.columns:
                    # Estimate volatility from price features
                    price_cols = [col for col in features.columns if any(x in col.lower() for x in ['close', 'price'])]
                    if price_cols:
                        price_col = price_cols[0]
                        features[col_name] = features[price_col].rolling(lookback).std()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return features
    
    def _create_duration_targets(self, price_data: pd.DataFrame,
                               signal_direction: np.ndarray,
                               volatility_data: np.ndarray = None) -> np.ndarray:
        """
        Create duration targets in minutes
        
        Args:
            price_data: OHLCV price data
            signal_direction: Signal directions (1 for buy, -1 for sell, 0 for hold)
            volatility_data: Volatility measurements
            
        Returns:
            Array of duration values in minutes
        """
        try:
            durations_minutes = np.zeros(len(price_data))
            
            for i in range(len(price_data) - 100):  # Leave room for movement detection
                direction = signal_direction[i]
                if direction == 0:  # No signal
                    continue
                
                entry_price = price_data['close'].iloc[i]
                
                # Define movement thresholds based on volatility
                if volatility_data is not None and i < len(volatility_data):
                    volatility = volatility_data[i]
                    threshold = max(0.001, volatility * 2)  # 2x volatility
                else:
                    threshold = 0.002  # Default 0.2% movement
                
                # Find when movement threshold is reached
                duration = 0
                for j in range(i + 1, min(i + 100, len(price_data))):
                    current_price = price_data['close'].iloc[j]
                    
                    if direction == 1:  # Buy signal - looking for upward movement
                        if current_price >= entry_price * (1 + threshold):
                            duration = j - i
                            break
                    else:  # Sell signal - looking for downward movement
                        if current_price <= entry_price * (1 - threshold):
                            duration = j - i
                            break
                
                # Convert to minutes (assuming 5-minute bars)
                durations_minutes[i] = duration * 5
                
                # Apply realistic constraints (5 minutes to 8 hours)
                durations_minutes[i] = max(5, min(480, durations_minutes[i]))
            
            return durations_minutes
            
        except Exception as e:
            logger.error(f"Error creating duration targets: {str(e)}")
            return np.zeros(len(price_data))
    
    def _create_time_series_features(self, features: pd.DataFrame) -> np.ndarray:
        """Create time series features for LSTM"""
        try:
            # Select most relevant features for time series
            time_series_cols = []
            
            # Price-related features
            price_keywords = ['close', 'open', 'high', 'low', 'price', 'sma', 'ema']
            for keyword in price_keywords:
                cols = [col for col in features.columns if keyword.lower() in col.lower()]
                time_series_cols.extend(cols[:5])  # Limit to 5 per category
            
            # Volatility features
            vol_keywords = ['atr', 'volatility', 'bb_width']
            for keyword in vol_keywords:
                cols = [col for col in features.columns if keyword.lower() in col.lower()]
                time_series_cols.extend(cols[:3])
            
            # Momentum features
            momentum_keywords = ['rsi', 'macd', 'momentum', 'roc']
            for keyword in momentum_keywords:
                cols = [col for col in features.columns if keyword.lower() in col.lower()]
                time_series_cols.extend(cols[:3])
            
            # Remove duplicates and ensure columns exist
            time_series_cols = list(set(time_series_cols))
            time_series_cols = [col for col in time_series_cols if col in features.columns]
            
            if not time_series_cols:
                # Fallback: use first 10 numeric columns
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                time_series_cols = list(numeric_cols[:10])
            
            return features[time_series_cols].values
            
        except Exception as e:
            logger.error(f"Error creating time series features: {str(e)}")
            return features.select_dtypes(include=[np.number]).values
    
    def train(self, features: pd.DataFrame, targets: np.ndarray,
              validation_split: float = 0.2) -> DurationTrainingResult:
        """
        Train the ensemble models
        
        Args:
            features: Training features
            targets: Duration targets in minutes
            validation_split: Fraction of data for validation
            
        Returns:
            Training results
        """
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available")
        
        try:
            # Prepare features
            features_clean = self._prepare_features(features)
            self.feature_names = list(features_clean.columns)
            
            # Check minimum samples
            if len(features_clean) < self.config['training']['min_samples']:
                raise ValueError(f"Insufficient training samples: {len(features_clean)}")
            
            # Scale targets if configured
            if self.config['training']['scale_targets']:
                targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
            else:
                targets_scaled = targets
            
            # Split data (ensuring enough for sequences)
            min_train_size = len(features_clean) - int(len(features_clean) * validation_split)
            split_idx = max(self.sequence_length + 1, min_train_size)
            
            X_train, X_val = features_clean[:split_idx], features_clean[split_idx:]
            y_train, y_val = targets_scaled[:split_idx], targets_scaled[split_idx:]
            
            logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
            logger.info(f"Duration statistics - Mean: {y_train.mean():.2f} min, Std: {y_train.std():.2f} min")
            
            # Prepare time series data for LSTM
            ts_features = self._create_time_series_features(features_clean)
            
            # Train LSTM if TensorFlow is available
            if tf is not None:
                self.models['lstm'] = LSTMNetwork(
                    sequence_length=self.sequence_length,
                    n_features=ts_features.shape[1],
                    config=self.config['lstm']
                )
                
                # Create sequences for LSTM
                X_lstm_train, y_lstm_train = self.models['lstm'].create_sequences(
                    ts_features[:split_idx], y_train
                )
                
                if len(X_val) > self.sequence_length:
                    X_lstm_val, y_lstm_val = self.models['lstm'].create_sequences(
                        ts_features[split_idx:], y_val[self.sequence_length:]
                    )
                    
                    self.models['lstm'].fit(
                        X_lstm_train, y_lstm_train,
                        X_lstm_val, y_lstm_val
                    )
                else:
                    self.models['lstm'].fit(X_lstm_train, y_lstm_train)
            
            # Train XGBoost
            train_data = xgb.DMatrix(X_train, label=y_train)
            
            if len(X_val) > 0:
                val_data = xgb.DMatrix(X_val, label=y_val)
                self.models['xgboost'] = xgb.train(
                    self.config['xgboost'],
                    train_data,
                    evals=[(val_data, 'validation')],
                    callbacks=[xgb.callback.EarlyStopping(self.config['xgboost']['early_stopping_rounds'])],
                    verbose_eval=False
                )
            else:
                self.models['xgboost'] = xgb.train(
                    self.config['xgboost'],
                    train_data,
                    verbose_eval=False
                )
            
            # Cross-validation for XGBoost
            cv_scores = []
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                cv_train_data = xgb.DMatrix(X_cv_train, label=y_cv_train)
                cv_model = xgb.train(
                    self.config['xgboost'],
                    cv_train_data,
                    num_boost_round=100,
                    verbose_eval=False
                )
                
                cv_pred = cv_model.predict(xgb.DMatrix(X_cv_val))
                cv_score = -mean_squared_error(y_cv_val, cv_pred)
                cv_scores.append(cv_score)
            
            # Validation predictions
            if len(X_val) > 0:
                val_pred = self.predict(X_val)
                
                # Unscale if needed
                if self.config['training']['scale_targets']:
                    y_val_unscaled = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                    val_pred_unscaled = self.target_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
                else:
                    y_val_unscaled = y_val
                    val_pred_unscaled = val_pred
                
                # Calculate metrics
                mse = mean_squared_error(y_val_unscaled, val_pred_unscaled)
                mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
                r2 = r2_score(y_val_unscaled, val_pred_unscaled)
                rmse = np.sqrt(mse)
            else:
                mse = mae = rmse = 0.0
                r2 = 0.0
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            self.is_trained = True
            
            result = DurationTrainingResult(
                mse=mse,
                mae=mae,
                r2_score=r2,
                rmse=rmse,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance
            )
            
            logger.info(f"Training completed - RMSE: {rmse:.2f} min, MAE: {mae:.2f} min, R²: {r2:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training duration predictor: {str(e)}")
            raise
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict expected duration using ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            if isinstance(features, pd.DataFrame):
                features_clean = self._prepare_features(features)
                # Ensure all training features are present
                missing_features = set(self.feature_names) - set(features_clean.columns)
                if missing_features:
                    for feature in missing_features:
                        features_clean[feature] = 0.0
                features_clean = features_clean[self.feature_names]
            else:
                features_clean = pd.DataFrame(features, columns=self.feature_names)
            
            predictions = {}
            
            # XGBoost prediction
            xgb_pred = self.models['xgboost'].predict(xgb.DMatrix(features_clean))
            predictions['xgboost'] = xgb_pred
            
            # LSTM prediction
            if 'lstm' in self.models and tf is not None:
                ts_features = self._create_time_series_features(features_clean)
                
                if len(ts_features) >= self.sequence_length:
                    # Use the last sequence_length rows for prediction
                    X_lstm = ts_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                    lstm_pred = self.models['lstm'].predict(X_lstm)
                    predictions['lstm'] = np.full(len(features_clean), lstm_pred[0])
                else:
                    # Fallback: pad with zeros or use XGBoost only
                    predictions['lstm'] = predictions['xgboost']
            else:
                # No LSTM available, use XGBoost prediction
                predictions['lstm'] = predictions['xgboost']
            
            # Ensemble prediction
            ensemble_pred = (
                predictions['lstm'] * self.ensemble_weights['lstm'] +
                predictions['xgboost'] * self.ensemble_weights['xgboost']
            )
            
            # Unscale if needed
            if self.config['training']['scale_targets']:
                ensemble_pred = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
            
            # Apply realistic constraints
            ensemble_pred = np.clip(ensemble_pred, 5, 480)  # 5 minutes to 8 hours
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error predicting duration: {str(e)}")
            return np.full(len(features), 60.0)  # Default 1 hour
    
    def predict_with_volatility_adjustment(self, features: Union[pd.DataFrame, np.ndarray], 
                                         current_volatility: float = None) -> DurationPrediction:
        """Predict duration with volatility adjustment and confidence"""
        try:
            prediction = self.predict(features)
            
            if len(prediction) == 1:
                pred_value = prediction[0]
                
                # Calculate model agreement
                if isinstance(features, pd.DataFrame):
                    features_clean = self._prepare_features(features)
                    features_clean = features_clean[self.feature_names]
                else:
                    features_clean = pd.DataFrame(features, columns=self.feature_names)
                
                xgb_pred = self.models['xgboost'].predict(xgb.DMatrix(features_clean))[0]
                
                # LSTM prediction
                if 'lstm' in self.models and tf is not None:
                    ts_features = self._create_time_series_features(features_clean)
                    if len(ts_features) >= self.sequence_length:
                        X_lstm = ts_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                        lstm_pred = self.models['lstm'].predict(X_lstm)[0]
                    else:
                        lstm_pred = xgb_pred
                else:
                    lstm_pred = xgb_pred
                
                # Unscale individual predictions if needed
                if self.config['training']['scale_targets']:
                    xgb_pred = self.target_scaler.inverse_transform([[xgb_pred]])[0][0]
                    lstm_pred = self.target_scaler.inverse_transform([[lstm_pred]])[0][0]
                
                # Model agreement
                agreement = 1.0 / (1.0 + abs(xgb_pred - lstm_pred) / max(xgb_pred, lstm_pred, 1))
                
                # Confidence interval
                uncertainty = abs(xgb_pred - lstm_pred)
                ci_lower = max(5, pred_value - uncertainty)
                ci_upper = min(480, pred_value + uncertainty)
                
                # Volatility adjustment
                if current_volatility is not None:
                    # Higher volatility typically means faster movements
                    volatility_factor = 1.0 / (1.0 + current_volatility * 10)
                    volatility_adjusted = pred_value * volatility_factor
                else:
                    volatility_adjusted = pred_value
                
                return DurationPrediction(
                    expected_duration_minutes=pred_value,
                    confidence_interval=(ci_lower, ci_upper),
                    model_agreement=agreement,
                    volatility_adjusted_duration=volatility_adjusted,
                    prediction_metadata={
                        'xgb_prediction': xgb_pred,
                        'lstm_prediction': lstm_pred,
                        'uncertainty': uncertainty,
                        'volatility_factor': 1.0 / (1.0 + (current_volatility or 0) * 10),
                        'ensemble_weights': self.ensemble_weights
                    }
                )
            else:
                return DurationPrediction(
                    expected_duration_minutes=prediction[0] if len(prediction) > 0 else 60.0,
                    confidence_interval=(30.0, 120.0),
                    model_agreement=0.0,
                    volatility_adjusted_duration=60.0,
                    prediction_metadata={}
                )
                
        except Exception as e:
            logger.error(f"Error in volatility-adjusted prediction: {str(e)}")
            return DurationPrediction(
                expected_duration_minutes=60.0,
                confidence_interval=(30.0, 120.0),
                model_agreement=0.0,
                volatility_adjusted_duration=60.0,
                prediction_metadata={}
            )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost (LSTM doesn't provide traditional importance)"""
        try:
            importance = {}
            
            # XGBoost importance
            if 'xgboost' in self.models:
                xgb_importance = self.models['xgboost'].get_score(importance_type='weight')
                total_importance = sum(xgb_importance.values())
                
                if total_importance > 0:
                    for feature, score in xgb_importance.items():
                        if feature.startswith('f'):
                            # XGBoost uses f0, f1, etc. - map back to feature names
                            feature_idx = int(feature[1:])
                            if feature_idx < len(self.feature_names):
                                feature_name = self.feature_names[feature_idx]
                                importance[feature_name] = score / total_importance
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'ensemble_weights': self.ensemble_weights,
                'sequence_length': self.sequence_length,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.target_scaler = model_data['target_scaler']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.ensemble_weights = model_data['ensemble_weights']
            self.sequence_length = model_data['sequence_length']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with time series characteristics
    n_samples = 1500
    n_features = 25
    
    # Generate features with some time series patterns
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some time-based features
    features['hour'] = np.random.randint(0, 24, n_samples)
    features['volatility'] = np.abs(np.random.randn(n_samples)) * 0.01
    
    # Generate realistic duration targets (in minutes)
    # Most movements complete within 30-120 minutes
    targets = np.random.gamma(2, 30, n_samples)  # Gamma distribution
    targets = np.clip(targets, 5, 480)  # Clip to realistic range
    
    # Initialize and train predictor
    predictor = DurationPredictor()
    
    print("Training Duration Predictor...")
    training_result = predictor.train(features, targets)
    
    print(f"Training Results:")
    print(f"  RMSE: {training_result.rmse:.2f} minutes")
    print(f"  MAE: {training_result.mae:.2f} minutes")
    print(f"  R² Score: {training_result.r2_score:.4f}")
    
    # Test prediction
    test_features = features.iloc[:1]
    prediction = predictor.predict_with_volatility_adjustment(
        test_features, 
        current_volatility=0.01
    )
    
    print(f"\nSample Prediction:")
    print(f"  Expected Duration: {prediction.expected_duration_minutes:.1f} minutes")
    print(f"  Confidence Interval: [{prediction.confidence_interval[0]:.1f}, {prediction.confidence_interval[1]:.1f}] minutes")
    print(f"  Model Agreement: {prediction.model_agreement:.4f}")
    print(f"  Volatility-Adjusted: {prediction.volatility_adjusted_duration:.1f} minutes")
    
    print("\nDuration Predictor implementation completed!")