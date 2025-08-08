"""
Profit Predictor
LightGBM + Neural Network ensemble for predicting expected profit in pips
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import lightgbm as lgb
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Model
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    lgb = None
    tf = None

logger = logging.getLogger(__name__)

@dataclass
class ProfitPrediction:
    """Container for profit prediction results"""
    expected_profit_pips: float
    confidence_interval: Tuple[float, float]
    model_agreement: float
    risk_adjusted_profit: float
    prediction_metadata: Dict[str, Any]

@dataclass
class ProfitTrainingResult:
    """Container for training results"""
    mse: float
    mae: float
    r2_score: float
    rmse: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]

class DeepNeuralNetwork:
    """Custom neural network for profit prediction"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        self.input_dim = input_dim
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self) -> Optional["tf.keras.Model"]:
        """Build the neural network architecture"""
        try:
            if tf is None:
                logger.warning("TensorFlow not available, cannot build neural network")
                return None
                
            # Input layer
            inputs = tf.keras.Input(shape=(self.input_dim,))
            
            # Feature extraction layers
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(64, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(32, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            # Output layer (regression)
            outputs = layers.Dense(1, activation='linear')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile with Adam optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building neural network: {str(e)}")
            return None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the neural network"""
        try:
            if tf is None:
                logger.warning("TensorFlow not available, skipping neural network training")
                return {}
                
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Build model
            self.model = self.build_model()
            if self.model is None:
                raise ValueError("Failed to build neural network")
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
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
            
            # Train the model
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.config.get('epochs', 200),
                batch_size=self.config.get('batch_size', 32),
                validation_data=validation_data,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            return {'history': history.history}
            
        except Exception as e:
            logger.error(f"Error training neural network: {str(e)}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None or tf is None:
                logger.warning("Neural network not available, returning zeros")
                return np.zeros(len(X))
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making neural network predictions: {str(e)}")
            return np.zeros(len(X))

class ProfitPredictor:
    """
    Ensemble regressor combining LightGBM and Neural Network
    to predict expected profit in pips
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the profit predictor
        
        Args:
            model_config: Configuration for the models
        """
        self.config = model_config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.ensemble_weights = {'lightgbm': 0.7, 'neural_network': 0.3}
        self.target_scaler = StandardScaler()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the models"""
        return {
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'n_estimators': 500,
                'random_state': 42,
                'verbose': -1
            },
            'neural_network': {
                'epochs': 200,
                'batch_size': 32,
                'learning_rate': 0.001
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
            # Remove infinite values and extreme outliers
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            for col in features.select_dtypes(include=[np.number]).columns:
                if features[col].isnull().sum() > 0:
                    features[col].fillna(features[col].median(), inplace=True)
            
            # Remove constant columns
            constant_cols = features.columns[features.nunique() <= 1]
            if len(constant_cols) > 0:
                features = features.drop(columns=constant_cols)
            
            # Cap extreme outliers
            for col in features.select_dtypes(include=[np.number]).columns:
                q01 = features[col].quantile(0.01)
                q99 = features[col].quantile(0.99)
                features[col] = features[col].clip(q01, q99)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return features
    
    def _create_profit_targets(self, price_data: pd.DataFrame, 
                             signal_direction: np.ndarray,
                             lookahead_periods: int = 50) -> np.ndarray:
        """
        Create profit targets in pips
        
        Args:
            price_data: OHLCV price data
            signal_direction: Signal directions (1 for buy, -1 for sell, 0 for hold)
            lookahead_periods: Periods to look ahead for profit calculation
            
        Returns:
            Array of profit values in pips
        """
        try:
            profits_pips = np.zeros(len(price_data))
            
            # Estimate pip value (depends on instrument)
            # For most forex pairs, 1 pip = 0.0001
            # For indices, we'll use 0.1 as pip value
            pip_value = 0.1
            
            for i in range(len(price_data) - lookahead_periods):
                direction = signal_direction[i]
                if direction == 0:  # No signal
                    continue
                
                entry_price = price_data['close'].iloc[i]
                
                # Find the best profit in the next periods
                future_prices = price_data['close'].iloc[i+1:i+lookahead_periods+1]
                
                if direction == 1:  # Buy signal
                    max_profit_price = future_prices.max()
                    profit_points = max_profit_price - entry_price
                else:  # Sell signal
                    min_profit_price = future_prices.min()
                    profit_points = entry_price - min_profit_price
                
                # Convert to pips
                profits_pips[i] = profit_points / pip_value
                
                # Apply realistic constraints
                profits_pips[i] = max(-100, min(100, profits_pips[i]))  # Cap at ±100 pips
            
            return profits_pips
            
        except Exception as e:
            logger.error(f"Error creating profit targets: {str(e)}")
            return np.zeros(len(price_data))
    
    def train(self, features: pd.DataFrame, targets: np.ndarray,
              validation_split: float = 0.2) -> ProfitTrainingResult:
        """
        Train the ensemble models
        
        Args:
            features: Training features
            targets: Profit targets in pips
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
            
            # Split data
            split_idx = int(len(features_clean) * (1 - validation_split))
            X_train, X_val = features_clean[:split_idx], features_clean[split_idx:]
            y_train, y_val = targets_scaled[:split_idx], targets_scaled[split_idx:]
            
            logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
            logger.info(f"Target statistics - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models['lightgbm'] = lgb.train(
                self.config['lightgbm'],
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Train Neural Network
            if tf is not None:
                self.models['neural_network'] = DeepNeuralNetwork(
                    input_dim=len(self.feature_names),
                    config=self.config['neural_network']
                )
                self.models['neural_network'].fit(
                    X_train.values, y_train,
                    X_val.values, y_val
                )
            else:
                # Fallback to sklearn MLPRegressor
                self.models['neural_network'] = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size=32,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                )
                
                # Scale features for sklearn MLP
                self.scalers['nn_scaler'] = StandardScaler()
                X_train_scaled = self.scalers['nn_scaler'].fit_transform(X_train)
                self.models['neural_network'].fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = []
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
            
            # CV for LightGBM
            lgb_cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                cv_train_data = lgb.Dataset(X_cv_train, label=y_cv_train)
                cv_model = lgb.train(
                    self.config['lightgbm'],
                    cv_train_data,
                    num_boost_round=100,
                    callbacks=[lgb.log_evaluation(0)]
                )
                
                cv_pred = cv_model.predict(X_cv_val)
                cv_score = -mean_squared_error(y_cv_val, cv_pred)
                lgb_cv_scores.append(cv_score)
            
            cv_scores.extend(lgb_cv_scores)
            
            # Validation predictions
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
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            self.is_trained = True
            
            result = ProfitTrainingResult(
                mse=mse,
                mae=mae,
                r2_score=r2,
                rmse=rmse,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance
            )
            
            logger.info(f"Training completed - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training profit predictor: {str(e)}")
            raise
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict expected profit using ensemble"""
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
            
            # Get predictions from each model
            predictions = {}
            
            # LightGBM prediction
            lgb_pred = self.models['lightgbm'].predict(features_clean)
            predictions['lightgbm'] = lgb_pred
            
            # Neural Network prediction
            if isinstance(self.models['neural_network'], DeepNeuralNetwork):
                nn_pred = self.models['neural_network'].predict(features_clean.values)
            else:
                # sklearn MLP
                features_scaled = self.scalers['nn_scaler'].transform(features_clean)
                nn_pred = self.models['neural_network'].predict(features_scaled)
            
            predictions['neural_network'] = nn_pred
            
            # Ensemble prediction
            ensemble_pred = (
                predictions['lightgbm'] * self.ensemble_weights['lightgbm'] +
                predictions['neural_network'] * self.ensemble_weights['neural_network']
            )
            
            # Unscale if needed
            if self.config['training']['scale_targets']:
                ensemble_pred = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error predicting profit: {str(e)}")
            return np.zeros(len(features))
    
    def predict_with_confidence(self, features: Union[pd.DataFrame, np.ndarray]) -> ProfitPrediction:
        """Predict profit with confidence intervals and risk adjustment"""
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
                
                lgb_pred = self.models['lightgbm'].predict(features_clean)[0]
                
                if isinstance(self.models['neural_network'], DeepNeuralNetwork):
                    nn_pred = self.models['neural_network'].predict(features_clean.values)[0]
                else:
                    features_scaled = self.scalers['nn_scaler'].transform(features_clean)
                    nn_pred = self.models['neural_network'].predict(features_scaled)[0]
                
                # Unscale individual predictions if needed
                if self.config['training']['scale_targets']:
                    lgb_pred = self.target_scaler.inverse_transform([[lgb_pred]])[0][0]
                    nn_pred = self.target_scaler.inverse_transform([[nn_pred]])[0][0]
                
                # Model agreement (higher when models agree)
                agreement = 1.0 / (1.0 + abs(lgb_pred - nn_pred))
                
                # Confidence interval (wider when models disagree)
                uncertainty = abs(lgb_pred - nn_pred)
                ci_lower = pred_value - uncertainty
                ci_upper = pred_value + uncertainty
                
                # Risk-adjusted profit (conservative estimate)
                risk_adjusted = pred_value * agreement
                
                return ProfitPrediction(
                    expected_profit_pips=pred_value,
                    confidence_interval=(ci_lower, ci_upper),
                    model_agreement=agreement,
                    risk_adjusted_profit=risk_adjusted,
                    prediction_metadata={
                        'lgb_prediction': lgb_pred,
                        'nn_prediction': nn_pred,
                        'uncertainty': uncertainty,
                        'ensemble_weights': self.ensemble_weights
                    }
                )
            else:
                return ProfitPrediction(
                    expected_profit_pips=prediction[0] if len(prediction) > 0 else 0.0,
                    confidence_interval=(0.0, 0.0),
                    model_agreement=0.0,
                    risk_adjusted_profit=0.0,
                    prediction_metadata={}
                )
                
        except Exception as e:
            logger.error(f"Error in confidence prediction: {str(e)}")
            return ProfitPrediction(
                expected_profit_pips=0.0,
                confidence_interval=(0.0, 0.0),
                model_agreement=0.0,
                risk_adjusted_profit=0.0,
                prediction_metadata={}
            )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from ensemble"""
        try:
            importance = {}
            
            # LightGBM importance
            if 'lightgbm' in self.models:
                lgb_importance = self.models['lightgbm'].feature_importance()
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(lgb_importance[i]) * self.ensemble_weights['lightgbm']
            
            # For neural networks, we can't get traditional feature importance
            # but we can estimate it using permutation importance or other methods
            # For now, we'll just use the LightGBM importance
            
            # Normalize
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v/total_importance for k, v in importance.items()}
            
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
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data
    n_samples = 2000
    n_features = 30
    
    # Generate features (simulating technical indicators and patterns)
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate realistic profit targets (most trades have small profits/losses)
    targets = np.random.normal(0, 10, n_samples)  # Mean 0, std 10 pips
    targets = np.clip(targets, -50, 50)  # Clip to realistic range
    
    # Initialize and train predictor
    predictor = ProfitPredictor()
    
    print("Training Profit Predictor...")
    training_result = predictor.train(features, targets)
    
    print(f"Training Results:")
    print(f"  RMSE: {training_result.rmse:.4f} pips")
    print(f"  MAE: {training_result.mae:.4f} pips") 
    print(f"  R² Score: {training_result.r2_score:.4f}")
    
    # Test prediction
    test_features = features.iloc[:1]
    prediction = predictor.predict_with_confidence(test_features)
    
    print(f"\nSample Prediction:")
    print(f"  Expected Profit: {prediction.expected_profit_pips:.2f} pips")
    print(f"  Confidence Interval: [{prediction.confidence_interval[0]:.2f}, {prediction.confidence_interval[1]:.2f}] pips")
    print(f"  Model Agreement: {prediction.model_agreement:.4f}")
    print(f"  Risk-Adjusted Profit: {prediction.risk_adjusted_profit:.2f} pips")
    
    print("\nProfit Predictor implementation completed!")