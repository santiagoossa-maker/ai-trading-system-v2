"""
Signal Quality Classifier
XGBoost + Random Forest ensemble for predicting signal success probability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Container for model prediction results"""
    probability: float
    confidence: float
    model_agreement: float
    feature_importance: Dict[str, float]
    prediction_metadata: Dict[str, Any]

@dataclass
class TrainingResult:
    """Container for training results"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]

class SignalQualityClassifier:
    """
    Ensemble classifier combining XGBoost and Random Forest
    to predict the probability of a trading signal being successful
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal quality classifier
        
        Args:
            model_config: Configuration for the models
        """
        self.config = model_config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.ensemble_weights = {'xgboost': 0.6, 'random_forest': 0.4}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the models"""
        return {
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 20,
                'verbose': False
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'training': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scale_features': True,
                'min_samples': 1000
            }
        }
    
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Processed feature DataFrame
        """
        try:
            # Remove infinite values and extreme outliers
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with median for numeric columns
            for col in features.select_dtypes(include=[np.number]).columns:
                if features[col].isnull().sum() > 0:
                    features[col].fillna(features[col].median(), inplace=True)
            
            # Remove constant columns
            constant_cols = features.columns[features.nunique() <= 1]
            if len(constant_cols) > 0:
                features = features.drop(columns=constant_cols)
                logger.info(f"Removed {len(constant_cols)} constant columns")
            
            # Cap extreme outliers (beyond 3 standard deviations)
            for col in features.select_dtypes(include=[np.number]).columns:
                mean_val = features[col].mean()
                std_val = features[col].std()
                if std_val > 0:
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    features[col] = features[col].clip(lower_bound, upper_bound)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return features
    
    def _create_target_variable(self, features: pd.DataFrame, price_data: pd.DataFrame, 
                              signal_direction: np.ndarray, lookahead_periods: int = 50) -> np.ndarray:
        """
        Create binary target variable for signal quality
        
        Args:
            features: Feature DataFrame
            price_data: OHLCV price data
            signal_direction: Array of signal directions (1 for buy, -1 for sell, 0 for hold)
            lookahead_periods: Periods to look ahead for profit calculation
            
        Returns:
            Binary target array (1 for successful signal, 0 for unsuccessful)
        """
        try:
            targets = np.zeros(len(features))
            
            for i in range(len(features) - lookahead_periods):
                direction = signal_direction[i]
                if direction == 0:  # No signal
                    continue
                
                current_price = price_data['close'].iloc[i]
                future_prices = price_data['close'].iloc[i+1:i+lookahead_periods+1]
                
                if direction == 1:  # Buy signal
                    # Check if price goes up by at least 0.1% before going down by 0.2%
                    profit_threshold = current_price * 1.001
                    loss_threshold = current_price * 0.998
                    
                    for future_price in future_prices:
                        if future_price >= profit_threshold:
                            targets[i] = 1
                            break
                        elif future_price <= loss_threshold:
                            targets[i] = 0
                            break
                
                elif direction == -1:  # Sell signal
                    # Check if price goes down by at least 0.1% before going up by 0.2%
                    profit_threshold = current_price * 0.999
                    loss_threshold = current_price * 1.002
                    
                    for future_price in future_prices:
                        if future_price <= profit_threshold:
                            targets[i] = 1
                            break
                        elif future_price >= loss_threshold:
                            targets[i] = 0
                            break
            
            return targets
            
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            return np.zeros(len(features))
    
    def train(self, features: pd.DataFrame, targets: np.ndarray, 
              validation_split: float = 0.2) -> TrainingResult:
        """
        Train the ensemble models
        
        Args:
            features: Training features
            targets: Binary targets (1 for successful signal, 0 for unsuccessful)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results
        """
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available")
        
        try:
            # Prepare features
            features_clean = self._prepare_features(features)
            self.feature_names = list(features_clean.columns)
            
            # Check minimum samples requirement
            if len(features_clean) < self.config['training']['min_samples']:
                raise ValueError(f"Insufficient training samples: {len(features_clean)} < {self.config['training']['min_samples']}")
            
            # Scale features if configured
            if self.config['training']['scale_features']:
                self.scalers['main'] = StandardScaler()
                features_scaled = self.scalers['main'].fit_transform(features_clean)
                features_scaled = pd.DataFrame(features_scaled, columns=features_clean.columns, index=features_clean.index)
            else:
                features_scaled = features_clean
            
            # Split data
            split_idx = int(len(features_scaled) * (1 - validation_split))
            X_train, X_val = features_scaled[:split_idx], features_scaled[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            logger.info(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
            logger.info(f"Class distribution - Train: {np.bincount(y_train.astype(int))}, Val: {np.bincount(y_val.astype(int))}")
            
            # Train XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(**self.config['xgboost'])
            self.models['xgboost'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Train Random Forest
            self.models['random_forest'] = RandomForestClassifier(**self.config['random_forest'])
            self.models['random_forest'].fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
            
            for model_name, model in self.models.items():
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
                cv_scores.extend(scores)
                logger.info(f"{model_name} CV AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            # Validation predictions
            val_pred_proba = self.predict_proba(X_val)
            val_pred = (val_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, val_pred_proba)
            
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            accuracy = accuracy_score(y_val, val_pred)
            precision = precision_score(y_val, val_pred, zero_division=0)
            recall = recall_score(y_val, val_pred, zero_division=0)
            f1 = f1_score(y_val, val_pred, zero_division=0)
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            self.is_trained = True
            
            result = TrainingResult(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc_score,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance
            )
            
            logger.info(f"Training completed - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training signal quality classifier: {str(e)}")
            raise
    
    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict signal success probability using ensemble
        
        Args:
            features: Input features
            
        Returns:
            Array of probabilities
        """
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
                features_clean = features
            
            # Scale if needed
            if 'main' in self.scalers:
                features_scaled = self.scalers['main'].transform(features_clean)
            else:
                features_scaled = features_clean
            
            # Get predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(features_scaled)
                predictions[model_name] = pred_proba[:, 1]  # Probability of positive class
            
            # Ensemble prediction
            ensemble_pred = (
                predictions['xgboost'] * self.ensemble_weights['xgboost'] +
                predictions['random_forest'] * self.ensemble_weights['random_forest']
            )
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {str(e)}")
            return np.zeros(len(features))
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray], 
                return_details: bool = False) -> Union[ModelPrediction, np.ndarray]:
        """
        Predict signal quality with detailed information
        
        Args:
            features: Input features
            return_details: Whether to return detailed prediction information
            
        Returns:
            ModelPrediction object or array of probabilities
        """
        try:
            probabilities = self.predict_proba(features)
            
            if not return_details:
                return probabilities
            
            if len(probabilities) == 1:
                prob = probabilities[0]
                
                # Calculate model agreement
                features_clean = self._prepare_features(features)
                if 'main' in self.scalers:
                    features_scaled = self.scalers['main'].transform(features_clean)
                else:
                    features_scaled = features_clean
                
                xgb_pred = self.models['xgboost'].predict_proba(features_scaled)[0, 1]
                rf_pred = self.models['random_forest'].predict_proba(features_scaled)[0, 1]
                agreement = 1.0 - abs(xgb_pred - rf_pred)
                
                # Feature importance for this prediction
                feature_importance = self._get_feature_importance()
                
                return ModelPrediction(
                    probability=prob,
                    confidence=agreement,
                    model_agreement=agreement,
                    feature_importance=feature_importance,
                    prediction_metadata={
                        'xgb_probability': xgb_pred,
                        'rf_probability': rf_pred,
                        'ensemble_weights': self.ensemble_weights
                    }
                )
            else:
                return probabilities
                
        except Exception as e:
            logger.error(f"Error in detailed prediction: {str(e)}")
            return ModelPrediction(
                probability=0.5,
                confidence=0.0,
                model_agreement=0.0,
                feature_importance={},
                prediction_metadata={}
            )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from ensemble"""
        try:
            importance = {}
            
            if 'xgboost' in self.models and hasattr(self.models['xgboost'], 'feature_importances_'):
                xgb_importance = self.models['xgboost'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(xgb_importance[i]) * self.ensemble_weights['xgboost']
            
            if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
                rf_importance = self.models['random_forest'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    if feature in importance:
                        importance[feature] += float(rf_importance[i]) * self.ensemble_weights['random_forest']
                    else:
                        importance[feature] = float(rf_importance[i]) * self.ensemble_weights['random_forest']
            
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
    n_features = 25
    
    # Generate features (simulating technical indicators)
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate realistic targets (successful signals are less common)
    targets = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Initialize and train classifier
    classifier = SignalQualityClassifier()
    
    print("Training Signal Quality Classifier...")
    training_result = classifier.train(features, targets)
    
    print(f"Training Results:")
    print(f"  Accuracy: {training_result.accuracy:.4f}")
    print(f"  Precision: {training_result.precision:.4f}")
    print(f"  Recall: {training_result.recall:.4f}")
    print(f"  F1 Score: {training_result.f1_score:.4f}")
    print(f"  AUC Score: {training_result.auc_score:.4f}")
    
    # Test prediction
    test_features = features.iloc[:5]
    predictions = classifier.predict(test_features, return_details=True)
    
    if isinstance(predictions, ModelPrediction):
        print(f"\nSample Prediction:")
        print(f"  Probability: {predictions.probability:.4f}")
        print(f"  Confidence: {predictions.confidence:.4f}")
        print(f"  Model Agreement: {predictions.model_agreement:.4f}")
    
    print("\nSignal Quality Classifier implementation completed!")