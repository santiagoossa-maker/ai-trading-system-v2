"""
Risk Assessor
Support Vector Regression + Random Forest ensemble for predicting maximum probable risk (%)
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
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import scipy.stats as stats
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Container for risk assessment results"""
    max_probable_risk_percent: float
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall: float
    confidence_level: float
    risk_factors: Dict[str, float]
    prediction_metadata: Dict[str, Any]

@dataclass
class RiskTrainingResult:
    """Container for training results"""
    mse: float
    mae: float
    r2_score: float
    rmse: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    risk_distribution_stats: Dict[str, float]

class RiskAssessor:
    """
    Ensemble regressor combining SVR and Random Forest
    to predict maximum probable risk percentage
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk assessor
        
        Args:
            model_config: Configuration for the models
        """
        self.config = model_config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.ensemble_weights = {'svr': 0.6, 'random_forest': 0.4}
        self.risk_distribution_params = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the models"""
        return {
            'svr': {
                'kernel': 'rbf',
                'C': 100,
                'gamma': 'scale',
                'epsilon': 0.01,
                'cache_size': 1000
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            'training': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scale_features': True,
                'use_robust_scaler': True,
                'min_samples': 1000
            }
        }
    
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction with risk-specific enhancements"""
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
            
            # Add risk-specific features
            self._add_risk_features(features)
            
            # Cap extreme outliers using robust statistics
            for col in features.select_dtypes(include=[np.number]).columns:
                q25 = features[col].quantile(0.25)
                q75 = features[col].quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 3 * iqr
                upper_bound = q75 + 3 * iqr
                features[col] = features[col].clip(lower_bound, upper_bound)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return features
    
    def _add_risk_features(self, features: pd.DataFrame):
        """Add risk-specific features to the dataset"""
        try:
            # Volatility clustering features
            vol_cols = [col for col in features.columns if 'volatility' in col.lower() or 'atr' in col.lower()]
            if vol_cols:
                vol_col = vol_cols[0]
                # Volatility regime
                features['vol_regime'] = (features[vol_col] / features[vol_col].rolling(50).mean()) - 1
                
                # Volatility trend
                features['vol_trend'] = features[vol_col].rolling(10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
                )
            
            # Correlation risk features
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 5:
                # Sample correlations (computational efficiency)
                sample_cols = list(numeric_cols[:10])
                corr_matrix = features[sample_cols].rolling(50).corr()
                
                # Average correlation
                if not corr_matrix.empty:
                    features['avg_correlation'] = corr_matrix.groupby(level=0).apply(
                        lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean()
                    )
            
            # Momentum risk features
            momentum_cols = [col for col in features.columns if any(keyword in col.lower() 
                           for keyword in ['rsi', 'momentum', 'roc', 'macd'])]
            if momentum_cols:
                # Momentum divergence
                momentum_col = momentum_cols[0]
                features['momentum_extremes'] = ((features[momentum_col] > features[momentum_col].quantile(0.9)) | 
                                               (features[momentum_col] < features[momentum_col].quantile(0.1))).astype(int)
            
            # Support/Resistance risk
            price_cols = [col for col in features.columns if any(keyword in col.lower() 
                         for keyword in ['close', 'price', 'high', 'low'])]
            if price_cols:
                price_col = price_cols[0]
                # Distance from recent highs/lows
                recent_high = features[price_col].rolling(50).max()
                recent_low = features[price_col].rolling(50).min()
                features['distance_from_high'] = (recent_high - features[price_col]) / features[price_col]
                features['distance_from_low'] = (features[price_col] - recent_low) / features[price_col]
            
            # Time-based risk factors
            if hasattr(features.index, 'hour'):
                features['risky_hour'] = features.index.hour.isin([0, 1, 22, 23]).astype(int)
            
        except Exception as e:
            logger.error(f"Error adding risk features: {str(e)}")
    
    def _create_risk_targets(self, price_data: pd.DataFrame,
                           signal_direction: np.ndarray,
                           lookahead_periods: int = 50) -> np.ndarray:
        """
        Create risk targets as maximum adverse excursion percentage
        
        Args:
            price_data: OHLCV price data
            signal_direction: Signal directions (1 for buy, -1 for sell, 0 for hold)
            lookahead_periods: Periods to look ahead for risk calculation
            
        Returns:
            Array of maximum adverse excursion percentages
        """
        try:
            max_risks = np.zeros(len(price_data))
            
            for i in range(len(price_data) - lookahead_periods):
                direction = signal_direction[i]
                if direction == 0:  # No signal
                    continue
                
                entry_price = price_data['close'].iloc[i]
                future_prices = price_data['close'].iloc[i+1:i+lookahead_periods+1]
                
                if direction == 1:  # Buy signal - risk is price going down
                    min_price = future_prices.min()
                    max_adverse_excursion = (entry_price - min_price) / entry_price * 100
                else:  # Sell signal - risk is price going up
                    max_price = future_prices.max()
                    max_adverse_excursion = (max_price - entry_price) / entry_price * 100
                
                # Also consider intraday risk using high/low data
                if 'high' in price_data.columns and 'low' in price_data.columns:
                    future_highs = price_data['high'].iloc[i+1:i+lookahead_periods+1]
                    future_lows = price_data['low'].iloc[i+1:i+lookahead_periods+1]
                    
                    if direction == 1:  # Buy signal
                        intraday_risk = (entry_price - future_lows.min()) / entry_price * 100
                    else:  # Sell signal
                        intraday_risk = (future_highs.max() - entry_price) / entry_price * 100
                    
                    max_adverse_excursion = max(max_adverse_excursion, intraday_risk)
                
                # Cap risk at reasonable levels (max 50% loss)
                max_risks[i] = max(0, min(50, max_adverse_excursion))
            
            return max_risks
            
        except Exception as e:
            logger.error(f"Error creating risk targets: {str(e)}")
            return np.zeros(len(price_data))
    
    def _calculate_risk_distribution_params(self, risk_values: np.ndarray) -> Dict[str, float]:
        """Calculate parameters for risk distribution"""
        try:
            # Remove zeros (no-signal cases)
            non_zero_risks = risk_values[risk_values > 0]
            
            if len(non_zero_risks) == 0:
                return {}
            
            # Basic statistics
            params = {
                'mean': np.mean(non_zero_risks),
                'std': np.std(non_zero_risks),
                'median': np.median(non_zero_risks),
                'q95': np.percentile(non_zero_risks, 95),
                'q99': np.percentile(non_zero_risks, 99),
                'max': np.max(non_zero_risks),
                'skewness': stats.skew(non_zero_risks),
                'kurtosis': stats.kurtosis(non_zero_risks)
            }
            
            # Fit distributions
            try:
                # Log-normal distribution (common for financial risks)
                if np.all(non_zero_risks > 0):
                    lognorm_params = stats.lognorm.fit(non_zero_risks)
                    params['lognorm_shape'] = lognorm_params[0]
                    params['lognorm_scale'] = lognorm_params[2]
                
                # Gamma distribution
                gamma_params = stats.gamma.fit(non_zero_risks)
                params['gamma_shape'] = gamma_params[0]
                params['gamma_scale'] = gamma_params[2]
                
            except Exception as e:
                logger.warning(f"Could not fit distributions: {str(e)}")
            
            return params
            
        except Exception as e:
            logger.error(f"Error calculating risk distribution params: {str(e)}")
            return {}
    
    def train(self, features: pd.DataFrame, targets: np.ndarray,
              validation_split: float = 0.2) -> RiskTrainingResult:
        """
        Train the ensemble models
        
        Args:
            features: Training features
            targets: Risk targets (max adverse excursion %)
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
            
            # Calculate risk distribution parameters
            self.risk_distribution_params = self._calculate_risk_distribution_params(targets)
            
            # Split data
            split_idx = int(len(features_clean) * (1 - validation_split))
            X_train, X_val = features_clean[:split_idx], features_clean[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            # Focus on non-zero risk samples for training
            non_zero_mask_train = y_train > 0
            non_zero_mask_val = y_val > 0
            
            if np.sum(non_zero_mask_train) < 100:
                logger.warning("Very few non-zero risk samples for training")
            
            logger.info(f"Training with {len(X_train)} samples ({np.sum(non_zero_mask_train)} with risk > 0)")
            logger.info(f"Risk statistics - Mean: {y_train[non_zero_mask_train].mean():.4f}%, "
                       f"Max: {y_train.max():.4f}%")
            
            # Scale features
            if self.config['training']['scale_features']:
                if self.config['training']['use_robust_scaler']:
                    self.scalers['main'] = RobustScaler()
                else:
                    self.scalers['main'] = StandardScaler()
                
                X_train_scaled = self.scalers['main'].fit_transform(X_train)
                X_val_scaled = self.scalers['main'].transform(X_val)
                
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Train SVR (focus on non-zero samples)
            svr_train_mask = non_zero_mask_train
            if np.sum(svr_train_mask) > 50:  # Minimum samples for SVR
                self.models['svr'] = SVR(**self.config['svr'])
                self.models['svr'].fit(X_train_scaled[svr_train_mask], y_train[svr_train_mask])
            else:
                logger.warning("Insufficient non-zero samples for SVR training")
                # Fallback: train on all samples but give higher weight to non-zero
                sample_weights = np.where(y_train > 0, 5.0, 1.0)  # Weight non-zero samples 5x more
                
                # SVR doesn't support sample weights directly, so we duplicate high-risk samples
                high_risk_indices = np.where(y_train > y_train.quantile(0.8))[0]
                if len(high_risk_indices) > 0:
                    # Add high-risk samples multiple times
                    augmented_X = np.vstack([X_train_scaled.values] + [X_train_scaled.iloc[high_risk_indices].values] * 3)
                    augmented_y = np.hstack([y_train] + [y_train[high_risk_indices]] * 3)
                    
                    self.models['svr'] = SVR(**self.config['svr'])
                    self.models['svr'].fit(augmented_X, augmented_y)
                else:
                    self.models['svr'] = SVR(**self.config['svr'])
                    self.models['svr'].fit(X_train_scaled, y_train)
            
            # Train Random Forest (can handle all samples including zeros)
            # Use sample weights to emphasize risky samples
            sample_weights = np.where(y_train > 0, 3.0, 1.0)
            
            self.models['random_forest'] = RandomForestRegressor(**self.config['random_forest'])
            self.models['random_forest'].fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            # Cross-validation
            cv_scores = []
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
            
            # CV for Random Forest (more stable with mixed zero/non-zero targets)
            rf_cv_scores = []
            for train_idx, val_idx in kf.split(X_train_scaled):
                X_cv_train, X_cv_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                cv_weights = np.where(y_cv_train > 0, 3.0, 1.0)
                
                cv_model = RandomForestRegressor(
                    n_estimators=100,  # Reduced for CV speed
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                cv_model.fit(X_cv_train, y_cv_train, sample_weight=cv_weights)
                
                cv_pred = cv_model.predict(X_cv_val)
                # Focus on non-zero samples for evaluation
                if np.sum(y_cv_val > 0) > 0:
                    cv_score = -mean_squared_error(y_cv_val[y_cv_val > 0], cv_pred[y_cv_val > 0])
                else:
                    cv_score = -mean_squared_error(y_cv_val, cv_pred)
                rf_cv_scores.append(cv_score)
            
            cv_scores.extend(rf_cv_scores)
            
            # Validation predictions
            val_pred = self.predict(X_val_scaled)
            
            # Calculate metrics (focus on samples with actual risk)
            if np.sum(non_zero_mask_val) > 0:
                val_true_nonzero = y_val[non_zero_mask_val]
                val_pred_nonzero = val_pred[non_zero_mask_val]
                
                mse = mean_squared_error(val_true_nonzero, val_pred_nonzero)
                mae = mean_absolute_error(val_true_nonzero, val_pred_nonzero)
                r2 = r2_score(val_true_nonzero, val_pred_nonzero)
            else:
                mse = mean_squared_error(y_val, val_pred)
                mae = mean_absolute_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
            
            rmse = np.sqrt(mse)
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            self.is_trained = True
            
            result = RiskTrainingResult(
                mse=mse,
                mae=mae,
                r2_score=r2,
                rmse=rmse,
                cross_val_scores=cv_scores,
                feature_importance=feature_importance,
                risk_distribution_stats=self.risk_distribution_params
            )
            
            logger.info(f"Training completed - RMSE: {rmse:.4f}%, MAE: {mae:.4f}%, R²: {r2:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training risk assessor: {str(e)}")
            raise
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict maximum probable risk using ensemble"""
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
            
            # Scale features
            if 'main' in self.scalers:
                features_scaled = self.scalers['main'].transform(features_clean)
                features_scaled = pd.DataFrame(features_scaled, columns=features_clean.columns)
            else:
                features_scaled = features_clean
            
            # Get predictions from each model
            predictions = {}
            
            # SVR prediction
            if 'svr' in self.models:
                svr_pred = self.models['svr'].predict(features_scaled)
                predictions['svr'] = svr_pred
            else:
                predictions['svr'] = np.zeros(len(features_scaled))
            
            # Random Forest prediction
            rf_pred = self.models['random_forest'].predict(features_scaled)
            predictions['random_forest'] = rf_pred
            
            # Ensemble prediction
            ensemble_pred = (
                predictions['svr'] * self.ensemble_weights['svr'] +
                predictions['random_forest'] * self.ensemble_weights['random_forest']
            )
            
            # Ensure non-negative risk predictions
            ensemble_pred = np.maximum(0, ensemble_pred)
            
            # Cap at reasonable maximum (50%)
            ensemble_pred = np.minimum(50, ensemble_pred)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error predicting risk: {str(e)}")
            return np.full(len(features), 5.0)  # Default 5% risk
    
    def assess_risk_comprehensive(self, features: Union[pd.DataFrame, np.ndarray],
                                current_volatility: float = None) -> RiskAssessment:
        """Comprehensive risk assessment with VaR and Expected Shortfall"""
        try:
            base_prediction = self.predict(features)
            
            if len(base_prediction) == 1:
                pred_value = base_prediction[0]
                
                # Calculate model predictions separately for confidence assessment
                if isinstance(features, pd.DataFrame):
                    features_clean = self._prepare_features(features)
                    features_clean = features_clean[self.feature_names]
                else:
                    features_clean = pd.DataFrame(features, columns=self.feature_names)
                
                if 'main' in self.scalers:
                    features_scaled = self.scalers['main'].transform(features_clean)
                else:
                    features_scaled = features_clean.values
                
                # Individual model predictions
                if 'svr' in self.models:
                    svr_pred = self.models['svr'].predict(features_scaled)[0]
                else:
                    svr_pred = pred_value
                
                rf_pred = self.models['random_forest'].predict(features_scaled)[0]
                
                # Model agreement (confidence)
                agreement = 1.0 / (1.0 + abs(svr_pred - rf_pred) / max(svr_pred, rf_pred, 0.01))
                
                # Calculate VaR using risk distribution parameters
                if self.risk_distribution_params:
                    # Use fitted distribution to estimate VaR
                    try:
                        if 'lognorm_shape' in self.risk_distribution_params:
                            # Log-normal VaR
                            var_95 = stats.lognorm.ppf(
                                0.95, 
                                self.risk_distribution_params['lognorm_shape'],
                                scale=self.risk_distribution_params['lognorm_scale']
                            )
                            var_99 = stats.lognorm.ppf(
                                0.99,
                                self.risk_distribution_params['lognorm_shape'],
                                scale=self.risk_distribution_params['lognorm_scale']
                            )
                        else:
                            # Fallback to empirical quantiles
                            var_95 = self.risk_distribution_params.get('q95', pred_value * 1.5)
                            var_99 = self.risk_distribution_params.get('q99', pred_value * 2.0)
                    except:
                        var_95 = self.risk_distribution_params.get('q95', pred_value * 1.5)
                        var_99 = self.risk_distribution_params.get('q99', pred_value * 2.0)
                else:
                    # Default VaR estimates
                    var_95 = pred_value * 1.5
                    var_99 = pred_value * 2.0
                
                # Expected Shortfall (average loss beyond VaR)
                expected_shortfall = var_99 * 1.2  # Conservative estimate
                
                # Risk factor decomposition
                risk_factors = self._decompose_risk_factors(features_clean)
                
                # Volatility adjustment
                if current_volatility is not None:
                    volatility_multiplier = 1.0 + current_volatility * 10
                    adjusted_risk = pred_value * volatility_multiplier
                    var_95 *= volatility_multiplier
                    var_99 *= volatility_multiplier
                    expected_shortfall *= volatility_multiplier
                else:
                    adjusted_risk = pred_value
                
                return RiskAssessment(
                    max_probable_risk_percent=adjusted_risk,
                    value_at_risk_95=var_95,
                    value_at_risk_99=var_99,
                    expected_shortfall=expected_shortfall,
                    confidence_level=agreement,
                    risk_factors=risk_factors,
                    prediction_metadata={
                        'svr_prediction': svr_pred,
                        'rf_prediction': rf_pred,
                        'base_prediction': pred_value,
                        'volatility_multiplier': 1.0 + (current_volatility or 0) * 10,
                        'ensemble_weights': self.ensemble_weights,
                        'risk_distribution_params': self.risk_distribution_params
                    }
                )
            else:
                return RiskAssessment(
                    max_probable_risk_percent=base_prediction[0] if len(base_prediction) > 0 else 5.0,
                    value_at_risk_95=10.0,
                    value_at_risk_99=15.0,
                    expected_shortfall=20.0,
                    confidence_level=0.0,
                    risk_factors={},
                    prediction_metadata={}
                )
                
        except Exception as e:
            logger.error(f"Error in comprehensive risk assessment: {str(e)}")
            return RiskAssessment(
                max_probable_risk_percent=5.0,
                value_at_risk_95=10.0,
                value_at_risk_99=15.0,
                expected_shortfall=20.0,
                confidence_level=0.0,
                risk_factors={},
                prediction_metadata={}
            )
    
    def _decompose_risk_factors(self, features: pd.DataFrame) -> Dict[str, float]:
        """Decompose risk into contributing factors"""
        try:
            risk_factors = {}
            
            # Volatility risk
            vol_cols = [col for col in features.columns if 'vol' in col.lower() or 'atr' in col.lower()]
            if vol_cols:
                vol_values = features[vol_cols].values.flatten()
                risk_factors['volatility'] = np.mean(vol_values) * 100
            
            # Momentum risk
            momentum_cols = [col for col in features.columns if any(kw in col.lower() 
                           for kw in ['rsi', 'momentum', 'macd'])]
            if momentum_cols:
                momentum_values = features[momentum_cols].values.flatten()
                risk_factors['momentum'] = np.std(momentum_values) * 10
            
            # Correlation risk
            if 'avg_correlation' in features.columns:
                risk_factors['correlation'] = float(features['avg_correlation'].iloc[0]) * 5
            
            # Time-based risk
            if 'risky_hour' in features.columns:
                risk_factors['time_of_day'] = float(features['risky_hour'].iloc[0]) * 2
            
            # Position risk (distance from support/resistance)
            if 'distance_from_high' in features.columns and 'distance_from_low' in features.columns:
                near_extreme = min(features['distance_from_high'].iloc[0], 
                                 features['distance_from_low'].iloc[0])
                risk_factors['position'] = (1.0 - near_extreme) * 3
            
            # Normalize to sum to approximately the predicted risk
            total_factor_risk = sum(risk_factors.values())
            if total_factor_risk > 0:
                normalization = 5.0 / total_factor_risk  # Normalize to ~5% total
                risk_factors = {k: v * normalization for k, v in risk_factors.items()}
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error decomposing risk factors: {str(e)}")
            return {}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from ensemble"""
        try:
            importance = {}
            
            # Random Forest importance
            if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
                rf_importance = self.models['random_forest'].feature_importances_
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(rf_importance[i]) * self.ensemble_weights['random_forest']
            
            # SVR doesn't provide feature importance directly
            # We could use permutation importance but it's computationally expensive
            # For now, just use Random Forest importance
            
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
                'risk_distribution_params': self.risk_distribution_params,
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
            self.risk_distribution_params = model_data['risk_distribution_params']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with realistic risk patterns
    n_samples = 2000
    n_features = 30
    
    # Generate features (simulating technical indicators)
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some risk-related features
    features['volatility'] = np.abs(np.random.randn(n_samples)) * 0.02
    features['avg_correlation'] = np.random.uniform(0.1, 0.9, n_samples)
    features['risky_hour'] = np.random.choice([0, 1], n_samples)
    
    # Generate realistic risk targets (mostly small risks, few large ones)
    # Using gamma distribution which is common for financial risks
    targets = np.random.gamma(2, 2, n_samples)  # Shape=2, scale=2
    targets = np.clip(targets, 0, 25)  # Cap at 25%
    
    # Initialize and train assessor
    assessor = RiskAssessor()
    
    print("Training Risk Assessor...")
    training_result = assessor.train(features, targets)
    
    print(f"Training Results:")
    print(f"  RMSE: {training_result.rmse:.4f}%")
    print(f"  MAE: {training_result.mae:.4f}%")
    print(f"  R² Score: {training_result.r2_score:.4f}")
    print(f"  Risk Distribution - Mean: {training_result.risk_distribution_stats.get('mean', 0):.2f}%, "
          f"95th percentile: {training_result.risk_distribution_stats.get('q95', 0):.2f}%")
    
    # Test prediction
    test_features = features.iloc[:1]
    assessment = assessor.assess_risk_comprehensive(
        test_features,
        current_volatility=0.02
    )
    
    print(f"\nSample Risk Assessment:")
    print(f"  Max Probable Risk: {assessment.max_probable_risk_percent:.2f}%")
    print(f"  VaR (95%): {assessment.value_at_risk_95:.2f}%")
    print(f"  VaR (99%): {assessment.value_at_risk_99:.2f}%")
    print(f"  Expected Shortfall: {assessment.expected_shortfall:.2f}%")
    print(f"  Confidence: {assessment.confidence_level:.4f}")
    print(f"  Risk Factors: {assessment.risk_factors}")
    
    print("\nRisk Assessor implementation completed!")