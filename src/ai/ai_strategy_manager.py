"""
AI Strategy Manager
Coordinates the 4 ensemble models and integrates with existing trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from pathlib import Path
import pickle

# Import our AI ensemble models
try:
    from ai.ensemble.signal_quality_classifier import SignalQualityClassifier, ModelPrediction
    from ai.ensemble.profit_predictor import ProfitPredictor, ProfitPrediction
    from ai.ensemble.duration_predictor import DurationPredictor, DurationPrediction
    from ai.ensemble.risk_assessor import RiskAssessor, RiskAssessment
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    AI_MODELS_AVAILABLE = False
    logger.warning(f"AI ensemble models not available: {str(e)}")

# Import existing strategy components
try:
    from strategies.multi_strategy_engine import MultiStrategyEngine, AggregatedSignal, SignalType
    from ai.feature_collector import FeatureCollector
    STRATEGY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    STRATEGY_COMPONENTS_AVAILABLE = False
    logger.warning(f"Strategy components not available: {str(e)}")

logger = logging.getLogger(__name__)

class AIDecisionType(Enum):
    """Types of AI-enhanced decisions"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class AISignalAnalysis:
    """Comprehensive AI analysis of a trading signal"""
    signal_quality_score: float  # 0-1, probability of success
    expected_profit_pips: float
    expected_duration_minutes: float
    max_risk_percent: float
    confidence_level: float
    risk_reward_ratio: float
    
    # Individual model outputs
    quality_prediction: Optional[ModelPrediction] = None
    profit_prediction: Optional[ProfitPrediction] = None
    duration_prediction: Optional[DurationPrediction] = None
    risk_assessment: Optional[RiskAssessment] = None

@dataclass 
class AITradingDecision:
    """Final AI-enhanced trading decision"""
    decision: AIDecisionType
    base_signal: Optional[AggregatedSignal]
    ai_analysis: AISignalAnalysis
    position_size_multiplier: float  # Adjustment to standard position size
    stop_loss_adjustment: float     # Adjustment to stop loss (%)
    take_profit_targets: List[float]  # Multiple TP levels in pips
    time_horizon_minutes: float     # Expected trade duration
    confidence_score: float         # Overall confidence (0-1)
    
    # Metadata
    processing_time_ms: float
    model_versions: Dict[str, str]
    feature_count: int

class AIStrategyManager:
    """
    Coordinates 4 AI ensemble models with existing trading strategies
    Provides intelligent signal enhancement and decision making
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI Strategy Manager
        
        Args:
            config: Configuration for AI models and strategy coordination
        """
        self.config = config or self._get_default_config()
        
        # Initialize AI models
        self.models = {}
        self.feature_collector = None
        self.strategy_engine = None
        
        # Model states
        self.models_trained = {}
        self.models_loaded = {}
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        # Threading for async operations
        self.lock = threading.RLock()
        
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'signal_quality': {
                    'enabled': True,
                    'weight': 0.3,
                    'min_confidence': 0.6
                },
                'profit_predictor': {
                    'enabled': True,
                    'weight': 0.25,
                    'min_expected_profit': 5.0  # pips
                },
                'duration_predictor': {
                    'enabled': True,
                    'weight': 0.2,
                    'max_duration': 480  # minutes (8 hours)
                },
                'risk_assessor': {
                    'enabled': True,
                    'weight': 0.25,
                    'max_risk': 10.0  # percent
                }
            },
            'decision_thresholds': {
                'strong_buy_score': 0.85,
                'buy_score': 0.7,
                'weak_buy_score': 0.55,
                'weak_sell_score': 0.45,
                'sell_score': 0.3,
                'strong_sell_score': 0.15
            },
            'risk_management': {
                'max_position_size_multiplier': 2.0,
                'min_position_size_multiplier': 0.1,
                'default_stop_loss': 2.0,  # percent
                'max_stop_loss_adjustment': 1.5,
                'min_risk_reward_ratio': 1.5
            },
            'paths': {
                'models_dir': 'models/',
                'features_cache': 'cache/features/'
            }
        }
    
    def _initialize_components(self):
        """Initialize AI models and supporting components"""
        try:
            if not AI_MODELS_AVAILABLE:
                logger.warning("AI ensemble models not available")
                self.models = {}
                return
            
            # Initialize AI models with error handling
            try:
                self.models['signal_quality'] = SignalQualityClassifier()
            except Exception as e:
                logger.warning(f"Failed to initialize signal quality classifier: {str(e)}")
                self.models['signal_quality'] = None
            
            try:
                self.models['profit_predictor'] = ProfitPredictor()
            except Exception as e:
                logger.warning(f"Failed to initialize profit predictor: {str(e)}")
                self.models['profit_predictor'] = None
            
            try:
                self.models['duration_predictor'] = DurationPredictor()
            except Exception as e:
                logger.warning(f"Failed to initialize duration predictor: {str(e)}")
                self.models['duration_predictor'] = None
            
            try:
                self.models['risk_assessor'] = RiskAssessor()
            except Exception as e:
                logger.warning(f"Failed to initialize risk assessor: {str(e)}")
                self.models['risk_assessor'] = None
            
            # Initialize feature collector
            if STRATEGY_COMPONENTS_AVAILABLE:
                try:
                    self.feature_collector = FeatureCollector()
                    self.strategy_engine = MultiStrategyEngine()
                except Exception as e:
                    logger.warning(f"Failed to initialize strategy components: {str(e)}")
                    self.feature_collector = None
                    self.strategy_engine = None
            
            # Track model states
            for model_name in self.models:
                if self.models[model_name] is not None:
                    self.models_trained[model_name] = False
                    self.models_loaded[model_name] = False
            
            logger.info("AI Strategy Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI Strategy Manager: {str(e)}")
    
    def load_trained_models(self, models_dir: str = None) -> Dict[str, bool]:
        """
        Load pre-trained AI models from disk
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            Dictionary showing which models were successfully loaded
        """
        models_dir = models_dir or self.config['paths']['models_dir']
        models_dir = Path(models_dir)
        
        load_results = {}
        
        for model_name, model in self.models.items():
            try:
                model_file = models_dir / f"{model_name}.pkl"
                if model_file.exists():
                    success = model.load_model(str(model_file))
                    if success:
                        self.models_loaded[model_name] = True
                        self.models_trained[model_name] = True
                        load_results[model_name] = True
                        logger.info(f"Loaded {model_name} model")
                    else:
                        load_results[model_name] = False
                        logger.error(f"Failed to load {model_name} model")
                else:
                    load_results[model_name] = False
                    logger.warning(f"Model file not found: {model_file}")
            except Exception as e:
                load_results[model_name] = False
                logger.error(f"Error loading {model_name}: {str(e)}")
        
        return load_results
    
    def analyze_signal(self, market_data: Dict[str, pd.DataFrame], 
                      symbol: str, 
                      base_signal: Optional[AggregatedSignal] = None) -> AISignalAnalysis:
        """
        Perform comprehensive AI analysis of a potential trading signal
        
        Args:
            market_data: Multi-timeframe market data
            symbol: Trading symbol
            base_signal: Optional base signal from traditional strategies
            
        Returns:
            Comprehensive AI analysis
        """
        try:
            start_time = time.time()
            
            # Generate features
            if self.feature_collector is None:
                raise ValueError("Feature collector not initialized")
            
            features = self.feature_collector.collect_all_features(market_data, symbol)
            if features.empty:
                logger.warning(f"No features generated for {symbol}")
                return self._create_default_analysis()
            
            # Get latest feature row for prediction
            latest_features = features.iloc[-1:] if len(features) > 0 else features
            
            # Initialize results
            quality_prediction = None
            profit_prediction = None
            duration_prediction = None
            risk_assessment = None
            
            # Signal Quality Analysis
            if (self.config['models']['signal_quality']['enabled'] and 
                self.models_trained.get('signal_quality', False) and
                self.models.get('signal_quality') is not None):
                try:
                    quality_prediction = self.models['signal_quality'].predict(
                        latest_features, return_details=True
                    )
                except Exception as e:
                    logger.error(f"Error in signal quality prediction: {str(e)}")
            
            # Profit Prediction
            if (self.config['models']['profit_predictor']['enabled'] and
                self.models_trained.get('profit_predictor', False) and
                self.models.get('profit_predictor') is not None):
                try:
                    profit_prediction = self.models['profit_predictor'].predict_with_confidence(
                        latest_features
                    )
                except Exception as e:
                    logger.error(f"Error in profit prediction: {str(e)}")
            
            # Duration Prediction
            if (self.config['models']['duration_predictor']['enabled'] and
                self.models_trained.get('duration_predictor', False) and
                self.models.get('duration_predictor') is not None):
                try:
                    # Get current volatility for adjustment
                    current_volatility = self._estimate_current_volatility(market_data, symbol)
                    duration_prediction = self.models['duration_predictor'].predict_with_volatility_adjustment(
                        latest_features, current_volatility
                    )
                except Exception as e:
                    logger.error(f"Error in duration prediction: {str(e)}")
            
            # Risk Assessment
            if (self.config['models']['risk_assessor']['enabled'] and
                self.models_trained.get('risk_assessor', False) and
                self.models.get('risk_assessor') is not None):
                try:
                    current_volatility = self._estimate_current_volatility(market_data, symbol)
                    risk_assessment = self.models['risk_assessor'].assess_risk_comprehensive(
                        latest_features, current_volatility
                    )
                except Exception as e:
                    logger.error(f"Error in risk assessment: {str(e)}")
            
            # Combine results
            signal_quality_score = (quality_prediction.probability 
                                   if quality_prediction else 0.5)
            expected_profit = (profit_prediction.expected_profit_pips 
                             if profit_prediction else 0.0)
            expected_duration = (duration_prediction.expected_duration_minutes 
                               if duration_prediction else 60.0)
            max_risk = (risk_assessment.max_probable_risk_percent 
                       if risk_assessment else 5.0)
            
            # Calculate overall confidence
            confidence_components = []
            if quality_prediction:
                confidence_components.append(quality_prediction.confidence)
            if profit_prediction:
                confidence_components.append(profit_prediction.model_agreement)
            if duration_prediction:
                confidence_components.append(duration_prediction.model_agreement)
            if risk_assessment:
                confidence_components.append(risk_assessment.confidence_level)
            
            confidence_level = np.mean(confidence_components) if confidence_components else 0.0
            
            # Calculate risk-reward ratio
            risk_reward_ratio = abs(expected_profit) / max(max_risk * 10, 1.0) if max_risk > 0 else 0.0
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            analysis = AISignalAnalysis(
                signal_quality_score=signal_quality_score,
                expected_profit_pips=expected_profit,
                expected_duration_minutes=expected_duration,
                max_risk_percent=max_risk,
                confidence_level=confidence_level,
                risk_reward_ratio=risk_reward_ratio,
                quality_prediction=quality_prediction,
                profit_prediction=profit_prediction,
                duration_prediction=duration_prediction,
                risk_assessment=risk_assessment
            )
            
            logger.debug(f"AI analysis completed for {symbol} in {processing_time:.1f}ms")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI signal analysis for {symbol}: {str(e)}")
            return self._create_default_analysis()
    
    def make_trading_decision(self, market_data: Dict[str, pd.DataFrame],
                            symbol: str,
                            base_signal: Optional[AggregatedSignal] = None) -> AITradingDecision:
        """
        Make an AI-enhanced trading decision
        
        Args:
            market_data: Multi-timeframe market data
            symbol: Trading symbol
            base_signal: Optional base signal from traditional strategies
            
        Returns:
            Final AI-enhanced trading decision
        """
        try:
            start_time = time.time()
            
            # Get AI analysis
            ai_analysis = self.analyze_signal(market_data, symbol, base_signal)
            
            # Calculate overall decision score
            decision_score = self._calculate_decision_score(ai_analysis, base_signal)
            
            # Map score to decision type
            decision = self._score_to_decision(decision_score)
            
            # Calculate position sizing adjustments
            position_multiplier = self._calculate_position_multiplier(ai_analysis)
            
            # Calculate stop loss adjustment
            stop_loss_adjustment = self._calculate_stop_loss_adjustment(ai_analysis)
            
            # Calculate take profit targets
            take_profit_targets = self._calculate_take_profit_targets(ai_analysis)
            
            # Overall confidence
            confidence_score = self._calculate_overall_confidence(ai_analysis, base_signal)
            
            processing_time = (time.time() - start_time) * 1000
            
            decision_obj = AITradingDecision(
                decision=decision,
                base_signal=base_signal,
                ai_analysis=ai_analysis,
                position_size_multiplier=position_multiplier,
                stop_loss_adjustment=stop_loss_adjustment,
                take_profit_targets=take_profit_targets,
                time_horizon_minutes=ai_analysis.expected_duration_minutes,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                model_versions={name: "1.0" for name in self.models.keys()},
                feature_count=len(self.feature_collector.feature_names) if self.feature_collector else 0
            )
            
            # Store prediction for performance tracking
            with self.lock:
                self.prediction_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'symbol': symbol,
                    'decision': decision.value,
                    'confidence': confidence_score,
                    'expected_profit': ai_analysis.expected_profit_pips,
                    'max_risk': ai_analysis.max_risk_percent
                })
                
                # Keep only last 1000 predictions
                if len(self.prediction_history) > 1000:
                    self.prediction_history = self.prediction_history[-1000:]
            
            logger.info(f"AI trading decision for {symbol}: {decision.value} "
                       f"(confidence: {confidence_score:.3f}, profit: {ai_analysis.expected_profit_pips:.1f} pips)")
            
            return decision_obj
            
        except Exception as e:
            logger.error(f"Error making trading decision for {symbol}: {str(e)}")
            return self._create_default_decision(base_signal)
    
    def _calculate_decision_score(self, analysis: AISignalAnalysis, 
                                base_signal: Optional[AggregatedSignal]) -> float:
        """Calculate weighted decision score from AI analysis"""
        try:
            weights = self.config['models']
            score = 0.5  # Neutral starting point
            
            # Signal quality component
            if weights['signal_quality']['enabled']:
                quality_weight = weights['signal_quality']['weight']
                quality_score = analysis.signal_quality_score
                score += (quality_score - 0.5) * quality_weight
            
            # Profit expectation component
            if weights['profit_predictor']['enabled']:
                profit_weight = weights['profit_predictor']['weight']
                # Normalize profit to -1 to 1 scale
                profit_normalized = np.tanh(analysis.expected_profit_pips / 20.0)
                score += profit_normalized * profit_weight
            
            # Duration component (shorter expected duration is better for most strategies)
            if weights['duration_predictor']['enabled']:
                duration_weight = weights['duration_predictor']['weight']
                # Prefer trades that complete within reasonable time
                max_duration = weights['duration_predictor']['max_duration']
                duration_score = max(0, 1 - analysis.expected_duration_minutes / max_duration)
                score += (duration_score - 0.5) * duration_weight
            
            # Risk component (lower risk is better)
            if weights['risk_assessor']['enabled']:
                risk_weight = weights['risk_assessor']['weight']
                max_risk = weights['risk_assessor']['max_risk']
                risk_score = max(0, 1 - analysis.max_risk_percent / max_risk)
                score += (risk_score - 0.5) * risk_weight
            
            # Base signal influence
            if base_signal:
                base_influence = 0.1  # 10% influence from traditional signals
                if base_signal.final_signal == SignalType.BUY:
                    score += base_signal.strength * base_influence
                elif base_signal.final_signal == SignalType.SELL:
                    score -= base_signal.strength * base_influence
            
            # Risk-reward ratio bonus
            if analysis.risk_reward_ratio > self.config['risk_management']['min_risk_reward_ratio']:
                score += 0.05  # 5% bonus for good risk-reward
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating decision score: {str(e)}")
            return 0.5
    
    def _score_to_decision(self, score: float) -> AIDecisionType:
        """Map decision score to decision type"""
        thresholds = self.config['decision_thresholds']
        
        if score >= thresholds['strong_buy_score']:
            return AIDecisionType.STRONG_BUY
        elif score >= thresholds['buy_score']:
            return AIDecisionType.BUY
        elif score >= thresholds['weak_buy_score']:
            return AIDecisionType.WEAK_BUY
        elif score <= thresholds['strong_sell_score']:
            return AIDecisionType.STRONG_SELL
        elif score <= thresholds['sell_score']:
            return AIDecisionType.SELL
        elif score <= thresholds['weak_sell_score']:
            return AIDecisionType.WEAK_SELL
        else:
            return AIDecisionType.HOLD
    
    def _calculate_position_multiplier(self, analysis: AISignalAnalysis) -> float:
        """Calculate position size multiplier based on AI analysis"""
        try:
            base_multiplier = 1.0
            
            # Adjust for signal quality
            quality_adjustment = analysis.signal_quality_score
            
            # Adjust for risk
            max_risk = self.config['models']['risk_assessor']['max_risk']
            risk_adjustment = max(0.1, 1.0 - analysis.max_risk_percent / max_risk)
            
            # Adjust for confidence
            confidence_adjustment = analysis.confidence_level
            
            # Combine adjustments
            multiplier = base_multiplier * quality_adjustment * risk_adjustment * confidence_adjustment
            
            # Apply limits
            min_multiplier = self.config['risk_management']['min_position_size_multiplier']
            max_multiplier = self.config['risk_management']['max_position_size_multiplier']
            
            return np.clip(multiplier, min_multiplier, max_multiplier)
            
        except Exception as e:
            logger.error(f"Error calculating position multiplier: {str(e)}")
            return 1.0
    
    def _calculate_stop_loss_adjustment(self, analysis: AISignalAnalysis) -> float:
        """Calculate stop loss adjustment based on risk assessment"""
        try:
            base_sl = self.config['risk_management']['default_stop_loss']
            max_adjustment = self.config['risk_management']['max_stop_loss_adjustment']
            
            # Adjust based on predicted risk
            risk_factor = analysis.max_risk_percent / 10.0  # Normalize around 10%
            adjustment = base_sl * (1.0 + risk_factor * max_adjustment)
            
            return min(adjustment, base_sl * (1 + max_adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating stop loss adjustment: {str(e)}")
            return self.config['risk_management']['default_stop_loss']
    
    def _calculate_take_profit_targets(self, analysis: AISignalAnalysis) -> List[float]:
        """Calculate multiple take profit targets"""
        try:
            expected_profit = analysis.expected_profit_pips
            
            if expected_profit <= 0:
                return [10.0]  # Default TP
            
            # Multiple targets at different levels
            targets = [
                expected_profit * 0.5,  # Conservative target
                expected_profit * 0.8,  # Medium target
                expected_profit * 1.2,  # Optimistic target
            ]
            
            return [max(5.0, target) for target in targets]
            
        except Exception as e:
            logger.error(f"Error calculating take profit targets: {str(e)}")
            return [10.0, 15.0, 20.0]
    
    def _calculate_overall_confidence(self, analysis: AISignalAnalysis,
                                    base_signal: Optional[AggregatedSignal]) -> float:
        """Calculate overall confidence in the decision"""
        try:
            confidence_components = [analysis.confidence_level]
            
            # Add base signal confidence if available
            if base_signal:
                confidence_components.append(base_signal.confidence)
            
            # Risk-reward ratio confidence
            if analysis.risk_reward_ratio > 2.0:
                confidence_components.append(0.8)
            elif analysis.risk_reward_ratio > 1.5:
                confidence_components.append(0.6)
            else:
                confidence_components.append(0.4)
            
            return np.mean(confidence_components)
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.5
    
    def _estimate_current_volatility(self, market_data: Dict[str, pd.DataFrame], 
                                   symbol: str) -> float:
        """Estimate current volatility for the symbol"""
        try:
            # Use M5 data for volatility estimation
            timeframe = 'M5' if 'M5' in market_data else list(market_data.keys())[0]
            df = market_data[timeframe]
            
            if len(df) < 20:
                return 0.01  # Default volatility
            
            # Calculate ATR-based volatility
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range[-14:])  # 14-period ATR
            
            # Normalize by price
            current_price = close[-1]
            volatility = atr / current_price if current_price > 0 else 0.01
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error estimating volatility: {str(e)}")
            return 0.01
    
    def _create_default_analysis(self) -> AISignalAnalysis:
        """Create default analysis when AI models fail"""
        return AISignalAnalysis(
            signal_quality_score=0.5,
            expected_profit_pips=0.0,
            expected_duration_minutes=60.0,
            max_risk_percent=5.0,
            confidence_level=0.0,
            risk_reward_ratio=1.0
        )
    
    def _create_default_decision(self, base_signal: Optional[AggregatedSignal]) -> AITradingDecision:
        """Create default decision when AI analysis fails"""
        return AITradingDecision(
            decision=AIDecisionType.HOLD,
            base_signal=base_signal,
            ai_analysis=self._create_default_analysis(),
            position_size_multiplier=1.0,
            stop_loss_adjustment=2.0,
            take_profit_targets=[10.0, 15.0, 20.0],
            time_horizon_minutes=60.0,
            confidence_score=0.0,
            processing_time_ms=0.0,
            model_versions={},
            feature_count=0
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get AI system status"""
        return {
            'models_loaded': self.models_loaded,
            'models_trained': self.models_trained,
            'feature_collector_ready': self.feature_collector is not None,
            'strategy_engine_ready': self.strategy_engine is not None,
            'prediction_history_count': len(self.prediction_history),
            'ai_models_available': AI_MODELS_AVAILABLE,
            'strategy_components_available': STRATEGY_COMPONENTS_AVAILABLE
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of AI predictions"""
        try:
            if not self.prediction_history:
                return {}
            
            df = pd.DataFrame(self.prediction_history)
            
            return {
                'total_predictions': len(df),
                'avg_confidence': df['confidence'].mean(),
                'avg_expected_profit': df['expected_profit'].mean(),
                'avg_max_risk': df['max_risk'].mean(),
                'decision_distribution': df['decision'].value_counts().to_dict(),
                'recent_activity': len(df[df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(hours=24)])
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    manager = AIStrategyManager()
    
    # Check system status
    status = manager.get_system_status()
    print("AI Strategy Manager Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Create sample market data
    sample_data = {
        'M5': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='5T'))
    }
    
    # Ensure OHLC consistency
    sample_data['M5']['high'] = np.maximum.reduce([
        sample_data['M5']['open'], 
        sample_data['M5']['high'], 
        sample_data['M5']['low'], 
        sample_data['M5']['close']
    ])
    sample_data['M5']['low'] = np.minimum.reduce([
        sample_data['M5']['open'], 
        sample_data['M5']['high'], 
        sample_data['M5']['low'], 
        sample_data['M5']['close']
    ])
    
    # Test AI analysis (will use defaults since models aren't trained)
    try:
        analysis = manager.analyze_signal(sample_data, "R_75")
        print(f"\nSample AI Analysis:")
        print(f"  Signal Quality: {analysis.signal_quality_score:.3f}")
        print(f"  Expected Profit: {analysis.expected_profit_pips:.1f} pips")
        print(f"  Expected Duration: {analysis.expected_duration_minutes:.1f} minutes")
        print(f"  Max Risk: {analysis.max_risk_percent:.2f}%")
        print(f"  Confidence: {analysis.confidence_level:.3f}")
        
        # Test trading decision
        decision = manager.make_trading_decision(sample_data, "R_75")
        print(f"\nSample Trading Decision:")
        print(f"  Decision: {decision.decision.value}")
        print(f"  Position Multiplier: {decision.position_size_multiplier:.2f}")
        print(f"  Stop Loss Adjustment: {decision.stop_loss_adjustment:.2f}%")
        print(f"  Take Profit Targets: {decision.take_profit_targets}")
        print(f"  Confidence: {decision.confidence_score:.3f}")
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
    
    print("\nAI Strategy Manager implementation completed!")