"""
Robustness Analysis
Comprehensive robustness testing across different market conditions and stress scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Market condition classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility" 
    LOW_VOLATILITY = "low_volatility"
    CRISIS_PERIOD = "crisis_period"
    RECOVERY_PERIOD = "recovery_period"

class StressScenario(Enum):
    """Stress testing scenarios"""
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_CHANGE = "regime_change"
    BLACK_SWAN = "black_swan"

@dataclass
class MarketPeriod:
    """Container for market period analysis"""
    period_id: str
    start_date: datetime
    end_date: datetime
    condition: MarketCondition
    data: pd.DataFrame
    performance: Dict[str, float]
    characteristics: Dict[str, float]

@dataclass
class StressTestResult:
    """Container for stress test results"""
    scenario: StressScenario
    stressed_returns: np.ndarray
    performance_impact: Dict[str, float]
    risk_metrics: Dict[str, float]
    recovery_time: Optional[int]
    survival_probability: float

@dataclass
class RobustnessResults:
    """Container for complete robustness analysis"""
    overall_robustness_score: float
    
    # Market condition analysis
    market_condition_performance: Dict[MarketCondition, Dict[str, float]]
    condition_stability: Dict[str, float]
    worst_performing_condition: MarketCondition
    best_performing_condition: MarketCondition
    
    # Stress testing results
    stress_test_results: Dict[StressScenario, StressTestResult]
    overall_stress_score: float
    stress_survival_rate: float
    
    # Parameter sensitivity
    parameter_sensitivity: Dict[str, float]
    stability_across_parameters: float
    
    # Time period analysis
    time_period_performance: List[MarketPeriod]
    performance_consistency: float
    regime_adaptability: float
    
    # Risk analysis
    tail_risk_analysis: Dict[str, float]
    correlation_risk: Dict[str, float]
    
    # Metadata
    analysis_timestamp: datetime
    robustness_metadata: Dict[str, Any]

class RobustnessAnalyzer:
    """
    Comprehensive robustness analyzer for trading strategies across different
    market conditions, stress scenarios, and parameter variations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize robustness analyzer
        
        Args:
            config: Configuration for robustness analysis
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for robustness analysis"""
        return {
            'market_conditions': {
                'bull_threshold': 0.15,      # 15% annual return
                'bear_threshold': -0.10,     # -10% annual return
                'high_vol_threshold': 0.25,   # 25% annualized volatility
                'low_vol_threshold': 0.10,    # 10% annualized volatility
                'min_period_days': 60        # Minimum days for condition
            },
            'stress_testing': {
                'crash_magnitude': -0.20,    # -20% crash
                'volatility_spike': 3.0,     # 3x normal volatility
                'liquidity_impact': 0.005,   # 0.5% additional cost
                'correlation_stress': 0.9    # High correlation scenario
            },
            'parameter_testing': {
                'variation_range': 0.2,      # ±20% parameter variation
                'test_points': 5,            # Number of test points per parameter
                'stability_threshold': 0.8   # Minimum stability score
            },
            'time_analysis': {
                'window_months': 6,          # Rolling window size
                'overlap_months': 3,         # Window overlap
                'min_observations': 50       # Minimum observations per window
            }
        }
    
    def run_robustness_analysis(self, 
                               data: pd.DataFrame,
                               strategy_function: Callable,
                               base_parameters: Dict[str, Any],
                               benchmark_data: Optional[pd.DataFrame] = None) -> RobustnessResults:
        """
        Run comprehensive robustness analysis
        
        Args:
            data: Historical price data
            strategy_function: Strategy function to test
            base_parameters: Base strategy parameters
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Complete robustness analysis results
        """
        try:
            logger.info("Starting comprehensive robustness analysis...")
            
            # Market condition analysis
            market_periods = self._identify_market_conditions(data)
            condition_performance = self._analyze_market_conditions(
                market_periods, strategy_function, base_parameters
            )
            
            # Stress testing
            stress_results = self._run_stress_tests(
                data, strategy_function, base_parameters
            )
            
            # Parameter sensitivity analysis
            param_sensitivity = self._analyze_parameter_sensitivity(
                data, strategy_function, base_parameters
            )
            
            # Time period analysis
            time_performance = self._analyze_time_periods(
                data, strategy_function, base_parameters
            )
            
            # Risk analysis
            tail_risk = self._analyze_tail_risks(
                data, strategy_function, base_parameters
            )
            
            # Calculate overall scores
            overall_robustness = self._calculate_overall_robustness(
                condition_performance, stress_results, param_sensitivity, time_performance
            )
            
            results = RobustnessResults(
                overall_robustness_score=overall_robustness,
                market_condition_performance=condition_performance,
                condition_stability=self._calculate_condition_stability(condition_performance),
                worst_performing_condition=self._find_worst_condition(condition_performance),
                best_performing_condition=self._find_best_condition(condition_performance),
                stress_test_results=stress_results,
                overall_stress_score=self._calculate_stress_score(stress_results),
                stress_survival_rate=self._calculate_survival_rate(stress_results),
                parameter_sensitivity=param_sensitivity,
                stability_across_parameters=self._calculate_param_stability(param_sensitivity),
                time_period_performance=time_performance,
                performance_consistency=self._calculate_time_consistency(time_performance),
                regime_adaptability=self._calculate_regime_adaptability(time_performance),
                tail_risk_analysis=tail_risk,
                correlation_risk={},  # Would be implemented for multi-asset strategies
                analysis_timestamp=datetime.now(),
                robustness_metadata=self._generate_metadata(data, base_parameters)
            )
            
            logger.info("Robustness analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in robustness analysis: {str(e)}")
            raise
    
    def _identify_market_conditions(self, data: pd.DataFrame) -> List[MarketPeriod]:
        """Identify different market condition periods"""
        try:
            periods = []
            window_days = 252  # 1 year rolling window
            min_period = self.config['market_conditions']['min_period_days']
            
            bull_threshold = self.config['market_conditions']['bull_threshold']
            bear_threshold = self.config['market_conditions']['bear_threshold']
            high_vol_threshold = self.config['market_conditions']['high_vol_threshold']
            low_vol_threshold = self.config['market_conditions']['low_vol_threshold']
            
            # Calculate rolling metrics
            returns = data['close'].pct_change().dropna()
            rolling_return = returns.rolling(window_days).apply(lambda x: (1 + x).prod() ** (252/len(x)) - 1)
            rolling_vol = returns.rolling(window_days).std() * np.sqrt(252)
            
            # Identify periods
            current_condition = None
            period_start = None
            
            for i, (date, ret, vol) in enumerate(zip(data.index, rolling_return, rolling_vol)):
                if pd.isna(ret) or pd.isna(vol):
                    continue
                
                # Determine condition
                if ret >= bull_threshold and vol <= high_vol_threshold:
                    condition = MarketCondition.BULL_MARKET
                elif ret <= bear_threshold:
                    condition = MarketCondition.BEAR_MARKET
                elif vol >= high_vol_threshold:
                    condition = MarketCondition.HIGH_VOLATILITY
                elif vol <= low_vol_threshold:
                    condition = MarketCondition.LOW_VOLATILITY
                else:
                    condition = MarketCondition.SIDEWAYS_MARKET
                
                # Check for condition change
                if condition != current_condition:
                    # End previous period
                    if current_condition is not None and period_start is not None:
                        period_end = date
                        period_days = (period_end - period_start).days
                        
                        if period_days >= min_period:
                            period_data = data[(data.index >= period_start) & (data.index <= period_end)]
                            
                            period = MarketPeriod(
                                period_id=f"{current_condition.value}_{period_start.strftime('%Y%m%d')}",
                                start_date=period_start,
                                end_date=period_end,
                                condition=current_condition,
                                data=period_data,
                                performance={},
                                characteristics=self._calculate_period_characteristics(period_data)
                            )
                            periods.append(period)
                    
                    # Start new period
                    current_condition = condition
                    period_start = date
            
            # Close final period
            if current_condition is not None and period_start is not None:
                period_end = data.index[-1]
                period_days = (period_end - period_start).days
                
                if period_days >= min_period:
                    period_data = data[(data.index >= period_start) & (data.index <= period_end)]
                    
                    period = MarketPeriod(
                        period_id=f"{current_condition.value}_{period_start.strftime('%Y%m%d')}",
                        start_date=period_start,
                        end_date=period_end,
                        condition=current_condition,
                        data=period_data,
                        performance={},
                        characteristics=self._calculate_period_characteristics(period_data)
                    )
                    periods.append(period)
            
            logger.info(f"Identified {len(periods)} market condition periods")
            return periods
            
        except Exception as e:
            logger.error(f"Error identifying market conditions: {str(e)}")
            return []
    
    def _calculate_period_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate characteristics of a market period"""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            characteristics = {
                'total_return': (data['close'].iloc[-1] / data['close'].iloc[0]) - 1,
                'annual_return': ((data['close'].iloc[-1] / data['close'].iloc[0]) ** (252 / len(data))) - 1,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'positive_days': (returns > 0).mean(),
                'avg_up_day': returns[returns > 0].mean() if (returns > 0).any() else 0,
                'avg_down_day': returns[returns < 0].mean() if (returns < 0).any() else 0
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error calculating period characteristics: {str(e)}")
            return {}
    
    def _analyze_market_conditions(self, market_periods: List[MarketPeriod],
                                 strategy_function: Callable,
                                 parameters: Dict[str, Any]) -> Dict[MarketCondition, Dict[str, float]]:
        """Analyze strategy performance across different market conditions"""
        try:
            condition_performance = {}
            
            for period in market_periods:
                try:
                    # Run strategy on this period
                    signals = strategy_function(period.data, parameters)
                    
                    if signals is None or len(signals) == 0:
                        continue
                    
                    # Calculate performance
                    performance = self._calculate_strategy_performance(period.data, signals)
                    
                    # Store results
                    if period.condition not in condition_performance:
                        condition_performance[period.condition] = []
                    
                    condition_performance[period.condition].append(performance)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing period {period.period_id}: {str(e)}")
                    continue
            
            # Aggregate performance by condition
            aggregated_performance = {}
            for condition, performances in condition_performance.items():
                if performances:
                    aggregated_performance[condition] = {
                        'avg_return': np.mean([p['total_return'] for p in performances]),
                        'avg_sharpe': np.mean([p['sharpe_ratio'] for p in performances]),
                        'avg_max_dd': np.mean([p['max_drawdown'] for p in performances]),
                        'win_rate': np.mean([p['win_rate'] for p in performances]),
                        'consistency': 1 - np.std([p['total_return'] for p in performances]),
                        'periods_count': len(performances)
                    }
            
            return aggregated_performance
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
    
    def _run_stress_tests(self, data: pd.DataFrame,
                         strategy_function: Callable,
                         parameters: Dict[str, Any]) -> Dict[StressScenario, StressTestResult]:
        """Run comprehensive stress tests"""
        try:
            stress_results = {}
            
            # Get baseline performance
            baseline_signals = strategy_function(data, parameters)
            baseline_performance = self._calculate_strategy_performance(data, baseline_signals)
            
            # Test each stress scenario
            scenarios = [
                StressScenario.MARKET_CRASH,
                StressScenario.VOLATILITY_SPIKE,
                StressScenario.LIQUIDITY_CRISIS,
                StressScenario.CORRELATION_BREAKDOWN
            ]
            
            for scenario in scenarios:
                try:
                    # Apply stress to data
                    stressed_data = self._apply_stress_scenario(data, scenario)
                    
                    # Run strategy on stressed data
                    stressed_signals = strategy_function(stressed_data, parameters)
                    
                    if stressed_signals is None:
                        continue
                    
                    # Calculate stressed performance
                    stressed_performance = self._calculate_strategy_performance(stressed_data, stressed_signals)
                    
                    # Calculate impact
                    performance_impact = {
                        'return_impact': stressed_performance['total_return'] - baseline_performance['total_return'],
                        'sharpe_impact': stressed_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio'],
                        'drawdown_impact': stressed_performance['max_drawdown'] - baseline_performance['max_drawdown']
                    }
                    
                    # Calculate survival probability
                    survival_prob = 1.0 if stressed_performance['total_return'] > -0.5 else 0.0
                    
                    stress_result = StressTestResult(
                        scenario=scenario,
                        stressed_returns=self._calculate_strategy_returns(stressed_data, stressed_signals),
                        performance_impact=performance_impact,
                        risk_metrics=self._calculate_stress_risk_metrics(stressed_performance),
                        recovery_time=None,  # Would need time series analysis
                        survival_probability=survival_prob
                    )
                    
                    stress_results[scenario] = stress_result
                    
                except Exception as e:
                    logger.warning(f"Error in stress test {scenario.value}: {str(e)}")
                    continue
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {str(e)}")
            return {}
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: StressScenario) -> pd.DataFrame:
        """Apply stress scenario to market data"""
        try:
            stressed_data = data.copy()
            
            if scenario == StressScenario.MARKET_CRASH:
                # Apply sudden crash
                crash_magnitude = self.config['stress_testing']['crash_magnitude']
                crash_day = len(data) // 2  # Middle of period
                
                multiplier = np.ones(len(data))
                multiplier[crash_day:] *= (1 + crash_magnitude)
                
                for col in ['open', 'high', 'low', 'close']:
                    stressed_data[col] *= multiplier
            
            elif scenario == StressScenario.VOLATILITY_SPIKE:
                # Increase volatility
                returns = data['close'].pct_change().fillna(0)
                vol_multiplier = self.config['stress_testing']['volatility_spike']
                
                stressed_returns = returns * vol_multiplier
                stressed_prices = data['close'].iloc[0] * (1 + stressed_returns).cumprod()
                
                # Scale OHLC proportionally
                price_ratio = stressed_prices / data['close']
                for col in ['open', 'high', 'low', 'close']:
                    stressed_data[col] *= price_ratio
            
            elif scenario == StressScenario.LIQUIDITY_CRISIS:
                # Increase bid-ask spreads (simulated as increased transaction costs)
                liquidity_impact = self.config['stress_testing']['liquidity_impact']
                
                # Add noise to prices to simulate wider spreads
                noise = np.random.normal(0, liquidity_impact, len(data))
                for col in ['open', 'high', 'low', 'close']:
                    stressed_data[col] *= (1 + noise)
            
            elif scenario == StressScenario.CORRELATION_BREAKDOWN:
                # For single asset, simulate regime change
                returns = data['close'].pct_change().fillna(0)
                
                # Reverse half of the returns to simulate correlation breakdown
                mid_point = len(returns) // 2
                modified_returns = returns.copy()
                modified_returns.iloc[mid_point:] *= -0.5
                
                stressed_prices = data['close'].iloc[0] * (1 + modified_returns).cumprod()
                price_ratio = stressed_prices / data['close']
                
                for col in ['open', 'high', 'low', 'close']:
                    stressed_data[col] *= price_ratio
            
            return stressed_data
            
        except Exception as e:
            logger.error(f"Error applying stress scenario {scenario.value}: {str(e)}")
            return data.copy()
    
    def _analyze_parameter_sensitivity(self, data: pd.DataFrame,
                                     strategy_function: Callable,
                                     base_parameters: Dict[str, Any]) -> Dict[str, float]:
        """Analyze sensitivity to parameter changes"""
        try:
            sensitivity_scores = {}
            variation_range = self.config['parameter_testing']['variation_range']
            test_points = self.config['parameter_testing']['test_points']
            
            # Get baseline performance
            baseline_signals = strategy_function(data, base_parameters)
            baseline_performance = self._calculate_strategy_performance(data, baseline_signals)
            baseline_sharpe = baseline_performance['sharpe_ratio']
            
            for param_name, base_value in base_parameters.items():
                try:
                    if not isinstance(base_value, (int, float)):
                        continue
                    
                    sharpe_values = []
                    
                    # Test parameter variations
                    for i in range(test_points):
                        variation = (i - test_points // 2) * variation_range / (test_points // 2)
                        test_value = base_value * (1 + variation)
                        
                        # Ensure positive values
                        if test_value <= 0:
                            continue
                        
                        test_parameters = base_parameters.copy()
                        test_parameters[param_name] = test_value
                        
                        # Run strategy with modified parameter
                        test_signals = strategy_function(data, test_parameters)
                        if test_signals is None:
                            continue
                        
                        test_performance = self._calculate_strategy_performance(data, test_signals)
                        sharpe_values.append(test_performance['sharpe_ratio'])
                    
                    # Calculate sensitivity
                    if len(sharpe_values) > 1:
                        sensitivity = np.std(sharpe_values) / abs(baseline_sharpe + 1e-8)
                        stability = 1 / (1 + sensitivity)  # Higher sensitivity = lower stability
                        sensitivity_scores[param_name] = stability
                    
                except Exception as e:
                    logger.warning(f"Error testing parameter {param_name}: {str(e)}")
                    continue
            
            return sensitivity_scores
            
        except Exception as e:
            logger.error(f"Error analyzing parameter sensitivity: {str(e)}")
            return {}
    
    def _analyze_time_periods(self, data: pd.DataFrame,
                            strategy_function: Callable,
                            parameters: Dict[str, Any]) -> List[MarketPeriod]:
        """Analyze performance across different time periods"""
        try:
            window_months = self.config['time_analysis']['window_months']
            overlap_months = self.config['time_analysis']['overlap_months']
            min_obs = self.config['time_analysis']['min_observations']
            
            periods = []
            window_days = window_months * 30
            step_days = (window_months - overlap_months) * 30
            
            start_idx = 0
            period_id = 0
            
            while start_idx + window_days < len(data):
                end_idx = start_idx + window_days
                
                # Extract period data
                period_data = data.iloc[start_idx:end_idx]
                
                if len(period_data) < min_obs:
                    start_idx += step_days
                    continue
                
                # Run strategy on this period
                try:
                    signals = strategy_function(period_data, parameters)
                    if signals is not None:
                        performance = self._calculate_strategy_performance(period_data, signals)
                    else:
                        performance = {}
                except:
                    performance = {}
                
                period = MarketPeriod(
                    period_id=f"period_{period_id}",
                    start_date=period_data.index[0],
                    end_date=period_data.index[-1],
                    condition=MarketCondition.SIDEWAYS_MARKET,  # Default
                    data=period_data,
                    performance=performance,
                    characteristics=self._calculate_period_characteristics(period_data)
                )
                
                periods.append(period)
                period_id += 1
                start_idx += step_days
            
            return periods
            
        except Exception as e:
            logger.error(f"Error analyzing time periods: {str(e)}")
            return []
    
    def _analyze_tail_risks(self, data: pd.DataFrame,
                          strategy_function: Callable,
                          parameters: Dict[str, Any]) -> Dict[str, float]:
        """Analyze tail risk characteristics"""
        try:
            signals = strategy_function(data, parameters)
            if signals is None:
                return {}
            
            returns = self._calculate_strategy_returns(data, signals)
            
            # Remove zero returns for analysis
            non_zero_returns = returns[returns != 0]
            
            if len(non_zero_returns) < 10:
                return {}
            
            # Calculate tail risk metrics
            var_95 = np.percentile(non_zero_returns, 5)
            var_99 = np.percentile(non_zero_returns, 1)
            
            # Expected shortfall
            es_95 = np.mean(non_zero_returns[non_zero_returns <= var_95])
            es_99 = np.mean(non_zero_returns[non_zero_returns <= var_99])
            
            # Tail ratio
            tail_ratio = abs(np.percentile(non_zero_returns, 95)) / abs(np.percentile(non_zero_returns, 5))
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'tail_ratio': tail_ratio,
                'left_tail_weight': np.mean(non_zero_returns < np.percentile(non_zero_returns, 10)),
                'right_tail_weight': np.mean(non_zero_returns > np.percentile(non_zero_returns, 90))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tail risks: {str(e)}")
            return {}
    
    def _calculate_strategy_performance(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        try:
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
            
            total_return = (1 + returns).prod() - 1
            
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Trade statistics
            trade_returns = returns[returns != 0]
            win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {str(e)}")
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> np.ndarray:
        """Calculate strategy returns from signals"""
        try:
            price_returns = data['close'].pct_change().fillna(0)
            
            # Align signals with returns
            if len(signals) != len(price_returns):
                min_len = min(len(signals), len(price_returns))
                signals = signals.iloc[:min_len]
                price_returns = price_returns.iloc[:min_len]
            
            strategy_returns = signals.shift(1) * price_returns
            return strategy_returns.fillna(0).values
            
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {str(e)}")
            return np.array([])
    
    def _calculate_max_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate maximum drawdown"""
        try:
            if isinstance(returns, pd.Series):
                returns = returns.values
            
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_stress_risk_metrics(self, performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk metrics specific to stress testing"""
        return {
            'stressed_sharpe': performance.get('sharpe_ratio', 0),
            'stressed_drawdown': performance.get('max_drawdown', 0),
            'stressed_return': performance.get('total_return', 0),
            'risk_adjusted_return': performance.get('total_return', 0) / max(performance.get('max_drawdown', 1), 0.01)
        }
    
    def _calculate_overall_robustness(self, condition_performance: Dict,
                                    stress_results: Dict,
                                    param_sensitivity: Dict,
                                    time_performance: List) -> float:
        """Calculate overall robustness score"""
        try:
            scores = []
            
            # Market condition stability
            if condition_performance:
                condition_scores = [perf.get('consistency', 0) for perf in condition_performance.values()]
                scores.append(np.mean(condition_scores))
            
            # Stress test survival
            if stress_results:
                survival_rates = [result.survival_probability for result in stress_results.values()]
                scores.append(np.mean(survival_rates))
            
            # Parameter stability
            if param_sensitivity:
                scores.append(np.mean(list(param_sensitivity.values())))
            
            # Time consistency
            if time_performance:
                time_returns = [p.performance.get('total_return', 0) for p in time_performance if p.performance]
                if time_returns:
                    time_consistency = 1 - (np.std(time_returns) / (abs(np.mean(time_returns)) + 1e-8))
                    scores.append(max(0, time_consistency))
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall robustness: {str(e)}")
            return 0.0
    
    def _calculate_condition_stability(self, condition_performance: Dict) -> Dict[str, float]:
        """Calculate stability metrics for market conditions"""
        stability = {}
        
        for condition, performance in condition_performance.items():
            stability[condition.value] = performance.get('consistency', 0)
        
        return stability
    
    def _find_worst_condition(self, condition_performance: Dict) -> MarketCondition:
        """Find worst performing market condition"""
        try:
            worst_score = float('inf')
            worst_condition = MarketCondition.SIDEWAYS_MARKET
            
            for condition, performance in condition_performance.items():
                score = performance.get('avg_sharpe', 0)
                if score < worst_score:
                    worst_score = score
                    worst_condition = condition
            
            return worst_condition
        except:
            return MarketCondition.SIDEWAYS_MARKET
    
    def _find_best_condition(self, condition_performance: Dict) -> MarketCondition:
        """Find best performing market condition"""
        try:
            best_score = float('-inf')
            best_condition = MarketCondition.SIDEWAYS_MARKET
            
            for condition, performance in condition_performance.items():
                score = performance.get('avg_sharpe', 0)
                if score > best_score:
                    best_score = score
                    best_condition = condition
            
            return best_condition
        except:
            return MarketCondition.SIDEWAYS_MARKET
    
    def _calculate_stress_score(self, stress_results: Dict) -> float:
        """Calculate overall stress test score"""
        try:
            if not stress_results:
                return 0.0
            
            scores = []
            for result in stress_results.values():
                # Score based on survival and performance impact
                survival_score = result.survival_probability
                impact_score = max(0, 1 + result.performance_impact.get('return_impact', -1))
                combined_score = (survival_score + impact_score) / 2
                scores.append(combined_score)
            
            return np.mean(scores)
        except:
            return 0.0
    
    def _calculate_survival_rate(self, stress_results: Dict) -> float:
        """Calculate overall survival rate across stress tests"""
        try:
            if not stress_results:
                return 0.0
            
            survival_rates = [result.survival_probability for result in stress_results.values()]
            return np.mean(survival_rates)
        except:
            return 0.0
    
    def _calculate_param_stability(self, param_sensitivity: Dict) -> float:
        """Calculate overall parameter stability"""
        try:
            if not param_sensitivity:
                return 0.0
            
            return np.mean(list(param_sensitivity.values()))
        except:
            return 0.0
    
    def _calculate_time_consistency(self, time_performance: List) -> float:
        """Calculate time-based performance consistency"""
        try:
            if not time_performance:
                return 0.0
            
            returns = [p.performance.get('total_return', 0) for p in time_performance if p.performance]
            
            if len(returns) < 2:
                return 0.0
            
            consistency = 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8))
            return max(0, consistency)
        except:
            return 0.0
    
    def _calculate_regime_adaptability(self, time_performance: List) -> float:
        """Calculate ability to adapt to regime changes"""
        try:
            if len(time_performance) < 3:
                return 0.0
            
            # Measure how quickly performance recovers after poor periods
            returns = [p.performance.get('total_return', 0) for p in time_performance if p.performance]
            
            recovery_scores = []
            for i in range(1, len(returns) - 1):
                if returns[i] < 0 and returns[i+1] > returns[i]:
                    recovery = (returns[i+1] - returns[i]) / abs(returns[i])
                    recovery_scores.append(min(recovery, 1.0))
            
            return np.mean(recovery_scores) if recovery_scores else 0.0
        except:
            return 0.0
    
    def _generate_metadata(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for robustness analysis"""
        try:
            return {
                'data_points': len(data),
                'analysis_period': f"{data.index[0]} to {data.index[-1]}",
                'base_parameters': parameters,
                'analysis_timestamp': datetime.now().isoformat(),
                'config': self.config
            }
        except:
            return {}

if __name__ == "__main__":
    # Example usage
    print("Robustness Analysis Test")
    print("=" * 40)
    
    # Create sample data with different market regimes
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Generate data with bull, bear, and sideways periods
    n_days = len(dates)
    returns = []
    
    # Bull market (first third)
    bull_days = n_days // 3
    bull_returns = np.random.normal(0.001, 0.015, bull_days)  # Positive bias, lower vol
    returns.extend(bull_returns)
    
    # Bear market (second third)
    bear_days = n_days // 3
    bear_returns = np.random.normal(-0.0005, 0.025, bear_days)  # Negative bias, higher vol
    returns.extend(bear_returns)
    
    # Sideways market (last third)
    sideways_days = n_days - bull_days - bear_days
    sideways_returns = np.random.normal(0, 0.012, sideways_days)  # No bias, medium vol
    returns.extend(sideways_returns)
    
    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        ohlc = [
            sample_data['open'].iloc[i],
            sample_data['high'].iloc[i],
            sample_data['low'].iloc[i],
            sample_data['close'].iloc[i]
        ]
        sample_data['high'].iloc[i] = max(ohlc)
        sample_data['low'].iloc[i] = min(ohlc)
    
    # Define a simple strategy
    def simple_trend_strategy(data, params):
        """Simple trend following strategy"""
        short_ma = params.get('short_ma', 10)
        long_ma = params.get('long_ma', 30)
        
        short_sma = data['close'].rolling(short_ma).mean()
        long_sma = data['close'].rolling(long_ma).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_sma > long_sma] = 1
        signals[short_sma < long_sma] = -1
        
        return signals
    
    # Define strategy parameters
    base_parameters = {
        'short_ma': 10,
        'long_ma': 30
    }
    
    # Run robustness analysis
    analyzer = RobustnessAnalyzer()
    
    try:
        print("Running comprehensive robustness analysis...")
        results = analyzer.run_robustness_analysis(
            data=sample_data,
            strategy_function=simple_trend_strategy,
            base_parameters=base_parameters
        )
        
        print(f"\nRobustness Analysis Results:")
        print(f"=" * 40)
        print(f"Overall Robustness Score: {results.overall_robustness_score:.3f}")
        print(f"Overall Stress Score: {results.overall_stress_score:.3f}")
        print(f"Stress Survival Rate: {results.stress_survival_rate:.1%}")
        print(f"Parameter Stability: {results.stability_across_parameters:.3f}")
        print(f"Performance Consistency: {results.performance_consistency:.3f}")
        print(f"Regime Adaptability: {results.regime_adaptability:.3f}")
        
        print(f"\nMarket Condition Performance:")
        for condition, performance in results.market_condition_performance.items():
            print(f"  {condition.value}:")
            print(f"    Avg Return: {performance.get('avg_return', 0):.2%}")
            print(f"    Avg Sharpe: {performance.get('avg_sharpe', 0):.3f}")
            print(f"    Consistency: {performance.get('consistency', 0):.3f}")
            print(f"    Periods: {performance.get('periods_count', 0)}")
        
        print(f"\nBest/Worst Conditions:")
        print(f"  Best: {results.best_performing_condition.value}")
        print(f"  Worst: {results.worst_performing_condition.value}")
        
        print(f"\nStress Test Results:")
        for scenario, result in results.stress_test_results.items():
            print(f"  {scenario.value}:")
            print(f"    Survival: {result.survival_probability:.1%}")
            print(f"    Return Impact: {result.performance_impact.get('return_impact', 0):.2%}")
            print(f"    Sharpe Impact: {result.performance_impact.get('sharpe_impact', 0):.3f}")
        
        if results.parameter_sensitivity:
            print(f"\nParameter Sensitivity:")
            for param, stability in results.parameter_sensitivity.items():
                print(f"  {param}: {stability:.3f}")
        
        if results.tail_risk_analysis:
            print(f"\nTail Risk Analysis:")
            print(f"  VaR 95%: {results.tail_risk_analysis.get('var_95', 0):.2%}")
            print(f"  VaR 99%: {results.tail_risk_analysis.get('var_99', 0):.2%}")
            print(f"  Expected Shortfall 95%: {results.tail_risk_analysis.get('expected_shortfall_95', 0):.2%}")
            print(f"  Tail Ratio: {results.tail_risk_analysis.get('tail_ratio', 0):.3f}")
        
        print(f"\nTime Period Analysis:")
        print(f"  Periods Analyzed: {len(results.time_period_performance)}")
        if results.time_period_performance:
            avg_return = np.mean([p.performance.get('total_return', 0) for p in results.time_period_performance if p.performance])
            print(f"  Average Period Return: {avg_return:.2%}")
        
    except Exception as e:
        print(f"Error in robustness analysis: {str(e)}")
    
    print("\nRobustness Analysis implementation completed!")