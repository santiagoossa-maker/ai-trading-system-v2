"""
Walk-Forward Analysis
Rigorous out-of-sample testing with rolling windows for strategy validation
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

class ValidationMethod(Enum):
    """Validation methods for walk-forward analysis"""
    FIXED_WINDOW = "fixed_window"
    EXPANDING_WINDOW = "expanding_window" 
    ROLLING_WINDOW = "rolling_window"

class OptimizationMetric(Enum):
    """Metrics for parameter optimization"""
    SHARPE_RATIO = "sharpe_ratio"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"

@dataclass
class WalkForwardPeriod:
    """Container for walk-forward analysis period"""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimal_parameters: Dict[str, Any]
    test_performance: Dict[str, float]

@dataclass
class BacktestResults:
    """Container for backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time series data
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_log: pd.DataFrame
    
    # Additional metrics
    start_date: datetime
    end_date: datetime
    total_days: int
    metadata: Dict[str, Any]

@dataclass
class WalkForwardResults:
    """Container for complete walk-forward analysis results"""
    overall_performance: BacktestResults
    period_results: List[WalkForwardPeriod]
    parameter_stability: Dict[str, float]
    degradation_analysis: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    optimization_path: pd.DataFrame
    validation_method: ValidationMethod
    analysis_metadata: Dict[str, Any]

class WalkForwardAnalyzer:
    """
    Sophisticated walk-forward analysis engine for robust strategy validation
    with rolling optimization windows and out-of-sample testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize walk-forward analyzer
        
        Args:
            config: Configuration for walk-forward analysis
        """
        self.config = config or self._get_default_config()
        self.analysis_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for walk-forward analysis"""
        return {
            'validation': {
                'method': ValidationMethod.ROLLING_WINDOW,
                'train_months': 6,
                'test_months': 1,
                'min_train_samples': 500,
                'overlap_allowed': False
            },
            'optimization': {
                'metric': OptimizationMetric.SHARPE_RATIO,
                'max_iterations': 100,
                'parameter_stability_threshold': 0.7,
                'convergence_tolerance': 1e-6
            },
            'performance': {
                'benchmark_return': 0.0,  # Risk-free rate
                'transaction_cost': 0.0001,  # 1 pip spread
                'slippage': 0.0001,
                'commission': 0.0
            },
            'risk': {
                'max_drawdown_limit': 0.2,  # 20%
                'var_confidence': 0.05,  # 95% VaR
                'max_leverage': 1.0
            }
        }
    
    def run_walk_forward_analysis(self, 
                                data: pd.DataFrame,
                                strategy_function: Callable,
                                parameter_ranges: Dict[str, List],
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> WalkForwardResults:
        """
        Run comprehensive walk-forward analysis
        
        Args:
            data: Historical price data with OHLCV columns
            strategy_function: Function that takes (data, parameters) and returns signals
            parameter_ranges: Dictionary of parameter names and their test ranges
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Complete walk-forward analysis results
        """
        try:
            logger.info("Starting walk-forward analysis...")
            
            # Prepare data
            analysis_data = self._prepare_data(data, start_date, end_date)
            
            # Generate walk-forward periods
            periods = self._generate_walk_forward_periods(analysis_data)
            logger.info(f"Generated {len(periods)} walk-forward periods")
            
            # Run analysis for each period
            period_results = []
            optimization_path = []
            
            for i, period in enumerate(periods):
                logger.info(f"Processing period {i+1}/{len(periods)}: "
                           f"{period.train_start.date()} to {period.test_end.date()}")
                
                # Optimize parameters on training data
                optimal_params = self._optimize_parameters(
                    period.train_data, strategy_function, parameter_ranges
                )
                
                # Test on out-of-sample data
                test_performance = self._backtest_strategy(
                    period.test_data, strategy_function, optimal_params
                )
                
                # Store results
                period_result = WalkForwardPeriod(
                    period_id=i,
                    train_start=period.train_start,
                    train_end=period.train_end,
                    test_start=period.test_start,
                    test_end=period.test_end,
                    train_data=period.train_data,
                    test_data=period.test_data,
                    optimal_parameters=optimal_params,
                    test_performance=test_performance
                )
                
                period_results.append(period_result)
                
                # Track optimization path
                optimization_path.append({
                    'period': i,
                    'train_start': period.train_start,
                    'test_start': period.test_start,
                    **optimal_params,
                    **test_performance
                })
            
            # Aggregate results
            overall_performance = self._aggregate_period_results(period_results, analysis_data)
            
            # Analyze parameter stability
            parameter_stability = self._analyze_parameter_stability(period_results)
            
            # Analyze performance degradation
            degradation_analysis = self._analyze_degradation(period_results)
            
            # Calculate out-of-sample performance
            oos_performance = self._calculate_oos_performance(period_results)
            
            results = WalkForwardResults(
                overall_performance=overall_performance,
                period_results=period_results,
                parameter_stability=parameter_stability,
                degradation_analysis=degradation_analysis,
                out_of_sample_performance=oos_performance,
                optimization_path=pd.DataFrame(optimization_path),
                validation_method=self.config['validation']['method'],
                analysis_metadata=self._generate_metadata(analysis_data, parameter_ranges)
            )
            
            logger.info("Walk-forward analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {str(e)}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Prepare and validate data for analysis"""
        try:
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter by date range
            analysis_data = data.copy()
            if start_date:
                analysis_data = analysis_data[analysis_data.index >= start_date]
            if end_date:
                analysis_data = analysis_data[analysis_data.index <= end_date]
            
            # Validate data consistency
            if (analysis_data['high'] < analysis_data['low']).any():
                logger.warning("Found inconsistent OHLC data (high < low)")
            
            # Remove any NaN values
            analysis_data = analysis_data.dropna()
            
            # Ensure minimum data requirements
            min_samples = self.config['validation']['min_train_samples'] * 2
            if len(analysis_data) < min_samples:
                raise ValueError(f"Insufficient data: {len(analysis_data)} < {min_samples} required")
            
            logger.info(f"Prepared {len(analysis_data)} data points for analysis")
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _generate_walk_forward_periods(self, data: pd.DataFrame) -> List[WalkForwardPeriod]:
        """Generate walk-forward analysis periods"""
        try:
            periods = []
            
            train_months = self.config['validation']['train_months']
            test_months = self.config['validation']['test_months']
            method = self.config['validation']['method']
            
            # Calculate period lengths in days (approximate)
            train_days = train_months * 30
            test_days = test_months * 30
            step_days = test_days  # Non-overlapping by default
            
            start_date = data.index[0]
            end_date = data.index[-1]
            
            current_start = start_date
            period_id = 0
            
            while True:
                # Calculate period boundaries
                train_start = current_start
                train_end = train_start + timedelta(days=train_days)
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=test_days)
                
                # Check if we have enough data
                if test_end > end_date:
                    break
                
                # Extract period data
                train_data = data[(data.index >= train_start) & (data.index <= train_end)]
                test_data = data[(data.index >= test_start) & (data.index <= test_end)]
                
                # Validate minimum samples
                if len(train_data) < self.config['validation']['min_train_samples']:
                    logger.warning(f"Skipping period {period_id}: insufficient training data")
                    current_start += timedelta(days=step_days)
                    continue
                
                if len(test_data) < 10:  # Minimum test samples
                    logger.warning(f"Skipping period {period_id}: insufficient test data")
                    current_start += timedelta(days=step_days)
                    continue
                
                period = WalkForwardPeriod(
                    period_id=period_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_data=train_data,
                    test_data=test_data,
                    optimal_parameters={},
                    test_performance={}
                )
                
                periods.append(period)
                period_id += 1
                
                # Move to next period
                if method == ValidationMethod.EXPANDING_WINDOW:
                    # Keep same start, extend training window
                    current_start = start_date
                    train_days += step_days * 30  # Extend training window
                else:
                    # Rolling window: move start forward
                    current_start += timedelta(days=step_days)
            
            return periods
            
        except Exception as e:
            logger.error(f"Error generating walk-forward periods: {str(e)}")
            return []
    
    def _optimize_parameters(self, train_data: pd.DataFrame,
                           strategy_function: Callable,
                           parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters on training data"""
        try:
            metric = self.config['optimization']['metric']
            max_iterations = self.config['optimization']['max_iterations']
            
            best_score = -np.inf if metric in [OptimizationMetric.SHARPE_RATIO, OptimizationMetric.PROFIT_FACTOR] else np.inf
            best_params = {}
            
            # Generate parameter combinations (grid search for simplicity)
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            # Limit combinations if too many
            if len(param_combinations) > max_iterations:
                np.random.shuffle(param_combinations)
                param_combinations = param_combinations[:max_iterations]
            
            logger.debug(f"Testing {len(param_combinations)} parameter combinations")
            
            for i, params in enumerate(param_combinations):
                try:
                    # Run backtest with these parameters
                    performance = self._backtest_strategy(train_data, strategy_function, params)
                    
                    # Extract optimization metric
                    if metric == OptimizationMetric.SHARPE_RATIO:
                        score = performance.get('sharpe_ratio', 0)
                        is_better = score > best_score
                    elif metric == OptimizationMetric.PROFIT_FACTOR:
                        score = performance.get('profit_factor', 0)
                        is_better = score > best_score
                    elif metric == OptimizationMetric.MAX_DRAWDOWN:
                        score = performance.get('max_drawdown', 1)
                        is_better = score < best_score  # Lower is better
                    elif metric == OptimizationMetric.WIN_RATE:
                        score = performance.get('win_rate', 0)
                        is_better = score > best_score
                    else:
                        score = performance.get('sharpe_ratio', 0)
                        is_better = score > best_score
                    
                    if is_better:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.debug(f"Error testing parameters {params}: {str(e)}")
                    continue
            
            if not best_params:
                # Fallback to default parameters
                best_params = {key: values[len(values)//2] for key, values in parameter_ranges.items()}
                logger.warning("No valid parameters found, using defaults")
            
            logger.debug(f"Optimal parameters: {best_params} (score: {best_score:.4f})")
            return best_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return {key: values[0] for key, values in parameter_ranges.items()}
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for testing"""
        try:
            import itertools
            
            keys = list(parameter_ranges.keys())
            values = list(parameter_ranges.values())
            
            combinations = []
            for combo in itertools.product(*values):
                param_dict = dict(zip(keys, combo))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {str(e)}")
            return []
    
    def _backtest_strategy(self, data: pd.DataFrame,
                          strategy_function: Callable,
                          parameters: Dict[str, Any]) -> Dict[str, float]:
        """Run backtest with given strategy and parameters"""
        try:
            # Generate signals using strategy function
            signals = strategy_function(data, parameters)
            
            if signals is None or len(signals) == 0:
                return self._get_default_performance()
            
            # Ensure signals align with data
            if len(signals) != len(data):
                logger.warning(f"Signal length mismatch: {len(signals)} vs {len(data)}")
                min_len = min(len(signals), len(data))
                signals = signals[:min_len]
                data = data.iloc[:min_len]
            
            # Calculate returns
            returns = self._calculate_strategy_returns(data, signals)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(returns, data)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return self._get_default_performance()
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns from price data and signals"""
        try:
            # Calculate price returns
            price_returns = data['close'].pct_change().fillna(0)
            
            # Apply transaction costs
            transaction_cost = self.config['performance']['transaction_cost']
            
            # Calculate position changes
            position_changes = signals.diff().abs()
            
            # Apply costs when position changes
            costs = position_changes * transaction_cost
            
            # Calculate strategy returns
            strategy_returns = (signals.shift(1) * price_returns) - costs
            
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {str(e)}")
            return pd.Series([0] * len(data), index=data.index)
    
    def _calculate_performance_metrics(self, returns: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if len(returns) == 0 or returns.std() == 0:
                return self._get_default_performance()
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.config['performance']['benchmark_return']) / volatility if volatility > 0 else 0
            
            # Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Trade statistics
            trades = self._analyze_trades(returns)
            
            # Additional ratios
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (annual_return - self.config['performance']['benchmark_return']) / downside_vol if downside_vol > 0 else 0
            
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            performance = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': trades['win_rate'],
                'profit_factor': trades['profit_factor'],
                'total_trades': trades['total_trades'],
                'avg_win': trades['avg_win'],
                'avg_loss': trades['avg_loss']
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._get_default_performance()
    
    def _analyze_trades(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze individual trades from returns"""
        try:
            # Identify trade periods (non-zero returns)
            trade_returns = returns[returns != 0]
            
            if len(trade_returns) == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_win': 0,
                    'avg_loss': 0
                }
            
            # Basic trade statistics
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            total_trades = len(trade_returns)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            
            total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0}
    
    def _get_default_performance(self) -> Dict[str, float]:
        """Get default performance metrics for failed backtests"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    
    def _aggregate_period_results(self, period_results: List[WalkForwardPeriod], 
                                 data: pd.DataFrame) -> BacktestResults:
        """Aggregate results from all periods"""
        try:
            if not period_results:
                raise ValueError("No period results to aggregate")
            
            # Combine all out-of-sample returns
            all_returns = []
            all_dates = []
            
            for period in period_results:
                period_returns = period.test_performance.get('returns', [])
                if isinstance(period_returns, (list, np.ndarray)) and len(period_returns) > 0:
                    all_returns.extend(period_returns)
                    # Generate dates for this period
                    period_dates = pd.date_range(
                        start=period.test_start,
                        end=period.test_end,
                        periods=len(period_returns)
                    )
                    all_dates.extend(period_dates)
            
            if not all_returns:
                # Fallback: use aggregated metrics
                total_return = np.mean([p.test_performance.get('total_return', 0) for p in period_results])
                sharpe_ratio = np.mean([p.test_performance.get('sharpe_ratio', 0) for p in period_results])
                max_drawdown = np.max([p.test_performance.get('max_drawdown', 0) for p in period_results])
                win_rate = np.mean([p.test_performance.get('win_rate', 0) for p in period_results])
                profit_factor = np.mean([p.test_performance.get('profit_factor', 0) for p in period_results])
                
                return BacktestResults(
                    total_return=total_return,
                    annual_return=total_return,  # Approximation
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    total_trades=sum(p.test_performance.get('total_trades', 0) for p in period_results),
                    winning_trades=0,
                    losing_trades=0,
                    avg_win=np.mean([p.test_performance.get('avg_win', 0) for p in period_results]),
                    avg_loss=np.mean([p.test_performance.get('avg_loss', 0) for p in period_results]),
                    largest_win=0.0,
                    largest_loss=0.0,
                    equity_curve=pd.Series([1.0]),
                    drawdown_series=pd.Series([0.0]),
                    trade_log=pd.DataFrame(),
                    start_date=period_results[0].test_start,
                    end_date=period_results[-1].test_end,
                    total_days=(period_results[-1].test_end - period_results[0].test_start).days,
                    metadata={}
                )
            
            # Create return series
            returns_series = pd.Series(all_returns, index=all_dates)
            
            # Calculate comprehensive metrics
            performance = self._calculate_performance_metrics(returns_series, data)
            
            # Create equity curve
            equity_curve = (1 + returns_series).cumprod()
            
            # Calculate drawdown series
            running_max = equity_curve.expanding().max()
            drawdown_series = (equity_curve - running_max) / running_max
            
            # Trade analysis
            trades_analysis = self._analyze_trades(returns_series)
            
            return BacktestResults(
                total_return=performance['total_return'],
                annual_return=performance['annual_return'],
                sharpe_ratio=performance['sharpe_ratio'],
                max_drawdown=performance['max_drawdown'],
                win_rate=performance['win_rate'],
                profit_factor=performance['profit_factor'],
                calmar_ratio=performance['calmar_ratio'],
                sortino_ratio=performance['sortino_ratio'],
                total_trades=int(trades_analysis['total_trades']),
                winning_trades=int(trades_analysis['total_trades'] * trades_analysis['win_rate']),
                losing_trades=int(trades_analysis['total_trades'] * (1 - trades_analysis['win_rate'])),
                avg_win=trades_analysis['avg_win'],
                avg_loss=trades_analysis['avg_loss'],
                largest_win=max(all_returns) if all_returns else 0.0,
                largest_loss=min(all_returns) if all_returns else 0.0,
                equity_curve=equity_curve,
                drawdown_series=drawdown_series,
                trade_log=pd.DataFrame(),  # Would be populated in real implementation
                start_date=period_results[0].test_start,
                end_date=period_results[-1].test_end,
                total_days=(period_results[-1].test_end - period_results[0].test_start).days,
                metadata={'periods_analyzed': len(period_results)}
            )
            
        except Exception as e:
            logger.error(f"Error aggregating period results: {str(e)}")
            raise
    
    def _analyze_parameter_stability(self, period_results: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Analyze stability of optimal parameters across periods"""
        try:
            if not period_results:
                return {}
            
            stability_metrics = {}
            
            # Get all parameter names
            all_params = set()
            for period in period_results:
                all_params.update(period.optimal_parameters.keys())
            
            # Calculate stability for each parameter
            for param_name in all_params:
                param_values = []
                for period in period_results:
                    if param_name in period.optimal_parameters:
                        param_values.append(period.optimal_parameters[param_name])
                
                if len(param_values) > 1:
                    # Calculate coefficient of variation as stability metric
                    if np.mean(param_values) != 0:
                        cv = np.std(param_values) / abs(np.mean(param_values))
                        stability = max(0, 1 - cv)  # Higher stability = lower variation
                    else:
                        stability = 1.0 if np.std(param_values) == 0 else 0.0
                else:
                    stability = 1.0
                
                stability_metrics[param_name] = stability
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing parameter stability: {str(e)}")
            return {}
    
    def _analyze_degradation(self, period_results: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Analyze performance degradation over time"""
        try:
            if len(period_results) < 2:
                return {}
            
            # Extract performance metrics over time
            sharpe_ratios = [p.test_performance.get('sharpe_ratio', 0) for p in period_results]
            returns = [p.test_performance.get('total_return', 0) for p in period_results]
            
            # Calculate trends
            periods = np.arange(len(period_results))
            
            # Sharpe ratio trend
            if len(sharpe_ratios) > 1:
                sharpe_trend = np.polyfit(periods, sharpe_ratios, 1)[0]
            else:
                sharpe_trend = 0.0
            
            # Return trend  
            if len(returns) > 1:
                return_trend = np.polyfit(periods, returns, 1)[0]
            else:
                return_trend = 0.0
            
            # Performance consistency
            sharpe_consistency = 1 - (np.std(sharpe_ratios) / (abs(np.mean(sharpe_ratios)) + 1e-6))
            return_consistency = 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-6))
            
            return {
                'sharpe_trend': sharpe_trend,
                'return_trend': return_trend,
                'sharpe_consistency': max(0, sharpe_consistency),
                'return_consistency': max(0, return_consistency),
                'overall_degradation': max(0, -sharpe_trend)  # Positive if degrading
            }
            
        except Exception as e:
            logger.error(f"Error analyzing degradation: {str(e)}")
            return {}
    
    def _calculate_oos_performance(self, period_results: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Calculate pure out-of-sample performance metrics"""
        try:
            if not period_results:
                return {}
            
            # Aggregate only out-of-sample test results
            oos_sharpe = np.mean([p.test_performance.get('sharpe_ratio', 0) for p in period_results])
            oos_return = np.mean([p.test_performance.get('total_return', 0) for p in period_results])
            oos_max_dd = np.max([p.test_performance.get('max_drawdown', 0) for p in period_results])
            oos_win_rate = np.mean([p.test_performance.get('win_rate', 0) for p in period_results])
            
            # Calculate consistency metrics
            sharpe_std = np.std([p.test_performance.get('sharpe_ratio', 0) for p in period_results])
            return_std = np.std([p.test_performance.get('total_return', 0) for p in period_results])
            
            return {
                'oos_sharpe_ratio': oos_sharpe,
                'oos_annual_return': oos_return,
                'oos_max_drawdown': oos_max_dd,
                'oos_win_rate': oos_win_rate,
                'oos_sharpe_std': sharpe_std,
                'oos_return_std': return_std,
                'oos_consistency': max(0, 1 - sharpe_std) if oos_sharpe != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating OOS performance: {str(e)}")
            return {}
    
    def _generate_metadata(self, data: pd.DataFrame, parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Generate metadata for the analysis"""
        try:
            return {
                'data_points': len(data),
                'start_date': data.index[0].isoformat(),
                'end_date': data.index[-1].isoformat(),
                'parameter_ranges': parameter_ranges,
                'validation_method': self.config['validation']['method'].value,
                'optimization_metric': self.config['optimization']['metric'].value,
                'train_months': self.config['validation']['train_months'],
                'test_months': self.config['validation']['test_months'],
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    print("Walk-Forward Analysis Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Generate synthetic price data
    returns = np.random.randn(len(dates)) * 0.01
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
    
    # Define a simple strategy function
    def simple_sma_strategy(data, params):
        """Simple SMA crossover strategy"""
        short_period = params.get('short_sma', 10)
        long_period = params.get('long_sma', 20)
        
        short_sma = data['close'].rolling(short_period).mean()
        long_sma = data['close'].rolling(long_period).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_sma > long_sma] = 1
        signals[short_sma < long_sma] = -1
        
        return signals
    
    # Define parameter ranges
    parameter_ranges = {
        'short_sma': [5, 10, 15],
        'long_sma': [20, 30, 40]
    }
    
    # Initialize analyzer
    analyzer = WalkForwardAnalyzer()
    
    try:
        # Run walk-forward analysis
        print("Running walk-forward analysis...")
        results = analyzer.run_walk_forward_analysis(
            data=sample_data,
            strategy_function=simple_sma_strategy,
            parameter_ranges=parameter_ranges,
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        print(f"\nWalk-Forward Analysis Results:")
        print(f"  Total Periods: {len(results.period_results)}")
        print(f"  Overall Sharpe Ratio: {results.overall_performance.sharpe_ratio:.4f}")
        print(f"  Overall Annual Return: {results.overall_performance.annual_return:.2%}")
        print(f"  Maximum Drawdown: {results.overall_performance.max_drawdown:.2%}")
        print(f"  Win Rate: {results.overall_performance.win_rate:.2%}")
        print(f"  Total Trades: {results.overall_performance.total_trades}")
        
        if results.parameter_stability:
            print(f"\n  Parameter Stability:")
            for param, stability in results.parameter_stability.items():
                print(f"    {param}: {stability:.3f}")
        
        if results.degradation_analysis:
            print(f"\n  Degradation Analysis:")
            for metric, value in results.degradation_analysis.items():
                print(f"    {metric}: {value:.4f}")
        
        if results.out_of_sample_performance:
            print(f"\n  Out-of-Sample Performance:")
            print(f"    OOS Sharpe Ratio: {results.out_of_sample_performance.get('oos_sharpe_ratio', 0):.4f}")
            print(f"    OOS Annual Return: {results.out_of_sample_performance.get('oos_annual_return', 0):.2%}")
            print(f"    OOS Consistency: {results.out_of_sample_performance.get('oos_consistency', 0):.3f}")
        
        print("\nWalk-Forward Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in walk-forward analysis: {str(e)}")
    
    print("\nWalk-Forward Analysis implementation completed!")