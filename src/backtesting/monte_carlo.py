"""
Monte Carlo Simulation
Advanced Monte Carlo analysis for trading strategy robustness testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimulationType(Enum):
    """Types of Monte Carlo simulations"""
    BOOTSTRAP_TRADES = "bootstrap_trades"
    BOOTSTRAP_RETURNS = "bootstrap_returns"
    PARAMETRIC_RETURNS = "parametric_returns"
    SCENARIO_ANALYSIS = "scenario_analysis"

class RiskMetric(Enum):
    """Risk metrics for Monte Carlo analysis"""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    TAIL_RATIO = "tail_ratio"
    WORST_CASE = "worst_case"

@dataclass
class SimulationPath:
    """Container for single simulation path"""
    path_id: int
    returns: np.ndarray
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    final_return: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    metadata: Dict[str, Any]

@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results"""
    # Summary statistics
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_95: float
    max_drawdown_99: float
    
    # Probability metrics
    prob_positive: float
    prob_target_return: float
    prob_exceed_drawdown: float
    
    # Distribution statistics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Confidence intervals
    return_ci_95: Tuple[float, float]
    return_ci_99: Tuple[float, float]
    sharpe_ci_95: Tuple[float, float]
    
    # Simulation details
    num_simulations: int
    simulation_type: SimulationType
    all_paths: List[SimulationPath]
    convergence_analysis: Dict[str, Any]
    
    # Metadata
    analysis_timestamp: datetime
    simulation_metadata: Dict[str, Any]

class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulator for trading strategy analysis with multiple
    simulation methods and comprehensive risk assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            config: Configuration for Monte Carlo simulation
        """
        self.config = config or self._get_default_config()
        self.simulation_cache = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Monte Carlo simulation"""
        return {
            'simulation': {
                'num_simulations': 1000,
                'confidence_levels': [0.95, 0.99],
                'target_return': 0.1,  # 10% annual return
                'max_drawdown_threshold': 0.2,  # 20% drawdown
                'parallel_processing': True,
                'max_workers': 4
            },
            'bootstrap': {
                'block_size': 'auto',  # For time series bootstrap
                'replacement': True,
                'preserve_autocorr': True
            },
            'parametric': {
                'distribution': 'normal',  # normal, t-distribution, skewed-t
                'estimate_parameters': True,
                'include_fat_tails': True
            },
            'convergence': {
                'check_convergence': True,
                'convergence_tolerance': 0.01,
                'min_simulations': 100,
                'convergence_window': 50
            }
        }
    
    def run_monte_carlo_analysis(self, 
                                returns: Union[pd.Series, np.ndarray],
                                simulation_type: SimulationType = SimulationType.BOOTSTRAP_RETURNS,
                                num_simulations: Optional[int] = None,
                                target_return: Optional[float] = None) -> MonteCarloResults:
        """
        Run comprehensive Monte Carlo analysis
        
        Args:
            returns: Historical returns or trade data
            simulation_type: Type of Monte Carlo simulation
            num_simulations: Number of simulations to run
            target_return: Target return for probability calculations
            
        Returns:
            Complete Monte Carlo analysis results
        """
        try:
            logger.info(f"Starting Monte Carlo analysis with {simulation_type.value}")
            
            # Prepare parameters
            num_sims = num_simulations or self.config['simulation']['num_simulations']
            target_ret = target_return or self.config['simulation']['target_return']
            
            # Validate inputs
            if len(returns) == 0:
                raise ValueError("No returns data provided")
            
            # Convert to numpy array if needed
            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = np.array(returns)
            
            # Remove any NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) < 10:
                raise ValueError("Insufficient return data for Monte Carlo analysis")
            
            # Run simulations
            if self.config['simulation']['parallel_processing']:
                simulation_paths = self._run_parallel_simulations(
                    returns_array, simulation_type, num_sims
                )
            else:
                simulation_paths = self._run_sequential_simulations(
                    returns_array, simulation_type, num_sims
                )
            
            # Analyze results
            results = self._analyze_simulation_results(
                simulation_paths, target_ret, simulation_type
            )
            
            logger.info(f"Monte Carlo analysis completed: {num_sims} simulations")
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {str(e)}")
            raise
    
    def _run_parallel_simulations(self, returns: np.ndarray, 
                                simulation_type: SimulationType,
                                num_simulations: int) -> List[SimulationPath]:
        """Run simulations in parallel"""
        try:
            max_workers = self.config['simulation']['max_workers']
            batch_size = max(1, num_simulations // max_workers)
            
            simulation_paths = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for i in range(0, num_simulations, batch_size):
                    end_idx = min(i + batch_size, num_simulations)
                    batch_size_actual = end_idx - i
                    
                    future = executor.submit(
                        self._run_simulation_batch,
                        returns, simulation_type, i, batch_size_actual
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        simulation_paths.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error in simulation batch: {str(e)}")
            
            return simulation_paths
            
        except Exception as e:
            logger.error(f"Error in parallel simulations: {str(e)}")
            return self._run_sequential_simulations(returns, simulation_type, num_simulations)
    
    def _run_sequential_simulations(self, returns: np.ndarray,
                                  simulation_type: SimulationType,
                                  num_simulations: int) -> List[SimulationPath]:
        """Run simulations sequentially"""
        try:
            return self._run_simulation_batch(returns, simulation_type, 0, num_simulations)
        except Exception as e:
            logger.error(f"Error in sequential simulations: {str(e)}")
            return []
    
    def _run_simulation_batch(self, returns: np.ndarray,
                            simulation_type: SimulationType,
                            start_id: int, batch_size: int) -> List[SimulationPath]:
        """Run a batch of simulations"""
        try:
            simulation_paths = []
            
            for i in range(batch_size):
                path_id = start_id + i
                
                # Generate simulation path
                if simulation_type == SimulationType.BOOTSTRAP_RETURNS:
                    sim_returns = self._bootstrap_returns(returns)
                elif simulation_type == SimulationType.BOOTSTRAP_TRADES:
                    sim_returns = self._bootstrap_trades(returns)
                elif simulation_type == SimulationType.PARAMETRIC_RETURNS:
                    sim_returns = self._parametric_simulation(returns)
                elif simulation_type == SimulationType.SCENARIO_ANALYSIS:
                    sim_returns = self._scenario_simulation(returns)
                else:
                    sim_returns = self._bootstrap_returns(returns)
                
                # Calculate path metrics
                path = self._calculate_path_metrics(sim_returns, path_id)
                simulation_paths.append(path)
            
            return simulation_paths
            
        except Exception as e:
            logger.error(f"Error in simulation batch: {str(e)}")
            return []
    
    def _bootstrap_returns(self, returns: np.ndarray) -> np.ndarray:
        """Bootstrap returns preserving time series properties"""
        try:
            n = len(returns)
            
            if self.config['bootstrap']['preserve_autocorr']:
                # Block bootstrap to preserve autocorrelation
                block_size = self._calculate_block_size(returns)
                
                num_blocks = int(np.ceil(n / block_size))
                bootstrapped = []
                
                for _ in range(num_blocks):
                    start_idx = np.random.randint(0, n - block_size + 1)
                    block = returns[start_idx:start_idx + block_size]
                    bootstrapped.extend(block)
                
                # Trim to original length
                sim_returns = np.array(bootstrapped[:n])
            else:
                # Simple bootstrap
                indices = np.random.choice(n, size=n, replace=True)
                sim_returns = returns[indices]
            
            return sim_returns
            
        except Exception as e:
            logger.error(f"Error in bootstrap returns: {str(e)}")
            return returns.copy()
    
    def _bootstrap_trades(self, returns: np.ndarray) -> np.ndarray:
        """Bootstrap individual trades"""
        try:
            # Identify trade returns (non-zero values)
            trade_returns = returns[returns != 0]
            
            if len(trade_returns) == 0:
                return returns.copy()
            
            # Bootstrap trade returns
            n_trades = len(trade_returns)
            sim_trades = np.random.choice(trade_returns, size=n_trades, replace=True)
            
            # Create return series with same structure as original
            sim_returns = np.zeros_like(returns)
            trade_indices = np.where(returns != 0)[0]
            
            if len(trade_indices) > 0:
                sim_returns[trade_indices] = sim_trades[:len(trade_indices)]
            
            return sim_returns
            
        except Exception as e:
            logger.error(f"Error in bootstrap trades: {str(e)}")
            return returns.copy()
    
    def _parametric_simulation(self, returns: np.ndarray) -> np.ndarray:
        """Parametric simulation based on estimated distribution"""
        try:
            # Remove zero returns for parameter estimation
            non_zero_returns = returns[returns != 0]
            
            if len(non_zero_returns) < 10:
                return self._bootstrap_returns(returns)
            
            # Estimate distribution parameters
            mean_ret = np.mean(non_zero_returns)
            std_ret = np.std(non_zero_returns)
            
            # Generate new returns
            if self.config['parametric']['distribution'] == 'normal':
                sim_non_zero = np.random.normal(mean_ret, std_ret, len(non_zero_returns))
            elif self.config['parametric']['distribution'] == 't-distribution':
                # Estimate degrees of freedom
                from scipy import stats
                df_est = 6.0  # Conservative estimate for fat tails
                sim_non_zero = stats.t.rvs(df_est, loc=mean_ret, scale=std_ret, size=len(non_zero_returns))
            else:
                sim_non_zero = np.random.normal(mean_ret, std_ret, len(non_zero_returns))
            
            # Create full return series
            sim_returns = np.zeros_like(returns)
            non_zero_indices = np.where(returns != 0)[0]
            
            if len(non_zero_indices) > 0:
                sim_returns[non_zero_indices] = sim_non_zero[:len(non_zero_indices)]
            
            return sim_returns
            
        except Exception as e:
            logger.error(f"Error in parametric simulation: {str(e)}")
            return self._bootstrap_returns(returns)
    
    def _scenario_simulation(self, returns: np.ndarray) -> np.ndarray:
        """Scenario-based simulation with stress testing"""
        try:
            # Create scenarios based on historical patterns
            scenarios = {
                'normal': {'prob': 0.7, 'vol_mult': 1.0, 'mean_mult': 1.0},
                'stress': {'prob': 0.2, 'vol_mult': 2.0, 'mean_mult': 0.5},
                'crisis': {'prob': 0.1, 'vol_mult': 3.0, 'mean_mult': -1.0}
            }
            
            # Select scenario
            scenario_probs = [s['prob'] for s in scenarios.values()]
            scenario_names = list(scenarios.keys())
            selected_scenario = np.random.choice(scenario_names, p=scenario_probs)
            
            scenario_params = scenarios[selected_scenario]
            
            # Modify returns based on scenario
            base_returns = self._bootstrap_returns(returns)
            mean_ret = np.mean(base_returns[base_returns != 0])
            std_ret = np.std(base_returns[base_returns != 0])
            
            # Apply scenario modifications
            modified_mean = mean_ret * scenario_params['mean_mult']
            modified_std = std_ret * scenario_params['vol_mult']
            
            # Generate scenario returns
            sim_returns = base_returns.copy()
            non_zero_mask = sim_returns != 0
            
            if np.any(non_zero_mask):
                sim_returns[non_zero_mask] = np.random.normal(
                    modified_mean, modified_std, np.sum(non_zero_mask)
                )
            
            return sim_returns
            
        except Exception as e:
            logger.error(f"Error in scenario simulation: {str(e)}")
            return self._bootstrap_returns(returns)
    
    def _calculate_block_size(self, returns: np.ndarray) -> int:
        """Calculate optimal block size for block bootstrap"""
        try:
            if self.config['bootstrap']['block_size'] == 'auto':
                # Use rule of thumb: block_size = n^(1/3)
                n = len(returns)
                block_size = max(1, int(n ** (1/3)))
                return min(block_size, n // 4)  # Cap at 25% of data
            else:
                return int(self.config['bootstrap']['block_size'])
        except:
            return max(1, len(returns) // 10)
    
    def _calculate_path_metrics(self, returns: np.ndarray, path_id: int) -> SimulationPath:
        """Calculate metrics for a simulation path"""
        try:
            # Equity curve
            equity_curve = np.cumprod(1 + returns)
            
            # Drawdown curve
            running_max = np.maximum.accumulate(equity_curve)
            drawdown_curve = (equity_curve - running_max) / running_max
            
            # Performance metrics
            final_return = equity_curve[-1] - 1
            max_drawdown = abs(np.min(drawdown_curve))
            
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Trade count
            trade_count = np.sum(returns != 0)
            
            return SimulationPath(
                path_id=path_id,
                returns=returns,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                final_return=final_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                trade_count=trade_count,
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Error calculating path metrics: {str(e)}")
            return SimulationPath(
                path_id=path_id,
                returns=returns,
                equity_curve=np.ones_like(returns),
                drawdown_curve=np.zeros_like(returns),
                final_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trade_count=0,
                metadata={}
            )
    
    def _analyze_simulation_results(self, paths: List[SimulationPath],
                                  target_return: float,
                                  simulation_type: SimulationType) -> MonteCarloResults:
        """Analyze simulation results and calculate statistics"""
        try:
            if not paths:
                raise ValueError("No simulation paths to analyze")
            
            # Extract metrics from all paths
            final_returns = np.array([path.final_return for path in paths])
            max_drawdowns = np.array([path.max_drawdown for path in paths])
            sharpe_ratios = np.array([path.sharpe_ratio for path in paths])
            
            # Summary statistics
            mean_return = np.mean(final_returns)
            median_return = np.median(final_returns)
            std_return = np.std(final_returns)
            min_return = np.min(final_returns)
            max_return = np.max(final_returns)
            
            # Risk metrics
            var_95 = np.percentile(final_returns, 5)
            var_99 = np.percentile(final_returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(final_returns[final_returns <= var_95])
            cvar_99 = np.mean(final_returns[final_returns <= var_99])
            
            # Drawdown statistics
            max_drawdown_95 = np.percentile(max_drawdowns, 95)
            max_drawdown_99 = np.percentile(max_drawdowns, 99)
            
            # Probability metrics
            prob_positive = np.mean(final_returns > 0)
            prob_target_return = np.mean(final_returns >= target_return)
            prob_exceed_drawdown = np.mean(max_drawdowns >= self.config['simulation']['max_drawdown_threshold'])
            
            # Distribution statistics
            from scipy import stats
            skewness = stats.skew(final_returns)
            kurtosis = stats.kurtosis(final_returns)
            jb_stat, jb_pvalue = stats.jarque_bera(final_returns)
            
            # Confidence intervals
            return_ci_95 = (np.percentile(final_returns, 2.5), np.percentile(final_returns, 97.5))
            return_ci_99 = (np.percentile(final_returns, 0.5), np.percentile(final_returns, 99.5))
            sharpe_ci_95 = (np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5))
            
            # Convergence analysis
            convergence_analysis = self._analyze_convergence(final_returns)
            
            results = MonteCarloResults(
                mean_return=mean_return,
                median_return=median_return,
                std_return=std_return,
                min_return=min_return,
                max_return=max_return,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown_95=max_drawdown_95,
                max_drawdown_99=max_drawdown_99,
                prob_positive=prob_positive,
                prob_target_return=prob_target_return,
                prob_exceed_drawdown=prob_exceed_drawdown,
                skewness=skewness,
                kurtosis=kurtosis,
                jarque_bera_stat=jb_stat,
                jarque_bera_pvalue=jb_pvalue,
                return_ci_95=return_ci_95,
                return_ci_99=return_ci_99,
                sharpe_ci_95=sharpe_ci_95,
                num_simulations=len(paths),
                simulation_type=simulation_type,
                all_paths=paths,
                convergence_analysis=convergence_analysis,
                analysis_timestamp=datetime.now(),
                simulation_metadata=self._generate_simulation_metadata(paths)
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing simulation results: {str(e)}")
            raise
    
    def _analyze_convergence(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze convergence of Monte Carlo simulation"""
        try:
            if not self.config['convergence']['check_convergence']:
                return {}
            
            window_size = self.config['convergence']['convergence_window']
            tolerance = self.config['convergence']['convergence_tolerance']
            
            # Calculate rolling statistics
            rolling_means = []
            rolling_stds = []
            
            for i in range(window_size, len(returns) + 1):
                window_data = returns[:i]
                rolling_means.append(np.mean(window_data))
                rolling_stds.append(np.std(window_data))
            
            # Check convergence
            if len(rolling_means) >= 2:
                final_mean = rolling_means[-1]
                mean_changes = [abs(m - final_mean) / abs(final_mean + 1e-8) for m in rolling_means[-10:]]
                converged = all(change < tolerance for change in mean_changes[-5:]) if len(mean_changes) >= 5 else False
            else:
                converged = False
            
            return {
                'converged': converged,
                'rolling_means': rolling_means[-20:],  # Last 20 values
                'rolling_stds': rolling_stds[-20:],
                'final_mean_stability': np.std(rolling_means[-10:]) if len(rolling_means) >= 10 else 0,
                'recommended_simulations': len(returns) if converged else len(returns) * 2
            }
            
        except Exception as e:
            logger.error(f"Error analyzing convergence: {str(e)}")
            return {}
    
    def _generate_simulation_metadata(self, paths: List[SimulationPath]) -> Dict[str, Any]:
        """Generate metadata for simulation"""
        try:
            return {
                'total_paths': len(paths),
                'avg_trade_count': np.mean([path.trade_count for path in paths]),
                'simulation_config': self.config,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating simulation metadata: {str(e)}")
            return {}
    
    def generate_risk_report(self, results: MonteCarloResults) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            report = {
                'executive_summary': {
                    'expected_return': results.mean_return,
                    'probability_of_loss': 1 - results.prob_positive,
                    'worst_case_scenario_95': results.var_95,
                    'expected_loss_given_worst_5_percent': results.cvar_95,
                    'maximum_expected_drawdown': results.max_drawdown_95
                },
                'detailed_risk_metrics': {
                    'value_at_risk': {
                        '95_percent': results.var_95,
                        '99_percent': results.var_99
                    },
                    'conditional_var': {
                        '95_percent': results.cvar_95,
                        '99_percent': results.cvar_99
                    },
                    'drawdown_risk': {
                        '95th_percentile': results.max_drawdown_95,
                        '99th_percentile': results.max_drawdown_99,
                        'probability_large_drawdown': results.prob_exceed_drawdown
                    }
                },
                'return_distribution': {
                    'mean': results.mean_return,
                    'median': results.median_return,
                    'standard_deviation': results.std_return,
                    'skewness': results.skewness,
                    'kurtosis': results.kurtosis,
                    'normality_test_pvalue': results.jarque_bera_pvalue
                },
                'confidence_intervals': {
                    'return_95_percent': results.return_ci_95,
                    'return_99_percent': results.return_ci_99,
                    'sharpe_95_percent': results.sharpe_ci_95
                },
                'scenario_probabilities': {
                    'probability_positive_return': results.prob_positive,
                    f'probability_exceed_target_{results.simulation_metadata.get("target_return", 0.1):.1%}': results.prob_target_return,
                    'probability_large_drawdown': results.prob_exceed_drawdown
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    print("Monte Carlo Simulation Test")
    print("=" * 40)
    
    # Create sample return data
    np.random.seed(42)
    
    # Generate realistic trading returns
    n_periods = 252 * 2  # 2 years of daily data
    base_return = 0.0005  # 0.05% daily base return
    volatility = 0.02  # 2% daily volatility
    
    # Generate returns with some autocorrelation and fat tails
    returns = []
    for i in range(n_periods):
        if i == 0:
            ret = np.random.normal(base_return, volatility)
        else:
            # Add some mean reversion
            momentum = 0.1 * returns[-1]  # 10% momentum
            mean_reversion = -0.05 * returns[-1]  # 5% mean reversion
            ret = base_return + momentum + mean_reversion + np.random.normal(0, volatility)
        
        returns.append(ret)
    
    returns = np.array(returns)
    
    # Add some zero returns (no trading days)
    trade_probability = 0.3  # 30% of days have trades
    trade_mask = np.random.random(len(returns)) < trade_probability
    returns = returns * trade_mask
    
    print(f"Generated {len(returns)} return observations")
    print(f"Trading days: {np.sum(trade_mask)} ({np.mean(trade_mask):.1%})")
    print(f"Mean return: {np.mean(returns[returns != 0]):.4f}")
    print(f"Volatility: {np.std(returns[returns != 0]):.4f}")
    
    # Initialize simulator
    simulator = MonteCarloSimulator()
    
    # Test different simulation types
    simulation_types = [
        SimulationType.BOOTSTRAP_RETURNS,
        SimulationType.BOOTSTRAP_TRADES,
        SimulationType.PARAMETRIC_RETURNS,
        SimulationType.SCENARIO_ANALYSIS
    ]
    
    for sim_type in simulation_types:
        try:
            print(f"\nRunning {sim_type.value} simulation...")
            
            results = simulator.run_monte_carlo_analysis(
                returns=returns,
                simulation_type=sim_type,
                num_simulations=500,  # Reduced for demo
                target_return=0.1  # 10% annual target
            )
            
            print(f"  Simulation Results:")
            print(f"    Mean Return: {results.mean_return:.2%}")
            print(f"    Std Return: {results.std_return:.2%}")
            print(f"    VaR (95%): {results.var_95:.2%}")
            print(f"    CVaR (95%): {results.cvar_95:.2%}")
            print(f"    Max DD (95%): {results.max_drawdown_95:.2%}")
            print(f"    Prob Positive: {results.prob_positive:.1%}")
            print(f"    Prob Target: {results.prob_target_return:.1%}")
            
            if results.convergence_analysis.get('converged'):
                print(f"    ✓ Simulation converged")
            else:
                print(f"    ⚠ Simulation may need more iterations")
        
        except Exception as e:
            print(f"    Error in {sim_type.value}: {str(e)}")
    
    # Generate detailed risk report for bootstrap simulation
    try:
        print(f"\nDetailed Risk Report (Bootstrap Returns):")
        print("=" * 50)
        
        bootstrap_results = simulator.run_monte_carlo_analysis(
            returns=returns,
            simulation_type=SimulationType.BOOTSTRAP_RETURNS,
            num_simulations=1000
        )
        
        risk_report = simulator.generate_risk_report(bootstrap_results)
        
        exec_summary = risk_report['executive_summary']
        print(f"Executive Summary:")
        print(f"  Expected Return: {exec_summary['expected_return']:.2%}")
        print(f"  Probability of Loss: {exec_summary['probability_of_loss']:.1%}")
        print(f"  Worst Case (95%): {exec_summary['worst_case_scenario_95']:.2%}")
        print(f"  Expected Loss in Worst 5%: {exec_summary['expected_loss_given_worst_5_percent']:.2%}")
        
        detailed_risk = risk_report['detailed_risk_metrics']
        print(f"\nRisk Metrics:")
        print(f"  VaR 95%/99%: {detailed_risk['value_at_risk']['95_percent']:.2%} / {detailed_risk['value_at_risk']['99_percent']:.2%}")
        print(f"  CVaR 95%/99%: {detailed_risk['conditional_var']['95_percent']:.2%} / {detailed_risk['conditional_var']['99_percent']:.2%}")
        print(f"  Max DD 95%/99%: {detailed_risk['drawdown_risk']['95th_percentile']:.2%} / {detailed_risk['drawdown_risk']['99th_percentile']:.2%}")
        
        return_dist = risk_report['return_distribution']
        print(f"\nReturn Distribution:")
        print(f"  Mean/Median: {return_dist['mean']:.2%} / {return_dist['median']:.2%}")
        print(f"  Skewness: {return_dist['skewness']:.3f}")
        print(f"  Kurtosis: {return_dist['kurtosis']:.3f}")
        print(f"  Normality p-value: {return_dist['normality_test_pvalue']:.4f}")
        
    except Exception as e:
        print(f"Error generating risk report: {str(e)}")
    
    print("\nMonte Carlo Simulation implementation completed!")