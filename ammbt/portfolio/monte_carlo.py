"""
Monte Carlo simulation for LP strategy stress testing.

Generates synthetic price paths using various stochastic models and runs
backtests across many simulations to produce confidence intervals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import numba
import pandas as pd


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    simulation_metrics: pd.DataFrame
    """Metrics DataFrame with one row per simulation."""

    confidence_intervals: Dict[str, tuple]
    """95% confidence intervals for key metrics: {metric: (lower, upper)}."""

    summary: pd.DataFrame
    """Summary statistics (mean, std, median, 5th/95th percentile) per metric."""

    n_simulations: int
    """Number of simulations run."""

    def __repr__(self) -> str:
        avg_pnl = self.summary.loc['mean', 'net_pnl'] if 'net_pnl' in self.summary.columns else 0
        return (
            f"MonteCarloResult(\n"
            f"  n_simulations={self.n_simulations},\n"
            f"  avg_net_pnl={avg_pnl:.2f},\n"
            f"  metrics={list(self.summary.columns)}\n"
            f")"
        )


@numba.jit(nopython=True, cache=True)
def _generate_gbm_path(
    initial_price: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """
    Generate Geometric Brownian Motion price path.

    dS/S = mu*dt + sigma*dW
    """
    np.random.seed(seed)
    prices = np.empty(n_steps, dtype=np.float64)
    prices[0] = initial_price

    for i in range(1, n_steps):
        z = np.random.randn()
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    return prices


@numba.jit(nopython=True, cache=True)
def _generate_jump_diffusion_path(
    initial_price: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """
    Generate Merton's Jump-Diffusion price path.

    dS/S = mu*dt + sigma*dW + J*dN
    """
    np.random.seed(seed)
    prices = np.empty(n_steps, dtype=np.float64)
    prices[0] = initial_price

    for i in range(1, n_steps):
        z = np.random.randn()
        # Poisson jump
        n_jumps = 0
        if np.random.random() < jump_intensity * dt:
            n_jumps = 1
        jump = 0.0
        if n_jumps > 0:
            jump = np.random.randn() * jump_std + jump_mean

        prices[i] = prices[i - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z + jump
        )
        prices[i] = max(1e-10, prices[i])

    return prices


@numba.jit(nopython=True, cache=True)
def _generate_mean_reverting_path(
    initial_price: float,
    theta: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """
    Generate Ornstein-Uhlenbeck mean-reverting price path.

    dX = theta*(mu - X)*dt + sigma*dW
    Price = exp(X) to keep positive.
    """
    np.random.seed(seed)
    prices = np.empty(n_steps, dtype=np.float64)
    x = np.log(initial_price)

    for i in range(n_steps):
        prices[i] = np.exp(x)
        if i < n_steps - 1:
            z = np.random.randn()
            x = x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * z

    return prices


def _prices_to_swaps(prices: np.ndarray, avg_volume: float = 1000.0) -> pd.DataFrame:
    """Convert a synthetic price path to a swap DataFrame."""
    n = len(prices)
    np.random.seed(None)  # Fresh randomness for volumes

    volumes = np.abs(np.random.normal(avg_volume, avg_volume * 0.2, n))
    price_changes = np.diff(prices, prepend=prices[0])

    # Direction: positive price change = buy (amount1 in), negative = sell (amount0 in)
    amount0 = np.where(price_changes < 0, volumes / prices, 0.0)
    amount1 = np.where(price_changes >= 0, volumes, 0.0)

    return pd.DataFrame({
        'amount0': amount0.astype(np.float64),
        'amount1': amount1.astype(np.float64),
        'price': prices.astype(np.float64),
    })


def _bootstrap_swaps(base_swaps: pd.DataFrame, n_steps: int, block_size: int = 50) -> pd.DataFrame:
    """Block bootstrap: resample blocks of swaps with replacement."""
    n_base = len(base_swaps)
    n_blocks = max(1, n_steps // block_size)

    blocks = []
    for _ in range(n_blocks):
        start = np.random.randint(0, max(1, n_base - block_size))
        block = base_swaps.iloc[start:start + block_size]
        blocks.append(block)

    result = pd.concat(blocks, ignore_index=True)
    return result.iloc[:n_steps].reset_index(drop=True)


def _fit_gbm_params(prices: np.ndarray) -> Dict[str, float]:
    """Fit GBM parameters from historical prices."""
    log_returns = np.diff(np.log(prices))
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    return {'mu': mu, 'sigma': sigma}


def monte_carlo(
    backtester,
    base_swaps: pd.DataFrame,
    n_simulations: int = 100,
    strategy: Optional[Union[Dict, pd.DataFrame]] = None,
    model: str = 'gbm',
    n_steps: Optional[int] = None,
    model_params: Optional[Dict] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation across synthetic price paths.

    Parameters
    ----------
    backtester : LPBacktester
        Configured backtester instance.
    base_swaps : pd.DataFrame
        Historical swap data (used for parameter fitting and bootstrap).
    n_simulations : int
        Number of simulation runs.
    strategy : dict or pd.DataFrame, optional
        Strategy to test. If None, uses a default single strategy.
    model : str
        Price path model: ``'gbm'``, ``'jump'``, ``'mean_revert'``, ``'bootstrap'``.
    n_steps : int, optional
        Length of each synthetic path. Defaults to len(base_swaps).
    model_params : dict, optional
        Override model parameters. If None, fitted from base_swaps.

    Returns
    -------
    MonteCarloResult
        Simulation results with confidence intervals.

    Examples
    --------
    >>> result = monte_carlo(bt, swaps, n_simulations=500, model='gbm')
    >>> print(result.confidence_intervals['net_pnl'])
    """
    if n_steps is None:
        n_steps = len(base_swaps)

    # Fit parameters from base data
    if 'price' in base_swaps.columns:
        base_prices = base_swaps['price'].values
    else:
        base_prices = np.ones(len(base_swaps))

    initial_price = base_prices[0]
    params = model_params or {}

    if model in ('gbm', 'jump', 'mean_revert') and not params:
        fitted = _fit_gbm_params(base_prices)
        params.setdefault('mu', fitted['mu'])
        params.setdefault('sigma', fitted['sigma'])

    avg_volume = np.abs(base_swaps['amount0']).mean()

    # Default strategy
    if strategy is None:
        strategy = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

    # Run simulations
    all_metrics = []
    for sim_idx in range(n_simulations):
        seed = sim_idx * 7 + 42

        if model == 'gbm':
            prices = _generate_gbm_path(
                initial_price,
                params.get('mu', 0.0),
                params.get('sigma', 0.02),
                n_steps,
                params.get('dt', 1.0),
                seed,
            )
            swaps = _prices_to_swaps(prices, avg_volume)

        elif model == 'jump':
            prices = _generate_jump_diffusion_path(
                initial_price,
                params.get('mu', 0.0),
                params.get('sigma', 0.02),
                params.get('jump_intensity', 0.1),
                params.get('jump_mean', 0.0),
                params.get('jump_std', 0.05),
                n_steps,
                params.get('dt', 1.0),
                seed,
            )
            swaps = _prices_to_swaps(prices, avg_volume)

        elif model == 'mean_revert':
            prices = _generate_mean_reverting_path(
                initial_price,
                params.get('theta', 0.1),
                params.get('mu', np.log(initial_price)),
                params.get('sigma', 0.02),
                n_steps,
                params.get('dt', 1.0),
                seed,
            )
            swaps = _prices_to_swaps(prices, avg_volume)

        elif model == 'bootstrap':
            np.random.seed(seed)
            swaps = _bootstrap_swaps(
                base_swaps, n_steps, params.get('block_size', 50)
            )

        else:
            raise ValueError(f"Unknown model: '{model}'. Use 'gbm', 'jump', 'mean_revert', 'bootstrap'.")

        try:
            result = backtester.run(swaps, strategy)
            row = result.metrics.iloc[0].to_dict()
            row['simulation_idx'] = sim_idx
            all_metrics.append(row)
        except Exception:
            continue

    if not all_metrics:
        raise RuntimeError("All simulations failed. Check strategy parameters and model settings.")

    sim_metrics = pd.DataFrame(all_metrics)

    # Compute confidence intervals and summary
    numeric_cols = sim_metrics.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'simulation_idx']

    ci = {}
    for col in numeric_cols:
        vals = sim_metrics[col].dropna()
        if len(vals) > 0:
            ci[col] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    summary = sim_metrics[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    return MonteCarloResult(
        simulation_metrics=sim_metrics,
        confidence_intervals=ci,
        summary=summary,
        n_simulations=len(all_metrics),
    )
