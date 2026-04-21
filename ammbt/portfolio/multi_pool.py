"""
Multi-pool portfolio backtester.

Run backtests across multiple pools simultaneously and compute
portfolio-level metrics including correlation and diversification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ammbt.portfolio.base import LPBacktester, BacktestResult


@dataclass
class PoolConfig:
    """Configuration for a single pool in the portfolio."""

    amm_type: str
    pool_params: Dict
    strategies: Union[Dict, pd.DataFrame]
    swaps: pd.DataFrame
    weight: float = 1.0
    name: Optional[str] = None


@dataclass
class PortfolioResult:
    """Results from multi-pool portfolio backtest."""

    pool_results: List[BacktestResult]
    """Individual BacktestResult per pool."""

    pool_names: List[str]
    """Names for each pool."""

    portfolio_metrics: Dict[str, float]
    """Aggregate portfolio-level metrics."""

    correlation_matrix: pd.DataFrame
    """Return correlation matrix across pools."""

    pool_weights: List[float]
    """Weight of each pool in the portfolio."""

    def summary(self) -> pd.DataFrame:
        """Summary of each pool's best strategy performance."""
        rows = []
        for i, (result, name, weight) in enumerate(
            zip(self.pool_results, self.pool_names, self.pool_weights)
        ):
            best_idx = result.metrics['net_pnl'].idxmax()
            row = result.metrics.iloc[best_idx].to_dict()
            row['pool'] = name
            row['weight'] = weight
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n_pools = len(self.pool_results)
        port_ret = self.portfolio_metrics.get('portfolio_return', 0)
        port_sharpe = self.portfolio_metrics.get('portfolio_sharpe', 0)
        return (
            f"PortfolioResult(\n"
            f"  n_pools={n_pools},\n"
            f"  portfolio_return={port_ret:.4f},\n"
            f"  portfolio_sharpe={port_sharpe:.4f}\n"
            f")"
        )


class PortfolioBacktester:
    """
    Multi-pool portfolio backtester.

    Add multiple pools with their own AMM type, parameters, strategies,
    and swap data. Run all pools and compute portfolio-level analytics.

    Examples
    --------
    >>> portfolio = PortfolioBacktester()
    >>> portfolio.add_pool('v3', pool_params, strategies_eth, swaps_eth, weight=0.6, name='ETH/USDC')
    >>> portfolio.add_pool('dlmm', pool_params2, strategies_sol, swaps_sol, weight=0.4, name='SOL/USDC')
    >>> result = portfolio.run()
    >>> print(result.portfolio_metrics)
    """

    def __init__(self):
        self.pools: List[PoolConfig] = []

    def add_pool(
        self,
        amm_type: str,
        pool_params: Optional[Dict] = None,
        strategies: Union[Dict, pd.DataFrame] = None,
        swaps: pd.DataFrame = None,
        weight: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a pool to the portfolio.

        Parameters
        ----------
        amm_type : str
            AMM type (``'v2'``, ``'v3'``, ``'dlmm'``).
        pool_params : dict, optional
            AMM-specific pool parameters.
        strategies : dict or pd.DataFrame
            Strategy parameters for this pool.
        swaps : pd.DataFrame
            Swap data for this pool.
        weight : float
            Portfolio weight (weights are normalized at run time).
        name : str, optional
            Human-readable pool name.
        """
        if strategies is None or swaps is None:
            raise ValueError("strategies and swaps are required")

        if name is None:
            name = f"pool_{len(self.pools)}"

        self.pools.append(PoolConfig(
            amm_type=amm_type,
            pool_params=pool_params or {},
            strategies=strategies,
            swaps=swaps,
            weight=weight,
            name=name,
        ))

    def run(self, strategy_idx: Optional[int] = 0) -> PortfolioResult:
        """
        Run all pool backtests and compute portfolio metrics.

        Parameters
        ----------
        strategy_idx : int, optional
            Which strategy index to use for portfolio-level return series.
            Default 0 (first strategy per pool).

        Returns
        -------
        PortfolioResult
            Portfolio-level results with correlation matrix and aggregate metrics.
        """
        if not self.pools:
            raise ValueError("No pools added. Call add_pool() first.")

        # Normalize weights
        total_weight = sum(p.weight for p in self.pools)
        weights = [p.weight / total_weight for p in self.pools]

        # Run each pool
        pool_results = []
        pool_names = []
        for pool_config in self.pools:
            bt = LPBacktester(amm_type=pool_config.amm_type, **pool_config.pool_params)
            result = bt.run(pool_config.swaps, pool_config.strategies)
            pool_results.append(result)
            pool_names.append(pool_config.name)

        # Extract return series for correlation
        return_series = {}
        for i, (result, name) in enumerate(zip(pool_results, pool_names)):
            idx = min(strategy_idx, result.positions.shape[1] - 1)
            prices = result.metadata.get('price_history')
            if prices is None and 'reserve0_history' in result.metadata:
                prices = result.metadata['reserve1_history'] / result.metadata['reserve0_history']
            if prices is None:
                prices = np.ones(result.positions.shape[0])

            amm_type = result.metadata.get('amm_type', 'v3')
            values = np.zeros(result.positions.shape[0])
            for t in range(result.positions.shape[0]):
                pos = result.positions[t, idx]
                if amm_type == 'v2':
                    values[t] = pos['token0_balance'] * prices[t] + pos['token1_balance']
                else:
                    values[t] = (
                        pos['token0_balance'] * prices[t] + pos['token1_balance'] +
                        pos['uncollected_fees_0'] * prices[t] + pos['uncollected_fees_1']
                    )

            # Compute returns
            returns = np.zeros(len(values) - 1)
            for t in range(1, len(values)):
                if values[t - 1] > 0:
                    returns[t - 1] = (values[t] - values[t - 1]) / values[t - 1]
            return_series[name] = returns

        # Align return series to common length (truncate to shortest)
        min_len = min(len(r) for r in return_series.values())
        aligned = {k: v[:min_len] for k, v in return_series.items()}
        returns_df = pd.DataFrame(aligned)

        # Correlation matrix
        correlation = returns_df.corr()

        # Portfolio returns (weighted sum)
        portfolio_returns = np.zeros(min_len)
        for i, name in enumerate(pool_names):
            portfolio_returns += weights[i] * aligned[name]

        # Portfolio metrics
        port_mean = portfolio_returns.mean()
        port_std = portfolio_returns.std()
        port_sharpe = port_mean / port_std * np.sqrt(252) if port_std > 0 else 0.0

        cumulative = np.cumprod(1 + portfolio_returns)
        port_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0

        cummax = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - cummax) / cummax
        port_max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0

        portfolio_metrics = {
            'portfolio_return': float(port_return),
            'portfolio_sharpe': float(port_sharpe),
            'portfolio_max_drawdown': float(port_max_dd),
            'portfolio_volatility': float(port_std * np.sqrt(252)),
            'n_pools': len(self.pools),
        }

        return PortfolioResult(
            pool_results=pool_results,
            pool_names=pool_names,
            portfolio_metrics=portfolio_metrics,
            correlation_matrix=correlation,
            pool_weights=weights,
        )
