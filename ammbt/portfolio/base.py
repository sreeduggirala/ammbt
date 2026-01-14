"""
Core backtesting engine.

Main entry point for running LP strategy backtests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from ammbt.amms.univ2 import UniswapV2Simulator
from ammbt.amms.univ3 import UniswapV3Simulator
from ammbt.amms.dlmm import MeteoraLMMSimulator
from ammbt.portfolio.analytics import calculate_metrics, calculate_capital_efficiency
from ammbt.base.array_wrapper import ArrayWrapper


# Strategy parameter dtypes
STRATEGY_PARAM_DTYPE_V2 = np.dtype([
    ('initial_capital', 'f8'),
    ('rebalance_threshold', 'f8'),
    ('rebalance_frequency', 'i4'),
])

STRATEGY_PARAM_DTYPE_V3 = np.dtype([
    ('initial_capital', 'f8'),
    ('tick_lower', 'i4'),
    ('tick_upper', 'i4'),
    ('rebalance_threshold', 'f8'),
    ('rebalance_frequency', 'i4'),
])

STRATEGY_PARAM_DTYPE_DLMM = np.dtype([
    ('initial_capital', 'f8'),
    ('bin_lower', 'i4'),
    ('bin_upper', 'i4'),
    ('liquidity_shape', 'i4'),  # 0=Spot, 1=Curve, 2=Bid-Ask
    ('rebalance_threshold', 'f8'),
    ('rebalance_frequency', 'i4'),
])


class BacktestResult:
    """
    Container for backtest results.

    Provides easy access to metrics, positions, and plotting.
    """

    def __init__(
        self,
        positions: np.ndarray,
        metrics: pd.DataFrame,
        capital_efficiency: pd.DataFrame,
        metadata: Dict[str, Any],
        wrapper: ArrayWrapper,
        strategy_params: pd.DataFrame,
    ):
        """
        Initialize backtest result.

        Parameters
        ----------
        positions : np.ndarray
            Position state array (n_swaps, n_strategies)
        metrics : pd.DataFrame
            Performance metrics per strategy
        capital_efficiency : pd.DataFrame
            Capital efficiency metrics
        metadata : dict
            Additional simulation metadata (reserve history, etc.)
        wrapper : ArrayWrapper
            Array wrapper for index/column info
        strategy_params : pd.DataFrame
            Strategy parameters used
        """
        self.positions = positions
        self.metrics = metrics
        self.capital_efficiency = capital_efficiency
        self.metadata = metadata
        self.wrapper = wrapper
        self.strategy_params = strategy_params

    def summary(self) -> pd.DataFrame:
        """
        Get summary of top strategies.

        Returns
        -------
        pd.DataFrame
            Top strategies by net PnL
        """
        summary = self.metrics.copy()
        summary['strategy_idx'] = summary.index
        return summary.sort_values('net_pnl', ascending=False)

    def get_position_history(self, strategy_idx: int = 0) -> pd.DataFrame:
        """
        Get position history for a specific strategy.

        Parameters
        ----------
        strategy_idx : int
            Strategy index

        Returns
        -------
        pd.DataFrame
            Position state over time
        """
        pos = self.positions[:, strategy_idx]

        df = pd.DataFrame({
            'liquidity': pos['liquidity'],
            'token0_balance': pos['token0_balance'],
            'token1_balance': pos['token1_balance'],
            'uncollected_fees_0': pos['uncollected_fees_0'],
            'uncollected_fees_1': pos['uncollected_fees_1'],
            'gas_spent': pos['gas_spent'],
            'is_active': pos['is_active'],
            'num_rebalances': pos['num_rebalances'],
        })

        if self.wrapper.index is not None:
            df.index = self.wrapper.index

        return df

    def __repr__(self) -> str:
        n_strategies = len(self.metrics)
        best_pnl = self.metrics['net_pnl'].max()
        worst_pnl = self.metrics['net_pnl'].min()
        avg_pnl = self.metrics['net_pnl'].mean()

        return (
            f"BacktestResult(\n"
            f"  n_strategies={n_strategies},\n"
            f"  best_pnl={best_pnl:.2f},\n"
            f"  worst_pnl={worst_pnl:.2f},\n"
            f"  avg_pnl={avg_pnl:.2f}\n"
            f")"
        )


class LPBacktester:
    """
    Main LP backtesting engine.

    Vectorized backtesting for AMM liquidity provider positions.

    Examples
    --------
    >>> import ammbt as amm
    >>> swaps = amm.generate_swaps(10000, volatility=0.02)
    >>> strategies = {
    ...     'initial_capital': [10000, 20000, 50000],
    ...     'rebalance_threshold': [0.05, 0.1, 0.0],
    ...     'rebalance_frequency': [100, 500, 0],
    ... }
    >>> bt = amm.LPBacktester(amm_type='v2')
    >>> results = bt.run(swaps, strategies)
    >>> print(results.summary())
    """

    def __init__(
        self,
        amm_type: str = 'v2',
        **amm_params,
    ):
        """
        Initialize backtester.

        Parameters
        ----------
        amm_type : str
            AMM type ('v2', 'v3', 'dlmm')
        **amm_params
            AMM-specific parameters
        """
        self.amm_type = amm_type

        # Initialize AMM simulator
        if amm_type == 'v2':
            self.simulator = UniswapV2Simulator(**amm_params)
        elif amm_type == 'v3':
            self.simulator = UniswapV3Simulator(**amm_params)
        elif amm_type == 'dlmm':
            self.simulator = MeteoraLMMSimulator(**amm_params)
        else:
            raise NotImplementedError(f"AMM type '{amm_type}' not yet implemented")

    def run(
        self,
        swaps: pd.DataFrame,
        strategies: Union[Dict[str, List], pd.DataFrame],
    ) -> BacktestResult:
        """
        Run backtest.

        Parameters
        ----------
        swaps : pd.DataFrame
            Swap event data with columns: amount0, amount1, price
        strategies : dict or pd.DataFrame
            Strategy parameters. Can be:
            - Dict with keys: initial_capital, rebalance_threshold, rebalance_frequency
            - DataFrame with those columns

        Returns
        -------
        BacktestResult
            Backtest results with metrics and position history

        Examples
        --------
        >>> strategies = {
        ...     'initial_capital': [10000, 20000],
        ...     'rebalance_threshold': [0.05, 0.1],
        ...     'rebalance_frequency': [100, 200],
        ... }
        >>> results = backtester.run(swaps, strategies)
        """
        # Convert strategies to structured array
        if isinstance(strategies, dict):
            # Ensure all lists have same length
            lengths = [len(v) for v in strategies.values()]
            if len(set(lengths)) != 1:
                raise ValueError("All strategy parameter lists must have same length")

            n_strategies = lengths[0]

            # Create DataFrame first
            strategy_df = pd.DataFrame(strategies)
        else:
            strategy_df = strategies
            n_strategies = len(strategy_df)

        # Convert to structured array based on AMM type
        if self.amm_type == 'v2':
            strategy_params = np.zeros(n_strategies, dtype=STRATEGY_PARAM_DTYPE_V2)
            strategy_params['initial_capital'] = strategy_df['initial_capital'].values
            strategy_params['rebalance_threshold'] = strategy_df.get('rebalance_threshold', 0.0).values
            strategy_params['rebalance_frequency'] = strategy_df.get('rebalance_frequency', 0).values
        elif self.amm_type == 'v3':
            strategy_params = np.zeros(n_strategies, dtype=STRATEGY_PARAM_DTYPE_V3)
            strategy_params['initial_capital'] = strategy_df['initial_capital'].values
            strategy_params['tick_lower'] = strategy_df['tick_lower'].values
            strategy_params['tick_upper'] = strategy_df['tick_upper'].values
            strategy_params['rebalance_threshold'] = strategy_df.get('rebalance_threshold', 0.0).values
            strategy_params['rebalance_frequency'] = strategy_df.get('rebalance_frequency', 0).values
        elif self.amm_type == 'dlmm':
            strategy_params = np.zeros(n_strategies, dtype=STRATEGY_PARAM_DTYPE_DLMM)
            strategy_params['initial_capital'] = strategy_df['initial_capital'].values
            strategy_params['bin_lower'] = strategy_df['bin_lower'].values
            strategy_params['bin_upper'] = strategy_df['bin_upper'].values
            strategy_params['liquidity_shape'] = strategy_df.get('liquidity_shape', 0).values
            strategy_params['rebalance_threshold'] = strategy_df.get('rebalance_threshold', 0.0).values
            strategy_params['rebalance_frequency'] = strategy_df.get('rebalance_frequency', 0).values
        else:
            raise NotImplementedError(f"AMM type '{self.amm_type}' not supported")

        # Validate swaps data
        required_cols = ['amount0', 'amount1']
        for col in required_cols:
            if col not in swaps.columns:
                raise ValueError(f"Missing required column: {col}")

        n_swaps = len(swaps)

        # Initialize positions
        positions = self.simulator.initialize_positions(
            n_strategies,
            n_swaps,
            strategy_params,
        )

        # Run simulation
        positions, metadata = self.simulator.simulate(
            swaps,
            positions,
            strategy_params,
        )

        # Extract prices
        if 'price' in swaps.columns:
            prices = swaps['price'].values
        elif 'price_history' in metadata:
            # V3 returns price history
            prices = metadata['price_history']
        elif 'reserve0_history' in metadata and 'reserve1_history' in metadata:
            # V2 - calculate from reserves
            prices = metadata['reserve1_history'] / metadata['reserve0_history']
        else:
            raise ValueError("No price data available in swaps or metadata")

        # Calculate metrics
        metrics = calculate_metrics(
            positions,
            prices,
            strategy_params['initial_capital'],
        )

        capital_efficiency = calculate_capital_efficiency(
            positions,
            prices,
        )

        # Combine metrics
        metrics = pd.concat([metrics, capital_efficiency], axis=1)

        # Create array wrapper
        wrapper = ArrayWrapper(
            index=swaps.index if hasattr(swaps, 'index') else pd.RangeIndex(n_swaps),
            columns=pd.RangeIndex(n_strategies),
            ndim=2,
        )

        # Create result object
        result = BacktestResult(
            positions=positions,
            metrics=metrics,
            capital_efficiency=capital_efficiency,
            metadata=metadata,
            wrapper=wrapper,
            strategy_params=strategy_df,
        )

        return result
