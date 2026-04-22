"""
Walk-forward analysis for LP strategy robustness testing.

Splits swap data into in-sample/out-of-sample windows, optimizes strategy
selection in-sample, and validates on out-of-sample data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    splits: List[Dict]
    """Per-split info: in-sample/out-of-sample indices, best strategy index."""

    in_sample_best: List[int]
    """Best strategy index selected in each in-sample window."""

    oos_metrics: pd.DataFrame
    """Out-of-sample metrics for the selected strategy in each split."""

    aggregate_metrics: Dict[str, float]
    """Averaged metrics across all out-of-sample periods."""

    def __repr__(self) -> str:
        n = len(self.splits)
        avg_ret = self.aggregate_metrics.get('total_return', 0)
        avg_sharpe = self.aggregate_metrics.get('sharpe', 0)
        return (
            f"WalkForwardResult(\n"
            f"  n_splits={n},\n"
            f"  avg_oos_return={avg_ret:.4f},\n"
            f"  avg_oos_sharpe={avg_sharpe:.4f}\n"
            f")"
        )


def walk_forward(
    backtester,
    swaps: pd.DataFrame,
    strategy_grid: Union[Dict, pd.DataFrame],
    n_splits: int = 5,
    train_pct: float = 0.6,
    optimization_metric: str = 'sharpe',
) -> WalkForwardResult:
    """
    Run walk-forward analysis.

    For each split:
    1. Run all strategies on in-sample data.
    2. Select the best strategy by ``optimization_metric``.
    3. Run that strategy on out-of-sample data.
    4. Record out-of-sample performance.

    Parameters
    ----------
    backtester : LPBacktester
        Configured backtester instance.
    swaps : pd.DataFrame
        Full swap dataset.
    strategy_grid : dict or pd.DataFrame
        Strategy parameters to test (full grid).
    n_splits : int
        Number of walk-forward windows.
    train_pct : float
        Fraction of each window used for in-sample (0 < train_pct < 1).
    optimization_metric : str
        Metric to optimize in-sample (e.g., ``'sharpe'``, ``'net_pnl'``,
        ``'total_return'``, ``'sortino'``).

    Returns
    -------
    WalkForwardResult
        Results with per-split and aggregate out-of-sample metrics.

    Examples
    --------
    >>> bt = LPBacktester(amm_type='v3')
    >>> result = walk_forward(bt, swaps, strategy_grid, n_splits=5)
    >>> print(result.aggregate_metrics)
    """
    if not 0 < train_pct < 1:
        raise ValueError(f"train_pct must be between 0 and 1, got {train_pct}")

    n_total = len(swaps)
    window_size = n_total // n_splits

    if window_size < 10:
        raise ValueError(
            f"Not enough data for {n_splits} splits "
            f"({n_total} swaps, {window_size} per window)"
        )

    # Convert strategy grid to DataFrame if dict
    if isinstance(strategy_grid, dict):
        strategy_df = pd.DataFrame(strategy_grid)
    else:
        strategy_df = strategy_grid

    splits = []
    in_sample_best = []
    oos_metrics_list = []

    for split_idx in range(n_splits):
        start = split_idx * window_size
        end = min(start + window_size, n_total)

        train_end = start + int((end - start) * train_pct)

        if train_end <= start or train_end >= end:
            continue

        # In-sample
        is_swaps = swaps.iloc[start:train_end].reset_index(drop=True)
        # Out-of-sample
        oos_swaps = swaps.iloc[train_end:end].reset_index(drop=True)

        if len(is_swaps) < 2 or len(oos_swaps) < 2:
            continue

        # Run all strategies in-sample
        is_result = backtester.run(is_swaps, strategy_df)

        # Select best strategy
        if optimization_metric not in is_result.metrics.columns:
            raise ValueError(
                f"Metric '{optimization_metric}' not found in results. "
                f"Available: {list(is_result.metrics.columns)}"
            )

        best_idx = is_result.metrics[optimization_metric].idxmax()
        in_sample_best.append(int(best_idx))

        # Extract single best strategy
        best_strategy = strategy_df.iloc[[best_idx]].reset_index(drop=True)

        # Run best strategy out-of-sample
        oos_result = backtester.run(oos_swaps, best_strategy)

        # Record
        split_info = {
            'split_idx': split_idx,
            'is_start': start,
            'is_end': train_end,
            'oos_start': train_end,
            'oos_end': end,
            'best_strategy_idx': int(best_idx),
            'is_metric_value': float(is_result.metrics[optimization_metric].iloc[best_idx]),
        }
        splits.append(split_info)

        oos_row = oos_result.metrics.iloc[0].copy()
        oos_row['split_idx'] = split_idx
        oos_metrics_list.append(oos_row)

    if not oos_metrics_list:
        raise ValueError("No valid splits produced. Check data size and n_splits.")

    oos_metrics = pd.DataFrame(oos_metrics_list).reset_index(drop=True)

    # Aggregate across splits
    numeric_cols = oos_metrics.select_dtypes(include=[np.number]).columns
    aggregate = {col: float(oos_metrics[col].mean()) for col in numeric_cols if col != 'split_idx'}

    return WalkForwardResult(
        splits=splits,
        in_sample_best=in_sample_best,
        oos_metrics=oos_metrics,
        aggregate_metrics=aggregate,
    )
