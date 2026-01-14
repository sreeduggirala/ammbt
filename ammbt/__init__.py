"""
AMMBT - AMM Backtesting Engine

High-performance vectorized backtesting for AMM liquidity provider positions.
"""

__version__ = "0.1.0"

from ammbt.portfolio.base import LPBacktester
from ammbt.data.synthetic import generate_swaps, generate_price_path
from ammbt.plotting.core import (
    plot_performance,
    plot_metrics_heatmap,
    plot_efficient_frontier,
    plot_pnl_distribution,
)

__all__ = [
    "LPBacktester",
    "generate_swaps",
    "generate_price_path",
    "plot_performance",
    "plot_metrics_heatmap",
    "plot_efficient_frontier",
    "plot_pnl_distribution",
]
