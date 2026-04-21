"""
AMMBT - AMM Backtesting Engine

High-performance vectorized backtesting for AMM liquidity provider positions.
"""

__version__ = "0.3.0"

# Core backtester
from ammbt.portfolio.base import LPBacktester, BacktestResult

# AMM simulators
from ammbt.amms.univ2 import UniswapV2Simulator
from ammbt.amms.univ3 import UniswapV3Simulator
from ammbt.amms.dlmm import MeteoraLMMSimulator

# Plotting
from ammbt.plotting.core import (
    plot_performance,
    plot_metrics_heatmap,
    plot_efficient_frontier,
    plot_pnl_distribution,
)

# Strategy generators
from ammbt.utils.strategies import (
    generate_tick_ranges,
    generate_tick_ranges_asymmetric,
    create_v3_strategy_grid,
    generate_bin_ranges,
    generate_bin_ranges_asymmetric,
    create_dlmm_strategy_grid,
)

# Math utilities
from ammbt.utils.math import (
    tick_to_price,
    price_to_tick,
    get_tick_spacing,
    tick_to_sqrt_price,
    sqrt_price_to_tick,
    bin_id_to_price,
    price_to_bin_id,
)

# Data loaders
from ammbt.data import SubgraphLoader, FileLoader

__all__ = [
    # Core
    "LPBacktester",
    "BacktestResult",
    # Simulators
    "UniswapV2Simulator",
    "UniswapV3Simulator",
    "MeteoraLMMSimulator",
    # Plotting
    "plot_performance",
    "plot_metrics_heatmap",
    "plot_efficient_frontier",
    "plot_pnl_distribution",
    # Strategy generators
    "generate_tick_ranges",
    "generate_tick_ranges_asymmetric",
    "create_v3_strategy_grid",
    "generate_bin_ranges",
    "generate_bin_ranges_asymmetric",
    "create_dlmm_strategy_grid",
    # Math
    "tick_to_price",
    "price_to_tick",
    "get_tick_spacing",
    "tick_to_sqrt_price",
    "sqrt_price_to_tick",
    "bin_id_to_price",
    "price_to_bin_id",
    # Data loaders
    "SubgraphLoader",
    "FileLoader",
]
