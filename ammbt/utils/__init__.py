"""Utility functions and helpers."""

from ammbt.utils.config import Config
from ammbt.utils.math import (
    sqrt_price_to_price,
    price_to_sqrt_price,
    tick_to_price,
    price_to_tick,
    # DLMM math
    bin_id_to_price,
    price_to_bin_id,
    calculate_dlmm_base_fee,
    calculate_dlmm_total_fee,
)
from ammbt.utils.strategies import (
    # V3 helpers
    generate_tick_ranges,
    generate_tick_ranges_asymmetric,
    create_v3_strategy_grid,
    range_width_from_ticks,
    optimal_rebalance_frequency,
    # DLMM helpers
    SHAPE_SPOT,
    SHAPE_CURVE,
    SHAPE_BID_ASK,
    generate_bin_ranges,
    generate_bin_ranges_asymmetric,
    create_dlmm_strategy_grid,
    bin_range_width_from_bins,
    estimate_dlmm_fee_apr,
    get_recommended_bin_step,
)

__all__ = [
    "Config",
    # V3 math
    "sqrt_price_to_price",
    "price_to_sqrt_price",
    "tick_to_price",
    "price_to_tick",
    # DLMM math
    "bin_id_to_price",
    "price_to_bin_id",
    "calculate_dlmm_base_fee",
    "calculate_dlmm_total_fee",
    # V3 strategies
    "generate_tick_ranges",
    "generate_tick_ranges_asymmetric",
    "create_v3_strategy_grid",
    "range_width_from_ticks",
    "optimal_rebalance_frequency",
    # DLMM strategies
    "SHAPE_SPOT",
    "SHAPE_CURVE",
    "SHAPE_BID_ASK",
    "generate_bin_ranges",
    "generate_bin_ranges_asymmetric",
    "create_dlmm_strategy_grid",
    "bin_range_width_from_bins",
    "estimate_dlmm_fee_apr",
    "get_recommended_bin_step",
]
