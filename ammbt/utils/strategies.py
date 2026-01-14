"""
Strategy generation utilities.

Helper functions for creating v3 tick ranges and strategy grids.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from ammbt.utils.math import price_to_tick, tick_to_price


def generate_tick_ranges(
    current_price: float = 1.0,
    tick_spacings: List[int] = None,
    range_widths_pct: List[float] = None,
    num_ranges: int = 10,
) -> List[Tuple[int, int]]:
    """
    Generate symmetric tick ranges around current price.

    Parameters
    ----------
    current_price : float
        Current price (token1/token0)
    tick_spacings : list of int, optional
        Tick spacings to use (depends on fee tier)
        Default: [10, 60, 200] for 0.05%, 0.3%, 1% fees
    range_widths_pct : list of float, optional
        Range widths as percentage of current price
        E.g., 0.10 = ±10% range
        Default: [0.02, 0.05, 0.10, 0.20, 0.50]
    num_ranges : int
        Number of ranges to generate per width

    Returns
    -------
    list of tuple
        List of (tick_lower, tick_upper) pairs

    Examples
    --------
    >>> ranges = generate_tick_ranges(current_price=1.0, range_widths_pct=[0.10, 0.20])
    >>> len(ranges)
    20
    """
    if tick_spacings is None:
        tick_spacings = [60]  # Default to 0.3% fee tier spacing

    if range_widths_pct is None:
        range_widths_pct = [0.02, 0.05, 0.10, 0.20, 0.50]

    current_tick = price_to_tick(current_price)
    ranges = []

    for width_pct in range_widths_pct:
        for spacing in tick_spacings:
            # Calculate price bounds
            lower_price = current_price * (1 - width_pct)
            upper_price = current_price * (1 + width_pct)

            # Convert to ticks
            tick_lower = price_to_tick(lower_price)
            tick_upper = price_to_tick(upper_price)

            # Round to nearest tick spacing
            tick_lower = (tick_lower // spacing) * spacing
            tick_upper = (tick_upper // spacing) * spacing

            # Ensure upper > lower
            if tick_upper <= tick_lower:
                tick_upper = tick_lower + spacing

            ranges.append((tick_lower, tick_upper))

    return ranges[:num_ranges]


def generate_tick_ranges_asymmetric(
    current_price: float,
    lower_ranges_pct: List[float],
    upper_ranges_pct: List[float],
    tick_spacing: int = 60,
) -> List[Tuple[int, int]]:
    """
    Generate asymmetric tick ranges (e.g., for directional strategies).

    Parameters
    ----------
    current_price : float
        Current price
    lower_ranges_pct : list of float
        Distance below current price (e.g., [0.05, 0.10])
    upper_ranges_pct : list of float
        Distance above current price (e.g., [0.20, 0.30])
    tick_spacing : int
        Tick spacing for rounding

    Returns
    -------
    list of tuple
        List of (tick_lower, tick_upper) pairs
    """
    ranges = []

    for lower_pct in lower_ranges_pct:
        for upper_pct in upper_ranges_pct:
            lower_price = current_price * (1 - lower_pct)
            upper_price = current_price * (1 + upper_pct)

            tick_lower = price_to_tick(lower_price)
            tick_upper = price_to_tick(upper_price)

            tick_lower = (tick_lower // tick_spacing) * tick_spacing
            tick_upper = (tick_upper // tick_spacing) * tick_spacing

            if tick_upper > tick_lower:
                ranges.append((tick_lower, tick_upper))

    return ranges


def create_v3_strategy_grid(
    initial_capitals: List[float],
    tick_ranges: List[Tuple[int, int]],
    rebalance_thresholds: List[float] = None,
    rebalance_frequencies: List[int] = None,
) -> pd.DataFrame:
    """
    Create a grid of v3 strategies.

    Parameters
    ----------
    initial_capitals : list of float
        Capital amounts to test
    tick_ranges : list of tuple
        (tick_lower, tick_upper) pairs
    rebalance_thresholds : list of float, optional
        Price deviation thresholds
    rebalance_frequencies : list of int, optional
        Time-based rebalancing frequencies

    Returns
    -------
    pd.DataFrame
        Strategy grid with columns: initial_capital, tick_lower, tick_upper,
        rebalance_threshold, rebalance_frequency

    Examples
    --------
    >>> ranges = generate_tick_ranges(1.0, range_widths_pct=[0.10, 0.20])
    >>> strategies = create_v3_strategy_grid(
    ...     initial_capitals=[10000, 50000],
    ...     tick_ranges=ranges,
    ... )
    """
    if rebalance_thresholds is None:
        rebalance_thresholds = [0.0]  # No price-based rebalancing

    if rebalance_frequencies is None:
        rebalance_frequencies = [0]  # No time-based rebalancing

    strategies = []

    for capital in initial_capitals:
        for tick_lower, tick_upper in tick_ranges:
            for threshold in rebalance_thresholds:
                for frequency in rebalance_frequencies:
                    strategies.append({
                        'initial_capital': capital,
                        'tick_lower': tick_lower,
                        'tick_upper': tick_upper,
                        'rebalance_threshold': threshold,
                        'rebalance_frequency': frequency,
                    })

    return pd.DataFrame(strategies)


def range_width_from_ticks(tick_lower: int, tick_upper: int) -> float:
    """
    Calculate range width in price terms.

    Parameters
    ----------
    tick_lower : int
        Lower tick
    tick_upper : int
        Upper tick

    Returns
    -------
    float
        Range width as percentage
    """
    price_lower = tick_to_price(tick_lower)
    price_upper = tick_to_price(tick_upper)

    return (price_upper - price_lower) / ((price_upper + price_lower) / 2)


def optimal_rebalance_frequency(
    range_width_pct: float,
    volatility: float,
) -> int:
    """
    Estimate optimal rebalancing frequency based on range width and volatility.

    Simple heuristic: rebalance when expected price movement exceeds range.

    Parameters
    ----------
    range_width_pct : float
        Range width as percentage (e.g., 0.10 for ±10%)
    volatility : float
        Daily volatility (e.g., 0.03 for 3%)

    Returns
    -------
    int
        Suggested number of swaps between rebalances

    Examples
    --------
    >>> # Narrow range (10%) with high volatility (5%)
    >>> optimal_rebalance_frequency(0.10, 0.05)
    100
    """
    # Days until expected movement equals range width
    # Assuming sqrt(time) scaling of volatility
    days_to_range = (range_width_pct / volatility) ** 2

    # Convert to swap count (assuming ~100 swaps/day)
    swap_count = int(days_to_range * 100)

    return max(10, min(10000, swap_count))  # Clamp to reasonable range
