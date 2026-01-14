"""
Strategy generation utilities.

Helper functions for creating v3 tick ranges, DLMM bin ranges, and strategy grids.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from ammbt.utils.math import price_to_tick, tick_to_price, price_to_bin_id, bin_id_to_price


# DLMM Liquidity shape constants
SHAPE_SPOT = 0
SHAPE_CURVE = 1
SHAPE_BID_ASK = 2


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


# =============================================================================
# DLMM (Meteora) Strategy Helpers
# =============================================================================


def generate_bin_ranges(
    current_price: float = 1.0,
    bin_step: int = 25,
    range_widths_pct: List[float] = None,
    num_ranges: int = 10,
) -> List[Tuple[int, int]]:
    """
    Generate symmetric bin ranges around current price for DLMM.

    Parameters
    ----------
    current_price : float
        Current price (token Y / token X)
    bin_step : int
        Bin step size in basis points (e.g., 10, 25, 100)
    range_widths_pct : list of float, optional
        Range widths as percentage of current price.
        E.g., 0.10 = ±10% range
        Default: [0.02, 0.05, 0.10, 0.20, 0.50]
    num_ranges : int
        Maximum number of ranges to return

    Returns
    -------
    list of tuple
        List of (bin_lower, bin_upper) pairs

    Examples
    --------
    >>> ranges = generate_bin_ranges(current_price=1.0, bin_step=25, range_widths_pct=[0.10])
    >>> ranges[0]  # (bin_lower, bin_upper)
    (-40, 39)
    """
    if range_widths_pct is None:
        range_widths_pct = [0.02, 0.05, 0.10, 0.20, 0.50]

    current_bin = price_to_bin_id(current_price, bin_step)
    ranges = []

    for width_pct in range_widths_pct:
        # Calculate price bounds
        lower_price = current_price * (1 - width_pct)
        upper_price = current_price * (1 + width_pct)

        # Convert to bin IDs
        bin_lower = price_to_bin_id(lower_price, bin_step)
        bin_upper = price_to_bin_id(upper_price, bin_step)

        # Ensure upper > lower
        if bin_upper <= bin_lower:
            bin_upper = bin_lower + 1

        ranges.append((bin_lower, bin_upper))

    return ranges[:num_ranges]


def generate_bin_ranges_asymmetric(
    current_price: float,
    lower_ranges_pct: List[float],
    upper_ranges_pct: List[float],
    bin_step: int = 25,
) -> List[Tuple[int, int]]:
    """
    Generate asymmetric bin ranges for directional DLMM strategies.

    Parameters
    ----------
    current_price : float
        Current price
    lower_ranges_pct : list of float
        Distance below current price (e.g., [0.05, 0.10])
    upper_ranges_pct : list of float
        Distance above current price (e.g., [0.20, 0.30])
    bin_step : int
        Bin step size in basis points

    Returns
    -------
    list of tuple
        List of (bin_lower, bin_upper) pairs

    Examples
    --------
    >>> ranges = generate_bin_ranges_asymmetric(
    ...     current_price=1.0,
    ...     lower_ranges_pct=[0.05],
    ...     upper_ranges_pct=[0.20],
    ...     bin_step=25
    ... )
    """
    ranges = []

    for lower_pct in lower_ranges_pct:
        for upper_pct in upper_ranges_pct:
            lower_price = current_price * (1 - lower_pct)
            upper_price = current_price * (1 + upper_pct)

            bin_lower = price_to_bin_id(lower_price, bin_step)
            bin_upper = price_to_bin_id(upper_price, bin_step)

            if bin_upper > bin_lower:
                ranges.append((bin_lower, bin_upper))

    return ranges


def create_dlmm_strategy_grid(
    initial_capitals: List[float],
    bin_ranges: List[Tuple[int, int]],
    liquidity_shapes: List[int] = None,
    rebalance_thresholds: List[float] = None,
    rebalance_frequencies: List[int] = None,
) -> pd.DataFrame:
    """
    Create a grid of DLMM strategies.

    Parameters
    ----------
    initial_capitals : list of float
        Capital amounts to test
    bin_ranges : list of tuple
        (bin_lower, bin_upper) pairs from generate_bin_ranges()
    liquidity_shapes : list of int, optional
        Liquidity distribution shapes:
        - 0 (SHAPE_SPOT): Uniform distribution
        - 1 (SHAPE_CURVE): Concentrated near center (bell curve)
        - 2 (SHAPE_BID_ASK): More at edges (inverse curve)
        Default: [0] (Spot only)
    rebalance_thresholds : list of float, optional
        Price deviation thresholds for rebalancing
        Default: [0.0] (no price-based rebalancing)
    rebalance_frequencies : list of int, optional
        Minimum swaps between rebalances
        Default: [0] (no time-based rebalancing)

    Returns
    -------
    pd.DataFrame
        Strategy grid with columns:
        - initial_capital
        - bin_lower
        - bin_upper
        - liquidity_shape
        - rebalance_threshold
        - rebalance_frequency

    Examples
    --------
    >>> ranges = generate_bin_ranges(1.0, bin_step=25, range_widths_pct=[0.10, 0.20])
    >>> strategies = create_dlmm_strategy_grid(
    ...     initial_capitals=[10000, 50000],
    ...     bin_ranges=ranges,
    ...     liquidity_shapes=[SHAPE_SPOT, SHAPE_CURVE],
    ... )
    >>> len(strategies)
    8
    """
    if liquidity_shapes is None:
        liquidity_shapes = [SHAPE_SPOT]

    if rebalance_thresholds is None:
        rebalance_thresholds = [0.0]

    if rebalance_frequencies is None:
        rebalance_frequencies = [0]

    strategies = []

    for capital in initial_capitals:
        for bin_lower, bin_upper in bin_ranges:
            for shape in liquidity_shapes:
                for threshold in rebalance_thresholds:
                    for frequency in rebalance_frequencies:
                        strategies.append({
                            'initial_capital': capital,
                            'bin_lower': bin_lower,
                            'bin_upper': bin_upper,
                            'liquidity_shape': shape,
                            'rebalance_threshold': threshold,
                            'rebalance_frequency': frequency,
                        })

    return pd.DataFrame(strategies)


def bin_range_width_from_bins(
    bin_lower: int,
    bin_upper: int,
    bin_step: int,
) -> float:
    """
    Calculate range width in price terms from bin bounds.

    Parameters
    ----------
    bin_lower : int
        Lower bin ID
    bin_upper : int
        Upper bin ID
    bin_step : int
        Bin step size

    Returns
    -------
    float
        Range width as percentage (e.g., 0.10 for 10%)

    Examples
    --------
    >>> bin_range_width_from_bins(-40, 40, 25)  # ~20% range
    0.2...
    """
    price_lower = bin_id_to_price(bin_lower, bin_step)
    price_upper = bin_id_to_price(bin_upper, bin_step)

    return (price_upper - price_lower) / ((price_upper + price_lower) / 2)


def estimate_dlmm_fee_apr(
    volume_daily: float,
    liquidity: float,
    base_fee: float = 0.0025,
    avg_variable_fee: float = 0.0005,
) -> float:
    """
    Estimate APR from DLMM fees.

    Simplified estimation based on volume and liquidity.

    Parameters
    ----------
    volume_daily : float
        Daily trading volume (in quote token)
    liquidity : float
        Total liquidity in the pool
    base_fee : float
        Base fee rate (e.g., 0.0025 for 0.25%)
    avg_variable_fee : float
        Average variable fee rate

    Returns
    -------
    float
        Estimated APR (e.g., 0.50 for 50%)

    Examples
    --------
    >>> estimate_dlmm_fee_apr(1_000_000, 10_000_000, 0.0025)
    0.091...  # ~9% APR
    """
    total_fee = base_fee + avg_variable_fee
    daily_fees = volume_daily * total_fee
    daily_return = daily_fees / liquidity if liquidity > 0 else 0
    apr = daily_return * 365
    return apr


def get_recommended_bin_step(volatility_daily: float) -> int:
    """
    Get recommended DLMM bin step based on volatility.

    Higher volatility pairs should use larger bin steps for stability.

    Parameters
    ----------
    volatility_daily : float
        Daily price volatility (e.g., 0.03 for 3%)

    Returns
    -------
    int
        Recommended bin step in basis points

    Examples
    --------
    >>> get_recommended_bin_step(0.01)  # Low volatility
    10
    >>> get_recommended_bin_step(0.05)  # High volatility
    100
    """
    if volatility_daily < 0.01:
        return 10  # Stablecoins
    elif volatility_daily < 0.02:
        return 25  # Low volatility
    elif volatility_daily < 0.04:
        return 50  # Medium volatility
    elif volatility_daily < 0.08:
        return 100  # High volatility
    else:
        return 200  # Very high volatility
