"""
Uniswap V3 (CLMM) simulator.

Concentrated Liquidity Market Maker with tick-based ranges.
Implements the math from the Uniswap v3 whitepaper.
"""

import numpy as np
import pandas as pd
import numba
from typing import Dict, Any, Tuple

from ammbt.amms.base import BaseAMMSimulator
from ammbt.utils.math import (
    tick_to_price,
    price_to_tick,
    tick_to_sqrt_price,
    get_amounts_for_liquidity,
    get_liquidity_for_amounts,
    get_tick_spacing,
    round_tick_to_spacing,
    floor_tick_to_spacing,
    ceil_tick_to_spacing,
    Q96,
)
from ammbt.utils.config import Config


# Maximum number of initialized ticks to track
MAX_TICKS = 4096


# V3 Position dtype
V3_POSITION_DTYPE = np.dtype([
    # Position parameters
    ('tick_lower', 'i4'),           # Lower tick bound
    ('tick_upper', 'i4'),           # Upper tick bound
    ('liquidity', 'f8'),            # Position liquidity

    # Current balances
    ('token0_balance', 'f8'),       # Current token0 amount
    ('token1_balance', 'f8'),       # Current token1 amount

    # Fee tracking
    ('fee_growth_inside_0_last', 'f8'),  # Last fee growth checkpoint token0
    ('fee_growth_inside_1_last', 'f8'),  # Last fee growth checkpoint token1
    ('uncollected_fees_0', 'f8'),   # Uncollected fees token0
    ('uncollected_fees_1', 'f8'),   # Uncollected fees token1

    # Position state
    ('is_active', 'bool'),          # Position active
    ('is_in_range', 'bool'),        # Currently in range
    ('gas_spent', 'f8'),            # Cumulative gas costs
    ('last_rebalance_idx', 'i4'),   # Last rebalance swap index
    ('num_rebalances', 'i4'),       # Rebalance count
])


@numba.jit(nopython=True, cache=True)
def _get_sqrt_ratio_at_tick(tick: int) -> float:
    """
    Calculate sqrt price (Q96 format) at a given tick.

    From Uniswap v3 whitepaper: sqrt(1.0001^tick) * 2^96

    Note: Returns float64 instead of int to avoid Numba overflow
    """
    price = tick_to_price(tick)
    Q96_FLOAT = 79228162514264337593543950336.0
    return np.sqrt(price) * Q96_FLOAT


# =============================================================================
# Tick Map Data Structures and Helper Functions
# =============================================================================


@numba.jit(nopython=True, cache=True)
def _build_tick_map(
    positions: np.ndarray,
    n_strategies: int,
    current_tick: int,
    tick_indices: np.ndarray,
    liquidity_net: np.ndarray,
    fee_growth_outside_0: np.ndarray,
    fee_growth_outside_1: np.ndarray,
    fee_growth_global_0: float,
    fee_growth_global_1: float,
) -> int:
    """
    Build tick map from position boundaries.

    Collects unique ticks from all position boundaries, sorts them,
    and computes the net liquidity delta at each tick.

    Parameters
    ----------
    positions : np.ndarray
        Position state array at timestep 0 (shape: n_strategies)
    n_strategies : int
        Number of strategies
    current_tick : int
        Current pool tick
    tick_indices : np.ndarray
        Output array for tick indices (pre-allocated)
    liquidity_net : np.ndarray
        Output array for net liquidity at each tick (pre-allocated)
    fee_growth_outside_0 : np.ndarray
        Output array for fee growth outside token0 (pre-allocated)
    fee_growth_outside_1 : np.ndarray
        Output array for fee growth outside token1 (pre-allocated)
    fee_growth_global_0 : float
        Global fee growth for token0
    fee_growth_global_1 : float
        Global fee growth for token1

    Returns
    -------
    int
        Number of initialized ticks
    """
    # Collect all tick boundaries from positions
    # Each position contributes tick_lower and tick_upper
    temp_ticks = np.zeros(n_strategies * 2, dtype=np.int32)
    temp_liq_lower = np.zeros(n_strategies * 2, dtype=np.float64)
    temp_liq_upper = np.zeros(n_strategies * 2, dtype=np.float64)

    tick_count = 0
    for j in range(n_strategies):
        if positions[j]['is_active']:
            tick_lower = positions[j]['tick_lower']
            tick_upper = positions[j]['tick_upper']
            liq = positions[j]['liquidity']

            # Add lower tick
            found_lower = False
            for k in range(tick_count):
                if temp_ticks[k] == tick_lower:
                    temp_liq_lower[k] += liq
                    found_lower = True
                    break
            if not found_lower:
                temp_ticks[tick_count] = tick_lower
                temp_liq_lower[tick_count] = liq
                tick_count += 1

            # Add upper tick
            found_upper = False
            for k in range(tick_count):
                if temp_ticks[k] == tick_upper:
                    temp_liq_upper[k] += liq
                    found_upper = True
                    break
            if not found_upper:
                temp_ticks[tick_count] = tick_upper
                temp_liq_upper[tick_count] = liq
                tick_count += 1

    if tick_count == 0:
        return 0

    # Sort ticks using simple bubble sort (small n)
    for i in range(tick_count):
        for k in range(i + 1, tick_count):
            if temp_ticks[k] < temp_ticks[i]:
                # Swap ticks
                temp_ticks[i], temp_ticks[k] = temp_ticks[k], temp_ticks[i]
                # Swap liquidity
                temp_liq_lower[i], temp_liq_lower[k] = temp_liq_lower[k], temp_liq_lower[i]
                temp_liq_upper[i], temp_liq_upper[k] = temp_liq_upper[k], temp_liq_upper[i]

    # Compute net liquidity at each tick and initialize fee_growth_outside
    num_ticks = min(tick_count, len(tick_indices))
    for i in range(num_ticks):
        tick = temp_ticks[i]
        tick_indices[i] = tick
        # Net liquidity: +L at lower boundary, -L at upper boundary
        liquidity_net[i] = temp_liq_lower[i] - temp_liq_upper[i]

        # Initialize fee_growth_outside per V3 spec:
        # If tick <= current_tick, fee_growth_outside = fee_growth_global
        # If tick > current_tick, fee_growth_outside = 0
        if tick <= current_tick:
            fee_growth_outside_0[i] = fee_growth_global_0
            fee_growth_outside_1[i] = fee_growth_global_1
        else:
            fee_growth_outside_0[i] = 0.0
            fee_growth_outside_1[i] = 0.0

    return num_ticks


@numba.jit(nopython=True, cache=True)
def _compute_active_liquidity(
    current_tick: int,
    tick_indices: np.ndarray,
    liquidity_net: np.ndarray,
    num_initialized_ticks: int,
) -> float:
    """
    Compute active liquidity by summing liquidity_net for all ticks <= current_tick.

    Parameters
    ----------
    current_tick : int
        Current pool tick
    tick_indices : np.ndarray
        Sorted array of tick indices
    liquidity_net : np.ndarray
        Net liquidity delta at each tick
    num_initialized_ticks : int
        Number of initialized ticks

    Returns
    -------
    float
        Active liquidity at current tick
    """
    liquidity = 0.0
    for i in range(num_initialized_ticks):
        if tick_indices[i] <= current_tick:
            liquidity += liquidity_net[i]
        else:
            break
    return max(0.0, liquidity)


@numba.jit(nopython=True, cache=True)
def _find_next_initialized_tick(
    tick_indices: np.ndarray,
    num_initialized_ticks: int,
    current_tick: int,
    zero_for_one: bool,
) -> Tuple[int, int]:
    """
    Find the next initialized tick in the swap direction using binary search.

    Parameters
    ----------
    tick_indices : np.ndarray
        Sorted array of tick indices
    num_initialized_ticks : int
        Number of initialized ticks
    current_tick : int
        Current pool tick
    zero_for_one : bool
        True if swapping token0 for token1 (price decreasing)

    Returns
    -------
    Tuple[int, int]
        (tick_value, tick_array_index) or (MIN_TICK/MAX_TICK, -1) if none found
    """
    MIN_TICK = -887272
    MAX_TICK = 887272

    if num_initialized_ticks == 0:
        if zero_for_one:
            return (MIN_TICK, -1)
        else:
            return (MAX_TICK, -1)

    if zero_for_one:
        # Looking for next tick below current_tick (price decreasing)
        # Binary search for largest tick < current_tick
        result_idx = -1
        left, right = 0, num_initialized_ticks - 1
        while left <= right:
            mid = (left + right) // 2
            if tick_indices[mid] < current_tick:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        if result_idx >= 0:
            return (tick_indices[result_idx], result_idx)
        else:
            return (MIN_TICK, -1)
    else:
        # Looking for next tick above current_tick (price increasing)
        # Binary search for smallest tick > current_tick
        result_idx = -1
        left, right = 0, num_initialized_ticks - 1
        while left <= right:
            mid = (left + right) // 2
            if tick_indices[mid] > current_tick:
                result_idx = mid
                right = mid - 1
            else:
                left = mid + 1

        if result_idx >= 0:
            return (tick_indices[result_idx], result_idx)
        else:
            return (MAX_TICK, -1)


@numba.jit(nopython=True, cache=True)
def _get_tick_index(
    tick: int,
    tick_indices: np.ndarray,
    num_initialized_ticks: int,
) -> int:
    """
    Get the array index for a tick value.

    Parameters
    ----------
    tick : int
        Tick value to find
    tick_indices : np.ndarray
        Sorted array of tick indices
    num_initialized_ticks : int
        Number of initialized ticks

    Returns
    -------
    int
        Array index or -1 if not found
    """
    # Binary search
    left, right = 0, num_initialized_ticks - 1
    while left <= right:
        mid = (left + right) // 2
        if tick_indices[mid] == tick:
            return mid
        elif tick_indices[mid] < tick:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# =============================================================================
# Tick-by-Tick Swap Processing
# =============================================================================


@numba.jit(nopython=True, cache=True)
def _compute_swap_step(
    sqrt_price_current_x96: float,
    liquidity: float,
    sqrt_price_target_x96: float,
    amount_remaining: float,
    fee_rate: float,
    zero_for_one: bool,
) -> Tuple[float, float, float, float]:
    """
    Compute a single swap step within one tick range.

    Uses Uniswap V3 whitepaper formulas for exact math.

    Parameters
    ----------
    sqrt_price_current_x96 : float
        Current sqrt price in Q96 format
    liquidity : float
        Active liquidity in range
    sqrt_price_target_x96 : float
        Target sqrt price at tick boundary
    amount_remaining : float
        Remaining amount to swap (always positive, after fee)
    fee_rate : float
        Fee rate as decimal (e.g., 0.003 for 0.3%)
    zero_for_one : bool
        True if swapping token0 for token1

    Returns
    -------
    Tuple[float, float, float, float]
        (new_sqrt_price_x96, amount_in_consumed, amount_out, fee_amount)
    """
    Q96_FLOAT = 79228162514264337593543950336.0

    if liquidity <= 0 or amount_remaining <= 0:
        return (sqrt_price_current_x96, 0.0, 0.0, 0.0)

    sqrt_price = sqrt_price_current_x96 / Q96_FLOAT
    sqrt_price_target = sqrt_price_target_x96 / Q96_FLOAT

    if zero_for_one:
        # Swapping token0 for token1 (price decreases)
        # delta_sqrt_P = amount0 * sqrt_P / L
        # new_sqrt_P = sqrt_P - delta_sqrt_P
        # But we need to account for the fee on input

        # Max amount0 to reach target: L * (sqrt_P - sqrt_P_target) / (sqrt_P * sqrt_P_target)
        if sqrt_price_target >= sqrt_price:
            # Target is at or above current, no swap needed
            return (sqrt_price_current_x96, 0.0, 0.0, 0.0)

        max_amount0_to_target = liquidity * (sqrt_price - sqrt_price_target) / (sqrt_price * sqrt_price_target)
        max_amount0_to_target = max(0.0, max_amount0_to_target)

        # Calculate gross input needed (input + fee = gross)
        # amount_remaining is the gross input, amount_after_fee = gross * (1 - fee)
        amount_after_fee = amount_remaining * (1.0 - fee_rate)

        if amount_after_fee >= max_amount0_to_target:
            # Reaches or exceeds target - use full amount to reach target
            amount0_used = max_amount0_to_target
            new_sqrt_price = sqrt_price_target
            # Calculate gross input for this amount
            amount_in_gross = amount0_used / (1.0 - fee_rate)
            fee_amount = amount_in_gross * fee_rate
        else:
            # Doesn't reach target - use all remaining amount
            amount0_used = amount_after_fee
            # new_sqrt_price = L * sqrt_P / (L + amount0 * sqrt_P)
            new_sqrt_price = (liquidity * sqrt_price) / (liquidity + amount0_used * sqrt_price)
            amount_in_gross = amount_remaining
            fee_amount = amount_remaining * fee_rate

        # Amount out: L * (sqrt_P_old - sqrt_P_new)
        amount_out = liquidity * (sqrt_price - new_sqrt_price)
        amount_out = max(0.0, amount_out)

        return (new_sqrt_price * Q96_FLOAT, amount_in_gross, amount_out, fee_amount)

    else:
        # Swapping token1 for token0 (price increases)
        # new_sqrt_P = sqrt_P + amount1 / L

        if sqrt_price_target <= sqrt_price:
            # Target is at or below current, no swap needed
            return (sqrt_price_current_x96, 0.0, 0.0, 0.0)

        # Max amount1 to reach target: L * (sqrt_P_target - sqrt_P)
        max_amount1_to_target = liquidity * (sqrt_price_target - sqrt_price)
        max_amount1_to_target = max(0.0, max_amount1_to_target)

        amount_after_fee = amount_remaining * (1.0 - fee_rate)

        if amount_after_fee >= max_amount1_to_target:
            # Reaches target
            amount1_used = max_amount1_to_target
            new_sqrt_price = sqrt_price_target
            amount_in_gross = amount1_used / (1.0 - fee_rate)
            fee_amount = amount_in_gross * fee_rate
        else:
            # Doesn't reach target
            amount1_used = amount_after_fee
            new_sqrt_price = sqrt_price + amount1_used / liquidity
            amount_in_gross = amount_remaining
            fee_amount = amount_remaining * fee_rate

        # Amount out: L * (1/sqrt_P_old - 1/sqrt_P_new)
        if sqrt_price > 0 and new_sqrt_price > 0:
            amount_out = liquidity * (1.0 / sqrt_price - 1.0 / new_sqrt_price)
        else:
            amount_out = 0.0
        amount_out = max(0.0, amount_out)

        return (new_sqrt_price * Q96_FLOAT, amount_in_gross, amount_out, fee_amount)


@numba.jit(nopython=True, cache=True)
def _cross_tick(
    tick_idx: int,
    liquidity_net: np.ndarray,
    fee_growth_outside_0: np.ndarray,
    fee_growth_outside_1: np.ndarray,
    fee_growth_global_0: float,
    fee_growth_global_1: float,
    zero_for_one: bool,
    current_liquidity: float,
) -> Tuple[float, float, float]:
    """
    Cross a tick boundary during a swap.

    Flips fee_growth_outside and applies liquidity delta.

    Parameters
    ----------
    tick_idx : int
        Index in tick arrays
    liquidity_net : np.ndarray
        Net liquidity at each tick
    fee_growth_outside_0 : np.ndarray
        Fee growth outside for token0
    fee_growth_outside_1 : np.ndarray
        Fee growth outside for token1
    fee_growth_global_0 : float
        Global fee growth for token0
    fee_growth_global_1 : float
        Global fee growth for token1
    zero_for_one : bool
        True if price is decreasing (token0 -> token1)
    current_liquidity : float
        Current active liquidity

    Returns
    -------
    Tuple[float, float, float]
        (new_liquidity, new_fee_growth_outside_0, new_fee_growth_outside_1)
    """
    if tick_idx < 0:
        return (current_liquidity, 0.0, 0.0)

    # Flip fee_growth_outside when crossing
    new_fg_outside_0 = fee_growth_global_0 - fee_growth_outside_0[tick_idx]
    new_fg_outside_1 = fee_growth_global_1 - fee_growth_outside_1[tick_idx]
    fee_growth_outside_0[tick_idx] = new_fg_outside_0
    fee_growth_outside_1[tick_idx] = new_fg_outside_1

    # Apply liquidity delta
    liq_net = liquidity_net[tick_idx]
    if zero_for_one:
        # Moving left (price decreasing), subtract liquidity
        new_liquidity = current_liquidity - liq_net
    else:
        # Moving right (price increasing), add liquidity
        new_liquidity = current_liquidity + liq_net

    return (max(0.0, new_liquidity), new_fg_outside_0, new_fg_outside_1)


@numba.jit(nopython=True, cache=True)
def _execute_swap(
    amount_specified: float,
    zero_for_one: bool,
    sqrt_price_x96: float,
    liquidity: float,
    current_tick: int,
    tick_indices: np.ndarray,
    liquidity_net: np.ndarray,
    fee_growth_outside_0: np.ndarray,
    fee_growth_outside_1: np.ndarray,
    num_initialized_ticks: int,
    fee_rate: float,
    fee_growth_global_0: float,
    fee_growth_global_1: float,
) -> Tuple[float, float, float, int, float, float, float]:
    """
    Execute a full swap with tick-by-tick processing.

    Parameters
    ----------
    amount_specified : float
        Amount to swap (positive = exact input)
    zero_for_one : bool
        True if swapping token0 for token1
    sqrt_price_x96 : float
        Current sqrt price
    liquidity : float
        Current active liquidity
    current_tick : int
        Current tick
    tick_indices : np.ndarray
        Initialized tick indices
    liquidity_net : np.ndarray
        Net liquidity at each tick
    fee_growth_outside_0 : np.ndarray
        Fee growth outside token0
    fee_growth_outside_1 : np.ndarray
        Fee growth outside token1
    num_initialized_ticks : int
        Number of initialized ticks
    fee_rate : float
        Fee rate as decimal
    fee_growth_global_0 : float
        Global fee growth token0
    fee_growth_global_1 : float
        Global fee growth token1

    Returns
    -------
    Tuple
        (new_sqrt_price, amount0_delta, amount1_delta, new_tick,
         new_fee_growth_global_0, new_fee_growth_global_1, new_liquidity)
    """
    Q96_FLOAT = 79228162514264337593543950336.0
    MIN_TICK = -887272
    MAX_TICK = 887272

    amount_remaining = abs(amount_specified)
    total_amount_in = 0.0
    total_amount_out = 0.0
    total_fee = 0.0

    # Maximum iterations to prevent infinite loop
    max_iterations = 500
    iteration = 0

    while amount_remaining > 1e-18 and iteration < max_iterations:
        iteration += 1

        # Find next initialized tick
        next_tick, next_tick_idx = _find_next_initialized_tick(
            tick_indices, num_initialized_ticks, current_tick, zero_for_one
        )

        # Calculate sqrt price at next tick
        if next_tick == MIN_TICK:
            sqrt_price_target_x96 = _get_sqrt_ratio_at_tick(MIN_TICK + 1)
        elif next_tick == MAX_TICK:
            sqrt_price_target_x96 = _get_sqrt_ratio_at_tick(MAX_TICK - 1)
        else:
            sqrt_price_target_x96 = _get_sqrt_ratio_at_tick(next_tick)

        # Execute swap step
        new_sqrt_price_x96, amount_in, amount_out, fee_amount = _compute_swap_step(
            sqrt_price_x96,
            liquidity,
            sqrt_price_target_x96,
            amount_remaining,
            fee_rate,
            zero_for_one,
        )

        total_amount_in += amount_in
        total_amount_out += amount_out
        total_fee += fee_amount
        amount_remaining -= amount_in

        # Update fee growth global
        if liquidity > 0 and fee_amount > 0:
            if zero_for_one:
                fee_growth_global_0 += fee_amount / liquidity
            else:
                fee_growth_global_1 += fee_amount / liquidity

        sqrt_price_x96 = new_sqrt_price_x96

        # Check if we hit the tick boundary
        at_boundary = False
        if zero_for_one:
            at_boundary = new_sqrt_price_x96 <= sqrt_price_target_x96 * 1.0000001
        else:
            at_boundary = new_sqrt_price_x96 >= sqrt_price_target_x96 * 0.9999999

        if at_boundary and next_tick_idx >= 0 and amount_remaining > 1e-18:
            # Cross the tick
            liquidity, _, _ = _cross_tick(
                next_tick_idx,
                liquidity_net,
                fee_growth_outside_0,
                fee_growth_outside_1,
                fee_growth_global_0,
                fee_growth_global_1,
                zero_for_one,
                liquidity,
            )
            if zero_for_one:
                current_tick = next_tick - 1
            else:
                current_tick = next_tick

    # Calculate final tick from price
    final_price = (sqrt_price_x96 / Q96_FLOAT) ** 2
    final_tick = price_to_tick(final_price)

    # Compute deltas based on swap direction
    if zero_for_one:
        amount0_delta = total_amount_in  # Input
        amount1_delta = -total_amount_out  # Output (negative)
    else:
        amount0_delta = -total_amount_out  # Output (negative)
        amount1_delta = total_amount_in  # Input

    return (
        sqrt_price_x96,
        amount0_delta,
        amount1_delta,
        final_tick,
        fee_growth_global_0,
        fee_growth_global_1,
        liquidity,
    )


@numba.jit(nopython=True, cache=True)
def _calculate_fee_growth_inside(
    tick_lower: int,
    tick_upper: int,
    current_tick: int,
    fee_growth_global_0: float,
    fee_growth_global_1: float,
    fee_growth_outside_lower_0: float,
    fee_growth_outside_lower_1: float,
    fee_growth_outside_upper_0: float,
    fee_growth_outside_upper_1: float,
) -> Tuple[float, float]:
    """
    Calculate fee growth inside a position's range.

    This is the core of v3's fee calculation logic.
    From whitepaper section 6.3.
    """
    # Fee growth below the range
    if current_tick >= tick_lower:
        fee_growth_below_0 = fee_growth_outside_lower_0
        fee_growth_below_1 = fee_growth_outside_lower_1
    else:
        fee_growth_below_0 = fee_growth_global_0 - fee_growth_outside_lower_0
        fee_growth_below_1 = fee_growth_global_1 - fee_growth_outside_lower_1

    # Fee growth above the range
    if current_tick < tick_upper:
        fee_growth_above_0 = fee_growth_outside_upper_0
        fee_growth_above_1 = fee_growth_outside_upper_1
    else:
        fee_growth_above_0 = fee_growth_global_0 - fee_growth_outside_upper_0
        fee_growth_above_1 = fee_growth_global_1 - fee_growth_outside_upper_1

    # Fee growth inside = global - below - above
    fee_growth_inside_0 = fee_growth_global_0 - fee_growth_below_0 - fee_growth_above_0
    fee_growth_inside_1 = fee_growth_global_1 - fee_growth_below_1 - fee_growth_above_1

    return fee_growth_inside_0, fee_growth_inside_1


@numba.jit(nopython=True, cache=True)
def _simulate_v3_swaps_nb(
    swap_amounts_0: np.ndarray,
    swap_amounts_1: np.ndarray,
    positions: np.ndarray,
    initial_sqrt_price_x96: float,
    initial_liquidity: float,
    fee_tier: int,
    rebalance_threshold: np.ndarray,
    rebalance_frequency: np.ndarray,
    tick_indices: np.ndarray,
    liquidity_net: np.ndarray,
    fee_growth_outside_0: np.ndarray,
    fee_growth_outside_1: np.ndarray,
    num_initialized_ticks: int,
    tick_spacing: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Numba-compiled simulation loop for Uniswap V3 with tick-by-tick processing.

    Parameters
    ----------
    swap_amounts_0 : np.ndarray
        Token0 amount for each swap
    swap_amounts_1 : np.ndarray
        Token1 amount for each swap
    positions : np.ndarray
        Position state array (n_swaps, n_strategies)
    initial_sqrt_price_x96 : float
        Starting sqrt price in Q96 format
    initial_liquidity : float
        Initial pool liquidity (base liquidity from external LPs)
    fee_tier : int
        Fee in basis points (500, 3000, 10000)
    rebalance_threshold : np.ndarray
        Price deviation threshold for rebalancing
    rebalance_frequency : np.ndarray
        Minimum swaps between rebalances
    tick_indices : np.ndarray
        Sorted array of initialized tick indices
    liquidity_net : np.ndarray
        Net liquidity delta at each tick
    fee_growth_outside_0 : np.ndarray
        Fee growth outside for token0 at each tick
    fee_growth_outside_1 : np.ndarray
        Fee growth outside for token1 at each tick
    num_initialized_ticks : int
        Number of initialized ticks
    tick_spacing : int
        Tick spacing for this fee tier

    Returns
    -------
    tuple
        (positions, sqrt_price_history, tick_history)
    """
    n_swaps = len(swap_amounts_0)
    n_strategies = positions.shape[1]

    # Track pool state over time (use float64 for sqrt_price to avoid overflow)
    sqrt_price_history = np.zeros(n_swaps, dtype=np.float64)
    liquidity_history = np.zeros(n_swaps)
    tick_history = np.zeros(n_swaps, dtype=np.int32)

    # Initialize pool state
    Q96_FLOAT = 79228162514264337593543950336.0
    sqrt_price_x96 = initial_sqrt_price_x96
    current_tick = price_to_tick((sqrt_price_x96 / Q96_FLOAT) ** 2)

    # Compute initial active liquidity from tick map + base liquidity
    liquidity = initial_liquidity + _compute_active_liquidity(
        current_tick, tick_indices, liquidity_net, num_initialized_ticks
    )

    # Fee growth accumulators (global)
    fee_growth_global_0 = 0.0
    fee_growth_global_1 = 0.0

    # Fee rate as fraction
    fee_rate = fee_tier / 1_000_000.0

    # Calculate initial price
    initial_price = (sqrt_price_x96 / Q96_FLOAT) ** 2

    # Main simulation loop
    for i in range(n_swaps):
        # Get current price
        current_price = (sqrt_price_x96 / Q96_FLOAT) ** 2

        # 1. Process swap with tick-by-tick execution
        amount0_in = max(0.0, swap_amounts_0[i])
        amount1_in = max(0.0, swap_amounts_1[i])

        if amount0_in > 0 and liquidity > 0:
            # Swapping token0 for token1 (price decreases)
            (
                sqrt_price_x96,
                _,  # amount0_delta
                _,  # amount1_delta
                current_tick,
                fee_growth_global_0,
                fee_growth_global_1,
                liquidity,
            ) = _execute_swap(
                amount0_in,
                True,  # zero_for_one
                sqrt_price_x96,
                liquidity,
                current_tick,
                tick_indices,
                liquidity_net,
                fee_growth_outside_0,
                fee_growth_outside_1,
                num_initialized_ticks,
                fee_rate,
                fee_growth_global_0,
                fee_growth_global_1,
            )

        elif amount1_in > 0 and liquidity > 0:
            # Swapping token1 for token0 (price increases)
            (
                sqrt_price_x96,
                _,  # amount0_delta
                _,  # amount1_delta
                current_tick,
                fee_growth_global_0,
                fee_growth_global_1,
                liquidity,
            ) = _execute_swap(
                amount1_in,
                False,  # zero_for_one
                sqrt_price_x96,
                liquidity,
                current_tick,
                tick_indices,
                liquidity_net,
                fee_growth_outside_0,
                fee_growth_outside_1,
                num_initialized_ticks,
                fee_rate,
                fee_growth_global_0,
                fee_growth_global_1,
            )

        # Ensure sqrt_price doesn't go to zero
        sqrt_price_x96 = max(1.0, sqrt_price_x96)

        # Store pool history
        sqrt_price_history[i] = sqrt_price_x96
        liquidity_history[i] = liquidity
        tick_history[i] = current_tick

        # 2. Update all positions
        for j in range(n_strategies):
            if not positions[i, j]['is_active']:
                if i > 0:
                    positions[i, j] = positions[i-1, j]
                continue

            pos = positions[i, j]
            tick_lower = pos['tick_lower']
            tick_upper = pos['tick_upper']
            pos_liquidity = pos['liquidity']

            # Check if position is in range
            is_in_range = (current_tick >= tick_lower) and (current_tick < tick_upper)
            positions[i, j]['is_in_range'] = is_in_range

            # Calculate current token amounts based on price and range
            sqrt_price_a = _get_sqrt_ratio_at_tick(tick_lower)
            sqrt_price_b = _get_sqrt_ratio_at_tick(tick_upper)

            amount0, amount1 = get_amounts_for_liquidity(
                sqrt_price_x96,
                sqrt_price_a,
                sqrt_price_b,
                pos_liquidity,
            )

            positions[i, j]['token0_balance'] = amount0
            positions[i, j]['token1_balance'] = amount1

            # Calculate fees earned using proper fee_growth_outside from tick map
            # Look up fee_growth_outside for lower and upper ticks
            lower_idx = _get_tick_index(tick_lower, tick_indices, num_initialized_ticks)
            upper_idx = _get_tick_index(tick_upper, tick_indices, num_initialized_ticks)

            fg_outside_lower_0 = 0.0
            fg_outside_lower_1 = 0.0
            fg_outside_upper_0 = 0.0
            fg_outside_upper_1 = 0.0

            if lower_idx >= 0:
                fg_outside_lower_0 = fee_growth_outside_0[lower_idx]
                fg_outside_lower_1 = fee_growth_outside_1[lower_idx]
            if upper_idx >= 0:
                fg_outside_upper_0 = fee_growth_outside_0[upper_idx]
                fg_outside_upper_1 = fee_growth_outside_1[upper_idx]

            fee_growth_inside_0, fee_growth_inside_1 = _calculate_fee_growth_inside(
                tick_lower,
                tick_upper,
                current_tick,
                fee_growth_global_0,
                fee_growth_global_1,
                fg_outside_lower_0,
                fg_outside_lower_1,
                fg_outside_upper_0,
                fg_outside_upper_1,
            )

            # Calculate fees accrued since last checkpoint
            if i > 0:
                fees_0 = pos_liquidity * (
                    fee_growth_inside_0 - pos['fee_growth_inside_0_last']
                )
                fees_1 = pos_liquidity * (
                    fee_growth_inside_1 - pos['fee_growth_inside_1_last']
                )

                positions[i, j]['uncollected_fees_0'] += max(0.0, fees_0)
                positions[i, j]['uncollected_fees_1'] += max(0.0, fees_1)

            # Update checkpoint
            positions[i, j]['fee_growth_inside_0_last'] = fee_growth_inside_0
            positions[i, j]['fee_growth_inside_1_last'] = fee_growth_inside_1

            # 3. Check rebalancing conditions
            price_deviation = abs(current_price - initial_price) / initial_price
            swaps_since_rebalance = i - pos['last_rebalance_idx']

            should_rebalance = False

            # Price-based rebalancing
            if rebalance_threshold[j] > 0 and price_deviation >= rebalance_threshold[j]:
                if swaps_since_rebalance >= rebalance_frequency[j]:
                    should_rebalance = True

            # Out of range rebalancing
            if not is_in_range and swaps_since_rebalance >= rebalance_frequency[j]:
                if rebalance_frequency[j] > 0:
                    should_rebalance = True

            if should_rebalance:
                # Execute rebalance
                gas_cost_usd = 100.0  # v3 rebalancing is more expensive
                positions[i, j]['gas_spent'] += gas_cost_usd
                positions[i, j]['last_rebalance_idx'] = i
                positions[i, j]['num_rebalances'] += 1

                # Re-center range around current price with proper tick spacing
                tick_width = tick_upper - tick_lower
                half_width = tick_width // 2

                # Round new ticks to tick spacing
                new_tick_lower = floor_tick_to_spacing(current_tick - half_width, tick_spacing)
                new_tick_upper = ceil_tick_to_spacing(current_tick + half_width, tick_spacing)

                # Ensure minimum width
                if new_tick_upper <= new_tick_lower:
                    new_tick_upper = new_tick_lower + tick_spacing

                positions[i, j]['tick_lower'] = new_tick_lower
                positions[i, j]['tick_upper'] = new_tick_upper
                positions[i, j]['is_in_range'] = True

                # Reset fee checkpoints
                positions[i, j]['fee_growth_inside_0_last'] = fee_growth_global_0
                positions[i, j]['fee_growth_inside_1_last'] = fee_growth_global_1

            # Copy forward for next iteration
            if i < n_swaps - 1:
                positions[i+1, j] = positions[i, j]

    return positions, sqrt_price_history, tick_history


class UniswapV3Simulator(BaseAMMSimulator):
    """
    Uniswap V3 CLMM simulator.

    Simulates concentrated liquidity positions with tick ranges.
    """

    def __init__(
        self,
        initial_sqrt_price_x96: float = None,
        initial_price: float = 1.0,
        initial_liquidity: float = 1_000_000.0,
        fee_tier: int = 3000,  # 0.3% in basis points
    ):
        """
        Initialize Uniswap V3 simulator.

        Parameters
        ----------
        initial_sqrt_price_x96 : float, optional
            Starting sqrt price in Q96 format
        initial_price : float
            Starting price (used if sqrt_price not provided)
        initial_liquidity : float
            Initial pool liquidity
        fee_tier : int
            Fee tier in basis points (500, 3000, 10000)
        """
        Q96_FLOAT = 79228162514264337593543950336.0
        if initial_sqrt_price_x96 is None:
            initial_sqrt_price_x96 = np.sqrt(initial_price) * Q96_FLOAT

        pool_params = {
            "initial_sqrt_price_x96": initial_sqrt_price_x96,
            "initial_price": initial_price,
            "initial_liquidity": initial_liquidity,
            "fee_tier": fee_tier,
        }
        super().__init__(pool_params)

    def initialize_positions(
        self,
        n_strategies: int,
        n_swaps: int,
        strategy_params: np.ndarray,
    ) -> np.ndarray:
        """
        Create initial position array for V3.

        Parameters
        ----------
        n_strategies : int
            Number of strategy variants
        n_swaps : int
            Number of time steps
        strategy_params : np.ndarray
            Strategy parameters with fields:
            - initial_capital: float
            - tick_lower: int
            - tick_upper: int
            - rebalance_threshold: float
            - rebalance_frequency: int

        Returns
        -------
        np.ndarray
            Initialized position array (n_swaps, n_strategies)
        """
        positions = np.zeros((n_swaps, n_strategies), dtype=V3_POSITION_DTYPE)

        Q96_FLOAT = 79228162514264337593543950336.0
        sqrt_price_x96 = self.pool_params["initial_sqrt_price_x96"]
        current_price = (sqrt_price_x96 / Q96_FLOAT) ** 2

        for j in range(n_strategies):
            capital = strategy_params[j]['initial_capital']
            tick_lower = strategy_params[j]['tick_lower']
            tick_upper = strategy_params[j]['tick_upper']

            # Calculate sqrt prices at tick bounds
            sqrt_price_a = _get_sqrt_ratio_at_tick(tick_lower)
            sqrt_price_b = _get_sqrt_ratio_at_tick(tick_upper)

            # Split capital 50/50 in USD terms
            amount0_initial = capital / 2.0 / current_price
            amount1_initial = capital / 2.0

            # Calculate liquidity for this amount
            liquidity = get_liquidity_for_amounts(
                sqrt_price_x96,
                sqrt_price_a,
                sqrt_price_b,
                amount0_initial,
                amount1_initial,
            )

            # Calculate actual amounts given the liquidity
            amount0, amount1 = get_amounts_for_liquidity(
                sqrt_price_x96,
                sqrt_price_a,
                sqrt_price_b,
                liquidity,
            )

            # Set initial position
            positions[0, j]['tick_lower'] = tick_lower
            positions[0, j]['tick_upper'] = tick_upper
            positions[0, j]['liquidity'] = liquidity
            positions[0, j]['token0_balance'] = amount0
            positions[0, j]['token1_balance'] = amount1
            positions[0, j]['is_active'] = True
            positions[0, j]['is_in_range'] = True
            positions[0, j]['last_rebalance_idx'] = 0

        return positions

    def simulate(
        self,
        swaps: pd.DataFrame,
        positions: np.ndarray,
        strategy_params: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run V3 simulation with tick-by-tick swap processing.

        Parameters
        ----------
        swaps : pd.DataFrame
            Swap events with columns: amount0, amount1
        positions : np.ndarray
            Position state array
        strategy_params : np.ndarray
            Strategy parameters

        Returns
        -------
        tuple
            (positions, metadata_dict)
        """
        # Extract swap data
        amount0 = swaps['amount0'].values
        amount1 = swaps['amount1'].values

        # Extract strategy parameters
        rebalance_threshold = strategy_params['rebalance_threshold']
        rebalance_frequency = strategy_params['rebalance_frequency'].astype(np.int32)

        # Get tick spacing for this fee tier
        fee_tier = self.pool_params['fee_tier']
        tick_spacing = get_tick_spacing(fee_tier)

        # Get initial state
        Q96_FLOAT = 79228162514264337593543950336.0
        sqrt_price_x96 = self.pool_params['initial_sqrt_price_x96']
        current_tick = price_to_tick((sqrt_price_x96 / Q96_FLOAT) ** 2)

        # Pre-allocate tick map arrays
        n_strategies = positions.shape[1]
        max_ticks = min(2 * n_strategies + 100, MAX_TICKS)

        tick_indices = np.zeros(max_ticks, dtype=np.int32)
        liquidity_net = np.zeros(max_ticks, dtype=np.float64)
        fee_growth_outside_0 = np.zeros(max_ticks, dtype=np.float64)
        fee_growth_outside_1 = np.zeros(max_ticks, dtype=np.float64)

        # Build tick map from initial positions
        num_initialized_ticks = _build_tick_map(
            positions[0],  # Initial positions
            n_strategies,
            current_tick,
            tick_indices,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            0.0,  # Initial fee_growth_global_0
            0.0,  # Initial fee_growth_global_1
        )

        # Run simulation
        positions, sqrt_price_hist, tick_hist = _simulate_v3_swaps_nb(
            amount0,
            amount1,
            positions,
            self.pool_params['initial_sqrt_price_x96'],
            self.pool_params['initial_liquidity'],
            fee_tier,
            rebalance_threshold,
            rebalance_frequency,
            tick_indices,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            num_initialized_ticks,
            tick_spacing,
        )

        # Convert sqrt prices to regular prices
        price_hist = (sqrt_price_hist / Q96_FLOAT) ** 2

        metadata = {
            'sqrt_price_history': sqrt_price_hist,
            'price_history': price_hist,
            'tick_history': tick_hist,
        }

        return positions, metadata
