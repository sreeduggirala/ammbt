"""
Pluggable rebalance strategies for AMM LP positions.

All strategies are implemented as Numba-compatible functions using integer
dispatch (since Numba nopython mode doesn't support Python classes).

Strategy types:
- STATIC (0): Re-center range with same width (current default behavior).
- VOL_ADAPTIVE (1): Widen range when volatility is high, narrow when low.
- ASYMMETRIC (2): Shift range based on price momentum.
"""

import numpy as np
import numba

# Strategy type constants
REBALANCE_STATIC = 0
REBALANCE_VOL_ADAPTIVE = 1
REBALANCE_ASYMMETRIC = 2


@numba.jit(nopython=True, cache=True)
def should_rebalance(
    strategy_type: int,
    current_price: float,
    initial_price: float,
    is_in_range: bool,
    rebalance_threshold: float,
    swaps_since_rebalance: int,
    rebalance_frequency: int,
    volatility: float,
) -> bool:
    """
    Check if a position should rebalance.

    Parameters
    ----------
    strategy_type : int
        REBALANCE_STATIC, REBALANCE_VOL_ADAPTIVE, or REBALANCE_ASYMMETRIC.
    current_price, initial_price : float
        Current and reference prices.
    is_in_range : bool
        Whether position is currently in range.
    rebalance_threshold : float
        Price deviation threshold.
    swaps_since_rebalance : int
        Swaps since last rebalance.
    rebalance_frequency : int
        Minimum swaps between rebalances.
    volatility : float
        Rolling realized volatility.

    Returns
    -------
    bool
        Whether to rebalance.
    """
    if rebalance_frequency <= 0 and rebalance_threshold <= 0:
        return False

    if swaps_since_rebalance < rebalance_frequency:
        return False

    price_deviation = abs(current_price - initial_price) / initial_price if initial_price > 0 else 0.0

    if strategy_type == REBALANCE_STATIC:
        # Standard: threshold OR out-of-range
        if rebalance_threshold > 0 and price_deviation >= rebalance_threshold:
            return True
        if not is_in_range and rebalance_frequency > 0:
            return True
        if rebalance_frequency > 0 and rebalance_threshold <= 0:
            return True
        return False

    elif strategy_type == REBALANCE_VOL_ADAPTIVE:
        # Higher vol → higher threshold (rebalance less often)
        adaptive_threshold = rebalance_threshold * (1.0 + volatility * 10.0)
        if price_deviation >= adaptive_threshold:
            return True
        if not is_in_range:
            return True
        return False

    elif strategy_type == REBALANCE_ASYMMETRIC:
        # Same trigger as static, but range computation differs
        if rebalance_threshold > 0 and price_deviation >= rebalance_threshold:
            return True
        if not is_in_range and rebalance_frequency > 0:
            return True
        return False

    return False


@numba.jit(nopython=True, cache=True)
def compute_new_tick_range(
    strategy_type: int,
    current_tick: int,
    old_tick_lower: int,
    old_tick_upper: int,
    tick_spacing: int,
    volatility: float,
    price_momentum: float,
) -> tuple:
    """
    Compute new tick range after rebalance.

    Parameters
    ----------
    strategy_type : int
        Rebalance strategy type.
    current_tick : int
        Current pool tick.
    old_tick_lower, old_tick_upper : int
        Previous range bounds.
    tick_spacing : int
        Tick spacing for the fee tier.
    volatility : float
        Rolling realized volatility.
    price_momentum : float
        Recent price direction (positive = up, negative = down).

    Returns
    -------
    tuple
        (new_tick_lower, new_tick_upper)
    """
    tick_width = old_tick_upper - old_tick_lower

    if strategy_type == REBALANCE_STATIC:
        # Re-center with same width
        half_width = tick_width // 2
        new_lower = _floor_tick(current_tick - half_width, tick_spacing)
        new_upper = _ceil_tick(current_tick + half_width, tick_spacing)

    elif strategy_type == REBALANCE_VOL_ADAPTIVE:
        # Widen range when vol is high, narrow when low
        vol_factor = 1.0 + volatility * 5.0  # Scale width by vol
        adjusted_width = int(tick_width * vol_factor)
        half_width = adjusted_width // 2
        new_lower = _floor_tick(current_tick - half_width, tick_spacing)
        new_upper = _ceil_tick(current_tick + half_width, tick_spacing)

    elif strategy_type == REBALANCE_ASYMMETRIC:
        # Shift range in direction of momentum
        half_width = tick_width // 2
        # Momentum shift: up to 25% of width
        shift = int(price_momentum * half_width * 0.25)
        center = current_tick + shift
        new_lower = _floor_tick(center - half_width, tick_spacing)
        new_upper = _ceil_tick(center + half_width, tick_spacing)

    else:
        new_lower = _floor_tick(current_tick - tick_width // 2, tick_spacing)
        new_upper = _ceil_tick(current_tick + tick_width // 2, tick_spacing)

    # Ensure minimum width
    if new_upper <= new_lower:
        new_upper = new_lower + tick_spacing

    return (new_lower, new_upper)


@numba.jit(nopython=True, cache=True)
def compute_new_bin_range(
    strategy_type: int,
    active_bin: int,
    old_bin_lower: int,
    old_bin_upper: int,
    volatility: float,
    price_momentum: float,
) -> tuple:
    """
    Compute new bin range after rebalance (for DLMM).

    Parameters
    ----------
    strategy_type : int
        Rebalance strategy type.
    active_bin : int
        Current active bin.
    old_bin_lower, old_bin_upper : int
        Previous range bounds.
    volatility : float
        Rolling realized volatility.
    price_momentum : float
        Recent price direction.

    Returns
    -------
    tuple
        (new_bin_lower, new_bin_upper)
    """
    bin_width = old_bin_upper - old_bin_lower

    if strategy_type == REBALANCE_STATIC:
        new_lower = active_bin - bin_width // 2
        new_upper = active_bin + bin_width // 2

    elif strategy_type == REBALANCE_VOL_ADAPTIVE:
        vol_factor = 1.0 + volatility * 5.0
        adjusted_width = int(bin_width * vol_factor)
        new_lower = active_bin - adjusted_width // 2
        new_upper = active_bin + adjusted_width // 2

    elif strategy_type == REBALANCE_ASYMMETRIC:
        shift = int(price_momentum * bin_width * 0.25)
        center = active_bin + shift
        new_lower = center - bin_width // 2
        new_upper = center + bin_width // 2

    else:
        new_lower = active_bin - bin_width // 2
        new_upper = active_bin + bin_width // 2

    if new_upper <= new_lower:
        new_upper = new_lower + 1

    return (new_lower, new_upper)


@numba.jit(nopython=True, cache=True)
def compute_rolling_volatility(
    prices: np.ndarray,
    current_idx: int,
    window: int,
) -> float:
    """
    Compute rolling realized volatility from price history.

    Parameters
    ----------
    prices : np.ndarray
        Price array (at least current_idx + 1 elements used).
    current_idx : int
        Current index in the price array.
    window : int
        Lookback window.

    Returns
    -------
    float
        Realized volatility (standard deviation of log returns).
    """
    start = max(0, current_idx - window)
    if current_idx - start < 2:
        return 0.0

    sum_r = 0.0
    sum_r2 = 0.0
    count = 0

    for k in range(start + 1, current_idx + 1):
        if prices[k] > 0 and prices[k - 1] > 0:
            log_ret = np.log(prices[k] / prices[k - 1])
            sum_r += log_ret
            sum_r2 += log_ret * log_ret
            count += 1

    if count < 2:
        return 0.0

    mean_r = sum_r / count
    variance = sum_r2 / count - mean_r * mean_r
    return np.sqrt(max(0.0, variance))


@numba.jit(nopython=True, cache=True)
def compute_price_momentum(
    prices: np.ndarray,
    current_idx: int,
    window: int,
) -> float:
    """
    Compute price momentum as normalized return over window.

    Returns value in [-1, 1] range.

    Parameters
    ----------
    prices : np.ndarray
        Price array.
    current_idx : int
        Current index.
    window : int
        Lookback window.

    Returns
    -------
    float
        Momentum signal in [-1, 1].
    """
    start = max(0, current_idx - window)
    if prices[start] <= 0 or prices[current_idx] <= 0:
        return 0.0

    raw_return = (prices[current_idx] - prices[start]) / prices[start]
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, raw_return))


@numba.jit(nopython=True, cache=True)
def _floor_tick(tick: int, spacing: int) -> int:
    """Floor tick to spacing."""
    if spacing <= 0:
        return tick
    if tick >= 0:
        return (tick // spacing) * spacing
    return -(((-tick + spacing - 1) // spacing) * spacing)


@numba.jit(nopython=True, cache=True)
def _ceil_tick(tick: int, spacing: int) -> int:
    """Ceil tick to spacing."""
    if spacing <= 0:
        return tick
    if tick >= 0:
        return ((tick + spacing - 1) // spacing) * spacing
    return -((-tick) // spacing) * spacing
