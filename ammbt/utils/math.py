"""
Mathematical utilities for AMM calculations.

Includes:
- Price/tick conversions (Uniswap v3 style)
- Sqrt price math
- Fixed-point arithmetic helpers
"""

import numpy as np
import numba


# Constants for Uniswap v3 math
Q96 = 2**96
Q128 = 2**128
Q192 = 2**192


@numba.jit(nopython=True, cache=True)
def sqrt_price_to_price(sqrt_price_x96: int) -> float:
    """
    Convert Uniswap v3 sqrt price (fixed point Q96) to regular price.

    Parameters
    ----------
    sqrt_price_x96 : int
        Square root of price in Q96 fixed point format

    Returns
    -------
    float
        Price (token1/token0)

    Examples
    --------
    >>> sqrt_price_x96 = 79228162514264337593543950336  # sqrt(1) in Q96
    >>> sqrt_price_to_price(sqrt_price_x96)
    1.0
    """
    sqrt_price = sqrt_price_x96 / Q96
    return sqrt_price ** 2


@numba.jit(nopython=True, cache=True)
def price_to_sqrt_price(price: float) -> int:
    """
    Convert regular price to Uniswap v3 sqrt price (Q96).

    Parameters
    ----------
    price : float
        Price (token1/token0)

    Returns
    -------
    int
        Square root of price in Q96 format

    Examples
    --------
    >>> price_to_sqrt_price(1.0)
    79228162514264337593543950336
    """
    return int(np.sqrt(price) * Q96)


@numba.jit(nopython=True, cache=True)
def tick_to_price(tick: int) -> float:
    """
    Convert tick to price.

    price = 1.0001^tick

    Parameters
    ----------
    tick : int
        Tick index

    Returns
    -------
    float
        Price at that tick

    Examples
    --------
    >>> tick_to_price(0)
    1.0
    >>> tick_to_price(1)  # ~1.0001
    1.0001
    """
    return 1.0001 ** tick


@numba.jit(nopython=True, cache=True)
def price_to_tick(price: float) -> int:
    """
    Convert price to nearest tick.

    tick = floor(log(price) / log(1.0001))

    Parameters
    ----------
    price : float
        Price

    Returns
    -------
    int
        Nearest tick

    Examples
    --------
    >>> price_to_tick(1.0)
    0
    >>> price_to_tick(1.0001)
    1
    """
    return int(np.floor(np.log(price) / np.log(1.0001)))


@numba.jit(nopython=True, cache=True)
def tick_to_sqrt_price(tick: int) -> int:
    """
    Convert tick directly to sqrt price in Q96 format.

    Parameters
    ----------
    tick : int
        Tick index

    Returns
    -------
    int
        Sqrt price in Q96 format
    """
    price = tick_to_price(tick)
    return int(np.sqrt(price) * Q96)


@numba.jit(nopython=True, cache=True)
def sqrt_price_to_tick(sqrt_price_x96: int) -> int:
    """
    Convert sqrt price to tick.

    Parameters
    ----------
    sqrt_price_x96 : int
        Sqrt price in Q96 format

    Returns
    -------
    int
        Tick index
    """
    price = sqrt_price_to_price(sqrt_price_x96)
    return price_to_tick(price)


# Liquidity math (Uniswap v3)


@numba.jit(nopython=True, cache=True)
def get_liquidity_for_amounts(
    sqrt_price_x96: float,
    sqrt_price_a_x96: float,
    sqrt_price_b_x96: float,
    amount0: float,
    amount1: float,
) -> float:
    """
    Calculate liquidity for given token amounts and price range.

    Parameters
    ----------
    sqrt_price_x96 : float
        Current sqrt price (Q96 format)
    sqrt_price_a_x96 : float
        Lower bound sqrt price (Q96 format)
    sqrt_price_b_x96 : float
        Upper bound sqrt price (Q96 format)
    amount0 : float
        Amount of token0
    amount1 : float
        Amount of token1

    Returns
    -------
    float
        Liquidity amount
    """
    Q96_FLOAT = 79228162514264337593543950336.0

    if sqrt_price_x96 <= sqrt_price_a_x96:
        # Price below range, all token0
        liquidity = (
            amount0
            * (sqrt_price_a_x96 / Q96_FLOAT)
            * (sqrt_price_b_x96 / Q96_FLOAT)
            / ((sqrt_price_b_x96 - sqrt_price_a_x96) / Q96_FLOAT)
        )
    elif sqrt_price_x96 < sqrt_price_b_x96:
        # Price in range
        liquidity0 = (
            amount0
            * (sqrt_price_x96 / Q96_FLOAT)
            * (sqrt_price_b_x96 / Q96_FLOAT)
            / ((sqrt_price_b_x96 - sqrt_price_x96) / Q96_FLOAT)
        )
        liquidity1 = amount1 / ((sqrt_price_x96 - sqrt_price_a_x96) / Q96_FLOAT)
        liquidity = min(liquidity0, liquidity1)
    else:
        # Price above range, all token1
        liquidity = amount1 / ((sqrt_price_b_x96 - sqrt_price_a_x96) / Q96_FLOAT)

    return liquidity


@numba.jit(nopython=True, cache=True)
def get_amounts_for_liquidity(
    sqrt_price_x96: float,
    sqrt_price_a_x96: float,
    sqrt_price_b_x96: float,
    liquidity: float,
) -> tuple:
    """
    Calculate token amounts for given liquidity and price range.

    Parameters
    ----------
    sqrt_price_x96 : float
        Current sqrt price (Q96 format)
    sqrt_price_a_x96 : float
        Lower bound sqrt price (Q96 format)
    sqrt_price_b_x96 : float
        Upper bound sqrt price (Q96 format)
    liquidity : float
        Liquidity amount

    Returns
    -------
    tuple
        (amount0, amount1)
    """
    Q96_FLOAT = 79228162514264337593543950336.0
    sqrt_price = sqrt_price_x96 / Q96_FLOAT
    sqrt_price_a = sqrt_price_a_x96 / Q96_FLOAT
    sqrt_price_b = sqrt_price_b_x96 / Q96_FLOAT

    if sqrt_price <= sqrt_price_a:
        # Price below range
        amount0 = liquidity * (sqrt_price_b - sqrt_price_a) / (sqrt_price_a * sqrt_price_b)
        amount1 = 0.0
    elif sqrt_price < sqrt_price_b:
        # Price in range
        amount0 = liquidity * (sqrt_price_b - sqrt_price) / (sqrt_price * sqrt_price_b)
        amount1 = liquidity * (sqrt_price - sqrt_price_a)
    else:
        # Price above range
        amount0 = 0.0
        amount1 = liquidity * (sqrt_price_b - sqrt_price_a)

    return (amount0, amount1)


# CPAMM (Uniswap v2) math


@numba.jit(nopython=True, cache=True)
def get_amount_out(amount_in: float, reserve_in: float, reserve_out: float, fee: float = 0.003) -> float:
    """
    Calculate output amount for CPAMM swap (Uniswap v2 style).

    x * y = k formula with fees

    Parameters
    ----------
    amount_in : float
        Input amount
    reserve_in : float
        Input token reserve
    reserve_out : float
        Output token reserve
    fee : float
        Fee rate (default 0.3%)

    Returns
    -------
    float
        Output amount
    """
    if amount_in <= 0:
        return 0.0
    if reserve_in <= 0 or reserve_out <= 0:
        return 0.0

    amount_in_with_fee = amount_in * (1.0 - fee)
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in + amount_in_with_fee
    return numerator / denominator


@numba.jit(nopython=True, cache=True)
def get_amount_in(amount_out: float, reserve_in: float, reserve_out: float, fee: float = 0.003) -> float:
    """
    Calculate input amount needed for desired output (CPAMM).

    Parameters
    ----------
    amount_out : float
        Desired output amount
    reserve_in : float
        Input token reserve
    reserve_out : float
        Output token reserve
    fee : float
        Fee rate

    Returns
    -------
    float
        Required input amount
    """
    if amount_out <= 0:
        return 0.0
    if amount_out >= reserve_out:
        return np.inf
    if reserve_in <= 0 or reserve_out <= 0:
        return np.inf

    numerator = reserve_in * amount_out
    denominator = (reserve_out - amount_out) * (1.0 - fee)
    return numerator / denominator


# =============================================================================
# DLMM (Meteora Dynamic Liquidity Market Maker) Math
# =============================================================================

# DLMM uses discrete price bins where price = (1 + bin_step/10000)^bin_id
# Each bin has a fixed price, and swaps within a bin have zero slippage


@numba.jit(nopython=True, cache=True)
def bin_id_to_price(bin_id: int, bin_step: int) -> float:
    """
    Convert DLMM bin ID to price.

    price = (1 + bin_step/10000)^bin_id

    Parameters
    ----------
    bin_id : int
        Bin identifier (can be negative)
    bin_step : int
        Bin step size in basis points (e.g., 10, 25, 100)

    Returns
    -------
    float
        Price at that bin

    Examples
    --------
    >>> bin_id_to_price(0, 100)
    1.0
    >>> bin_id_to_price(1, 100)  # ~1.01
    1.01
    """
    base = 1.0 + bin_step / 10000.0
    return base ** bin_id


@numba.jit(nopython=True, cache=True)
def price_to_bin_id(price: float, bin_step: int) -> int:
    """
    Convert price to nearest DLMM bin ID.

    bin_id = round(log(price) / log(1 + bin_step/10000))

    Parameters
    ----------
    price : float
        Price
    bin_step : int
        Bin step size in basis points

    Returns
    -------
    int
        Nearest bin ID

    Examples
    --------
    >>> price_to_bin_id(1.0, 100)
    0
    >>> price_to_bin_id(1.01, 100)
    1
    """
    if price <= 0:
        return -2**31  # Min int32
    base = 1.0 + bin_step / 10000.0
    return int(np.round(np.log(price) / np.log(base)))


@numba.jit(nopython=True, cache=True)
def get_bin_price_bounds(bin_id: int, bin_step: int) -> tuple:
    """
    Get lower and upper price bounds for a bin.

    A bin covers prices from its lower edge to upper edge.

    Parameters
    ----------
    bin_id : int
        Bin identifier
    bin_step : int
        Bin step size in basis points

    Returns
    -------
    tuple
        (lower_price, upper_price)
    """
    base = 1.0 + bin_step / 10000.0
    # Bin price is at center, bounds are half step away
    center_price = base ** bin_id
    lower_price = base ** (bin_id - 0.5)
    upper_price = base ** (bin_id + 0.5)
    return (lower_price, upper_price)


@numba.jit(nopython=True, cache=True)
def calculate_dlmm_base_fee(bin_step: int, base_factor: int) -> float:
    """
    Calculate DLMM base fee.

    base_fee = bin_step * base_factor / 10^10

    Parameters
    ----------
    bin_step : int
        Bin step size in basis points
    base_factor : int
        Base fee factor (typically 5000-15000)

    Returns
    -------
    float
        Base fee as decimal (e.g., 0.003 for 0.3%)

    Examples
    --------
    >>> calculate_dlmm_base_fee(100, 10000)  # 1% bins, factor 10000
    0.001
    """
    return bin_step * base_factor / 1e10


@numba.jit(nopython=True, cache=True)
def calculate_dlmm_variable_fee(
    volatility_accumulator: float,
    bin_step: int,
    variable_fee_control: int,
    max_volatility_accumulator: int,
) -> float:
    """
    Calculate DLMM variable fee based on volatility.

    The variable fee increases with recent price volatility to
    compensate LPs for increased impermanent loss risk.

    Parameters
    ----------
    volatility_accumulator : float
        Accumulated volatility (from recent swaps)
    bin_step : int
        Bin step size in basis points
    variable_fee_control : int
        Variable fee control parameter
    max_volatility_accumulator : int
        Maximum volatility accumulator value

    Returns
    -------
    float
        Variable fee as decimal
    """
    if max_volatility_accumulator == 0:
        return 0.0

    # Clamp volatility
    vol = min(volatility_accumulator, float(max_volatility_accumulator))

    # Variable fee formula
    variable_fee = (vol * bin_step * variable_fee_control) / 1e18

    return variable_fee


@numba.jit(nopython=True, cache=True)
def calculate_dlmm_total_fee(
    bin_step: int,
    base_factor: int,
    volatility_accumulator: float,
    variable_fee_control: int,
    max_volatility_accumulator: int,
) -> float:
    """
    Calculate total DLMM swap fee (base + variable).

    Parameters
    ----------
    bin_step : int
        Bin step size
    base_factor : int
        Base fee factor
    volatility_accumulator : float
        Current volatility accumulator
    variable_fee_control : int
        Variable fee control parameter
    max_volatility_accumulator : int
        Max volatility accumulator

    Returns
    -------
    float
        Total fee as decimal
    """
    base = calculate_dlmm_base_fee(bin_step, base_factor)
    variable = calculate_dlmm_variable_fee(
        volatility_accumulator,
        bin_step,
        variable_fee_control,
        max_volatility_accumulator,
    )
    return base + variable


@numba.jit(nopython=True, cache=True)
def update_volatility_accumulator(
    current_accumulator: float,
    bins_crossed: int,
    volatility_reference: int,
    decay_period: int,
    time_elapsed: int,
) -> float:
    """
    Update DLMM volatility accumulator after a swap.

    The accumulator increases when bins are crossed and decays over time.

    Parameters
    ----------
    current_accumulator : float
        Current volatility accumulator value
    bins_crossed : int
        Number of bins crossed in this swap
    volatility_reference : int
        Reference volatility (used for scaling)
    decay_period : int
        Time period for decay (in seconds)
    time_elapsed : int
        Time since last update (in seconds)

    Returns
    -------
    float
        Updated volatility accumulator
    """
    # Decay existing accumulator
    if decay_period > 0 and time_elapsed > 0:
        decay_factor = max(0.0, 1.0 - time_elapsed / decay_period)
        decayed = current_accumulator * decay_factor
    else:
        decayed = current_accumulator

    # Add new volatility from bins crossed
    new_volatility = abs(bins_crossed) * volatility_reference

    return decayed + new_volatility


@numba.jit(nopython=True, cache=True)
def get_dlmm_composition_at_price(
    liquidity: float,
    bin_id: int,
    active_bin_id: int,
    bin_step: int,
) -> tuple:
    """
    Calculate token composition for a bin at current price.

    Similar to V3 concentrated liquidity, the composition depends on
    whether the bin is above, below, or at the active price.

    Parameters
    ----------
    liquidity : float
        Liquidity in the bin
    bin_id : int
        The bin's ID
    active_bin_id : int
        Currently active bin (where price is)
    bin_step : int
        Bin step size

    Returns
    -------
    tuple
        (amount_x, amount_y) token amounts
    """
    price = bin_id_to_price(bin_id, bin_step)

    if bin_id < active_bin_id:
        # Bin is below active price - all token Y (quote)
        return (0.0, liquidity)
    elif bin_id > active_bin_id:
        # Bin is above active price - all token X (base)
        return (liquidity / price, 0.0)
    else:
        # Active bin - mixed composition (50/50 at bin center)
        active_price = bin_id_to_price(active_bin_id, bin_step)
        amount_x = (liquidity / 2.0) / active_price
        amount_y = liquidity / 2.0
        return (amount_x, amount_y)


@numba.jit(nopython=True, cache=True)
def calculate_liquidity_distribution_spot(
    total_liquidity: float,
    center_bin: int,
    radius: int,
) -> tuple:
    """
    Calculate uniform liquidity distribution (Spot shape).

    Distributes liquidity evenly across bins.

    Parameters
    ----------
    total_liquidity : float
        Total liquidity to distribute
    center_bin : int
        Center bin ID
    radius : int
        Number of bins on each side

    Returns
    -------
    tuple
        (bin_ids array, liquidity_per_bin array)
    """
    n_bins = 2 * radius + 1
    liquidity_per_bin = total_liquidity / n_bins

    # Return as parallel arrays for Numba compatibility
    bin_ids = np.empty(n_bins, dtype=np.int32)
    liquidities = np.empty(n_bins, dtype=np.float64)

    for i in range(n_bins):
        bin_ids[i] = center_bin - radius + i
        liquidities[i] = liquidity_per_bin

    return (bin_ids, liquidities)


@numba.jit(nopython=True, cache=True)
def calculate_liquidity_distribution_curve(
    total_liquidity: float,
    center_bin: int,
    radius: int,
) -> tuple:
    """
    Calculate concentrated liquidity distribution (Curve shape).

    More liquidity near center, less at edges (bell curve).

    Parameters
    ----------
    total_liquidity : float
        Total liquidity to distribute
    center_bin : int
        Center bin ID
    radius : int
        Number of bins on each side

    Returns
    -------
    tuple
        (bin_ids array, liquidity_per_bin array)
    """
    n_bins = 2 * radius + 1

    # Calculate weights (Gaussian-like distribution)
    weights = np.empty(n_bins, dtype=np.float64)
    sigma = radius / 2.0 if radius > 0 else 1.0

    total_weight = 0.0
    for i in range(n_bins):
        distance = i - radius  # Distance from center
        weight = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))
        weights[i] = weight
        total_weight += weight

    # Normalize and allocate liquidity
    bin_ids = np.empty(n_bins, dtype=np.int32)
    liquidities = np.empty(n_bins, dtype=np.float64)

    for i in range(n_bins):
        bin_ids[i] = center_bin - radius + i
        liquidities[i] = total_liquidity * (weights[i] / total_weight)

    return (bin_ids, liquidities)


@numba.jit(nopython=True, cache=True)
def calculate_liquidity_distribution_bid_ask(
    total_liquidity: float,
    center_bin: int,
    radius: int,
    is_bid: bool,
) -> tuple:
    """
    Calculate bid-ask liquidity distribution.

    Inverse curve - more liquidity at edges for DCA strategies.

    Parameters
    ----------
    total_liquidity : float
        Total liquidity to distribute
    center_bin : int
        Center bin ID
    radius : int
        Number of bins on each side
    is_bid : bool
        True for bid side (below center), False for ask side (above)

    Returns
    -------
    tuple
        (bin_ids array, liquidity_per_bin array)
    """
    n_bins = radius + 1  # One-sided distribution

    # Inverse weights (more at edges)
    weights = np.empty(n_bins, dtype=np.float64)
    total_weight = 0.0

    for i in range(n_bins):
        # Linear increasing weight away from center
        weight = float(i + 1)
        weights[i] = weight
        total_weight += weight

    # Normalize and allocate
    bin_ids = np.empty(n_bins, dtype=np.int32)
    liquidities = np.empty(n_bins, dtype=np.float64)

    for i in range(n_bins):
        if is_bid:
            bin_ids[i] = center_bin - i  # Below center
        else:
            bin_ids[i] = center_bin + i  # Above center
        liquidities[i] = total_liquidity * (weights[i] / total_weight)

    return (bin_ids, liquidities)
