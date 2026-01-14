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
