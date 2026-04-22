"""
Loss-Versus-Rebalancing (LVR) calculation.

LVR measures the cost of providing liquidity to an AMM due to stale pricing
relative to an external reference market. It is the standard academic metric
for LP costs (Milionis et al., 2022).

Two methods:
- Exact: requires external reference prices (CEX). Computes arbitrage profits.
- Approximate: uses realized volatility. LVR ~ sigma^2 * L * dt / 2.
"""

import numpy as np
import numba
from typing import Dict, Optional


@numba.jit(nopython=True, cache=True)
def _calculate_exact_lvr(
    pool_prices: np.ndarray,
    reference_prices: np.ndarray,
    fee_rate: float,
    liquidity: np.ndarray,
) -> tuple:
    """
    Calculate exact LVR given external reference (CEX) prices.

    LVR at each step = max(0, |p_ref - p_pool| - fee) * arb_quantity.
    Simplified: arb profit when reference price diverges from pool price
    beyond the fee threshold.

    Parameters
    ----------
    pool_prices : np.ndarray
        Pool prices from the simulation.
    reference_prices : np.ndarray
        External reference prices (e.g., CEX mid-price).
    fee_rate : float
        Pool fee rate (e.g., 0.003 for 0.3%).
    liquidity : np.ndarray
        Active liquidity at each step.

    Returns
    -------
    tuple
        (cumulative_lvr, total_lvr)
    """
    n = len(pool_prices)
    cumulative_lvr = np.zeros(n)
    total_lvr = 0.0

    for i in range(1, n):
        if pool_prices[i] <= 0 or reference_prices[i] <= 0:
            cumulative_lvr[i] = total_lvr
            continue

        price_diff = abs(reference_prices[i] - pool_prices[i])
        fee_threshold = fee_rate * pool_prices[i]

        if price_diff > fee_threshold and liquidity[i] > 0:
            # Arbitrageur profit: trade at pool price, hedge at reference
            effective_diff = price_diff - fee_threshold
            arb_quantity = effective_diff * liquidity[i] / pool_prices[i]
            lvr_step = effective_diff * arb_quantity
            total_lvr += lvr_step

        cumulative_lvr[i] = total_lvr

    return cumulative_lvr, total_lvr


@numba.jit(nopython=True, cache=True)
def _calculate_approximate_lvr(
    prices: np.ndarray,
    fee_rate: float,
    liquidity: np.ndarray,
    window: int,
) -> tuple:
    """
    Approximate LVR using realized volatility.

    Uses the continuous-time approximation: LVR ~ sigma^2 * L * dt / 2.

    Parameters
    ----------
    prices : np.ndarray
        Pool prices.
    fee_rate : float
        Pool fee rate.
    liquidity : np.ndarray
        Active liquidity at each step.
    window : int
        Rolling window for volatility estimation.

    Returns
    -------
    tuple
        (cumulative_lvr, total_lvr)
    """
    n = len(prices)
    cumulative_lvr = np.zeros(n)
    total_lvr = 0.0

    for i in range(1, n):
        # Rolling realized variance
        start = max(0, i - window)
        count = i - start
        if count < 2 or liquidity[i] <= 0:
            cumulative_lvr[i] = total_lvr
            continue

        # Compute variance of log returns in window
        sum_r = 0.0
        sum_r2 = 0.0
        valid = 0
        for k in range(start + 1, i + 1):
            if prices[k] > 0 and prices[k - 1] > 0:
                log_ret = np.log(prices[k] / prices[k - 1])
                sum_r += log_ret
                sum_r2 += log_ret * log_ret
                valid += 1

        if valid < 2:
            cumulative_lvr[i] = total_lvr
            continue

        mean_r = sum_r / valid
        variance = sum_r2 / valid - mean_r * mean_r

        # LVR step: sigma^2 * L / 2
        lvr_step = max(0.0, variance) * liquidity[i] / 2.0
        total_lvr += lvr_step
        cumulative_lvr[i] = total_lvr

    return cumulative_lvr, total_lvr


def calculate_lvr(
    prices: np.ndarray,
    fee_rate: float,
    liquidity: np.ndarray,
    reference_prices: Optional[np.ndarray] = None,
    window: int = 100,
) -> Dict[str, object]:
    """
    Calculate Loss-Versus-Rebalancing (LVR).

    If ``reference_prices`` are provided, uses the exact method.
    Otherwise, uses the approximate method based on realized volatility.

    Parameters
    ----------
    prices : np.ndarray
        Pool prices from simulation.
    fee_rate : float
        Pool fee rate (e.g., 0.003).
    liquidity : np.ndarray
        Active liquidity at each time step. If scalar-like, broadcasts.
    reference_prices : np.ndarray, optional
        External reference prices (e.g., CEX). If provided, uses exact method.
    window : int
        Rolling window for volatility estimation (approximate method only).

    Returns
    -------
    dict
        - ``cumulative_lvr``: np.ndarray, cumulative LVR over time.
        - ``total_lvr``: float, total LVR.
        - ``method``: str, ``'exact'`` or ``'approximate'``.

    Examples
    --------
    >>> lvr = calculate_lvr(prices, fee_rate=0.003, liquidity=np.full(len(prices), 1e6))
    >>> print(f"Total LVR: {lvr['total_lvr']:.2f}")
    """
    prices = np.asarray(prices, dtype=np.float64)
    liquidity = np.asarray(liquidity, dtype=np.float64)

    # Broadcast scalar liquidity
    if liquidity.ndim == 0 or len(liquidity) == 1:
        liquidity = np.full(len(prices), float(liquidity.flat[0]))

    if reference_prices is not None:
        reference_prices = np.asarray(reference_prices, dtype=np.float64)
        cumulative, total = _calculate_exact_lvr(
            prices, reference_prices, fee_rate, liquidity
        )
        method = 'exact'
    else:
        cumulative, total = _calculate_approximate_lvr(
            prices, fee_rate, liquidity, window
        )
        method = 'approximate'

    return {
        'cumulative_lvr': cumulative,
        'total_lvr': total,
        'method': method,
    }
