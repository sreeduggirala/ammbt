"""
Portfolio analytics and performance metrics.

Calculate IL, fees, PnL, Sharpe, capital efficiency, etc.
All operations are vectorized across strategy dimension.
"""

import numpy as np
import pandas as pd
import numba
from typing import Tuple


@numba.jit(nopython=True, cache=True)
def calculate_hold_value(
    initial_amount0: float,
    initial_amount1: float,
    final_price: float,
) -> float:
    """
    Calculate value of holding (not providing liquidity).

    Parameters
    ----------
    initial_amount0 : float
        Initial token0 amount
    initial_amount1 : float
        Initial token1 amount
    final_price : float
        Final price (token1/token0)

    Returns
    -------
    float
        Hold value in token1 terms
    """
    return initial_amount0 * final_price + initial_amount1


@numba.jit(nopython=True, cache=True)
def calculate_position_value(
    token0_balance: float,
    token1_balance: float,
    uncollected_fees_0: float,
    uncollected_fees_1: float,
    price: float,
) -> float:
    """
    Calculate total position value.

    Parameters
    ----------
    token0_balance : float
        Token0 amount
    token1_balance : float
        Token1 amount
    uncollected_fees_0 : float
        Uncollected fees in token0
    uncollected_fees_1 : float
        Uncollected fees in token1
    price : float
        Current price (token1/token0)

    Returns
    -------
    float
        Total value in token1 terms
    """
    return (
        token0_balance * price +
        token1_balance +
        uncollected_fees_0 * price +
        uncollected_fees_1
    )


@numba.jit(nopython=True, cache=True)
def calculate_impermanent_loss(
    lp_value: float,
    hold_value: float,
) -> float:
    """
    Calculate impermanent loss as percentage.

    IL = (LP value - Hold value) / Hold value

    Parameters
    ----------
    lp_value : float
        Value from LP position
    hold_value : float
        Value from holding

    Returns
    -------
    float
        Impermanent loss (negative = loss, positive = gain)
    """
    if hold_value == 0:
        return 0.0
    return (lp_value - hold_value) / hold_value


def calculate_metrics(
    positions: np.ndarray,
    prices: np.ndarray,
    initial_capital: np.ndarray,
    amm_type: str = 'v3',
) -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics (vectorized).

    Parameters
    ----------
    positions : np.ndarray
        Position state array (n_swaps, n_strategies)
    prices : np.ndarray
        Price series (n_swaps,)
    initial_capital : np.ndarray
        Initial capital for each strategy (n_strategies,)

    Returns
    -------
    pd.DataFrame
        Metrics with one row per strategy:
        - final_value: Total value at end
        - total_fees: Fees collected
        - il: Impermanent loss
        - net_pnl: Net profit/loss after gas
        - gross_pnl: Profit/loss before gas
        - gas_costs: Total gas spent
        - num_rebalances: Number of rebalances
        - sharpe: Sharpe ratio
        - sortino: Sortino ratio
        - max_drawdown: Maximum drawdown
        - total_return: Total return percentage
    """
    n_swaps, n_strategies = positions.shape
    final_price = prices[-1]
    initial_price = prices[0]

    metrics = {}

    # Calculate for each strategy (vectorized where possible)
    final_values = np.zeros(n_strategies)
    total_fees = np.zeros(n_strategies)
    il = np.zeros(n_strategies)
    hold_values = np.zeros(n_strategies)
    gross_pnl = np.zeros(n_strategies)
    net_pnl = np.zeros(n_strategies)

    for j in range(n_strategies):
        # Final position state
        final_pos = positions[-1, j]

        # Calculate final value (AMM-type-aware)
        if amm_type == 'v2':
            # V2: fees auto-compound into reserves, so token balances already
            # include fees. uncollected_fees_0/1 are cumulative trackers only.
            final_value = (
                final_pos['token0_balance'] * final_price +
                final_pos['token1_balance']
            )
        else:
            # V3/DLMM: uncollected fees are real uncollected value
            final_value = calculate_position_value(
                final_pos['token0_balance'],
                final_pos['token1_balance'],
                final_pos['uncollected_fees_0'],
                final_pos['uncollected_fees_1'],
                final_price,
            )
        final_values[j] = final_value

        # Total fees (informational — cumulative tracker for all AMM types)
        total_fees[j] = (
            final_pos['uncollected_fees_0'] * final_price +
            final_pos['uncollected_fees_1']
        )

        # Hold value (what if we just held the initial tokens)
        initial_pos = positions[0, j]
        hold_value = calculate_hold_value(
            initial_pos['token0_balance'],
            initial_pos['token1_balance'],
            final_price,
        )
        hold_values[j] = hold_value

        # Impermanent loss: compare LP value vs hold value directly
        il[j] = calculate_impermanent_loss(final_value, hold_value)

        # PnL
        gross_pnl[j] = final_value - initial_capital[j]
        net_pnl[j] = gross_pnl[j] - final_pos['gas_spent']

    # Calculate time-series metrics
    returns_series = np.zeros((n_swaps - 1, n_strategies))
    values_series = np.zeros((n_swaps, n_strategies))

    for j in range(n_strategies):
        for i in range(n_swaps):
            if amm_type == 'v2':
                # V2: fees already in token balances
                values_series[i, j] = (
                    positions[i, j]['token0_balance'] * prices[i] +
                    positions[i, j]['token1_balance']
                )
            else:
                values_series[i, j] = calculate_position_value(
                    positions[i, j]['token0_balance'],
                    positions[i, j]['token1_balance'],
                    positions[i, j]['uncollected_fees_0'],
                    positions[i, j]['uncollected_fees_1'],
                    prices[i],
                )

        # Calculate returns
        for i in range(1, n_swaps):
            if values_series[i-1, j] > 0:
                returns_series[i-1, j] = (
                    values_series[i, j] - values_series[i-1, j]
                ) / values_series[i-1, j]

    # Sharpe ratio (vectorized)
    returns_mean = returns_series.mean(axis=0)
    returns_std = returns_series.std(axis=0)
    sharpe = np.where(
        returns_std > 0,
        returns_mean / returns_std * np.sqrt(252),  # Annualized
        0.0
    )

    # Sortino ratio (downside deviation)
    downside_returns = np.where(returns_series < 0, returns_series, 0)
    downside_std = np.sqrt((downside_returns ** 2).mean(axis=0))
    sortino = np.where(
        downside_std > 0,
        returns_mean / downside_std * np.sqrt(252),
        0.0
    )

    # Maximum drawdown (vectorized)
    cummax = np.maximum.accumulate(values_series, axis=0)
    drawdowns = (values_series - cummax) / cummax
    max_drawdown = drawdowns.min(axis=0)

    # Total return
    total_return = (final_values - initial_capital) / initial_capital

    # VaR (Value at Risk) at 95% and 99% confidence
    if n_swaps > 2:
        var_95 = np.percentile(returns_series, 5, axis=0)
        var_99 = np.percentile(returns_series, 1, axis=0)
    else:
        var_95 = np.zeros(n_strategies)
        var_99 = np.zeros(n_strategies)

    # CVaR (Conditional VaR / Expected Shortfall)
    cvar_95 = np.zeros(n_strategies)
    cvar_99 = np.zeros(n_strategies)
    for j in range(n_strategies):
        mask_95 = returns_series[:, j] <= var_95[j]
        if mask_95.any():
            cvar_95[j] = returns_series[mask_95, j].mean()
        else:
            cvar_95[j] = var_95[j]
        mask_99 = returns_series[:, j] <= var_99[j]
        if mask_99.any():
            cvar_99[j] = returns_series[mask_99, j].mean()
        else:
            cvar_99[j] = var_99[j]

    # Calmar ratio (return / max drawdown)
    calmar = np.where(max_drawdown != 0, total_return / np.abs(max_drawdown), 0.0)

    # Omega ratio (sum of gains / sum of losses)
    omega = np.zeros(n_strategies)
    for j in range(n_strategies):
        gains = returns_series[returns_series[:, j] > 0, j].sum()
        losses = abs(returns_series[returns_series[:, j] < 0, j].sum())
        omega[j] = gains / losses if losses > 0 else 0.0

    # Win rate
    n_returns = max(n_swaps - 1, 1)
    win_rate = (returns_series > 0).sum(axis=0) / n_returns

    # Profit factor (gross gains / gross losses)
    profit_factor = np.zeros(n_strategies)
    for j in range(n_strategies):
        gains = returns_series[returns_series[:, j] > 0, j].sum()
        losses = abs(returns_series[returns_series[:, j] < 0, j].sum())
        profit_factor[j] = gains / losses if losses > 0 else 0.0

    # Recovery time (swaps from max drawdown to recovery)
    recovery_time = np.zeros(n_strategies, dtype=np.int32)
    for j in range(n_strategies):
        dd_idx = np.argmin(drawdowns[:, j])
        peak_before = cummax[dd_idx, j]
        recovered = False
        for k in range(dd_idx, n_swaps):
            if values_series[k, j] >= peak_before:
                recovery_time[j] = k - dd_idx
                recovered = True
                break
        if not recovered:
            recovery_time[j] = n_swaps - dd_idx

    # HODL comparison
    hodl_return = np.where(
        initial_capital > 0,
        (hold_values - initial_capital) / initial_capital,
        0.0,
    )
    lp_vs_hodl = total_return - hodl_return

    # Assemble metrics DataFrame
    metrics_df = pd.DataFrame({
        'final_value': final_values,
        'total_fees': total_fees,
        'il': il,
        'il_pct': il * 100,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'gas_costs': positions[-1, :]['gas_spent'],
        'num_rebalances': positions[-1, :]['num_rebalances'],
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'hold_value': hold_values,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'calmar': calmar,
        'omega': omega,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'recovery_time': recovery_time,
        'hodl_return': hodl_return,
        'lp_vs_hodl': lp_vs_hodl,
    })

    return metrics_df


def calculate_capital_efficiency(
    positions: np.ndarray,
    prices: np.ndarray,
    amm_type: str = 'v3',
) -> pd.DataFrame:
    """
    Calculate capital efficiency metrics.

    For V3/DLMM: percentage of time in range, etc.
    For V2: always 100% (full range)

    Parameters
    ----------
    positions : np.ndarray
        Position state array
    prices : np.ndarray
        Price series

    Returns
    -------
    pd.DataFrame
        Capital efficiency metrics per strategy
    """
    n_swaps, n_strategies = positions.shape

    # For V2, always active
    pct_in_range = np.full(n_strategies, 1.0)

    # Average utilization (value / initial value over time)
    utilization = np.zeros(n_strategies)

    for j in range(n_strategies):
        initial_value = (
            positions[0, j]['token0_balance'] * prices[0] +
            positions[0, j]['token1_balance']
        )

        if initial_value > 0:
            avg_value = 0
            for i in range(n_swaps):
                if amm_type == 'v2':
                    value = (
                        positions[i, j]['token0_balance'] * prices[i] +
                        positions[i, j]['token1_balance']
                    )
                else:
                    value = calculate_position_value(
                        positions[i, j]['token0_balance'],
                        positions[i, j]['token1_balance'],
                        positions[i, j]['uncollected_fees_0'],
                        positions[i, j]['uncollected_fees_1'],
                        prices[i],
                    )
                avg_value += value / n_swaps

            utilization[j] = avg_value / initial_value

    return pd.DataFrame({
        'pct_time_in_range': pct_in_range * 100,
        'avg_utilization': utilization,
    })
