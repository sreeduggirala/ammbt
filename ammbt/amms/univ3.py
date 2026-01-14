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
    Q96,
)
from ammbt.utils.config import Config


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Numba-compiled simulation loop for Uniswap V3.

    Parameters
    ----------
    swap_amounts_0 : np.ndarray
        Token0 amount for each swap
    swap_amounts_1 : np.ndarray
        Token1 amount for each swap
    positions : np.ndarray
        Position state array (n_swaps, n_strategies)
    initial_sqrt_price_x96 : int
        Starting sqrt price in Q96 format
    initial_liquidity : float
        Initial pool liquidity
    fee_tier : int
        Fee in basis points (500, 3000, 10000)
    rebalance_threshold : np.ndarray
        Price deviation threshold for rebalancing
    rebalance_frequency : np.ndarray
        Minimum swaps between rebalances

    Returns
    -------
    tuple
        (positions, sqrt_price_history, liquidity_history)
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
    liquidity = initial_liquidity
    current_tick = price_to_tick((sqrt_price_x96 / Q96_FLOAT) ** 2)

    # Fee growth accumulators (global)
    fee_growth_global_0 = 0.0
    fee_growth_global_1 = 0.0

    # Fee growth outside (simplified - assume no other LPs)
    # In real implementation, would track per tick
    fee_growth_outside_0 = np.zeros(2**20)  # Large tick range
    fee_growth_outside_1 = np.zeros(2**20)

    # Fee rate as fraction
    fee_rate = fee_tier / 1_000_000.0

    # Calculate initial price
    initial_price = (sqrt_price_x96 / Q96_FLOAT) ** 2

    # Main simulation loop
    for i in range(n_swaps):
        # Get current price
        current_price = (sqrt_price_x96 / Q96_FLOAT) ** 2

        # 1. Process swap
        amount0_in = max(0.0, swap_amounts_0[i])
        amount1_in = max(0.0, swap_amounts_1[i])

        if amount0_in > 0 and liquidity > 0:
            # Swapping token0 for token1
            # Calculate price impact
            fee_amount = amount0_in * fee_rate
            amount0_after_fee = amount0_in - fee_amount

            # Simplified swap math (exact would require tick crossing)
            # ΔsqrtP = Δy / L (for token1 out)
            # For small swaps, approximate
            sqrt_price_delta = amount0_after_fee / liquidity
            new_sqrt_price = sqrt_price_x96 + sqrt_price_delta * Q96_FLOAT

            # Update fee growth
            if liquidity > 0:
                fee_growth_global_0 += fee_amount / liquidity

        elif amount1_in > 0 and liquidity > 0:
            # Swapping token1 for token0
            fee_amount = amount1_in * fee_rate
            amount1_after_fee = amount1_in - fee_amount

            sqrt_price_delta = -amount1_after_fee / liquidity
            new_sqrt_price = sqrt_price_x96 + sqrt_price_delta * Q96_FLOAT

            if liquidity > 0:
                fee_growth_global_1 += fee_amount / liquidity
        else:
            new_sqrt_price = sqrt_price_x96

        # Update pool state
        sqrt_price_x96 = max(1.0, new_sqrt_price)  # Prevent zero
        new_price = (sqrt_price_x96 / Q96_FLOAT) ** 2
        current_tick = price_to_tick(new_price)

        # Store pool history
        sqrt_price_history[i] = sqrt_price_x96
        liquidity_history[i] = liquidity
        tick_history[i] = current_tick

        # 2. Update all positions (vectorized across strategies)
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

            # Calculate fees earned (only if in range previously)
            if is_in_range or i == 0:
                # Simplified: assume fee growth outside is zero
                fee_growth_inside_0, fee_growth_inside_1 = _calculate_fee_growth_inside(
                    tick_lower,
                    tick_upper,
                    current_tick,
                    fee_growth_global_0,
                    fee_growth_global_1,
                    0.0, 0.0,  # fee_growth_outside_lower
                    0.0, 0.0,  # fee_growth_outside_upper
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
                # In v3, this means burning old position and minting new one
                gas_cost_usd = 100.0  # v3 rebalancing is more expensive
                positions[i, j]['gas_spent'] += gas_cost_usd
                positions[i, j]['last_rebalance_idx'] = i
                positions[i, j]['num_rebalances'] += 1

                # Simplified: re-center range around current price
                # Keep same width
                tick_width = tick_upper - tick_lower
                new_tick_lower = current_tick - tick_width // 2
                new_tick_upper = current_tick + tick_width // 2

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
        Run V3 simulation.

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

        # Run simulation
        positions, sqrt_price_hist, tick_hist = _simulate_v3_swaps_nb(
            amount0,
            amount1,
            positions,
            self.pool_params['initial_sqrt_price_x96'],
            self.pool_params['initial_liquidity'],
            self.pool_params['fee_tier'],
            rebalance_threshold,
            rebalance_frequency,
        )

        # Convert sqrt prices to regular prices
        Q96_FLOAT = 79228162514264337593543950336.0
        price_hist = (sqrt_price_hist / Q96_FLOAT) ** 2

        metadata = {
            'sqrt_price_history': sqrt_price_hist,
            'price_history': price_hist,
            'tick_history': tick_hist,
        }

        return positions, metadata
