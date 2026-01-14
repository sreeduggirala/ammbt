"""
Uniswap V2 (CPAMM) simulator.

Constant Product Market Maker: x * y = k
Full-range liquidity, fixed 0.3% fee.
"""

import numpy as np
import pandas as pd
import numba
from typing import Dict, Any, Tuple

from ammbt.amms.base import BaseAMMSimulator
from ammbt.utils.math import get_amount_out
from ammbt.utils.config import Config


# Define structured array dtype for V2 positions
V2_POSITION_DTYPE = np.dtype([
    ('liquidity', 'f8'),           # LP tokens owned
    ('token0_balance', 'f8'),      # Current token0 amount
    ('token1_balance', 'f8'),      # Current token1 amount
    ('uncollected_fees_0', 'f8'),  # Fees in token0
    ('uncollected_fees_1', 'f8'),  # Fees in token1
    ('gas_spent', 'f8'),           # Cumulative gas costs
    ('is_active', 'bool'),         # Position status
    ('last_rebalance_idx', 'i4'),  # Last rebalance swap index
    ('num_rebalances', 'i4'),      # Count of rebalances
])


@numba.jit(nopython=True, cache=True)
def _simulate_v2_swaps_nb(
    swap_amounts_0: np.ndarray,
    swap_amounts_1: np.ndarray,
    positions: np.ndarray,
    initial_reserve0: float,
    initial_reserve1: float,
    fee_rate: float,
    rebalance_threshold: np.ndarray,
    rebalance_frequency: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Numba-compiled simulation loop for Uniswap V2.

    Sequential over time (swaps), vectorized over strategies.

    Parameters
    ----------
    swap_amounts_0 : np.ndarray
        Token0 amount for each swap (negative = out)
    swap_amounts_1 : np.ndarray
        Token1 amount for each swap
    positions : np.ndarray
        Position state array (n_swaps, n_strategies)
    initial_reserve0 : float
        Starting reserve of token0
    initial_reserve1 : float
        Starting reserve of token1
    fee_rate : float
        Pool fee rate (0.003 for v2)
    rebalance_threshold : np.ndarray
        Price deviation threshold for rebalancing (n_strategies,)
    rebalance_frequency : np.ndarray
        Minimum swaps between rebalances (n_strategies,)

    Returns
    -------
    tuple
        (positions, reserve0_history, reserve1_history)
    """
    n_swaps = len(swap_amounts_0)
    n_strategies = positions.shape[1]

    # Track pool reserves over time
    reserve0_history = np.zeros(n_swaps)
    reserve1_history = np.zeros(n_swaps)

    # Initialize pool state
    reserve0 = initial_reserve0
    reserve1 = initial_reserve1
    total_supply = 0.0

    # Calculate total liquidity from all positions
    for j in range(n_strategies):
        if positions[0, j]['is_active']:
            total_supply += positions[0, j]['liquidity']

    # Initial price
    initial_price = reserve1 / reserve0 if reserve0 > 0 else 1.0

    # Main simulation loop (sequential over time)
    for i in range(n_swaps):
        # 1. Process swap and update pool reserves
        amount0_in = max(0.0, swap_amounts_0[i])
        amount1_in = max(0.0, swap_amounts_1[i])

        if amount0_in > 0:
            # Swapping token0 for token1
            amount1_out = get_amount_out(amount0_in, reserve0, reserve1, fee_rate)
            reserve0 += amount0_in
            reserve1 -= amount1_out
        elif amount1_in > 0:
            # Swapping token1 for token0
            amount0_out = get_amount_out(amount1_in, reserve1, reserve0, fee_rate)
            reserve1 += amount1_in
            reserve0 -= amount0_out

        # Track reserves
        reserve0_history[i] = reserve0
        reserve1_history[i] = reserve1

        # Current price
        current_price = reserve1 / reserve0 if reserve0 > 0 else 1.0

        # 2. Update all positions (vectorized across strategies)
        for j in range(n_strategies):
            if not positions[i, j]['is_active']:
                # Copy forward inactive position
                if i > 0:
                    positions[i, j] = positions[i-1, j]
                continue

            # Get position's liquidity share
            liquidity_share = positions[i, j]['liquidity'] / total_supply if total_supply > 0 else 0.0

            # Update token balances based on current pool ratio
            # For full-range LP: amount0 = liquidity * reserve0 / total_supply
            positions[i, j]['token0_balance'] = liquidity_share * reserve0
            positions[i, j]['token1_balance'] = liquidity_share * reserve1

            # Accumulate fees (proportional to liquidity share)
            # Fee is already taken from swap, it stays in pool
            # LP's value increases as reserves grow relative to k
            if amount0_in > 0:
                fee_amount = amount0_in * fee_rate
                positions[i, j]['uncollected_fees_0'] += fee_amount * liquidity_share
            elif amount1_in > 0:
                fee_amount = amount1_in * fee_rate
                positions[i, j]['uncollected_fees_1'] += fee_amount * liquidity_share

            # 3. Check rebalancing conditions
            price_deviation = abs(current_price - initial_price) / initial_price
            swaps_since_rebalance = i - positions[i, j]['last_rebalance_idx']

            should_rebalance = False

            # Rebalance if price threshold exceeded
            if rebalance_threshold[j] > 0 and price_deviation >= rebalance_threshold[j]:
                if swaps_since_rebalance >= rebalance_frequency[j]:
                    should_rebalance = True

            # Rebalance if frequency reached (time-based)
            if rebalance_frequency[j] > 0 and swaps_since_rebalance >= rebalance_frequency[j]:
                if rebalance_threshold[j] <= 0:  # Pure time-based rebalancing
                    should_rebalance = True

            if should_rebalance:
                # Execute rebalance
                # In V2, rebalancing means removing and re-adding liquidity
                # This incurs gas costs
                # For simplicity: assume fixed gas cost per rebalance
                gas_cost_usd = 50.0  # TODO: Make this configurable
                positions[i, j]['gas_spent'] += gas_cost_usd

                # Update rebalance tracking
                positions[i, j]['last_rebalance_idx'] = i
                positions[i, j]['num_rebalances'] += 1

                # Collect fees on rebalance
                # (In practice, fees are auto-compounded in v2)
                # positions[i, j]['uncollected_fees_0'] = 0.0
                # positions[i, j]['uncollected_fees_1'] = 0.0

            # Copy forward for next iteration
            if i < n_swaps - 1:
                positions[i+1, j] = positions[i, j]

    return positions, reserve0_history, reserve1_history


class UniswapV2Simulator(BaseAMMSimulator):
    """
    Uniswap V2 CPAMM simulator.

    Simulates full-range liquidity provision with constant product formula.
    """

    def __init__(
        self,
        initial_reserve0: float = 1_000_000.0,
        initial_reserve1: float = 1_000_000.0,
        fee_rate: float = None,
    ):
        """
        Initialize Uniswap V2 simulator.

        Parameters
        ----------
        initial_reserve0 : float
            Starting reserve of token0
        initial_reserve1 : float
            Starting reserve of token1
        fee_rate : float, optional
            Pool fee rate (default: 0.003)
        """
        if fee_rate is None:
            fee_rate = Config.get("defaults.v2_fee", 0.003)

        pool_params = {
            "initial_reserve0": initial_reserve0,
            "initial_reserve1": initial_reserve1,
            "fee_rate": fee_rate,
        }
        super().__init__(pool_params)

    def initialize_positions(
        self,
        n_strategies: int,
        n_swaps: int,
        strategy_params: np.ndarray,
    ) -> np.ndarray:
        """
        Create initial position array for V2.

        Parameters
        ----------
        n_strategies : int
            Number of strategy variants
        n_swaps : int
            Number of time steps
        strategy_params : np.ndarray
            Strategy parameters with fields:
            - initial_capital: float
            - rebalance_threshold: float
            - rebalance_frequency: int

        Returns
        -------
        np.ndarray
            Initialized position array (n_swaps, n_strategies)
        """
        positions = np.zeros((n_swaps, n_strategies), dtype=V2_POSITION_DTYPE)

        # Initialize each strategy's starting position
        initial_price = (
            self.pool_params["initial_reserve1"] / self.pool_params["initial_reserve0"]
        )

        for j in range(n_strategies):
            capital = strategy_params[j]['initial_capital']

            # Split capital 50/50 for V2 (full range at current price)
            amount0 = capital / 2.0 / initial_price  # token0
            amount1 = capital / 2.0  # token1 (assuming token1 is USD)

            # Calculate liquidity tokens
            # L = sqrt(amount0 * amount1)
            liquidity = np.sqrt(amount0 * amount1)

            # Set initial position
            positions[0, j]['liquidity'] = liquidity
            positions[0, j]['token0_balance'] = amount0
            positions[0, j]['token1_balance'] = amount1
            positions[0, j]['is_active'] = True
            positions[0, j]['last_rebalance_idx'] = 0

        return positions

    def simulate(
        self,
        swaps: pd.DataFrame,
        positions: np.ndarray,
        strategy_params: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run V2 simulation.

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
        positions, reserve0_hist, reserve1_hist = _simulate_v2_swaps_nb(
            amount0,
            amount1,
            positions,
            self.pool_params['initial_reserve0'],
            self.pool_params['initial_reserve1'],
            self.pool_params['fee_rate'],
            rebalance_threshold,
            rebalance_frequency,
        )

        metadata = {
            'reserve0_history': reserve0_hist,
            'reserve1_history': reserve1_hist,
        }

        return positions, metadata
