"""
Meteora DLMM (Dynamic Liquidity Market Maker) simulator.

Bin-based concentrated liquidity with dynamic fees.
Implements the Meteora DLMM mechanics for Solana.

Reference: https://docs.meteora.ag/overview/products/dlmm/what-is-dlmm
"""

import numpy as np
import pandas as pd
import numba
from typing import Dict, Any, Tuple

from ammbt.amms.base import BaseAMMSimulator
from ammbt.utils.math import (
    bin_id_to_price,
    price_to_bin_id,
    calculate_dlmm_total_fee,
    update_volatility_accumulator,
    get_dlmm_composition_at_price,
    calculate_liquidity_distribution_spot,
    calculate_liquidity_distribution_curve,
    calculate_liquidity_distribution_bid_ask,
)


# Liquidity shape enum values
SHAPE_SPOT = 0
SHAPE_CURVE = 1
SHAPE_BID_ASK = 2


# DLMM Position dtype
DLMM_POSITION_DTYPE = np.dtype([
    # Position parameters
    ('bin_lower', 'i4'),            # Lower bin bound
    ('bin_upper', 'i4'),            # Upper bin bound
    ('liquidity', 'f8'),            # Total position liquidity
    ('liquidity_shape', 'i4'),      # 0=Spot, 1=Curve, 2=Bid-Ask

    # Current balances
    ('token_x_balance', 'f8'),      # Current token X (base) amount
    ('token_y_balance', 'f8'),      # Current token Y (quote) amount

    # Fee tracking
    ('uncollected_fees_x', 'f8'),   # Uncollected fees token X
    ('uncollected_fees_y', 'f8'),   # Uncollected fees token Y
    ('fee_growth_global_x', 'f8'),  # Global fee growth X
    ('fee_growth_global_y', 'f8'),  # Global fee growth Y

    # Position state
    ('is_active', 'bool'),          # Position active
    ('is_in_range', 'bool'),        # Active bin within position range
    ('gas_spent', 'f8'),            # Cumulative gas costs
    ('last_rebalance_idx', 'i4'),   # Last rebalance swap index
    ('num_rebalances', 'i4'),       # Rebalance count

    # For V2-compatible analytics
    ('token0_balance', 'f8'),       # Alias for token_x_balance
    ('token1_balance', 'f8'),       # Alias for token_y_balance
    ('uncollected_fees_0', 'f8'),   # Alias for uncollected_fees_x
    ('uncollected_fees_1', 'f8'),   # Alias for uncollected_fees_y
])


@numba.jit(nopython=True, cache=True)
def _get_bin_liquidity(
    bin_id: int,
    bin_lower: int,
    bin_upper: int,
    total_liquidity: float,
    shape: int,
) -> float:
    """
    Get liquidity for a specific bin based on distribution shape.

    Parameters
    ----------
    bin_id : int
        Bin to query
    bin_lower : int
        Lower bin bound
    bin_upper : int
        Upper bin bound
    total_liquidity : float
        Total position liquidity
    shape : int
        Liquidity shape (0=Spot, 1=Curve, 2=Bid-Ask)

    Returns
    -------
    float
        Liquidity in this bin
    """
    if bin_id < bin_lower or bin_id > bin_upper:
        return 0.0

    n_bins = bin_upper - bin_lower + 1
    center_bin = (bin_lower + bin_upper) // 2
    radius = (bin_upper - bin_lower) // 2

    if shape == SHAPE_SPOT:
        # Uniform distribution
        return total_liquidity / n_bins

    elif shape == SHAPE_CURVE:
        # Gaussian-like distribution
        distance = bin_id - center_bin
        sigma = max(radius / 2.0, 1.0)
        weight = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))

        # Calculate normalization factor
        total_weight = 0.0
        for i in range(n_bins):
            d = (bin_lower + i) - center_bin
            total_weight += np.exp(-(d ** 2) / (2.0 * sigma ** 2))

        return total_liquidity * (weight / total_weight)

    elif shape == SHAPE_BID_ASK:
        # Inverse curve (more at edges)
        distance_from_center = abs(bin_id - center_bin)
        weight = distance_from_center + 1.0

        # Normalization
        total_weight = 0.0
        for i in range(n_bins):
            d = abs((bin_lower + i) - center_bin)
            total_weight += d + 1.0

        return total_liquidity * (weight / total_weight)

    return total_liquidity / n_bins  # Default to uniform


@numba.jit(nopython=True, cache=True)
def _calculate_position_value(
    bin_lower: int,
    bin_upper: int,
    active_bin: int,
    total_liquidity: float,
    shape: int,
    bin_step: int,
) -> Tuple[float, float]:
    """
    Calculate total token amounts for a position.

    Sums composition across all bins in the position.

    Parameters
    ----------
    bin_lower : int
        Lower bin bound
    bin_upper : int
        Upper bin bound
    active_bin : int
        Currently active bin
    total_liquidity : float
        Total position liquidity
    shape : int
        Liquidity distribution shape
    bin_step : int
        Bin step size

    Returns
    -------
    tuple
        (total_x, total_y)
    """
    total_x = 0.0
    total_y = 0.0

    for bin_id in range(bin_lower, bin_upper + 1):
        bin_liquidity = _get_bin_liquidity(
            bin_id, bin_lower, bin_upper, total_liquidity, shape
        )

        if bin_liquidity > 0:
            amount_x, amount_y = get_dlmm_composition_at_price(
                bin_liquidity, bin_id, active_bin, bin_step
            )
            total_x += amount_x
            total_y += amount_y

    return (total_x, total_y)


@numba.jit(nopython=True, cache=True)
def _simulate_dlmm_swaps_nb(
    swap_amounts_x: np.ndarray,
    swap_amounts_y: np.ndarray,
    timestamps: np.ndarray,
    positions: np.ndarray,
    initial_price: float,
    bin_step: int,
    base_factor: int,
    variable_fee_control: int,
    max_volatility_accumulator: int,
    volatility_reference: int,
    decay_period: int,
    rebalance_threshold: np.ndarray,
    rebalance_frequency: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Numba-compiled simulation loop for DLMM.

    Parameters
    ----------
    swap_amounts_x : np.ndarray
        Token X amount for each swap
    swap_amounts_y : np.ndarray
        Token Y amount for each swap
    timestamps : np.ndarray
        Timestamps for each swap
    positions : np.ndarray
        Position state array (n_swaps, n_strategies)
    initial_price : float
        Starting price
    bin_step : int
        Bin step in basis points
    base_factor : int
        Base fee factor
    variable_fee_control : int
        Variable fee control parameter
    max_volatility_accumulator : int
        Max volatility accumulator
    volatility_reference : int
        Volatility reference for accumulator updates
    decay_period : int
        Decay period in seconds
    rebalance_threshold : np.ndarray
        Price deviation threshold per strategy
    rebalance_frequency : np.ndarray
        Min swaps between rebalances per strategy

    Returns
    -------
    tuple
        (positions, price_history, active_bin_history, fee_history)
    """
    n_swaps = len(swap_amounts_x)
    n_strategies = positions.shape[1]

    # Track pool state
    price_history = np.zeros(n_swaps, dtype=np.float64)
    active_bin_history = np.zeros(n_swaps, dtype=np.int32)
    fee_history = np.zeros(n_swaps, dtype=np.float64)

    # Initialize pool state
    current_price = initial_price
    active_bin = price_to_bin_id(initial_price, bin_step)
    volatility_accumulator = 0.0
    last_timestamp = timestamps[0] if len(timestamps) > 0 else 0

    # Global fee growth accumulators
    fee_growth_global_x = 0.0
    fee_growth_global_y = 0.0

    # Simplified: assume uniform pool liquidity
    pool_liquidity = 1_000_000.0

    # Main simulation loop
    for i in range(n_swaps):
        current_timestamp = timestamps[i]
        time_elapsed = int(current_timestamp - last_timestamp)

        # 1. Process swap
        amount_x_in = max(0.0, swap_amounts_x[i])
        amount_y_in = max(0.0, swap_amounts_y[i])

        # Calculate current fee
        current_fee = calculate_dlmm_total_fee(
            bin_step,
            base_factor,
            volatility_accumulator,
            variable_fee_control,
            max_volatility_accumulator,
        )
        fee_history[i] = current_fee

        old_bin = active_bin

        if amount_x_in > 0 and pool_liquidity > 0:
            # Swapping X for Y (selling X)
            fee_amount = amount_x_in * current_fee
            amount_after_fee = amount_x_in - fee_amount

            # Price impact (simplified linear model)
            price_impact = amount_after_fee / pool_liquidity * 0.01
            current_price = current_price * (1 - price_impact)
            current_price = max(1e-10, current_price)

            # Update fee growth
            if pool_liquidity > 0:
                fee_growth_global_x += fee_amount / pool_liquidity

        elif amount_y_in > 0 and pool_liquidity > 0:
            # Swapping Y for X (buying X)
            fee_amount = amount_y_in * current_fee
            amount_after_fee = amount_y_in - fee_amount

            # Price impact
            price_impact = amount_after_fee / pool_liquidity * 0.01
            current_price = current_price * (1 + price_impact)

            # Update fee growth
            if pool_liquidity > 0:
                fee_growth_global_y += fee_amount / pool_liquidity

        # Update active bin
        active_bin = price_to_bin_id(current_price, bin_step)
        bins_crossed = abs(active_bin - old_bin)

        # Update volatility accumulator
        volatility_accumulator = update_volatility_accumulator(
            volatility_accumulator,
            bins_crossed,
            volatility_reference,
            decay_period,
            time_elapsed,
        )

        # Clamp volatility
        volatility_accumulator = min(
            volatility_accumulator,
            float(max_volatility_accumulator)
        )

        # Store history
        price_history[i] = current_price
        active_bin_history[i] = active_bin
        last_timestamp = current_timestamp

        # 2. Update all positions
        for j in range(n_strategies):
            if not positions[i, j]['is_active']:
                if i > 0:
                    positions[i, j] = positions[i-1, j]
                continue

            pos = positions[i, j]
            bin_lower = pos['bin_lower']
            bin_upper = pos['bin_upper']
            pos_liquidity = pos['liquidity']
            shape = pos['liquidity_shape']

            # Check if position is in range
            is_in_range = (active_bin >= bin_lower) and (active_bin <= bin_upper)
            positions[i, j]['is_in_range'] = is_in_range

            # Calculate current token amounts
            amount_x, amount_y = _calculate_position_value(
                bin_lower, bin_upper, active_bin,
                pos_liquidity, shape, bin_step
            )

            positions[i, j]['token_x_balance'] = amount_x
            positions[i, j]['token_y_balance'] = amount_y
            # Set aliases for compatibility
            positions[i, j]['token0_balance'] = amount_x
            positions[i, j]['token1_balance'] = amount_y

            # Calculate fees earned (if in range)
            if is_in_range and i > 0:
                # Simplified: proportional to liquidity in active bin
                bin_liq = _get_bin_liquidity(
                    active_bin, bin_lower, bin_upper, pos_liquidity, shape
                )
                if bin_liq > 0 and pool_liquidity > 0:
                    share = bin_liq / pool_liquidity

                    # Fee deltas
                    delta_fee_x = (fee_growth_global_x - pos['fee_growth_global_x']) * bin_liq
                    delta_fee_y = (fee_growth_global_y - pos['fee_growth_global_y']) * bin_liq

                    positions[i, j]['uncollected_fees_x'] += max(0.0, delta_fee_x)
                    positions[i, j]['uncollected_fees_y'] += max(0.0, delta_fee_y)
                    positions[i, j]['uncollected_fees_0'] = positions[i, j]['uncollected_fees_x']
                    positions[i, j]['uncollected_fees_1'] = positions[i, j]['uncollected_fees_y']

            # Update fee growth checkpoints
            positions[i, j]['fee_growth_global_x'] = fee_growth_global_x
            positions[i, j]['fee_growth_global_y'] = fee_growth_global_y

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
                # Execute rebalance (re-center around current price)
                gas_cost_usd = 0.5  # Solana gas is much cheaper
                positions[i, j]['gas_spent'] += gas_cost_usd
                positions[i, j]['last_rebalance_idx'] = i
                positions[i, j]['num_rebalances'] += 1

                # Re-center range
                bin_width = bin_upper - bin_lower
                new_bin_lower = active_bin - bin_width // 2
                new_bin_upper = active_bin + bin_width // 2

                positions[i, j]['bin_lower'] = new_bin_lower
                positions[i, j]['bin_upper'] = new_bin_upper
                positions[i, j]['is_in_range'] = True

                # Reset fee checkpoints
                positions[i, j]['fee_growth_global_x'] = fee_growth_global_x
                positions[i, j]['fee_growth_global_y'] = fee_growth_global_y

            # Copy forward for next iteration
            if i < n_swaps - 1:
                positions[i+1, j] = positions[i, j]

    return positions, price_history, active_bin_history, fee_history


class MeteoraLMMSimulator(BaseAMMSimulator):
    """
    Meteora DLMM simulator.

    Simulates bin-based concentrated liquidity with dynamic fees.

    Parameters
    ----------
    initial_price : float
        Starting price (Y/X)
    bin_step : int
        Bin step in basis points (e.g., 10, 25, 100)
    base_factor : int
        Base fee factor (typical: 10000)
    variable_fee_control : int
        Variable fee control (typical: 40000)
    max_volatility_accumulator : int
        Max volatility accumulator (typical: 350000)
    volatility_reference : int
        Volatility reference (typical: 10000)
    decay_period : int
        Volatility decay period in seconds (typical: 600)

    Examples
    --------
    >>> simulator = MeteoraLMMSimulator(
    ...     initial_price=1.0,
    ...     bin_step=25,  # 0.25% bins
    ... )
    """

    def __init__(
        self,
        initial_price: float = 1.0,
        bin_step: int = 25,
        base_factor: int = 10000,
        variable_fee_control: int = 40000,
        max_volatility_accumulator: int = 350000,
        volatility_reference: int = 10000,
        decay_period: int = 600,
    ):
        pool_params = {
            "initial_price": initial_price,
            "bin_step": bin_step,
            "base_factor": base_factor,
            "variable_fee_control": variable_fee_control,
            "max_volatility_accumulator": max_volatility_accumulator,
            "volatility_reference": volatility_reference,
            "decay_period": decay_period,
        }
        super().__init__(pool_params)

    def initialize_positions(
        self,
        n_strategies: int,
        n_swaps: int,
        strategy_params: np.ndarray,
    ) -> np.ndarray:
        """
        Create initial position array for DLMM.

        Parameters
        ----------
        n_strategies : int
            Number of strategy variants
        n_swaps : int
            Number of time steps
        strategy_params : np.ndarray
            Strategy parameters with fields:
            - initial_capital: float
            - bin_lower: int
            - bin_upper: int
            - liquidity_shape: int (0=Spot, 1=Curve, 2=Bid-Ask)
            - rebalance_threshold: float
            - rebalance_frequency: int

        Returns
        -------
        np.ndarray
            Initialized position array (n_swaps, n_strategies)
        """
        positions = np.zeros((n_swaps, n_strategies), dtype=DLMM_POSITION_DTYPE)

        initial_price = self.pool_params["initial_price"]
        bin_step = self.pool_params["bin_step"]
        active_bin = price_to_bin_id(initial_price, bin_step)

        for j in range(n_strategies):
            capital = strategy_params[j]['initial_capital']
            bin_lower = strategy_params[j]['bin_lower']
            bin_upper = strategy_params[j]['bin_upper']
            shape = strategy_params[j]['liquidity_shape']

            # Calculate initial token amounts
            # Liquidity = capital (in Y terms)
            liquidity = capital

            amount_x, amount_y = _calculate_position_value(
                bin_lower, bin_upper, active_bin,
                liquidity, shape, bin_step
            )

            # Set initial position
            positions[0, j]['bin_lower'] = bin_lower
            positions[0, j]['bin_upper'] = bin_upper
            positions[0, j]['liquidity'] = liquidity
            positions[0, j]['liquidity_shape'] = shape
            positions[0, j]['token_x_balance'] = amount_x
            positions[0, j]['token_y_balance'] = amount_y
            positions[0, j]['token0_balance'] = amount_x
            positions[0, j]['token1_balance'] = amount_y
            positions[0, j]['is_active'] = True
            positions[0, j]['is_in_range'] = (active_bin >= bin_lower) and (active_bin <= bin_upper)
            positions[0, j]['last_rebalance_idx'] = 0

        return positions

    def simulate(
        self,
        swaps: pd.DataFrame,
        positions: np.ndarray,
        strategy_params: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run DLMM simulation.

        Parameters
        ----------
        swaps : pd.DataFrame
            Swap events with columns: amount0 (X), amount1 (Y)
            Optional: timestamp (Unix seconds)
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
        amount_x = swaps['amount0'].values.astype(np.float64)
        amount_y = swaps['amount1'].values.astype(np.float64)

        # Get timestamps (default to sequential if not provided)
        if 'timestamp' in swaps.columns:
            timestamps = swaps['timestamp'].values.astype(np.float64)
        else:
            timestamps = np.arange(len(swaps), dtype=np.float64)

        # Extract strategy parameters
        rebalance_threshold = strategy_params['rebalance_threshold'].astype(np.float64)
        rebalance_frequency = strategy_params['rebalance_frequency'].astype(np.int32)

        # Run simulation
        positions, price_hist, bin_hist, fee_hist = _simulate_dlmm_swaps_nb(
            amount_x,
            amount_y,
            timestamps,
            positions,
            self.pool_params['initial_price'],
            self.pool_params['bin_step'],
            self.pool_params['base_factor'],
            self.pool_params['variable_fee_control'],
            self.pool_params['max_volatility_accumulator'],
            self.pool_params['volatility_reference'],
            self.pool_params['decay_period'],
            rebalance_threshold,
            rebalance_frequency,
        )

        metadata = {
            'price_history': price_hist,
            'active_bin_history': bin_hist,
            'fee_history': fee_hist,
            'bin_step': self.pool_params['bin_step'],
        }

        return positions, metadata


# Alias for consistency
DLMMSimulator = MeteoraLMMSimulator
