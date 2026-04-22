"""
Event recording system for simulation events.

Records swaps, rebalances, fee collections, and range changes during
simulation. Events are stored in a pre-allocated structured numpy array
for Numba compatibility.
"""

import numpy as np
import numba
import pandas as pd


# Event type constants (matching enums.py EventType)
EVENT_SWAP = 0
EVENT_MINT = 1
EVENT_BURN = 2
EVENT_COLLECT = 3
EVENT_REBALANCE = 4

EVENT_TYPE_NAMES = {
    EVENT_SWAP: 'swap',
    EVENT_MINT: 'mint',
    EVENT_BURN: 'burn',
    EVENT_COLLECT: 'collect',
    EVENT_REBALANCE: 'rebalance',
}

# Structured dtype for event records
EVENT_DTYPE = np.dtype([
    ('swap_idx', 'i4'),         # Which swap triggered this event
    ('strategy_idx', 'i4'),     # Which strategy (-1 for pool-level events)
    ('event_type', 'i4'),       # EVENT_SWAP, EVENT_REBALANCE, etc.
    ('price', 'f8'),            # Pool price at event time
    ('token0_amount', 'f8'),    # Token0 involved
    ('token1_amount', 'f8'),    # Token1 involved
    ('gas_cost', 'f8'),         # Gas cost for this event
    ('extra_0', 'f8'),          # Extra field (e.g., new tick_lower)
    ('extra_1', 'f8'),          # Extra field (e.g., new tick_upper)
])


@numba.jit(nopython=True, cache=True)
def record_event(
    event_log: np.ndarray,
    event_count: int,
    swap_idx: int,
    strategy_idx: int,
    event_type: int,
    price: float,
    token0_amount: float,
    token1_amount: float,
    gas_cost: float = 0.0,
    extra_0: float = 0.0,
    extra_1: float = 0.0,
) -> int:
    """
    Record a single event into the pre-allocated log.

    Parameters
    ----------
    event_log : np.ndarray
        Pre-allocated event array (EVENT_DTYPE).
    event_count : int
        Current number of recorded events.
    swap_idx, strategy_idx, event_type, price, token0_amount,
    token1_amount, gas_cost, extra_0, extra_1 : various
        Event fields.

    Returns
    -------
    int
        Updated event count.
    """
    if event_count < len(event_log):
        event_log[event_count]['swap_idx'] = swap_idx
        event_log[event_count]['strategy_idx'] = strategy_idx
        event_log[event_count]['event_type'] = event_type
        event_log[event_count]['price'] = price
        event_log[event_count]['token0_amount'] = token0_amount
        event_log[event_count]['token1_amount'] = token1_amount
        event_log[event_count]['gas_cost'] = gas_cost
        event_log[event_count]['extra_0'] = extra_0
        event_log[event_count]['extra_1'] = extra_1
        return event_count + 1
    return event_count


def events_to_dataframe(event_log: np.ndarray, event_count: int) -> pd.DataFrame:
    """
    Convert the structured event array to a pandas DataFrame.

    Parameters
    ----------
    event_log : np.ndarray
        Event log array (EVENT_DTYPE).
    event_count : int
        Number of valid events.

    Returns
    -------
    pd.DataFrame
        Event log as DataFrame with human-readable event type names.
    """
    if event_count == 0:
        return pd.DataFrame(columns=[
            'swap_idx', 'strategy_idx', 'event_type', 'event_name',
            'price', 'token0_amount', 'token1_amount', 'gas_cost',
            'extra_0', 'extra_1',
        ])

    valid = event_log[:event_count]
    df = pd.DataFrame({
        'swap_idx': valid['swap_idx'],
        'strategy_idx': valid['strategy_idx'],
        'event_type': valid['event_type'],
        'price': valid['price'],
        'token0_amount': valid['token0_amount'],
        'token1_amount': valid['token1_amount'],
        'gas_cost': valid['gas_cost'],
        'extra_0': valid['extra_0'],
        'extra_1': valid['extra_1'],
    })
    df['event_name'] = df['event_type'].map(EVENT_TYPE_NAMES).fillna('unknown')
    return df


def allocate_event_log(n_swaps: int, n_strategies: int) -> np.ndarray:
    """
    Pre-allocate an event log array.

    Estimates max events as n_swaps + 2 * n_strategies * (n_swaps / 100).

    Parameters
    ----------
    n_swaps : int
        Number of swap events.
    n_strategies : int
        Number of strategies.

    Returns
    -------
    np.ndarray
        Pre-allocated event array.
    """
    # Swaps + estimated rebalances + fee collections
    estimated = n_swaps + 2 * n_strategies * max(n_swaps // 100, 10)
    return np.zeros(estimated, dtype=EVENT_DTYPE)
