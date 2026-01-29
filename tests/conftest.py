"""
Pytest fixtures for AMM backtesting tests.
"""

import numpy as np
import pandas as pd
import pytest


def _create_swap_df(n_swaps: int, initial_price: float, seed: int) -> pd.DataFrame:
    """Create a simple swap DataFrame for testing."""
    np.random.seed(seed)

    # Simple random walk for prices
    returns = np.random.normal(0, 0.01, n_swaps)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Random volumes
    volumes = np.abs(np.random.normal(1000, 200, n_swaps))

    # Random buy/sell direction
    is_buy = np.random.random(n_swaps) > 0.5

    amount0 = np.where(is_buy, volumes / prices, -volumes * prices)
    amount1 = np.where(is_buy, -volumes, volumes)

    timestamps = 1700000000 + np.arange(n_swaps) * 15

    return pd.DataFrame({
        'amount0': amount0.astype(np.float64),
        'amount1': amount1.astype(np.float64),
        'price': prices.astype(np.float64),
        'timestamp': timestamps.astype(np.int64),
        'volume': volumes.astype(np.float64),
    })


def _create_price_path(n_steps: int, initial_price: float, seed: int) -> np.ndarray:
    """Create a simple price path for testing."""
    np.random.seed(seed)
    returns = np.random.normal(0, 0.02, n_steps)
    prices = initial_price * np.exp(np.cumsum(returns))
    prices[0] = initial_price
    return prices


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def small_swap_df(seed):
    """Small swap DataFrame for unit tests."""
    return _create_swap_df(n_swaps=100, initial_price=1.0, seed=seed)


@pytest.fixture
def medium_swap_df(seed):
    """Medium swap DataFrame for integration tests."""
    return _create_swap_df(n_swaps=1000, initial_price=1.0, seed=seed)


@pytest.fixture
def large_swap_df(seed):
    """Large swap DataFrame for performance tests."""
    return _create_swap_df(n_swaps=10000, initial_price=1.0, seed=seed)


@pytest.fixture
def v2_strategy_params():
    """Strategy parameters for V2 backtests."""
    return {
        'initial_capital': [10000.0, 20000.0, 50000.0],
        'rebalance_threshold': [0.05, 0.1, 0.0],
        'rebalance_frequency': [100, 200, 0],
    }


@pytest.fixture
def v3_strategy_params():
    """Strategy parameters for V3 backtests."""
    return {
        'initial_capital': [10000.0, 20000.0],
        'tick_lower': [-1000, -500],
        'tick_upper': [1000, 500],
        'rebalance_threshold': [0.05, 0.1],
        'rebalance_frequency': [100, 200],
    }


@pytest.fixture
def dlmm_strategy_params():
    """Strategy parameters for DLMM backtests."""
    return {
        'initial_capital': [10000.0, 20000.0],
        'bin_lower': [-10, -5],
        'bin_upper': [10, 5],
        'liquidity_shape': [0, 1],  # Spot, Curve
        'rebalance_threshold': [0.05, 0.1],
        'rebalance_frequency': [100, 200],
    }


@pytest.fixture
def price_path_gbm(seed):
    """Price path for testing."""
    return _create_price_path(n_steps=1000, initial_price=1.0, seed=seed)


@pytest.fixture
def price_path_jump(seed):
    """Price path for testing."""
    return _create_price_path(n_steps=1000, initial_price=1.0, seed=seed + 1)


@pytest.fixture
def price_path_ou(seed):
    """Price path for testing."""
    return _create_price_path(n_steps=1000, initial_price=1.0, seed=seed + 2)
