"""
Pytest fixtures for AMM backtesting tests.
"""

import numpy as np
import pandas as pd
import pytest

from ammbt.data.synthetic import generate_swaps, generate_price_path


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def small_swap_df(seed):
    """Small swap DataFrame for unit tests."""
    return generate_swaps(
        n_swaps=100,
        initial_price=1.0,
        base_volume=1000.0,
        price_volatility=0.01,
        seed=seed,
    )


@pytest.fixture
def medium_swap_df(seed):
    """Medium swap DataFrame for integration tests."""
    return generate_swaps(
        n_swaps=1000,
        initial_price=1.0,
        base_volume=1000.0,
        price_volatility=0.02,
        seed=seed,
    )


@pytest.fixture
def large_swap_df(seed):
    """Large swap DataFrame for performance tests."""
    return generate_swaps(
        n_swaps=10000,
        initial_price=1.0,
        base_volume=1000.0,
        price_volatility=0.02,
        seed=seed,
    )


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
    """GBM price path for testing."""
    return generate_price_path(
        n_steps=1000,
        initial_price=1.0,
        volatility=0.02,
        model='gbm',
        seed=seed,
    )


@pytest.fixture
def price_path_jump(seed):
    """Jump-diffusion price path for testing."""
    return generate_price_path(
        n_steps=1000,
        initial_price=1.0,
        volatility=0.02,
        model='jump',
        jump_intensity=0.1,
        seed=seed,
    )


@pytest.fixture
def price_path_ou(seed):
    """Ornstein-Uhlenbeck price path for testing."""
    return generate_price_path(
        n_steps=1000,
        initial_price=1.0,
        volatility=0.02,
        model='ou',
        mean_reversion_speed=0.1,
        seed=seed,
    )
