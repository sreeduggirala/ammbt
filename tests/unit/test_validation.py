"""Unit tests for input validation."""

import numpy as np
import pandas as pd
import pytest

from ammbt import LPBacktester
from ammbt.utils.math import (
    price_to_tick,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    Q96,
)


def _make_swaps(n=100):
    """Create minimal valid swap DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'amount0': np.random.normal(100, 20, n).astype(np.float64),
        'amount1': np.random.normal(-100, 20, n).astype(np.float64),
        'price': np.ones(n, dtype=np.float64),
    })


class TestSwapValidation:
    def test_empty_swaps_raises(self):
        bt = LPBacktester(amm_type='v2')
        swaps = pd.DataFrame({'amount0': pd.Series(dtype='float64'),
                              'amount1': pd.Series(dtype='float64')})
        strategies = {
            'initial_capital': [1000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="must not be empty"):
            bt.run(swaps, strategies)

    def test_missing_columns_raises(self):
        bt = LPBacktester(amm_type='v2')
        swaps = pd.DataFrame({'price': [1.0, 2.0]})
        strategies = {
            'initial_capital': [1000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="Missing required column"):
            bt.run(swaps, strategies)


class TestStrategyValidation:
    def test_zero_capital_raises(self):
        bt = LPBacktester(amm_type='v2')
        swaps = _make_swaps(10)
        strategies = {
            'initial_capital': [0.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            bt.run(swaps, strategies)

    def test_negative_capital_raises(self):
        bt = LPBacktester(amm_type='v2')
        swaps = _make_swaps(10)
        strategies = {
            'initial_capital': [-100.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            bt.run(swaps, strategies)

    def test_tick_lower_gte_tick_upper_raises(self):
        bt = LPBacktester(amm_type='v3')
        swaps = _make_swaps(10)
        strategies = {
            'initial_capital': [1000.0],
            'tick_lower': [100],
            'tick_upper': [100],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="tick_lower.*must be less than tick_upper"):
            bt.run(swaps, strategies)

    def test_bin_lower_gte_bin_upper_raises(self):
        bt = LPBacktester(amm_type='dlmm')
        swaps = _make_swaps(10)
        strategies = {
            'initial_capital': [1000.0],
            'bin_lower': [5],
            'bin_upper': [5],
            'liquidity_shape': [0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        with pytest.raises(ValueError, match="bin_lower.*must be less than bin_upper"):
            bt.run(swaps, strategies)


class TestDivisionByZeroGuards:
    def test_price_to_tick_zero_price(self):
        tick = price_to_tick(0.0)
        assert tick == -887272

    def test_price_to_tick_negative_price(self):
        tick = price_to_tick(-1.0)
        assert tick == -887272

    def test_get_liquidity_for_amounts_zero_width(self):
        Q96F = float(Q96)
        liq = get_liquidity_for_amounts(
            1.0 * Q96F, 1.0 * Q96F, 1.0 * Q96F, 100.0, 100.0
        )
        assert liq == 0.0

    def test_get_amounts_for_liquidity_zero_sqrt_price(self):
        a0, a1 = get_amounts_for_liquidity(0.0, 0.0, 0.0, 1000.0)
        assert a0 == 0.0
        assert a1 == 0.0

    def test_get_amounts_for_liquidity_inverted_range(self):
        Q96F = float(Q96)
        a0, a1 = get_amounts_for_liquidity(
            1.0 * Q96F, 2.0 * Q96F, 1.0 * Q96F, 1000.0
        )
        assert a0 == 0.0
        assert a1 == 0.0
