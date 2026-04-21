"""Unit tests for rebalancing logic across all AMM types."""

import numpy as np
import pandas as pd
import pytest

from ammbt import LPBacktester
from ammbt.utils.math import get_tick_spacing


def _make_trending_swaps(n, direction='down'):
    """Create swaps that consistently move price in one direction."""
    np.random.seed(42)
    if direction == 'down':
        amount0 = np.full(n, 5000.0, dtype=np.float64)
        amount1 = np.zeros(n, dtype=np.float64)
    else:
        amount0 = np.zeros(n, dtype=np.float64)
        amount1 = np.full(n, 5000.0, dtype=np.float64)

    return pd.DataFrame({
        'amount0': amount0,
        'amount1': amount1,
        'price': np.ones(n, dtype=np.float64),
        'timestamp': (1700000000 + np.arange(n) * 15).astype(np.int64),
    })


class TestV2Rebalancing:
    def test_no_rebalance_when_disabled(self):
        """No rebalance when threshold=0 and frequency=0."""
        bt = LPBacktester(amm_type='v2')
        swaps = _make_trending_swaps(100)
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        result = bt.run(swaps, strategies)
        assert result.positions[-1, 0]['num_rebalances'] == 0
        assert result.positions[-1, 0]['gas_spent'] == 0.0

    def test_gas_accumulation(self):
        """Gas should accumulate as num_rebalances * gas_cost."""
        bt = LPBacktester(amm_type='v2')
        swaps = _make_trending_swaps(200)
        gas_cost = 75.0
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.001],  # Very sensitive
            'rebalance_frequency': [1],
            'gas_cost_usd': [gas_cost],
        }
        result = bt.run(swaps, strategies)
        n_rebalances = result.positions[-1, 0]['num_rebalances']
        expected_gas = n_rebalances * gas_cost
        assert result.positions[-1, 0]['gas_spent'] == pytest.approx(expected_gas)

    def test_custom_gas_cost(self):
        """User-specified gas_cost_usd should be used."""
        bt = LPBacktester(amm_type='v2')
        swaps = _make_trending_swaps(200)
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.001],
            'rebalance_frequency': [1],
            'gas_cost_usd': [123.45],
        }
        result = bt.run(swaps, strategies)
        n_rebalances = result.positions[-1, 0]['num_rebalances']
        if n_rebalances > 0:
            assert result.positions[-1, 0]['gas_spent'] == pytest.approx(
                n_rebalances * 123.45
            )

    def test_frequency_respected(self):
        """Cannot rebalance more often than rebalance_frequency."""
        bt = LPBacktester(amm_type='v2')
        swaps = _make_trending_swaps(50)
        # threshold=0 + frequency=10 means time-based only
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [10],
        }
        result = bt.run(swaps, strategies)
        # With pure time-based and threshold=0, no rebalance should happen
        # because the threshold check (line 150) requires threshold > 0
        # and the frequency check (line 155-157) requires threshold <= 0
        # which is True here -- so rebalances happen every 10 swaps
        n_reb = result.positions[-1, 0]['num_rebalances']
        # Should be approximately 50/10 = 5 (minus the first window)
        assert n_reb <= 5


class TestV3Rebalancing:
    def test_tick_spacing_respected(self):
        """After rebalance, ticks should be multiples of tick_spacing."""
        bt = LPBacktester(amm_type='v3', fee_tier=3000)
        tick_spacing = get_tick_spacing(3000)
        swaps = _make_trending_swaps(50)
        strategies = {
            'initial_capital': [10000.0],
            'tick_lower': [-600],
            'tick_upper': [600],
            'rebalance_threshold': [0.001],
            'rebalance_frequency': [1],
        }
        result = bt.run(swaps, strategies)

        final_lower = result.positions[-1, 0]['tick_lower']
        final_upper = result.positions[-1, 0]['tick_upper']

        if result.positions[-1, 0]['num_rebalances'] > 0:
            assert final_lower % tick_spacing == 0, \
                f"tick_lower {final_lower} not aligned to spacing {tick_spacing}"
            assert final_upper % tick_spacing == 0, \
                f"tick_upper {final_upper} not aligned to spacing {tick_spacing}"

    def test_out_of_range_triggers_rebalance(self):
        """Position going out of range should trigger rebalance."""
        bt = LPBacktester(amm_type='v3')
        # Narrow range that will go out of range quickly
        swaps = _make_trending_swaps(50)
        strategies = {
            'initial_capital': [10000.0],
            'tick_lower': [-60],
            'tick_upper': [60],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [1],  # Allow rebalance every swap
        }
        result = bt.run(swaps, strategies)
        # Should have rebalanced at least once
        assert result.positions[-1, 0]['num_rebalances'] > 0


class TestDLMMRebalancing:
    def test_bin_re_centering(self):
        """After rebalance, bins should be centered around active bin."""
        bt = LPBacktester(amm_type='dlmm')
        swaps = _make_trending_swaps(100)
        strategies = {
            'initial_capital': [10000.0],
            'bin_lower': [-5],
            'bin_upper': [5],
            'liquidity_shape': [0],
            'rebalance_threshold': [0.001],
            'rebalance_frequency': [1],
        }
        result = bt.run(swaps, strategies)
        n_reb = result.positions[-1, 0]['num_rebalances']
        if n_reb > 0:
            # Width should be preserved
            final_width = (
                result.positions[-1, 0]['bin_upper'] -
                result.positions[-1, 0]['bin_lower']
            )
            assert final_width == 10  # Original was 5 - (-5) = 10
