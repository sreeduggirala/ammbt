"""Unit tests for analytics module."""

import numpy as np
import pandas as pd
import pytest

from ammbt.portfolio.analytics import (
    calculate_hold_value,
    calculate_position_value,
    calculate_impermanent_loss,
    calculate_metrics,
    calculate_capital_efficiency,
)
from ammbt.amms.univ2 import V2_POSITION_DTYPE
from ammbt.amms.univ3 import V3_POSITION_DTYPE


class TestCalculateHoldValue:
    def test_basic(self):
        assert calculate_hold_value(100.0, 50.0, 2.0) == 250.0

    def test_zero_amounts(self):
        assert calculate_hold_value(0.0, 0.0, 1.0) == 0.0

    def test_price_change(self):
        # 10 token0 at initial_price=1, final_price=4 + 10 token1 = 50
        assert calculate_hold_value(10.0, 10.0, 4.0) == 50.0


class TestCalculatePositionValue:
    def test_no_fees(self):
        val = calculate_position_value(10.0, 20.0, 0.0, 0.0, 2.0)
        assert val == 40.0  # 10*2 + 20

    def test_with_fees(self):
        val = calculate_position_value(10.0, 20.0, 5.0, 3.0, 2.0)
        assert val == 53.0  # 10*2 + 20 + 5*2 + 3


class TestCalculateImpermanentLoss:
    def test_no_loss(self):
        assert calculate_impermanent_loss(100.0, 100.0) == 0.0

    def test_loss(self):
        il = calculate_impermanent_loss(90.0, 100.0)
        assert il == pytest.approx(-0.10)

    def test_gain(self):
        il = calculate_impermanent_loss(110.0, 100.0)
        assert il == pytest.approx(0.10)

    def test_zero_hold(self):
        assert calculate_impermanent_loss(100.0, 0.0) == 0.0


class TestCalculateMetricsV2:
    """Test that V2 metrics do NOT double-count fees."""

    def _make_v2_positions(self, n_swaps, n_strategies):
        """Build V2 position arrays with known values."""
        positions = np.zeros((n_swaps, n_strategies), dtype=V2_POSITION_DTYPE)
        for j in range(n_strategies):
            for i in range(n_swaps):
                positions[i, j]['token0_balance'] = 10.0
                positions[i, j]['token1_balance'] = 20.0
                positions[i, j]['uncollected_fees_0'] = 5.0
                positions[i, j]['uncollected_fees_1'] = 3.0
                positions[i, j]['is_active'] = True
                positions[i, j]['liquidity'] = 100.0
        return positions

    def test_v2_final_value_excludes_fee_fields(self):
        """V2 final_value = token0*price + token1 (NOT adding uncollected fees)."""
        positions = self._make_v2_positions(10, 1)
        prices = np.ones(10) * 2.0
        initial_capital = np.array([40.0])  # 10*2 + 20

        metrics = calculate_metrics(positions, prices, initial_capital, amm_type='v2')
        # V2: final_value = 10*2 + 20 = 40 (NOT 40 + 5*2 + 3 = 53)
        assert metrics['final_value'].iloc[0] == pytest.approx(40.0)

    def test_v3_final_value_includes_fee_fields(self):
        """V3 final_value = token0*price + token1 + fees."""
        positions = np.zeros((10, 1), dtype=V3_POSITION_DTYPE)
        for i in range(10):
            positions[i, 0]['token0_balance'] = 10.0
            positions[i, 0]['token1_balance'] = 20.0
            positions[i, 0]['uncollected_fees_0'] = 5.0
            positions[i, 0]['uncollected_fees_1'] = 3.0
            positions[i, 0]['is_active'] = True
            positions[i, 0]['liquidity'] = 100.0
            positions[i, 0]['tick_lower'] = -1000
            positions[i, 0]['tick_upper'] = 1000

        prices = np.ones(10) * 2.0
        initial_capital = np.array([53.0])

        metrics = calculate_metrics(positions, prices, initial_capital, amm_type='v3')
        # V3: final_value = 10*2 + 20 + 5*2 + 3 = 53
        assert metrics['final_value'].iloc[0] == pytest.approx(53.0)

    def test_il_uses_final_value_directly(self):
        """IL should use final_value directly, not subtract fees first."""
        positions = self._make_v2_positions(10, 1)
        prices = np.ones(10) * 2.0
        initial_capital = np.array([40.0])

        metrics = calculate_metrics(positions, prices, initial_capital, amm_type='v2')
        # hold_value = 10*2 + 20 = 40 (initial balances at final price)
        # final_value for V2 = 40
        # IL = (40 - 40) / 40 = 0.0
        assert metrics['il'].iloc[0] == pytest.approx(0.0)


class TestCalculateMetricsEdgeCases:
    def test_sharpe_with_zero_std(self):
        """Sharpe should be 0 when returns have zero std."""
        positions = np.zeros((10, 1), dtype=V2_POSITION_DTYPE)
        for i in range(10):
            positions[i, 0]['token0_balance'] = 10.0
            positions[i, 0]['token1_balance'] = 10.0
            positions[i, 0]['is_active'] = True
        prices = np.ones(10)
        initial_capital = np.array([20.0])

        metrics = calculate_metrics(positions, prices, initial_capital, amm_type='v2')
        assert metrics['sharpe'].iloc[0] == 0.0

    def test_total_return(self):
        """Total return = (final - initial) / initial."""
        positions = np.zeros((5, 1), dtype=V2_POSITION_DTYPE)
        for i in range(5):
            positions[i, 0]['token0_balance'] = 10.0
            positions[i, 0]['token1_balance'] = 10.0
            positions[i, 0]['is_active'] = True
        # Final price = 2 -> final_value = 10*2 + 10 = 30
        prices = np.linspace(1.0, 2.0, 5)
        initial_capital = np.array([20.0])

        metrics = calculate_metrics(positions, prices, initial_capital, amm_type='v2')
        assert metrics['total_return'].iloc[0] == pytest.approx(0.5)  # (30-20)/20


class TestCalculateCapitalEfficiency:
    def test_v2_always_in_range(self):
        positions = np.zeros((5, 1), dtype=V2_POSITION_DTYPE)
        for i in range(5):
            positions[i, 0]['token0_balance'] = 10.0
            positions[i, 0]['token1_balance'] = 10.0
            positions[i, 0]['is_active'] = True
        prices = np.ones(5)
        eff = calculate_capital_efficiency(positions, prices, amm_type='v2')
        assert eff['pct_time_in_range'].iloc[0] == 100.0

    def test_utilization_stable_price(self):
        """With stable prices and balances, utilization should be ~1.0."""
        positions = np.zeros((5, 1), dtype=V2_POSITION_DTYPE)
        for i in range(5):
            positions[i, 0]['token0_balance'] = 10.0
            positions[i, 0]['token1_balance'] = 10.0
            positions[i, 0]['is_active'] = True
        prices = np.ones(5)
        eff = calculate_capital_efficiency(positions, prices, amm_type='v2')
        assert eff['avg_utilization'].iloc[0] == pytest.approx(1.0)
