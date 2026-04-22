"""Unit tests for DLMM simulator."""

import numpy as np
import pandas as pd
import pytest

from ammbt.amms.dlmm import (
    _get_bin_liquidity,
    _calculate_position_value,
    MeteoraLMMSimulator,
    DLMM_POSITION_DTYPE,
    SHAPE_SPOT,
    SHAPE_CURVE,
    SHAPE_BID_ASK,
)
from ammbt.utils.math import (
    bin_id_to_price,
    price_to_bin_id,
    calculate_dlmm_total_fee,
    calculate_dlmm_base_fee,
    calculate_dlmm_variable_fee,
    update_volatility_accumulator,
)
from ammbt import LPBacktester


class TestGetBinLiquidity:
    def test_spot_uniform(self):
        """Spot shape distributes liquidity uniformly."""
        total = 1000.0
        n_bins = 11  # bins -5 to 5
        expected_per_bin = total / n_bins
        for bid in range(-5, 6):
            liq = _get_bin_liquidity(bid, -5, 5, total, SHAPE_SPOT)
            assert liq == pytest.approx(expected_per_bin, rel=1e-6)

    def test_curve_center_has_most(self):
        """Curve shape has most liquidity at center."""
        center_liq = _get_bin_liquidity(0, -5, 5, 1000.0, SHAPE_CURVE)
        edge_liq = _get_bin_liquidity(-5, -5, 5, 1000.0, SHAPE_CURVE)
        assert center_liq > edge_liq

    def test_bid_ask_edges_have_most(self):
        """Bid-ask shape has more liquidity at edges."""
        center_liq = _get_bin_liquidity(0, -5, 5, 1000.0, SHAPE_BID_ASK)
        edge_liq = _get_bin_liquidity(5, -5, 5, 1000.0, SHAPE_BID_ASK)
        assert edge_liq > center_liq

    def test_out_of_range_returns_zero(self):
        assert _get_bin_liquidity(-10, -5, 5, 1000.0, SHAPE_SPOT) == 0.0
        assert _get_bin_liquidity(10, -5, 5, 1000.0, SHAPE_SPOT) == 0.0

    def test_total_liquidity_sums_correctly(self):
        """Total across all bins should equal total_liquidity for all shapes."""
        for shape in [SHAPE_SPOT, SHAPE_CURVE, SHAPE_BID_ASK]:
            total = 0.0
            for bid in range(-5, 6):
                total += _get_bin_liquidity(bid, -5, 5, 1000.0, shape)
            assert total == pytest.approx(1000.0, rel=1e-4)


class TestVolatilityAccumulator:
    def test_decay_over_time(self):
        """Accumulator should decay when no bins crossed."""
        initial = 100.0
        result = update_volatility_accumulator(
            initial, 0, 10000, 600, 300  # half decay period
        )
        assert result < initial
        assert result == pytest.approx(50.0, rel=1e-6)

    def test_full_decay(self):
        """Full decay period should zero out accumulator."""
        result = update_volatility_accumulator(100.0, 0, 10000, 600, 600)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_bins_crossed_increases(self):
        """Crossing bins should increase accumulator."""
        result = update_volatility_accumulator(0.0, 5, 10000, 600, 0)
        assert result == 50000.0  # 5 * 10000

    def test_decay_plus_increase(self):
        """Decay existing + add new volatility."""
        result = update_volatility_accumulator(100.0, 2, 10000, 600, 300)
        # Decay: 100 * 0.5 = 50, New: 2 * 10000 = 20000
        assert result == pytest.approx(20050.0, rel=1e-6)


class TestDynamicFee:
    def test_base_fee_only(self):
        # bin_step=100 * base_factor=10000 / 1e10 = 0.0001
        fee = calculate_dlmm_base_fee(100, 10000)
        assert fee == pytest.approx(0.0001)

    def test_total_fee_no_volatility(self):
        fee = calculate_dlmm_total_fee(100, 10000, 0.0, 40000, 350000)
        base = calculate_dlmm_base_fee(100, 10000)
        assert fee == pytest.approx(base)

    def test_total_fee_with_volatility(self):
        """Fee should increase with volatility."""
        fee_no_vol = calculate_dlmm_total_fee(100, 10000, 0.0, 40000, 350000)
        fee_with_vol = calculate_dlmm_total_fee(100, 10000, 100000.0, 40000, 350000)
        assert fee_with_vol > fee_no_vol

    def test_variable_fee_zero_max_vol(self):
        """Variable fee should be 0 when max_volatility_accumulator is 0."""
        var_fee = calculate_dlmm_variable_fee(100.0, 100, 40000, 0)
        assert var_fee == 0.0


class TestPriceImpactClamp:
    """Test that price_impact is clamped to prevent negative prices."""

    def test_large_swap_does_not_produce_negative_price(self):
        """Simulate a massive swap and verify price stays positive."""
        bt = LPBacktester(amm_type='dlmm')
        # Create swaps with extremely large amounts
        n = 10
        swaps = pd.DataFrame({
            'amount0': np.full(n, 1e12, dtype=np.float64),  # Huge sell pressure
            'amount1': np.zeros(n, dtype=np.float64),
            'price': np.ones(n, dtype=np.float64),
            'timestamp': (1700000000 + np.arange(n) * 15).astype(np.int64),
        })
        strategies = {
            'initial_capital': [10000.0],
            'bin_lower': [-10],
            'bin_upper': [10],
            'liquidity_shape': [0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        result = bt.run(swaps, strategies)
        # Price should remain positive
        assert (result.metadata['price_history'] > 0).all()


class TestDLMMBacktestWithShapes:
    """Test that all liquidity shapes produce valid results."""

    def _run_dlmm(self, shape):
        bt = LPBacktester(amm_type='dlmm')
        np.random.seed(42)
        n = 100
        swaps = pd.DataFrame({
            'amount0': np.random.normal(100, 20, n).astype(np.float64),
            'amount1': np.random.normal(-100, 20, n).astype(np.float64),
            'price': np.ones(n, dtype=np.float64),
            'timestamp': (1700000000 + np.arange(n) * 15).astype(np.int64),
        })
        strategies = {
            'initial_capital': [10000.0],
            'bin_lower': [-10],
            'bin_upper': [10],
            'liquidity_shape': [shape],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }
        return bt.run(swaps, strategies)

    def test_spot_shape(self):
        result = self._run_dlmm(SHAPE_SPOT)
        assert result is not None
        assert not np.isnan(result.metrics['net_pnl'].iloc[0])

    def test_curve_shape(self):
        result = self._run_dlmm(SHAPE_CURVE)
        assert result is not None
        assert not np.isnan(result.metrics['net_pnl'].iloc[0])

    def test_bid_ask_shape(self):
        result = self._run_dlmm(SHAPE_BID_ASK)
        assert result is not None
        assert not np.isnan(result.metrics['net_pnl'].iloc[0])


class TestBinPriceConversions:
    def test_roundtrip(self):
        """bin_id -> price -> bin_id should be identity."""
        for bid in [-100, -1, 0, 1, 100]:
            price = bin_id_to_price(bid, 25)
            recovered = price_to_bin_id(price, 25)
            assert recovered == bid

    def test_zero_price(self):
        """price_to_bin_id(0) should return min int."""
        result = price_to_bin_id(0.0, 25)
        assert result == -2**31
