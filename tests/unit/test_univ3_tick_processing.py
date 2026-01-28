"""
Unit tests for Uniswap V3 tick-by-tick swap processing.

Tests the new tick map, swap step, and tick crossing functions.
"""

import numpy as np
import pytest

from ammbt.amms.univ3 import (
    _get_sqrt_ratio_at_tick,
    _compute_swap_step,
    _find_next_initialized_tick,
    _get_tick_index,
    _compute_active_liquidity,
    _cross_tick,
    _execute_swap,
    V3_POSITION_DTYPE,
)
from ammbt.utils.math import (
    tick_to_price,
    price_to_tick,
    Q96,
)


Q96_FLOAT = 79228162514264337593543950336.0


class TestComputeSwapStep:
    """Tests for _compute_swap_step function."""

    def test_swap_step_zero_liquidity(self):
        """Test that zero liquidity returns no swap."""
        sqrt_price = 1.0 * Q96_FLOAT
        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, 0.0, 0.9 * Q96_FLOAT, 100.0, 0.003, True
        )
        assert new_sqrt_price == sqrt_price
        assert amount_in == 0.0
        assert amount_out == 0.0
        assert fee == 0.0

    def test_swap_step_zero_amount(self):
        """Test that zero amount returns no swap."""
        sqrt_price = 1.0 * Q96_FLOAT
        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, 1000.0, 0.9 * Q96_FLOAT, 0.0, 0.003, True
        )
        assert new_sqrt_price == sqrt_price
        assert amount_in == 0.0
        assert amount_out == 0.0
        assert fee == 0.0

    def test_swap_step_token0_for_token1(self):
        """Test swap step for token0 -> token1 (price decreases)."""
        sqrt_price = 1.0 * Q96_FLOAT
        sqrt_price_target = 0.9 * Q96_FLOAT
        liquidity = 10000.0
        amount = 100.0
        fee_rate = 0.003

        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, liquidity, sqrt_price_target, amount, fee_rate, True
        )

        # Price should decrease (token0 -> token1)
        assert new_sqrt_price < sqrt_price
        assert amount_in > 0
        assert amount_out > 0
        assert fee > 0
        assert fee == pytest.approx(amount_in * fee_rate, rel=1e-6)

    def test_swap_step_token1_for_token0(self):
        """Test swap step for token1 -> token0 (price increases)."""
        sqrt_price = 1.0 * Q96_FLOAT
        sqrt_price_target = 1.1 * Q96_FLOAT
        liquidity = 10000.0
        amount = 100.0
        fee_rate = 0.003

        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, liquidity, sqrt_price_target, amount, fee_rate, False
        )

        # Price should increase (token1 -> token0)
        assert new_sqrt_price > sqrt_price
        assert amount_in > 0
        assert amount_out > 0
        assert fee > 0

    def test_swap_step_reaches_target(self):
        """Test swap step that reaches the target price."""
        sqrt_price = 1.0 * Q96_FLOAT
        sqrt_price_target = 0.99 * Q96_FLOAT
        liquidity = 10000.0
        # Large amount that should reach target
        amount = 10000.0
        fee_rate = 0.003

        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, liquidity, sqrt_price_target, amount, fee_rate, True
        )

        # Should have reached target (within tolerance)
        assert new_sqrt_price <= sqrt_price_target * 1.00001
        # Should not have consumed all input
        assert amount_in < amount

    def test_swap_step_target_wrong_direction(self):
        """Test that target in wrong direction returns no swap."""
        sqrt_price = 1.0 * Q96_FLOAT
        # Target above current for zero_for_one=True (invalid)
        sqrt_price_target = 1.1 * Q96_FLOAT

        new_sqrt_price, amount_in, amount_out, fee = _compute_swap_step(
            sqrt_price, 1000.0, sqrt_price_target, 100.0, 0.003, True
        )

        assert new_sqrt_price == sqrt_price
        assert amount_in == 0.0


class TestFindNextInitializedTick:
    """Tests for _find_next_initialized_tick function."""

    def test_find_next_tick_zero_for_one(self):
        """Test finding next tick when price is decreasing."""
        tick_indices = np.array([-100, 0, 100, 200], dtype=np.int32)
        num_ticks = 4

        # Current tick at 150, should find 100
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, 150, True)
        assert tick == 100
        assert idx == 2

        # Current tick at 100, should find 0
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, 100, True)
        assert tick == 0
        assert idx == 1

    def test_find_next_tick_one_for_zero(self):
        """Test finding next tick when price is increasing."""
        tick_indices = np.array([-100, 0, 100, 200], dtype=np.int32)
        num_ticks = 4

        # Current tick at 50, should find 100
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, 50, False)
        assert tick == 100
        assert idx == 2

        # Current tick at 100, should find 200
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, 100, False)
        assert tick == 200
        assert idx == 3

    def test_find_next_tick_no_ticks(self):
        """Test finding next tick with empty tick array."""
        tick_indices = np.array([], dtype=np.int32)

        tick, idx = _find_next_initialized_tick(tick_indices, 0, 0, True)
        assert tick == -887272  # MIN_TICK
        assert idx == -1

        tick, idx = _find_next_initialized_tick(tick_indices, 0, 0, False)
        assert tick == 887272  # MAX_TICK
        assert idx == -1

    def test_find_next_tick_beyond_range(self):
        """Test finding next tick when no tick exists in direction."""
        tick_indices = np.array([0, 100], dtype=np.int32)
        num_ticks = 2

        # Looking left from tick -100, nothing there
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, -100, True)
        assert tick == -887272
        assert idx == -1

        # Looking right from tick 200, nothing there
        tick, idx = _find_next_initialized_tick(tick_indices, num_ticks, 200, False)
        assert tick == 887272
        assert idx == -1


class TestGetTickIndex:
    """Tests for _get_tick_index function."""

    def test_get_tick_index_found(self):
        """Test finding an existing tick."""
        tick_indices = np.array([-100, 0, 100, 200], dtype=np.int32)
        assert _get_tick_index(0, tick_indices, 4) == 1
        assert _get_tick_index(-100, tick_indices, 4) == 0
        assert _get_tick_index(200, tick_indices, 4) == 3

    def test_get_tick_index_not_found(self):
        """Test finding a non-existent tick."""
        tick_indices = np.array([-100, 0, 100, 200], dtype=np.int32)
        assert _get_tick_index(50, tick_indices, 4) == -1
        assert _get_tick_index(-50, tick_indices, 4) == -1
        assert _get_tick_index(300, tick_indices, 4) == -1


class TestComputeActiveLiquidity:
    """Tests for _compute_active_liquidity function."""

    def test_active_liquidity_simple(self):
        """Test computing active liquidity."""
        tick_indices = np.array([-100, 0, 100], dtype=np.int32)
        # Position from tick -100 to 100
        # liquidity_net = +L at -100, -L at 100
        liquidity_net = np.array([1000.0, 0.0, -1000.0], dtype=np.float64)

        # At tick 50 (between 0 and 100), should have 1000 liquidity
        liq = _compute_active_liquidity(50, tick_indices, liquidity_net, 3)
        assert liq == 1000.0

        # At tick -150 (below -100), should have 0 liquidity
        liq = _compute_active_liquidity(-150, tick_indices, liquidity_net, 3)
        assert liq == 0.0

        # At tick 150 (above 100), sum is 1000 + 0 - 1000 = 0
        liq = _compute_active_liquidity(150, tick_indices, liquidity_net, 3)
        assert liq == 0.0

    def test_active_liquidity_overlapping_positions(self):
        """Test active liquidity with overlapping positions."""
        # Two positions: [-100, 100] with L=1000 and [-50, 50] with L=500
        tick_indices = np.array([-100, -50, 50, 100], dtype=np.int32)
        liquidity_net = np.array([1000.0, 500.0, -500.0, -1000.0], dtype=np.float64)

        # At tick 0, both positions are active
        liq = _compute_active_liquidity(0, tick_indices, liquidity_net, 4)
        assert liq == 1500.0

        # At tick 75, only first position is active
        liq = _compute_active_liquidity(75, tick_indices, liquidity_net, 4)
        assert liq == 1000.0


class TestCrossTick:
    """Tests for _cross_tick function."""

    def test_cross_tick_zero_for_one(self):
        """Test crossing tick when price decreases."""
        liquidity_net = np.array([1000.0], dtype=np.float64)
        fee_growth_outside_0 = np.array([0.1], dtype=np.float64)
        fee_growth_outside_1 = np.array([0.2], dtype=np.float64)

        new_liq, fg0, fg1 = _cross_tick(
            0,  # tick_idx
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            0.5,  # fee_growth_global_0
            0.6,  # fee_growth_global_1
            True,  # zero_for_one (price decreasing)
            2000.0,  # current_liquidity
        )

        # Liquidity should decrease by liquidity_net
        assert new_liq == 1000.0  # 2000 - 1000
        # Fee growth outside should be flipped
        assert fee_growth_outside_0[0] == pytest.approx(0.4, rel=1e-6)  # 0.5 - 0.1
        assert fee_growth_outside_1[0] == pytest.approx(0.4, rel=1e-6)  # 0.6 - 0.2

    def test_cross_tick_one_for_zero(self):
        """Test crossing tick when price increases."""
        liquidity_net = np.array([1000.0], dtype=np.float64)
        fee_growth_outside_0 = np.array([0.1], dtype=np.float64)
        fee_growth_outside_1 = np.array([0.2], dtype=np.float64)

        new_liq, _, _ = _cross_tick(
            0,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            0.5,
            0.6,
            False,  # one_for_zero (price increasing)
            2000.0,
        )

        # Liquidity should increase by liquidity_net
        assert new_liq == 3000.0  # 2000 + 1000


class TestExecuteSwap:
    """Tests for _execute_swap function (full swap execution)."""

    def test_execute_swap_no_tick_crossing(self):
        """Test a swap that doesn't cross any ticks."""
        tick_indices = np.array([-1000, 1000], dtype=np.int32)
        liquidity_net = np.array([10000.0, -10000.0], dtype=np.float64)
        fee_growth_outside_0 = np.zeros(2, dtype=np.float64)
        fee_growth_outside_1 = np.zeros(2, dtype=np.float64)

        sqrt_price_x96 = 1.0 * Q96_FLOAT  # Price = 1.0
        liquidity = 10000.0
        current_tick = 0

        # Small swap that won't cross ticks
        result = _execute_swap(
            10.0,  # amount
            True,  # zero_for_one
            sqrt_price_x96,
            liquidity,
            current_tick,
            tick_indices,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            2,  # num_initialized_ticks
            0.003,  # fee_rate
            0.0,  # fee_growth_global_0
            0.0,  # fee_growth_global_1
        )

        new_sqrt_price, amount0_delta, amount1_delta, new_tick, fg0, fg1, new_liq = result

        # Price should decrease
        assert new_sqrt_price < sqrt_price_x96
        # Amount0 delta should be positive (input)
        assert amount0_delta > 0
        # Amount1 delta should be negative (output)
        assert amount1_delta < 0
        # Fee growth should increase
        assert fg0 > 0

    def test_execute_swap_with_tick_crossing(self):
        """Test a swap that crosses an initialized tick."""
        # Position from tick -100 to tick 100
        tick_indices = np.array([-100, 100], dtype=np.int32)
        liquidity_net = np.array([10000.0, -10000.0], dtype=np.float64)
        fee_growth_outside_0 = np.zeros(2, dtype=np.float64)
        fee_growth_outside_1 = np.zeros(2, dtype=np.float64)

        sqrt_price_x96 = 1.0 * Q96_FLOAT  # Price = 1.0, tick = 0
        liquidity = 10000.0
        current_tick = 0

        # Large swap that should cross the lower tick boundary
        result = _execute_swap(
            50000.0,  # large amount
            True,  # zero_for_one (price decreasing)
            sqrt_price_x96,
            liquidity,
            current_tick,
            tick_indices,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            2,  # num_initialized_ticks
            0.003,  # fee_rate
            0.0,  # fee_growth_global_0
            0.0,  # fee_growth_global_1
        )

        new_sqrt_price, _, _, new_tick, _, _, new_liq = result

        # Price should be significantly lower
        new_price = (new_sqrt_price / Q96_FLOAT) ** 2
        assert new_price < 0.999  # Price decreased noticeably

    def test_execute_swap_token1_for_token0(self):
        """Test swap in opposite direction (token1 -> token0)."""
        tick_indices = np.array([-100, 100], dtype=np.int32)
        liquidity_net = np.array([10000.0, -10000.0], dtype=np.float64)
        fee_growth_outside_0 = np.zeros(2, dtype=np.float64)
        fee_growth_outside_1 = np.zeros(2, dtype=np.float64)

        sqrt_price_x96 = 1.0 * Q96_FLOAT
        liquidity = 10000.0

        result = _execute_swap(
            100.0,
            False,  # one_for_zero (price increasing)
            sqrt_price_x96,
            liquidity,
            0,  # current_tick
            tick_indices,
            liquidity_net,
            fee_growth_outside_0,
            fee_growth_outside_1,
            2,
            0.003,
            0.0,
            0.0,
        )

        new_sqrt_price, amount0_delta, amount1_delta, _, _, fg1, _ = result

        # Price should increase
        assert new_sqrt_price > sqrt_price_x96
        # Amount1 delta should be positive (input)
        assert amount1_delta > 0
        # Amount0 delta should be negative (output)
        assert amount0_delta < 0
        # Fee growth for token1 should increase
        assert fg1 > 0


class TestIntegration:
    """Integration tests for the full V3 simulator."""

    def test_simple_simulation(self):
        """Test a simple simulation with the V3 simulator."""
        from ammbt.amms.univ3 import UniswapV3Simulator
        import pandas as pd

        # Create simulator
        sim = UniswapV3Simulator(
            initial_price=1.0,
            initial_liquidity=1000.0,  # Base liquidity
            fee_tier=3000,
        )

        # Create strategy params
        strategy_dtype = np.dtype([
            ('initial_capital', 'f8'),
            ('tick_lower', 'i4'),
            ('tick_upper', 'i4'),
            ('rebalance_threshold', 'f8'),
            ('rebalance_frequency', 'i4'),
        ])
        strategy_params = np.array([
            (10000.0, -1000, 1000, 0.0, 0),
        ], dtype=strategy_dtype)

        # Create positions
        n_swaps = 10
        positions = sim.initialize_positions(1, n_swaps, strategy_params)

        # Create swaps
        swaps = pd.DataFrame({
            'amount0': [100.0] * 5 + [0.0] * 5,
            'amount1': [0.0] * 5 + [100.0] * 5,
        })

        # Run simulation
        positions, metadata = sim.simulate(swaps, positions, strategy_params)

        # Verify results
        assert 'price_history' in metadata
        assert len(metadata['price_history']) == n_swaps

        # Price should change due to swaps
        price_hist = metadata['price_history']
        # First half: token0 swaps (price decreases)
        assert price_hist[4] < price_hist[0]
        # Second half: token1 swaps (price increases)
        assert price_hist[9] > price_hist[4]

    def test_tick_spacing_rebalance(self):
        """Test that rebalancing respects tick spacing."""
        from ammbt.amms.univ3 import UniswapV3Simulator
        from ammbt.utils.math import get_tick_spacing
        import pandas as pd

        # Use 0.3% fee tier (tick spacing = 60)
        fee_tier = 3000
        tick_spacing = get_tick_spacing(fee_tier)
        assert tick_spacing == 60

        sim = UniswapV3Simulator(
            initial_price=1.0,
            initial_liquidity=1000.0,
            fee_tier=fee_tier,
        )

        strategy_dtype = np.dtype([
            ('initial_capital', 'f8'),
            ('tick_lower', 'i4'),
            ('tick_upper', 'i4'),
            ('rebalance_threshold', 'f8'),
            ('rebalance_frequency', 'i4'),
        ])
        # Use tick-spacing-aligned bounds
        strategy_params = np.array([
            (10000.0, -600, 600, 0.01, 1),  # Low threshold, rebalance every swap
        ], dtype=strategy_dtype)

        n_swaps = 20
        positions = sim.initialize_positions(1, n_swaps, strategy_params)

        # Create swaps that will trigger rebalancing
        swaps = pd.DataFrame({
            'amount0': [5000.0] * n_swaps,
            'amount1': [0.0] * n_swaps,
        })

        positions, _ = sim.simulate(swaps, positions, strategy_params)

        # Check that final tick bounds are multiples of tick spacing
        final_tick_lower = positions[-1, 0]['tick_lower']
        final_tick_upper = positions[-1, 0]['tick_upper']

        assert final_tick_lower % tick_spacing == 0, \
            f"tick_lower {final_tick_lower} not a multiple of {tick_spacing}"
        assert final_tick_upper % tick_spacing == 0, \
            f"tick_upper {final_tick_upper} not a multiple of {tick_spacing}"
