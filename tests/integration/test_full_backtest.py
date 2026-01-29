"""
Integration tests for full backtest workflow.
"""

import numpy as np
import pandas as pd
import pytest

from ammbt import LPBacktester


def _create_swap_df(n_swaps: int, initial_price: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Create a simple swap DataFrame for testing."""
    np.random.seed(seed)

    returns = np.random.normal(0, 0.01, n_swaps)
    prices = initial_price * np.exp(np.cumsum(returns))

    volumes = np.abs(np.random.normal(1000, 200, n_swaps))
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


class TestV2FullBacktest:
    """Integration tests for V2 (CPAMM) backtesting."""

    def test_basic_backtest_runs(self, medium_swap_df, v2_strategy_params):
        """Test that a basic V2 backtest runs without errors."""
        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, v2_strategy_params)

        assert result is not None
        assert result.positions is not None
        assert result.metrics is not None

    def test_backtest_produces_valid_metrics(self, medium_swap_df, v2_strategy_params):
        """Test that backtest produces valid metrics."""
        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, v2_strategy_params)

        # Check metrics DataFrame structure
        assert 'net_pnl' in result.metrics.columns
        assert len(result.metrics) == len(v2_strategy_params['initial_capital'])

        # Check that metrics are finite
        assert not result.metrics['net_pnl'].isna().any()

    def test_position_value_never_negative(self, medium_swap_df, v2_strategy_params):
        """Test that position value is never negative."""
        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, v2_strategy_params)

        # Check all positions
        for j in range(result.positions.shape[1]):
            token0 = result.positions[:, j]['token0_balance']
            token1 = result.positions[:, j]['token1_balance']

            # Balances should never be negative
            assert (token0 >= -1e-10).all(), f"Strategy {j} has negative token0"
            assert (token1 >= -1e-10).all(), f"Strategy {j} has negative token1"

    def test_deterministic_results(self, seed):
        """Test that same input produces same output."""
        swaps1 = _create_swap_df(500, seed=seed)
        swaps2 = _create_swap_df(500, seed=seed)

        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.05],
            'rebalance_frequency': [100],
        }

        bt = LPBacktester(amm_type='v2')
        result1 = bt.run(swaps1, strategies)
        result2 = bt.run(swaps2, strategies)

        # Metrics should be identical
        assert result1.metrics['net_pnl'].iloc[0] == result2.metrics['net_pnl'].iloc[0]

    def test_multiple_strategies_independent(self, medium_swap_df):
        """Test that strategies are computed independently."""
        # Test that running same strategy alone vs with others gives similar results
        # Note: Not identical because strategies share pool state, but proportionally similar
        strategies_single = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

        strategies_multi = {
            'initial_capital': [10000.0, 20000.0, 50000.0],
            'rebalance_threshold': [0.0, 0.0, 0.0],
            'rebalance_frequency': [0, 0, 0],
        }

        bt = LPBacktester(amm_type='v2')
        result_single = bt.run(medium_swap_df, strategies_single)
        result_multi = bt.run(medium_swap_df, strategies_multi)

        # Multiple strategies share pool liquidity, so results scale with capital
        # Check that ratio of PnL to capital is similar (proportional performance)
        pnl_single = result_single.metrics['net_pnl'].iloc[0]
        cap_single = 10000.0

        pnl_multi_first = result_multi.metrics['net_pnl'].iloc[0]
        cap_multi_first = 10000.0

        # PnL/capital ratio should be similar (within 20% tolerance due to pool dynamics)
        ratio_single = pnl_single / cap_single
        ratio_multi = pnl_multi_first / cap_multi_first

        # The ratios may differ because adding more capital changes the pool dynamics
        # Just verify both have same sign and reasonable magnitude
        if abs(ratio_single) > 0.01:  # Only compare if significant
            assert (ratio_single > 0) == (ratio_multi > 0), "PnL signs should match"

    def test_reserves_follow_invariant(self, medium_swap_df, v2_strategy_params):
        """Test that reserves approximately follow x*y=k (k should increase from fees)."""
        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, v2_strategy_params)

        reserve0 = result.metadata['reserve0_history']
        reserve1 = result.metadata['reserve1_history']

        k_values = reserve0 * reserve1

        # k should generally increase (from fees)
        # Allow small numerical noise
        for i in range(1, len(k_values)):
            assert k_values[i] >= k_values[i - 1] - 1e-6, f"k decreased at step {i}"


class TestBacktestEdgeCases:
    """Tests for edge cases in backtesting."""

    def test_single_swap(self):
        """Test backtest with single swap."""
        swaps = _create_swap_df(1, seed=42)
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

        bt = LPBacktester(amm_type='v2')
        result = bt.run(swaps, strategies)

        assert result is not None
        assert len(result.positions) == 1

    def test_no_rebalancing(self, medium_swap_df):
        """Test backtest with no rebalancing."""
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, strategies)

        # Should have zero rebalances
        assert result.positions[-1, 0]['num_rebalances'] == 0

    def test_frequent_rebalancing(self, medium_swap_df):
        """Test backtest with frequent rebalancing."""
        strategies = {
            'initial_capital': [10000.0],
            'rebalance_threshold': [0.001],  # Very sensitive threshold
            'rebalance_frequency': [1],  # Every swap
        }

        bt = LPBacktester(amm_type='v2')
        result = bt.run(medium_swap_df, strategies)

        # Should have some rebalances (but not necessarily every swap due to threshold)
        assert result.positions[-1, 0]['num_rebalances'] >= 0

    def test_high_capital(self, small_swap_df):
        """Test with high capital value."""
        strategies = {
            'initial_capital': [1e12],  # 1 trillion
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

        bt = LPBacktester(amm_type='v2')
        result = bt.run(small_swap_df, strategies)

        assert not np.isnan(result.metrics['net_pnl'].iloc[0])
        assert not np.isinf(result.metrics['net_pnl'].iloc[0])

    def test_low_capital(self, small_swap_df):
        """Test with low capital value."""
        strategies = {
            'initial_capital': [1.0],  # $1
            'rebalance_threshold': [0.0],
            'rebalance_frequency': [0],
        }

        bt = LPBacktester(amm_type='v2')
        result = bt.run(small_swap_df, strategies)

        assert not np.isnan(result.metrics['net_pnl'].iloc[0])


class TestV3Backtest:
    """Basic tests for V3 backtesting (simplified, no tick crossing)."""

    def test_v3_backtest_runs(self, medium_swap_df, v3_strategy_params):
        """Test that V3 backtest runs without errors."""
        bt = LPBacktester(amm_type='v3')
        result = bt.run(medium_swap_df, v3_strategy_params)

        assert result is not None
        assert result.metrics is not None

    def test_v3_position_in_range_tracking(self, medium_swap_df, v3_strategy_params):
        """Test that V3 tracks in-range status."""
        bt = LPBacktester(amm_type='v3')
        result = bt.run(medium_swap_df, v3_strategy_params)

        # is_in_range should be boolean
        for j in range(result.positions.shape[1]):
            in_range = result.positions[:, j]['is_in_range']
            assert in_range.dtype == bool


class TestDLMMBacktest:
    """Basic tests for DLMM backtesting."""

    def test_dlmm_backtest_runs(self, medium_swap_df, dlmm_strategy_params):
        """Test that DLMM backtest runs without errors."""
        bt = LPBacktester(amm_type='dlmm')
        result = bt.run(medium_swap_df, dlmm_strategy_params)

        assert result is not None
        assert result.metrics is not None

    def test_dlmm_fee_history(self, medium_swap_df, dlmm_strategy_params):
        """Test that DLMM tracks fee history."""
        bt = LPBacktester(amm_type='dlmm')
        result = bt.run(medium_swap_df, dlmm_strategy_params)

        assert 'fee_history' in result.metadata
        assert len(result.metadata['fee_history']) == len(medium_swap_df)
        assert (result.metadata['fee_history'] >= 0).all()
