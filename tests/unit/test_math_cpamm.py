"""
Unit tests for CPAMM (Uniswap V2) math functions.

Tests get_amount_out, get_amount_in, and related edge cases.
"""

import numpy as np
import pytest

from ammbt.utils.math import get_amount_out, get_amount_in


class TestGetAmountOut:
    """Tests for get_amount_out function."""

    def test_basic_swap(self):
        """Test basic swap calculation."""
        # 1000 reserve each, swap 100 in
        amount_out = get_amount_out(100.0, 1000.0, 1000.0, 0.003)

        # Should get ~90.6 out (accounting for fee and price impact)
        assert amount_out > 0
        assert amount_out < 100  # Due to fee and slippage

        # Verify math: (100 * 0.997) * 1000 / (1000 + 100 * 0.997)
        expected = (100 * 0.997) * 1000 / (1000 + 100 * 0.997)
        assert abs(amount_out - expected) < 1e-10

    def test_zero_input_returns_zero(self):
        """Test that zero input returns zero output."""
        amount_out = get_amount_out(0.0, 1000.0, 1000.0, 0.003)
        assert amount_out == 0.0

    def test_negative_input_returns_zero(self):
        """Test that negative input returns zero output."""
        amount_out = get_amount_out(-100.0, 1000.0, 1000.0, 0.003)
        assert amount_out == 0.0

    def test_zero_reserve_in_returns_zero(self):
        """Test that zero reserve_in returns zero."""
        amount_out = get_amount_out(100.0, 0.0, 1000.0, 0.003)
        assert amount_out == 0.0

    def test_zero_reserve_out_returns_zero(self):
        """Test that zero reserve_out returns zero."""
        amount_out = get_amount_out(100.0, 1000.0, 0.0, 0.003)
        assert amount_out == 0.0

    def test_negative_reserve_returns_zero(self):
        """Test that negative reserves return zero."""
        assert get_amount_out(100.0, -1000.0, 1000.0, 0.003) == 0.0
        assert get_amount_out(100.0, 1000.0, -1000.0, 0.003) == 0.0

    def test_large_swap_price_impact(self):
        """Test that large swaps have significant price impact."""
        small_swap = get_amount_out(10.0, 1000.0, 1000.0, 0.003)
        large_swap = get_amount_out(500.0, 1000.0, 1000.0, 0.003)

        # Price per unit should be worse for large swap
        price_small = small_swap / 10.0
        price_large = large_swap / 500.0

        assert price_large < price_small

    def test_huge_swap_approaches_reserve(self):
        """Test that huge swaps approach but never exceed reserve_out."""
        # Swap equal to reserve_in
        amount_out = get_amount_out(1000.0, 1000.0, 1000.0, 0.003)
        assert amount_out < 1000.0  # Cannot exceed reserve

        # Even larger swap
        huge_out = get_amount_out(1_000_000.0, 1000.0, 1000.0, 0.003)
        assert huge_out < 1000.0
        assert huge_out > amount_out  # But still more than smaller swap

    def test_zero_fee(self):
        """Test swap with zero fee."""
        amount_out_no_fee = get_amount_out(100.0, 1000.0, 1000.0, 0.0)
        amount_out_with_fee = get_amount_out(100.0, 1000.0, 1000.0, 0.003)

        assert amount_out_no_fee > amount_out_with_fee

    def test_high_fee(self):
        """Test swap with high fee."""
        amount_out_low_fee = get_amount_out(100.0, 1000.0, 1000.0, 0.003)
        amount_out_high_fee = get_amount_out(100.0, 1000.0, 1000.0, 0.03)

        assert amount_out_high_fee < amount_out_low_fee

    def test_asymmetric_reserves(self):
        """Test swap with asymmetric reserves."""
        # More reserve_out means better price
        amount_out_balanced = get_amount_out(100.0, 1000.0, 1000.0, 0.003)
        amount_out_more_out = get_amount_out(100.0, 1000.0, 2000.0, 0.003)

        assert amount_out_more_out > amount_out_balanced

    def test_output_never_nan(self):
        """Test that output is never NaN for valid inputs."""
        test_cases = [
            (100.0, 1000.0, 1000.0),
            (0.0, 1000.0, 1000.0),
            (100.0, 0.0, 1000.0),
            (100.0, 1000.0, 0.0),
            (1e-10, 1000.0, 1000.0),
            (1e10, 1000.0, 1000.0),
        ]

        for amount_in, reserve_in, reserve_out in test_cases:
            result = get_amount_out(amount_in, reserve_in, reserve_out, 0.003)
            assert not np.isnan(result), f"NaN for inputs: {amount_in}, {reserve_in}, {reserve_out}"

    def test_output_never_negative(self):
        """Test that output is never negative."""
        test_cases = [
            (100.0, 1000.0, 1000.0),
            (-100.0, 1000.0, 1000.0),
            (100.0, -1000.0, 1000.0),
            (0.0, 0.0, 0.0),
        ]

        for amount_in, reserve_in, reserve_out in test_cases:
            result = get_amount_out(amount_in, reserve_in, reserve_out, 0.003)
            assert result >= 0, f"Negative result for inputs: {amount_in}, {reserve_in}, {reserve_out}"


class TestGetAmountIn:
    """Tests for get_amount_in function."""

    def test_basic_reverse(self):
        """Test basic reverse calculation."""
        # Get amount in for a desired output
        amount_in = get_amount_in(90.0, 1000.0, 1000.0, 0.003)

        assert amount_in > 0
        assert amount_in > 90  # Need more input than output (fee + slippage)

    def test_zero_output_returns_zero(self):
        """Test that zero output returns zero input."""
        amount_in = get_amount_in(0.0, 1000.0, 1000.0, 0.003)
        assert amount_in == 0.0

    def test_negative_output_returns_zero(self):
        """Test that negative output returns zero input."""
        amount_in = get_amount_in(-100.0, 1000.0, 1000.0, 0.003)
        assert amount_in == 0.0

    def test_output_exceeds_reserve_returns_inf(self):
        """Test that requesting more than reserve returns infinity."""
        amount_in = get_amount_in(1001.0, 1000.0, 1000.0, 0.003)
        assert np.isinf(amount_in)

    def test_zero_reserve_returns_inf(self):
        """Test that zero reserves return infinity."""
        assert np.isinf(get_amount_in(100.0, 0.0, 1000.0, 0.003))
        assert np.isinf(get_amount_in(100.0, 1000.0, 0.0, 0.003))

    def test_roundtrip_consistency(self):
        """Test that get_amount_in and get_amount_out are consistent."""
        reserve_in = 1000.0
        reserve_out = 1000.0
        fee = 0.003

        # Get output for input of 100
        amount_out = get_amount_out(100.0, reserve_in, reserve_out, fee)

        # Get required input for that output
        amount_in_calculated = get_amount_in(amount_out, reserve_in, reserve_out, fee)

        # Should be close to original input
        assert abs(amount_in_calculated - 100.0) < 0.01


class TestConstantProductInvariant:
    """Tests for x*y=k invariant preservation."""

    def test_k_increases_after_swap(self):
        """Test that k increases after a swap (due to fees)."""
        reserve0 = 1000.0
        reserve1 = 1000.0
        k_before = reserve0 * reserve1

        amount_in = 100.0
        amount_out = get_amount_out(amount_in, reserve0, reserve1, 0.003)

        new_reserve0 = reserve0 + amount_in
        new_reserve1 = reserve1 - amount_out
        k_after = new_reserve0 * new_reserve1

        # k should increase due to fee
        assert k_after > k_before

    def test_k_unchanged_without_fee(self):
        """Test that k is preserved without fees."""
        reserve0 = 1000.0
        reserve1 = 1000.0
        k_before = reserve0 * reserve1

        amount_in = 100.0
        amount_out = get_amount_out(amount_in, reserve0, reserve1, 0.0)

        new_reserve0 = reserve0 + amount_in
        new_reserve1 = reserve1 - amount_out
        k_after = new_reserve0 * new_reserve1

        # k should be approximately preserved (small numerical error)
        assert abs(k_after - k_before) / k_before < 1e-10


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_small_amounts(self):
        """Test with very small amounts."""
        amount_out = get_amount_out(1e-15, 1000.0, 1000.0, 0.003)
        assert amount_out >= 0
        assert not np.isnan(amount_out)
        assert not np.isinf(amount_out)

    def test_very_large_amounts(self):
        """Test with very large amounts."""
        amount_out = get_amount_out(1e15, 1000.0, 1000.0, 0.003)
        assert amount_out >= 0
        assert amount_out < 1000.0  # Cannot exceed reserve
        assert not np.isnan(amount_out)

    def test_very_large_reserves(self):
        """Test with very large reserves."""
        amount_out = get_amount_out(100.0, 1e15, 1e15, 0.003)
        assert amount_out > 0
        assert not np.isnan(amount_out)
        assert not np.isinf(amount_out)

    def test_very_small_reserves(self):
        """Test with very small reserves."""
        amount_out = get_amount_out(1e-10, 1e-8, 1e-8, 0.003)
        assert amount_out >= 0
        assert not np.isnan(amount_out)

    def test_mixed_magnitudes(self):
        """Test with mixed magnitudes."""
        # Small input, large reserves
        out1 = get_amount_out(1e-10, 1e10, 1e10, 0.003)
        assert out1 >= 0
        assert not np.isnan(out1)

        # Large input, small reserves (but valid)
        out2 = get_amount_out(1e5, 1e6, 1e6, 0.003)
        assert out2 >= 0
        assert out2 < 1e6
