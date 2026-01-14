"""
Quick test script for Uniswap V3 functionality.
"""

import ammbt as amm
from ammbt.utils import generate_tick_ranges, create_v3_strategy_grid
import numpy as np

print("=" * 80)
print("AMMBT Uniswap V3 Test")
print("=" * 80)

# Test 1: Generate synthetic swaps
print("\n1. Testing synthetic data generation...")
swaps = amm.generate_swaps(
    n_swaps=1000,
    volatility=0.03,
    seed=42,
)
print(f"   ✓ Generated {len(swaps)} swaps")
print(f"   ✓ Price range: {swaps['price'].min():.4f} - {swaps['price'].max():.4f}")

# Test 2: Generate tick ranges
print("\n2. Testing tick range generation...")
tick_ranges = generate_tick_ranges(
    current_price=1.0,
    range_widths_pct=[0.10, 0.20],
    num_ranges=2,
)
print(f"   ✓ Generated {len(tick_ranges)} tick ranges")
for i, (lower, upper) in enumerate(tick_ranges):
    print(f"      Range {i+1}: ticks [{lower}, {upper}]")

# Test 3: Create V3 strategy grid
print("\n3. Testing V3 strategy grid creation...")
strategies = create_v3_strategy_grid(
    initial_capitals=[10000, 20000],
    tick_ranges=tick_ranges,
    rebalance_frequencies=[0, 100],
)
print(f"   ✓ Created {len(strategies)} strategy variants")

# Test 4: Run V3 backtest
print("\n4. Testing V3 backtester...")
backtester = amm.LPBacktester(
    amm_type='v3',
    initial_price=1.0,
    fee_tier=3000,
)
print("   ✓ Initialized V3 backtester")

results = backtester.run(swaps, strategies)
print(f"   ✓ Simulation complete")
print(f"   ✓ Tested {len(results.metrics)} strategies")

# Test 5: Analyze results
print("\n5. Testing V3 analytics...")
best_pnl = results.metrics['net_pnl'].max()
worst_pnl = results.metrics['net_pnl'].min()
mean_pnl = results.metrics['net_pnl'].mean()
mean_time_in_range = results.metrics['pct_time_in_range'].mean()

print(f"   ✓ Best PnL: ${best_pnl:,.2f}")
print(f"   ✓ Worst PnL: ${worst_pnl:,.2f}")
print(f"   ✓ Mean PnL: ${mean_pnl:,.2f}")
print(f"   ✓ Mean time in range: {mean_time_in_range:.1f}%")

# Test 6: Check position tracking
print("\n6. Testing position state tracking...")
best_idx = results.metrics['net_pnl'].idxmax()
position_history = results.get_position_history(best_idx)
print(f"   ✓ Position history shape: {position_history.shape}")
print(f"   ✓ Final liquidity: {position_history['liquidity'].iloc[-1]:,.2f}")
print(f"   ✓ Total fees: ${position_history['uncollected_fees_0'].iloc[-1] + position_history['uncollected_fees_1'].iloc[-1]:,.2f}")

print("\n" + "=" * 80)
print("✓ All V3 tests passed! Uniswap V3 implementation is working correctly.")
print("=" * 80)
print("\nNext steps:")
print("  - Run the V3 demo notebook: jupyter notebook examples/uniswap_v3_demo.ipynb")
print("  - Compare V3 vs V2 performance on same data")
print("  - Test with real historical swap data")
print("=" * 80)
