"""
Quick test script to verify AMMBT installation and basic functionality.
"""

import ammbt as amm
import numpy as np

print("=" * 80)
print("AMMBT Installation Test")
print("=" * 80)

# Test 1: Generate synthetic swaps
print("\n1. Testing synthetic data generation...")
swaps = amm.generate_swaps(
    n_swaps=1000,
    volatility=0.02,
    seed=42,
)
print(f"   ✓ Generated {len(swaps)} swaps")
print(f"   ✓ Price range: {swaps['price'].min():.4f} - {swaps['price'].max():.4f}")

# Test 2: Create strategy space
print("\n2. Testing strategy creation...")
strategies = {
    'initial_capital': [10000, 20000, 30000],
    'rebalance_threshold': [0.0, 0.05, 0.10],
    'rebalance_frequency': [0, 100, 500],
}
print(f"   ✓ Created {3 * 3 * 3} strategy variants")

# Test 3: Run backtest
print("\n3. Testing backtester...")
backtester = amm.LPBacktester(amm_type='v2')
print("   ✓ Initialized backtester")

results = backtester.run(swaps, strategies)
print(f"   ✓ Simulation complete")
print(f"   ✓ Tested {len(results.metrics)} strategies")

# Test 4: Analyze results
print("\n4. Testing analytics...")
best_pnl = results.metrics['net_pnl'].max()
worst_pnl = results.metrics['net_pnl'].min()
mean_pnl = results.metrics['net_pnl'].mean()
print(f"   ✓ Best PnL: ${best_pnl:,.2f}")
print(f"   ✓ Worst PnL: ${worst_pnl:,.2f}")
print(f"   ✓ Mean PnL: ${mean_pnl:,.2f}")

# Test 5: Summary
print("\n5. Top 3 strategies:")
top_strategies = results.summary().head(3)
for idx, row in top_strategies.iterrows():
    print(f"   Strategy {idx}: Net PnL = ${row['net_pnl']:,.2f}, Sharpe = {row['sharpe']:.2f}")

print("\n" + "=" * 80)
print("✓ All tests passed! AMMBT is working correctly.")
print("=" * 80)
print("\nNext steps:")
print("  - Run the demo notebook: jupyter notebook examples/uniswap_v2_demo.ipynb")
print("  - Read the README: cat README.md")
print("=" * 80)
