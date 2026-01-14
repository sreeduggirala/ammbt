# ammBT - AMM Backtesting Engine

High-performance vectorized backtesting engine for AMM liquidity provider positions.

Inspired by [vectorbt](https://github.com/polakowo/vectorbt), ammBT enables testing thousands of LP strategies simultaneously using vectorized operations and Numba compilation.

## Features

- **Multi-AMM Support**: Uniswap v2 (CPAMM), Uniswap v3 (CLMM), Meteora (DLMM)
- **Vectorized Backtesting**: Test thousands of strategy variants in seconds
- **Path-Dependent Simulation**: Accurate swap-by-swap pool state simulation
- **Comprehensive Metrics**: IL, fees, PnL, Sharpe ratio, capital efficiency
- **Interactive Visualization**: Plotly-based charts and heatmaps

## Architecture

ammBT uses a hybrid approach:

- **Sequential (Numba)**: Swap processing through time (path-dependent)
- **Vectorized (NumPy)**: Parallel strategy evaluation across parameter space

This allows testing N strategies with time complexity of O(swaps) instead of O(swaps × N).

## Supported AMM Types

### Uniswap v2 (CPAMM)

- Constant product formula: x × y = k
- Full-range liquidity
- Fixed 0.3% fee

### Uniswap v3 (CLMM)

- Concentrated liquidity with tick ranges
- Multiple fee tiers (0.05%, 0.3%, 1%)
- Capital efficiency through range orders

### Meteora DLMM

- Bin-based liquidity distribution
- Dynamic fees based on volatility
- Solana-native implementation

## Status

⚠️ **Work in Progress** - This project is under active development.

## Installation (Development)

```bash
git clone <repo>
cd ammbt
pip install -r requirements.txt
pip install -e .
python test_install.py
```

## Quick Start

```python
import ammbt as amm

# Generate synthetic swap data
swaps = amm.generate_swaps(
    n_swaps=10000,
    volatility=0.02,
    drift=0.0
)

# Define strategy space
strategies = {
    'initial_capital': [10000, 50000, 100000],
    'rebalance_threshold': [0.0, 0.05, 0.10],
    'rebalance_frequency': [0, 100, 500],
}

# Run backtest
backtester = amm.LPBacktester(amm_type='v2')
results = backtester.run(swaps, strategies)

# Analyze
print(results.summary())
amm.plot_performance(results, strategy_idx=0).show()
```

See `examples/uniswap_v2_demo.ipynb` for full walkthrough.

## Project Status

- [x] Architecture design
- [x] Uniswap v2 implementation (COMPLETE)
- [x] Uniswap v3 implementation (COMPLETE)
- [ ] Meteora DLMM implementation
- [ ] Data loaders for real swap data
- [ ] Record system for event tracking

## License

MIT
