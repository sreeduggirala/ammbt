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
python -c "import ammbt; print('ammbt installed successfully')"
```

## Quick Start

```python
import pandas as pd
import numpy as np
import ammbt as amm

# Create swap data (or load from your data source)
np.random.seed(42)
n_swaps = 1000
prices = 1000 * np.exp(np.cumsum(np.random.randn(n_swaps) * 0.01))
swaps = pd.DataFrame({
    'amount0': np.random.randn(n_swaps) * 10,
    'amount1': np.random.randn(n_swaps) * 10000,
    'price': prices,
})

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

## Using Real Swap Data

Currently, users should source their own swap data from The Graph, Dune Analytics,
or similar providers. The swap DataFrame should contain at minimum:

- `amount0`: Token 0 amount (positive = buy, negative = sell)
- `amount1`: Token 1 amount
- `price`: Current pool price

For Uniswap V3, additional fields enable tick-by-tick fee tracking:

- `tick`: Current tick after swap
- `liquidity`: Pool liquidity at time of swap
- `sqrt_price_x96`: Precise price representation

**Note**: Data loader classes (`UniswapV3Loader`, `UniswapV2Loader`) are planned
for a future release.

## Simulation Capabilities

ammBT provides comprehensive LP position simulation:

- **Impermanent Loss (IL)**: Calculated by comparing LP value vs hold value
- **Rebalancing**: Threshold and frequency-based rebalancing with gas tracking
- **In-range Fee Accrual**: V3 has complete tick-based fee tracking
- **Comprehensive Metrics**: Sharpe ratio, Sortino ratio, max drawdown, capital efficiency
- **Tick-by-tick V3 Processing**: Accurate fee growth simulation per swap

## Project Status

- [x] Architecture design
- [x] Uniswap v2 implementation (COMPLETE)
- [x] Uniswap v3 implementation (COMPLETE)
- [x] Meteora DLMM implementation (COMPLETE)
- [ ] Data loaders for real swap data (Planned)
- [ ] Record system for event tracking

## License

MIT
