# Changelog

All notable changes to ammbt will be documented in this file.

## [0.3.0] - Unreleased

### Added
- **Data loaders**: `SubgraphLoader` (The Graph V2/V3), `FileLoader` (CSV/Parquet) with schema validation
- **Dynamic pool liquidity**: Optional `liquidity` column in swap DataFrame for time-varying pool depth
- **New risk metrics**: VaR (95%/99%), CVaR, Calmar ratio, Omega ratio, win rate, profit factor, recovery time, HODL comparison (`lp_vs_hodl`)
- **LVR calculation** (`ammbt.portfolio.lvr`): Loss-Versus-Rebalancing with exact (CEX reference) and approximate (realized vol) methods
- **Event recording system** (`ammbt.portfolio.events`): Structured event log for swaps, rebalances, fee collections
- **Serialization** (`ammbt.portfolio.io`): `save_result()` / `load_result()` with npz and parquet formats
- **Walk-forward analysis** (`ammbt.portfolio.walk_forward`): In-sample/out-of-sample strategy robustness testing
- **Monte Carlo simulation** (`ammbt.portfolio.monte_carlo`): GBM, jump-diffusion, mean-reverting, and bootstrap price path models
- **Multi-pool portfolio** (`ammbt.portfolio.multi_pool`): `PortfolioBacktester` with correlation matrix and portfolio-level Sharpe
- **Pluggable rebalance strategies** (`ammbt.utils.rebalance`): Static, volatility-adaptive, and asymmetric re-centering
- **Expanded public API**: Simulators, strategy generators, math utilities, and data loaders all exported from `ammbt`
- **GitHub Actions CI**: Python 3.9/3.10/3.11 matrix with ruff linting, mypy type checking, and pytest
- **53 new tests**: analytics, validation, DLMM simulator, rebalancing (155 total)

### Fixed
- **V2 fee double-counting**: Metrics now AMM-type-aware; V2 token balances already include auto-compounded fees
- **IL calculation**: Uses `final_value` directly instead of incorrectly subtracting fees first
- **Division-by-zero guards**: `price_to_tick(0)`, `get_liquidity_for_amounts` with zero-width range, `get_amounts_for_liquidity` with zero sqrt prices
- **DLMM negative prices**: Price impact clamped to max 99% to prevent sign flip
- **O(n²) tick sort**: Replaced selection sort with insertion sort in V3 tick map building
- **Version mismatch**: `__init__.py` now matches `pyproject.toml` (0.2.0)

### Changed
- **Gas costs configurable**: `gas_cost_usd` field in all strategy param dtypes (defaults: V2=$50, V3=$100, DLMM=$0.50)
- **Input validation**: `LPBacktester.run()` validates non-empty swaps, positive capital, and valid tick/bin ranges

## [0.2.0] - 2024-12-01

### Added
- Tick-by-tick swap processing for Uniswap V3 simulator
- Meteora DLMM simulator with dynamic fees and volatility accumulator
- Strategy grid generators for V3 and DLMM
- Plotly visualization (performance, heatmap, efficient frontier, PnL distribution)

## [0.1.0] - 2024-10-01

### Added
- Initial release
- Uniswap V2 (CPAMM) simulator
- Basic metrics (Sharpe, Sortino, max drawdown, IL)
- Vectorized multi-strategy backtesting with Numba JIT
