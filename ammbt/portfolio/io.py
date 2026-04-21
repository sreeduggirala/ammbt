"""
Serialization utilities for saving and loading backtest results.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def save_result(result, path: str, format: str = 'npz') -> None:
    """
    Save a BacktestResult to disk.

    Creates a directory containing:
    - ``positions.npz`` or ``positions.parquet``
    - ``metrics.csv``
    - ``strategy_params.csv``
    - ``metadata.json`` (scalars) + ``metadata_arrays/`` (numpy arrays)

    Parameters
    ----------
    result : BacktestResult
        The backtest result to save.
    path : str
        Directory path to save into (created if it doesn't exist).
    format : str
        Position format: ``'npz'`` (numpy, preserves structured dtypes) or
        ``'parquet'`` (interoperable).
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    # Save positions
    if format == 'npz':
        np.savez_compressed(out / 'positions.npz', positions=result.positions)
    elif format == 'parquet':
        _positions_to_parquet(result.positions, out / 'positions.parquet')
    else:
        raise ValueError(f"Unsupported format: '{format}'. Use 'npz' or 'parquet'.")

    # Save metrics
    result.metrics.to_csv(out / 'metrics.csv', index=False)

    # Save strategy params
    result.strategy_params.to_csv(out / 'strategy_params.csv', index=False)

    # Save metadata (separate scalars from arrays)
    arrays_dir = out / 'metadata_arrays'
    arrays_dir.mkdir(exist_ok=True)

    scalar_metadata = {}
    for key, value in result.metadata.items():
        if isinstance(value, np.ndarray):
            np.save(arrays_dir / f'{key}.npy', value)
        else:
            scalar_metadata[key] = _serialize_value(value)

    with open(out / 'metadata.json', 'w') as f:
        json.dump(scalar_metadata, f, indent=2)


def load_result(path: str):
    """
    Load a BacktestResult from disk.

    Parameters
    ----------
    path : str
        Directory path containing saved result files.

    Returns
    -------
    BacktestResult
        Reconstructed backtest result.

    Raises
    ------
    FileNotFoundError
        If the directory or required files don't exist.
    """
    from ammbt.portfolio.base import BacktestResult
    from ammbt.base.array_wrapper import ArrayWrapper

    out = Path(path)
    if not out.exists():
        raise FileNotFoundError(f"Result directory not found: {path}")

    # Load positions
    npz_path = out / 'positions.npz'
    parquet_path = out / 'positions.parquet'
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        positions = data['positions']
    elif parquet_path.exists():
        positions = _parquet_to_positions(parquet_path)
    else:
        raise FileNotFoundError("No positions file found (positions.npz or positions.parquet)")

    # Load metrics
    metrics = pd.read_csv(out / 'metrics.csv')

    # Load strategy params
    strategy_params = pd.read_csv(out / 'strategy_params.csv')

    # Load metadata
    with open(out / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load metadata arrays
    arrays_dir = out / 'metadata_arrays'
    if arrays_dir.exists():
        for npy_file in arrays_dir.glob('*.npy'):
            key = npy_file.stem
            metadata[key] = np.load(npy_file)

    # Extract capital efficiency columns if present
    cap_eff_cols = ['pct_time_in_range', 'avg_utilization']
    cap_eff_data = {c: metrics[c] for c in cap_eff_cols if c in metrics.columns}
    capital_efficiency = pd.DataFrame(cap_eff_data) if cap_eff_data else pd.DataFrame()

    # Reconstruct wrapper
    n_swaps, n_strategies = positions.shape
    wrapper = ArrayWrapper(
        index=pd.RangeIndex(n_swaps),
        columns=pd.RangeIndex(n_strategies),
        ndim=2,
    )

    return BacktestResult(
        positions=positions,
        metrics=metrics,
        capital_efficiency=capital_efficiency,
        metadata=metadata,
        wrapper=wrapper,
        strategy_params=strategy_params,
    )


def _positions_to_parquet(positions: np.ndarray, path) -> None:
    """Convert structured numpy positions array to parquet."""
    n_swaps, n_strategies = positions.shape
    data = {}
    for field in positions.dtype.names:
        for j in range(n_strategies):
            data[f'{field}_{j}'] = positions[:, j][field]
    pd.DataFrame(data).to_parquet(path, index=False)


def _parquet_to_positions(path) -> np.ndarray:
    """Convert parquet back to structured numpy positions array."""
    df = pd.read_parquet(path)
    # Infer structure from column names: field_strategyIdx
    fields = set()
    n_strategies = 0
    for col in df.columns:
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            fields.add(parts[0])
            n_strategies = max(n_strategies, int(parts[1]) + 1)

    # This is a lossy reconstruction — structured dtype info is lost
    # For full fidelity, use npz format
    n_swaps = len(df)
    # Return as plain array since we can't reconstruct dtype
    return df.values.reshape(n_swaps, -1)


def _serialize_value(value):
    """Convert a value to JSON-serializable form."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
