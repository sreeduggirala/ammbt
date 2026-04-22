"""File-based swap data loader (CSV, Parquet)."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ammbt.data.base import BaseSwapLoader


class FileLoader(BaseSwapLoader):
    """
    Load swap data from CSV or Parquet files.

    Supports column mapping to rename source columns to the standard
    ammbt schema (``amount0``, ``amount1``, etc.).

    Parameters
    ----------
    column_mapping : dict, optional
        Maps source column names to standard names.
        E.g., ``{'swap_amount_0': 'amount0', 'swap_amount_1': 'amount1'}``.

    Examples
    --------
    >>> loader = FileLoader(column_mapping={'tokenAmount0': 'amount0', 'tokenAmount1': 'amount1'})
    >>> swaps = loader.load(path='swaps.csv')
    >>> swaps.columns
    Index(['amount0', 'amount1', ...])
    """

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        self.column_mapping = column_mapping or {}

    def load(
        self,
        path: str,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load swap data from a CSV or Parquet file.

        File format is auto-detected from the extension.

        Parameters
        ----------
        path : str
            Path to the data file (.csv, .parquet, .pq).
        start_row : int, optional
            First row to include (0-indexed, after header).
        end_row : int, optional
            Last row to include (exclusive).
        **kwargs
            Additional arguments passed to ``pd.read_csv`` or ``pd.read_parquet``.

        Returns
        -------
        pd.DataFrame
            Validated swap data.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is unsupported or required columns are missing.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = filepath.suffix.lower()

        if suffix == '.csv':
            df = pd.read_csv(path, **kwargs)
        elif suffix in ('.parquet', '.pq'):
            df = pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported: .csv, .parquet, .pq"
            )

        # Apply column mapping
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)

        # Slice rows if requested
        if start_row is not None or end_row is not None:
            df = df.iloc[start_row:end_row]

        return self.validate(df)

    @staticmethod
    def detect_amm_type(df: pd.DataFrame) -> str:
        """
        Heuristic to detect AMM type from column names.

        Parameters
        ----------
        df : pd.DataFrame
            Swap data.

        Returns
        -------
        str
            Detected AMM type: ``'v2'``, ``'v3'``, or ``'dlmm'``.
        """
        cols = set(df.columns)
        if {'tick', 'sqrtPriceX96'} & cols:
            return 'v3'
        if {'active_bin', 'bin_step', 'bin_id'} & cols:
            return 'dlmm'
        return 'v2'
