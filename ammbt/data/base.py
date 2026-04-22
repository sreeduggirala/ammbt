"""Base swap data loader interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Standard column schema for swap DataFrames
REQUIRED_COLUMNS = ['amount0', 'amount1']
OPTIONAL_COLUMNS = ['price', 'timestamp', 'block_number', 'tx_hash', 'liquidity']
COLUMN_DTYPES: Dict[str, type] = {
    'amount0': np.float64,
    'amount1': np.float64,
    'price': np.float64,
    'timestamp': np.int64,
    'block_number': np.int64,
    'tx_hash': str,
    'liquidity': np.float64,
}


class BaseSwapLoader(ABC):
    """
    Abstract base class for swap data loaders.

    All loaders produce a standardized DataFrame with at minimum
    ``amount0`` and ``amount1`` columns (float64).

    Optional columns: ``price``, ``timestamp``, ``block_number``,
    ``tx_hash``, ``liquidity``.
    """

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load swap data and return a standardized DataFrame.

        Returns
        -------
        pd.DataFrame
            Swap data with at least ``amount0`` and ``amount1`` columns.
        """
        pass

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize a swap DataFrame.

        Checks required columns exist, casts dtypes, and drops rows
        with NaN in required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Raw swap data.

        Returns
        -------
        pd.DataFrame
            Validated and normalized swap data.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        # Check required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"DataFrame has columns: {list(df.columns)}"
            )

        # Cast dtypes for known columns
        for col, dtype in COLUMN_DTYPES.items():
            if col in df.columns and dtype != str:
                df[col] = df[col].astype(dtype)

        # Drop rows with NaN in required columns
        df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid rows after dropping NaN values in required columns")

        return df
