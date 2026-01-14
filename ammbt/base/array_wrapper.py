"""
Array wrapper class that maintains index/column metadata alongside numpy arrays.

Similar to vectorbt's array wrapper, this allows us to:
- Track strategy parameters alongside position state
- Broadcast operations across strategy dimensions
- Convert back to pandas DataFrames with proper labels
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple


class ArrayWrapper:
    """
    Wraps numpy arrays with index and column information.

    Core concept: We store position state as raw numpy arrays for speed,
    but track what each dimension represents for interpretability.

    Dimensions:
    - axis 0: time (swap events)
    - axis 1: strategy variants
    - axis 2: state variables (optional, for structured arrays)
    """

    def __init__(
        self,
        index: Optional[pd.Index] = None,
        columns: Optional[pd.Index] = None,
        ndim: int = 2,
    ):
        """
        Parameters
        ----------
        index : pd.Index, optional
            Row index (typically timestamps or swap numbers)
        columns : pd.Index, optional
            Column index (strategy variant labels)
        ndim : int
            Number of dimensions (2 for most cases)
        """
        self._index = index
        self._columns = columns
        self._ndim = ndim

    @property
    def index(self) -> Optional[pd.Index]:
        """Time/swap index."""
        return self._index

    @property
    def columns(self) -> Optional[pd.Index]:
        """Strategy index."""
        return self._columns

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape derived from index/columns."""
        if self.ndim == 1:
            return (len(self.index),) if self.index is not None else (0,)
        elif self.ndim == 2:
            return (
                len(self.index) if self.index is not None else 0,
                len(self.columns) if self.columns is not None else 0,
            )
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def wrap(self, arr: np.ndarray) -> pd.DataFrame:
        """
        Convert numpy array to DataFrame with proper index/columns.

        Parameters
        ----------
        arr : np.ndarray
            Array to wrap (shape must match wrapper dimensions)

        Returns
        -------
        pd.DataFrame
            Wrapped data with index and columns
        """
        if arr.ndim != self.ndim:
            raise ValueError(
                f"Array ndim {arr.ndim} doesn't match wrapper ndim {self.ndim}"
            )

        if self.ndim == 1:
            return pd.Series(arr, index=self.index)
        elif self.ndim == 2:
            return pd.DataFrame(arr, index=self.index, columns=self.columns)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def wrap_reduced(self, arr: np.ndarray, axis: int = 0) -> Union[pd.Series, pd.DataFrame]:
        """
        Wrap array after reduction along an axis.

        Parameters
        ----------
        arr : np.ndarray
            Reduced array
        axis : int
            Which axis was reduced (0 = time, 1 = strategies)

        Returns
        -------
        pd.Series or pd.DataFrame
            Wrapped reduced data
        """
        if axis == 0 and self.ndim == 2:
            # Reduced over time, left with strategy dimension
            return pd.Series(arr, index=self.columns)
        elif axis == 1 and self.ndim == 2:
            # Reduced over strategies, left with time dimension
            return pd.Series(arr, index=self.index)
        else:
            raise ValueError(f"Invalid axis {axis} for ndim {self.ndim}")

    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int, ...],
        index: Optional[pd.Index] = None,
        columns: Optional[pd.Index] = None,
    ) -> "ArrayWrapper":
        """
        Create wrapper from shape, generating default indices if needed.

        Parameters
        ----------
        shape : tuple
            Array shape
        index : pd.Index, optional
            Row index
        columns : pd.Index, optional
            Column index

        Returns
        -------
        ArrayWrapper
        """
        ndim = len(shape)

        if index is None and ndim >= 1:
            index = pd.RangeIndex(shape[0])
        if columns is None and ndim >= 2:
            columns = pd.RangeIndex(shape[1])

        return cls(index=index, columns=columns, ndim=ndim)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ArrayWrapper":
        """
        Create wrapper from existing DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe

        Returns
        -------
        ArrayWrapper
        """
        return cls(
            index=df.index,
            columns=df.columns,
            ndim=2,
        )

    def __repr__(self) -> str:
        return (
            f"ArrayWrapper(shape={self.shape}, "
            f"index={'[...]' if self.index is not None else None}, "
            f"columns={'[...]' if self.columns is not None else None})"
        )
