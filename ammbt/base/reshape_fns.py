"""
Broadcasting and reshaping utilities for multi-dimensional arrays.

These functions enable vectorized operations across strategy dimensions.
"""

import numpy as np
from typing import Tuple, List, Union


def broadcast(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    Broadcast multiple arrays to compatible shapes.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to broadcast

    Returns
    -------
    list of np.ndarray
        Broadcasted arrays

    Examples
    --------
    >>> a = np.array([1, 2, 3])  # (3,)
    >>> b = np.array([[1], [2], [3]])  # (3, 1)
    >>> a_bc, b_bc = broadcast(a, b)
    >>> a_bc.shape, b_bc.shape
    ((3, 3), (3, 3))
    """
    return list(np.broadcast_arrays(*arrays))


def broadcast_to_shape(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Broadcast array to target shape.

    Parameters
    ----------
    arr : np.ndarray
        Source array
    shape : tuple
        Target shape

    Returns
    -------
    np.ndarray
        Broadcasted array

    Examples
    --------
    >>> a = np.array([1, 2, 3])  # (3,)
    >>> broadcast_to_shape(a, (5, 3))  # (5, 3)
    """
    return np.broadcast_to(arr, shape)


def tile_to_strategies(arr: np.ndarray, n_strategies: int) -> np.ndarray:
    """
    Tile a 1D time series across strategy dimension.

    Parameters
    ----------
    arr : np.ndarray
        1D array (time series)
    n_strategies : int
        Number of strategy variants

    Returns
    -------
    np.ndarray
        2D array (n_time, n_strategies)

    Examples
    --------
    >>> prices = np.array([100, 101, 102])
    >>> tile_to_strategies(prices, 5).shape
    (3, 5)
    """
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D")

    return np.tile(arr[:, np.newaxis], (1, n_strategies))


def repeat_for_each_strategy(values: np.ndarray, n_times: int) -> np.ndarray:
    """
    Repeat strategy-level values for each time step.

    Parameters
    ----------
    values : np.ndarray
        1D array (strategy-level values)
    n_times : int
        Number of time steps

    Returns
    -------
    np.ndarray
        2D array (n_times, n_strategies)

    Examples
    --------
    >>> strategy_params = np.array([100, 200, 300])  # 3 strategies
    >>> repeat_for_each_strategy(strategy_params, 1000).shape
    (1000, 3)
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got {values.ndim}D")

    return np.tile(values, (n_times, 1))


def to_1d(arr: np.ndarray, order: str = 'C') -> np.ndarray:
    """
    Flatten array to 1D.

    Parameters
    ----------
    arr : np.ndarray
        Array to flatten
    order : str
        'C' for row-major (default), 'F' for column-major

    Returns
    -------
    np.ndarray
        Flattened array
    """
    return arr.ravel(order=order)


def to_2d(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Expand 1D array to 2D along specified axis.

    Parameters
    ----------
    arr : np.ndarray
        1D array
    axis : int
        Axis to expand (0 or 1)

    Returns
    -------
    np.ndarray
        2D array

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> to_2d(a, axis=0).shape
    (3, 1)
    >>> to_2d(a, axis=1).shape
    (1, 3)
    """
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D")

    if axis == 0:
        return arr[:, np.newaxis]
    elif axis == 1:
        return arr[np.newaxis, :]
    else:
        raise ValueError(f"Invalid axis {axis}, must be 0 or 1")
