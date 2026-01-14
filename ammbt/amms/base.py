"""Base AMM simulator interface."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any


class BaseAMMSimulator(ABC):
    """
    Abstract base class for AMM simulators.

    All AMM types (v2, v3, DLMM) implement this interface.
    """

    def __init__(self, pool_params: Dict[str, Any]):
        """
        Initialize simulator with pool parameters.

        Parameters
        ----------
        pool_params : dict
            Pool-specific parameters (reserves, fee tier, etc.)
        """
        self.pool_params = pool_params

    @abstractmethod
    def simulate(
        self,
        swaps: pd.DataFrame,
        positions: np.ndarray,
        strategy_params: np.ndarray,
    ) -> np.ndarray:
        """
        Run simulation across all swaps and strategies.

        Parameters
        ----------
        swaps : pd.DataFrame
            Swap event data
        positions : np.ndarray
            Position state array (n_swaps, n_strategies)
        strategy_params : np.ndarray
            Strategy parameters (n_strategies,)

        Returns
        -------
        np.ndarray
            Updated position state
        """
        pass

    @abstractmethod
    def initialize_positions(
        self,
        n_strategies: int,
        n_swaps: int,
        strategy_params: np.ndarray,
    ) -> np.ndarray:
        """
        Create initial position array.

        Parameters
        ----------
        n_strategies : int
            Number of strategy variants
        n_swaps : int
            Number of time steps
        strategy_params : np.ndarray
            Strategy parameters

        Returns
        -------
        np.ndarray
            Initialized position array
        """
        pass
