"""
Data loading utilities for AMM swap data.

Provides loaders for various data sources:
- SubgraphLoader: Load from The Graph (Uniswap V2/V3 subgraphs)
- FileLoader: Load from CSV/Parquet files
"""

from ammbt.data.base import BaseSwapLoader
from ammbt.data.file_loader import FileLoader
from ammbt.data.subgraph import SubgraphLoader
from ammbt.data.stream import SwapStream, IncrementalSimulator

__all__ = [
    "BaseSwapLoader",
    "FileLoader",
    "SubgraphLoader",
    "SwapStream",
    "IncrementalSimulator",
]
