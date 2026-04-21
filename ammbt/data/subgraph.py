"""The Graph / Subgraph swap data loader for Uniswap V2 and V3."""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ammbt.data.base import BaseSwapLoader

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


# Default subgraph endpoints (hosted service — may require API key for decentralized gateway)
UNISWAP_V2_SUBGRAPH = (
    "https://gateway.thegraph.com/api/subgraphs/id/A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"
)
UNISWAP_V3_SUBGRAPH = (
    "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)


class SubgraphLoader(BaseSwapLoader):
    """
    Load swap data from Uniswap subgraphs via The Graph.

    Supports two modes:
    - ``'swaps'``: Individual swap events (accurate, slower).
    - ``'hourly'``: Pool hour data snapshots (fast, approximate).

    Parameters
    ----------
    subgraph_url : str, optional
        Custom subgraph URL. Defaults to Uniswap V3 mainnet.
    api_key : str, optional
        The Graph API key (required for the decentralized gateway).
    mode : str
        ``'swaps'`` for individual events or ``'hourly'`` for pool snapshots.
    rate_limit : float
        Minimum seconds between API requests.

    Examples
    --------
    >>> loader = SubgraphLoader(api_key='your-key', mode='swaps')
    >>> swaps = loader.load(pool_address='0x8ad5...', start_block=18000000, end_block=18010000)
    """

    def __init__(
        self,
        subgraph_url: Optional[str] = None,
        api_key: Optional[str] = None,
        mode: str = 'swaps',
        rate_limit: float = 0.2,
    ):
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for SubgraphLoader. "
                "Install it with: pip install ammbt[data]"
            )

        self.subgraph_url = subgraph_url or UNISWAP_V3_SUBGRAPH
        self.api_key = api_key
        self.mode = mode
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

        if mode not in ('swaps', 'hourly'):
            raise ValueError(f"mode must be 'swaps' or 'hourly', got '{mode}'")

    def load(
        self,
        pool_address: str,
        start_block: int = 0,
        end_block: Optional[int] = None,
        max_results: int = 10000,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load swap data from a Uniswap subgraph.

        Parameters
        ----------
        pool_address : str
            Pool contract address (checksummed or lowercase).
        start_block : int
            Starting block number.
        end_block : int, optional
            Ending block number. If None, fetches up to latest.
        max_results : int
            Maximum number of results to fetch.

        Returns
        -------
        pd.DataFrame
            Validated swap data.
        """
        pool_address = pool_address.lower()

        if self.mode == 'swaps':
            records = self._query_swaps(pool_address, start_block, end_block, max_results)
        else:
            records = self._query_hourly(pool_address, start_block, end_block, max_results)

        if not records:
            raise ValueError(
                f"No data returned for pool {pool_address} "
                f"between blocks {start_block} and {end_block}"
            )

        df = pd.DataFrame(records)
        return self.validate(df)

    def _query_swaps(
        self,
        pool_address: str,
        start_block: int,
        end_block: Optional[int],
        max_results: int,
    ) -> List[Dict]:
        """Fetch individual swap events with pagination."""
        all_records: List[Dict] = []
        skip = 0
        page_size = 1000  # The Graph max per query

        while len(all_records) < max_results:
            block_filter = f'blockNumber_gte: "{start_block}"'
            if end_block is not None:
                block_filter += f', blockNumber_lte: "{end_block}"'

            query = f"""
            {{
                swaps(
                    first: {page_size},
                    skip: {skip},
                    orderBy: timestamp,
                    orderDirection: asc,
                    where: {{
                        pool: "{pool_address}",
                        {block_filter}
                    }}
                ) {{
                    amount0
                    amount1
                    sqrtPriceX96
                    tick
                    timestamp
                    transaction {{
                        blockNumber
                    }}
                }}
            }}
            """

            data = self._execute_query(query)
            swaps = data.get('data', {}).get('swaps', [])

            if not swaps:
                break

            for swap in swaps:
                record = {
                    'amount0': float(swap['amount0']),
                    'amount1': float(swap['amount1']),
                    'timestamp': int(swap['timestamp']),
                }
                if swap.get('sqrtPriceX96'):
                    sqrt_price = float(swap['sqrtPriceX96'])
                    Q96 = 79228162514264337593543950336.0
                    record['price'] = (sqrt_price / Q96) ** 2
                if swap.get('transaction', {}).get('blockNumber'):
                    record['block_number'] = int(swap['transaction']['blockNumber'])
                all_records.append(record)

            if len(swaps) < page_size:
                break

            skip += page_size

        return all_records[:max_results]

    def _query_hourly(
        self,
        pool_address: str,
        start_block: int,
        end_block: Optional[int],
        max_results: int,
    ) -> List[Dict]:
        """Fetch poolHourData snapshots and convert to synthetic swaps."""
        all_records: List[Dict] = []
        skip = 0
        page_size = 1000

        while len(all_records) < max_results:
            query = f"""
            {{
                poolHourDatas(
                    first: {page_size},
                    skip: {skip},
                    orderBy: periodStartUnix,
                    orderDirection: asc,
                    where: {{
                        pool: "{pool_address}"
                    }}
                ) {{
                    periodStartUnix
                    volumeToken0
                    volumeToken1
                    tick
                    sqrtPrice
                    liquidity
                    token0Price
                    token1Price
                }}
            }}
            """

            data = self._execute_query(query)
            hours = data.get('data', {}).get('poolHourDatas', [])

            if not hours:
                break

            for hour in hours:
                vol0 = float(hour.get('volumeToken0', 0))
                vol1 = float(hour.get('volumeToken1', 0))
                price = float(hour.get('token0Price', 0))
                timestamp = int(hour.get('periodStartUnix', 0))
                liquidity = float(hour.get('liquidity', 0))

                if vol0 == 0 and vol1 == 0:
                    continue

                record = {
                    'amount0': vol0,
                    'amount1': -vol1 if vol0 > 0 else vol1,
                    'price': price if price > 0 else np.nan,
                    'timestamp': timestamp,
                }
                if liquidity > 0:
                    record['liquidity'] = liquidity
                all_records.append(record)

            if len(hours) < page_size:
                break

            skip += page_size

        return all_records[:max_results]

    def _execute_query(self, query: str) -> Dict:
        """Execute a GraphQL query with rate limiting."""
        self._rate_limit_wait()

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        response = requests.post(
            self.subgraph_url,
            json={'query': query},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
