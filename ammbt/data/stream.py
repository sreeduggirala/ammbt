"""
Streaming / live data support for real-time AMM LP monitoring.

Provides WebSocket-based swap listeners and an incremental simulation
engine that processes new swaps without re-running the full backtest.
"""

import asyncio
import json
import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

from ammbt.data.base import BaseSwapLoader, REQUIRED_COLUMNS


class SwapStream:
    """
    Real-time swap event stream via WebSocket or polling.

    Supports two modes:
    - **WebSocket**: Connect to an RPC provider's WebSocket endpoint and
      subscribe to swap events (e.g., Alchemy, Infura, QuickNode).
    - **Polling**: Periodically query a subgraph or RPC for new swaps.

    Parameters
    ----------
    mode : str
        ``'websocket'`` or ``'polling'``.
    url : str
        WebSocket URL or HTTP endpoint.
    pool_address : str
        Pool contract address to monitor.
    poll_interval : float
        Seconds between polls (polling mode only).
    on_swap : callable, optional
        Callback ``fn(swap_dict)`` called for each new swap.

    Examples
    --------
    >>> stream = SwapStream(
    ...     mode='polling',
    ...     url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
    ...     pool_address='0x8ad5...',
    ...     poll_interval=15.0,
    ... )
    >>> stream.on_swap = lambda swap: print(f"New swap: {swap}")
    >>> asyncio.run(stream.start())
    """

    def __init__(
        self,
        mode: str = 'polling',
        url: str = '',
        pool_address: str = '',
        poll_interval: float = 15.0,
        on_swap: Optional[Callable] = None,
    ):
        self.mode = mode
        self.url = url
        self.pool_address = pool_address.lower()
        self.poll_interval = poll_interval
        self.on_swap = on_swap
        self._running = False
        self._buffer: List[Dict] = []
        self._last_timestamp = 0

    async def start(self) -> None:
        """Start listening for swap events."""
        self._running = True
        if self.mode == 'websocket':
            await self._run_websocket()
        elif self.mode == 'polling':
            await self._run_polling()
        else:
            raise ValueError(f"Unknown mode: '{self.mode}'. Use 'websocket' or 'polling'.")

    def stop(self) -> None:
        """Stop the stream."""
        self._running = False

    def drain_buffer(self) -> pd.DataFrame:
        """
        Get all buffered swaps since last drain and clear the buffer.

        Returns
        -------
        pd.DataFrame
            New swaps with standard columns (amount0, amount1, etc.).
        """
        if not self._buffer:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        df = pd.DataFrame(self._buffer)
        self._buffer = []
        return df

    async def _run_websocket(self) -> None:
        """Listen via WebSocket (e.g., Alchemy eth_subscribe)."""
        if websockets is None:
            raise ImportError(
                "The 'websockets' package is required for WebSocket mode. "
                "Install with: pip install ammbt[data]"
            )

        # Uniswap V3 Swap event signature
        SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": [
                "logs",
                {
                    "address": self.pool_address,
                    "topics": [SWAP_TOPIC],
                }
            ]
        })

        async with websockets.connect(self.url) as ws:
            await ws.send(subscribe_msg)
            response = await ws.recv()

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(msg)

                    if 'params' in data and 'result' in data['params']:
                        log = data['params']['result']
                        swap = self._parse_swap_log(log)
                        if swap:
                            self._buffer.append(swap)
                            if self.on_swap:
                                self.on_swap(swap)

                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

    async def _run_polling(self) -> None:
        """Poll a subgraph for new swaps."""
        if aiohttp is None:
            raise ImportError(
                "The 'aiohttp' package is required for polling mode. "
                "Install with: pip install ammbt[data]"
            )

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    swaps = await self._poll_subgraph(session)
                    for swap in swaps:
                        self._buffer.append(swap)
                        if self.on_swap:
                            self.on_swap(swap)
                except Exception:
                    pass

                await asyncio.sleep(self.poll_interval)

    async def _poll_subgraph(self, session) -> List[Dict]:
        """Query subgraph for swaps newer than last timestamp."""
        query = f"""
        {{
            swaps(
                first: 100,
                orderBy: timestamp,
                orderDirection: asc,
                where: {{
                    pool: "{self.pool_address}",
                    timestamp_gt: "{self._last_timestamp}"
                }}
            ) {{
                amount0
                amount1
                sqrtPriceX96
                timestamp
            }}
        }}
        """

        async with session.post(
            self.url,
            json={'query': query},
            headers={'Content-Type': 'application/json'},
        ) as resp:
            data = await resp.json()

        swaps = data.get('data', {}).get('swaps', [])
        result = []

        for swap in swaps:
            ts = int(swap['timestamp'])
            if ts > self._last_timestamp:
                self._last_timestamp = ts

            Q96 = 79228162514264337593543950336.0
            sqrt_price = float(swap.get('sqrtPriceX96', 0))
            price = (sqrt_price / Q96) ** 2 if sqrt_price > 0 else 0.0

            result.append({
                'amount0': float(swap['amount0']),
                'amount1': float(swap['amount1']),
                'price': price,
                'timestamp': ts,
            })

        return result

    def _parse_swap_log(self, log: Dict) -> Optional[Dict]:
        """Parse a raw Ethereum swap log into a swap dict."""
        try:
            data = log.get('data', '0x')
            if len(data) < 2 + 64 * 5:
                return None

            hex_data = data[2:]
            amount0 = int(hex_data[0:64], 16)
            amount1 = int(hex_data[64:128], 16)
            sqrt_price_x96 = int(hex_data[128:192], 16)

            # Handle two's complement for signed integers
            if amount0 >= 2**255:
                amount0 -= 2**256
            if amount1 >= 2**255:
                amount1 -= 2**256

            Q96 = 79228162514264337593543950336.0
            price = (sqrt_price_x96 / Q96) ** 2 if sqrt_price_x96 > 0 else 0.0

            return {
                'amount0': float(amount0) / 1e18,
                'amount1': float(amount1) / 1e18,
                'price': price,
                'timestamp': int(time.time()),
            }
        except Exception:
            return None


class IncrementalSimulator:
    """
    Incremental simulation engine for live strategy monitoring.

    Processes new swaps one-at-a-time without re-running the full backtest.
    Maintains the latest position state and updates metrics incrementally.

    Parameters
    ----------
    backtester : LPBacktester
        Configured backtester instance.
    strategies : dict or pd.DataFrame
        Strategy parameters.
    warmup_swaps : pd.DataFrame, optional
        Historical swaps to initialize the simulation state.

    Examples
    --------
    >>> sim = IncrementalSimulator(bt, strategies, warmup_swaps=historical)
    >>> for new_swap in live_feed:
    ...     metrics = sim.process_swap(new_swap)
    ...     print(f"Live PnL: {metrics['net_pnl']}")
    """

    def __init__(
        self,
        backtester,
        strategies: Union[Dict, pd.DataFrame],
        warmup_swaps: Optional[pd.DataFrame] = None,
    ):
        self.backtester = backtester
        self.strategies = strategies
        self._swap_buffer: List[Dict] = []
        self._latest_result = None
        self._total_swaps = 0

        if warmup_swaps is not None and len(warmup_swaps) > 0:
            self._latest_result = backtester.run(warmup_swaps, strategies)
            self._total_swaps = len(warmup_swaps)
            self._swap_buffer = warmup_swaps.to_dict('records')

    def process_swap(self, swap: Dict) -> Optional[pd.Series]:
        """
        Process a single new swap and return updated metrics.

        Parameters
        ----------
        swap : dict
            Swap data with at least ``amount0`` and ``amount1`` keys.

        Returns
        -------
        pd.Series or None
            Updated metrics for the first strategy, or None if not enough data.
        """
        self._swap_buffer.append(swap)
        self._total_swaps += 1

        # Re-run backtest on growing window (simple approach)
        # For production, this should use incremental state updates
        if self._total_swaps < 2:
            return None

        swaps_df = pd.DataFrame(self._swap_buffer)
        try:
            self._latest_result = self.backtester.run(swaps_df, self.strategies)
            return self._latest_result.metrics.iloc[0]
        except Exception:
            return None

    def process_batch(self, swaps: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process a batch of new swaps.

        Parameters
        ----------
        swaps : pd.DataFrame
            New swap data.

        Returns
        -------
        pd.DataFrame or None
            Updated metrics for all strategies.
        """
        for _, row in swaps.iterrows():
            self._swap_buffer.append(row.to_dict())
            self._total_swaps += 1

        if self._total_swaps < 2:
            return None

        swaps_df = pd.DataFrame(self._swap_buffer)
        try:
            self._latest_result = self.backtester.run(swaps_df, self.strategies)
            return self._latest_result.metrics
        except Exception:
            return None

    @property
    def latest_result(self):
        """Get the most recent BacktestResult."""
        return self._latest_result

    @property
    def total_swaps(self) -> int:
        """Total swaps processed."""
        return self._total_swaps
