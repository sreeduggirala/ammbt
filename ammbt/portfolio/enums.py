"""Enumerations for portfolio events and states."""

from enum import IntEnum


class EventType(IntEnum):
    """Types of events that can occur during simulation."""

    SWAP = 0  # Swap executed in pool
    MINT = 1  # Position created/liquidity added
    BURN = 2  # Position closed/liquidity removed
    COLLECT = 3  # Fees collected
    REBALANCE = 4  # Position rebalanced


class PositionStatus(IntEnum):
    """Status of an LP position."""

    INACTIVE = 0  # Position not yet created or already closed
    ACTIVE = 1  # Position is live and potentially earning fees
    OUT_OF_RANGE = 2  # Position is active but price is out of range (v3/DLMM)
    IN_RANGE = 3  # Position is active and in range (v3/DLMM)
