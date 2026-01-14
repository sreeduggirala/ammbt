"""Global configuration settings."""

from typing import Dict, Any


class Config:
    """
    Global configuration for AMMBT.

    Similar to vectorbt's config system.
    """

    _config: Dict[str, Any] = {
        # Precision settings
        "precision": {
            "price_decimals": 18,
            "fee_decimals": 6,
        },
        # Default pool parameters
        "defaults": {
            "v2_fee": 0.003,  # 0.3%
            "v3_fee_tiers": [500, 3000, 10000],  # 0.05%, 0.3%, 1%
            "v3_tick_spacing": {500: 10, 3000: 60, 10000: 200},
        },
        # Performance settings
        "numba": {
            "parallel": False,  # Enable parallel compilation
            "cache": True,  # Cache compiled functions
            "fastmath": True,  # Enable fast math optimizations
        },
        # Plotting settings
        "plotting": {
            "theme": "plotly_dark",
            "width": 1200,
            "height": 600,
        },
    }

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = cls._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split(".")
        config = cls._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @classmethod
    def reset(cls) -> None:
        """Reset to default configuration."""
        cls._config = {
            "precision": {
                "price_decimals": 18,
                "fee_decimals": 6,
            },
            "defaults": {
                "v2_fee": 0.003,
                "v3_fee_tiers": [500, 3000, 10000],
                "v3_tick_spacing": {500: 10, 3000: 60, 10000: 200},
            },
            "numba": {
                "parallel": False,
                "cache": True,
                "fastmath": True,
            },
            "plotting": {
                "theme": "plotly_dark",
                "width": 1200,
                "height": 600,
            },
        }
