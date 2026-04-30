"""
Data pipeline for Aave v3 integration and historical price feeds.

This module provides:
- Aave v3 subgraph client for protocol data
- Historical price feeds for calibration
- Data caching utilities
"""

from crypto_fht.data.aave_client import (
    AaveV3Client,
    ReserveData,
    UserPositionData,
)
from crypto_fht.data.price_feeds import (
    PriceFeedClient,
    HistoricalPriceData,
)
from crypto_fht.data.cache import DataCache

__all__ = [
    "AaveV3Client",
    "ReserveData",
    "UserPositionData",
    "PriceFeedClient",
    "HistoricalPriceData",
    "DataCache",
]
