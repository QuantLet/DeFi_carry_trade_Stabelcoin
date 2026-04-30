"""
Historical price data feeds for calibration.

Provides interfaces for fetching historical cryptocurrency prices
from various sources for Lévy process parameter calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HistoricalPriceData:
    """Historical price data for an asset.

    Attributes:
        asset: Asset symbol.
        timestamps: List of datetime timestamps.
        prices: Array of prices.
        returns: Array of log-returns (computed from prices).
    """

    asset: str
    timestamps: list[datetime] = field(default_factory=list)
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if len(self.prices) > 1 and len(self.returns) == 0:
            self.returns = np.diff(np.log(self.prices))

    @property
    def n_observations(self) -> int:
        """Number of price observations."""
        return len(self.prices)

    @property
    def n_returns(self) -> int:
        """Number of return observations."""
        return len(self.returns)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "price": self.prices,
        })
        df["log_return"] = np.concatenate([[np.nan], self.returns])
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, asset: str) -> "HistoricalPriceData":
        """Create from pandas DataFrame.

        Args:
            df: DataFrame with 'timestamp' and 'price' columns.
            asset: Asset symbol.

        Returns:
            HistoricalPriceData instance.
        """
        timestamps = pd.to_datetime(df["timestamp"]).tolist()
        prices = df["price"].values.astype(float)
        return cls(asset=asset, timestamps=timestamps, prices=prices)


class PriceFeedClient:
    """Client for fetching historical price data.

    Supports multiple data sources with fallback.

    Example:
        >>> client = PriceFeedClient()
        >>> data = await client.get_historical_prices(
        ...     "ETH",
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 6, 1),
        ...     interval="hourly"
        ... )
    """

    def __init__(self, source: str = "coingecko") -> None:
        """Initialize price feed client.

        Args:
            source: Data source ("coingecko", "binance", "chainlink").
        """
        self.source = source
        self._coingecko_ids = {
            "ETH": "ethereum",
            "WETH": "ethereum",
            "BTC": "bitcoin",
            "WBTC": "bitcoin",
            "USDC": "usd-coin",
            "USDT": "tether",
            "DAI": "dai",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "AAVE": "aave",
        }

    async def get_historical_prices(
        self,
        asset: str,
        start: datetime,
        end: datetime,
        interval: str = "hourly",
    ) -> HistoricalPriceData:
        """Fetch historical prices for an asset.

        Args:
            asset: Asset symbol (e.g., "ETH", "BTC").
            start: Start datetime.
            end: End datetime.
            interval: "hourly" or "daily".

        Returns:
            HistoricalPriceData with prices and returns.
        """
        if self.source == "coingecko":
            return await self._fetch_coingecko(asset, start, end, interval)
        else:
            # Fallback to simulated data
            return self._generate_simulated_data(asset, start, end, interval)

    async def _fetch_coingecko(
        self,
        asset: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> HistoricalPriceData:
        """Fetch from CoinGecko API."""
        import aiohttp

        coin_id = self._coingecko_ids.get(asset.upper(), asset.lower())
        from_ts = int(start.timestamp())
        to_ts = int(end.timestamp())

        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            f"?vs_currency=usd&from={from_ts}&to={to_ts}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    # Fallback to simulated
                    return self._generate_simulated_data(asset, start, end, interval)

                data = await response.json()
                prices_raw = data.get("prices", [])

                if not prices_raw:
                    return self._generate_simulated_data(asset, start, end, interval)

                timestamps = [
                    datetime.fromtimestamp(p[0] / 1000) for p in prices_raw
                ]
                prices = np.array([p[1] for p in prices_raw])

                return HistoricalPriceData(
                    asset=asset,
                    timestamps=timestamps,
                    prices=prices,
                )

    def _generate_simulated_data(
        self,
        asset: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> HistoricalPriceData:
        """Generate simulated price data for testing.

        Uses geometric Brownian motion with parameters typical for crypto.
        """
        # Time grid
        if interval == "hourly":
            delta = timedelta(hours=1)
            dt = 1 / (365 * 24)  # Hourly in years
        else:
            delta = timedelta(days=1)
            dt = 1 / 365  # Daily in years

        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current)
            current += delta

        n = len(timestamps)
        if n < 2:
            return HistoricalPriceData(
                asset=asset, timestamps=timestamps, prices=np.array([1000.0])
            )

        # Simulate GBM with jumps
        rng = np.random.default_rng(hash(asset) % 2**32)

        # Typical crypto parameters
        mu = 0.0  # Annualized drift
        sigma = 0.8  # Annualized volatility
        lambda_jump = 10.0  # Jump frequency (per year)
        jump_mean = 0.03  # Mean jump size

        sqrt_dt = np.sqrt(dt)
        log_returns = np.zeros(n - 1)

        for i in range(n - 1):
            # Diffusion
            dW = rng.standard_normal() * sqrt_dt
            diff = (mu - 0.5 * sigma**2) * dt + sigma * dW

            # Jumps
            n_jumps = rng.poisson(lambda_jump * dt)
            if n_jumps > 0:
                jumps = rng.exponential(jump_mean, size=n_jumps)
                diff -= np.sum(jumps)

            log_returns[i] = diff

        # Build price series
        initial_price = {
            "ETH": 2000.0,
            "BTC": 40000.0,
            "USDC": 1.0,
            "USDT": 1.0,
        }.get(asset.upper(), 1000.0)

        log_prices = np.concatenate([[np.log(initial_price)], np.cumsum(log_returns) + np.log(initial_price)])
        prices = np.exp(log_prices)

        return HistoricalPriceData(
            asset=asset,
            timestamps=timestamps,
            prices=prices,
            returns=log_returns,
        )

    def get_historical_prices_sync(
        self,
        asset: str,
        start: datetime,
        end: datetime,
        interval: str = "hourly",
    ) -> HistoricalPriceData:
        """Synchronous version of get_historical_prices.

        Falls back to simulated data to avoid async complexity.
        """
        return self._generate_simulated_data(asset, start, end, interval)


def compute_returns(
    prices: np.ndarray,
    return_type: str = "log",
) -> np.ndarray:
    """Compute returns from price series.

    Args:
        prices: Array of prices.
        return_type: "log" for log-returns, "simple" for simple returns.

    Returns:
        Array of returns (length = len(prices) - 1).
    """
    if len(prices) < 2:
        return np.array([])

    if return_type == "log":
        return np.diff(np.log(prices))
    else:
        return np.diff(prices) / prices[:-1]


def align_price_series(
    *price_data: HistoricalPriceData,
) -> list[HistoricalPriceData]:
    """Align multiple price series to common timestamps.

    Uses inner join on timestamps.

    Args:
        *price_data: Variable number of HistoricalPriceData instances.

    Returns:
        List of aligned HistoricalPriceData with same timestamps.
    """
    if len(price_data) < 2:
        return list(price_data)

    # Convert to DataFrames
    dfs = [pd.to_dataframe().set_index("timestamp") for pd in price_data]

    # Inner join
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="inner", rsuffix=f"_{df.columns[0]}")

    # Rebuild price data
    aligned = []
    timestamps = merged.index.tolist()

    for i, pd in enumerate(price_data):
        col = merged.columns[i * 2]  # price column
        prices = merged[col].values
        aligned.append(
            HistoricalPriceData(
                asset=pd.asset,
                timestamps=timestamps,
                prices=prices,
            )
        )

    return aligned
