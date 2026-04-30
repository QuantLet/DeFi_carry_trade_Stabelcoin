"""
Aave v3 protocol data client.

Fetches data from:
1. Aave subgraph (historical data, positions, liquidations)
2. On-chain data (current state)

Supports Ethereum mainnet with extensibility for other networks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp


@dataclass
class ReserveData:
    """Aave v3 reserve (asset) data.

    Attributes:
        asset: Underlying asset address.
        symbol: Asset symbol (e.g., "WETH", "USDC").
        decimals: Token decimals.
        ltv: Loan-to-value ratio (as decimal, e.g., 0.80 for 80%).
        liquidation_threshold: Liquidation threshold (as decimal).
        liquidation_bonus: Liquidation bonus (as decimal, e.g., 0.05 for 5%).
        borrow_cap: Maximum borrow amount (None if unlimited).
        supply_cap: Maximum supply amount (None if unlimited).
        available_liquidity: Available liquidity in token units.
        total_debt: Total borrowed amount.
        utilization_rate: Current utilization (0-1).
        variable_borrow_rate: Variable borrow APR.
        stable_borrow_rate: Stable borrow APR.
        supply_apy: Supply APY.
        price_usd: Price in USD.
        is_active: Whether reserve is active.
        is_frozen: Whether reserve is frozen.
    """

    asset: str
    symbol: str
    decimals: int
    ltv: float
    liquidation_threshold: float
    liquidation_bonus: float
    borrow_cap: float | None
    supply_cap: float | None
    available_liquidity: float
    total_debt: float
    utilization_rate: float
    variable_borrow_rate: float
    stable_borrow_rate: float
    supply_apy: float
    price_usd: float
    is_active: bool = True
    is_frozen: bool = False


@dataclass
class UserPositionData:
    """User's position in Aave v3.

    Attributes:
        user_address: Ethereum address.
        total_collateral_usd: Total collateral value in USD.
        total_debt_usd: Total debt value in USD.
        available_borrows_usd: Available borrowing capacity in USD.
        health_factor: Current health factor.
        ltv: Current loan-to-value ratio.
        collateral_positions: List of collateral position details.
        debt_positions: List of debt position details.
    """

    user_address: str
    total_collateral_usd: float
    total_debt_usd: float
    available_borrows_usd: float
    health_factor: float
    ltv: float
    collateral_positions: list[dict[str, Any]] = field(default_factory=list)
    debt_positions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiquidationEvent:
    """Historical liquidation event.

    Attributes:
        timestamp: Unix timestamp of liquidation.
        block_number: Block number.
        user: Liquidated user address.
        liquidator: Liquidator address.
        collateral_asset: Collateral asset symbol.
        debt_asset: Debt asset symbol.
        collateral_amount: Seized collateral amount.
        debt_amount: Repaid debt amount.
        collateral_price_usd: Collateral price at liquidation.
        debt_price_usd: Debt price at liquidation.
    """

    timestamp: int
    block_number: int
    user: str
    liquidator: str
    collateral_asset: str
    debt_asset: str
    collateral_amount: float
    debt_amount: float
    collateral_price_usd: float = 0.0
    debt_price_usd: float = 0.0


class AaveV3Client:
    """Client for Aave v3 protocol data.

    Fetches data from Aave subgraphs and on-chain sources.

    Attributes:
        network: Network identifier ("mainnet", "arbitrum", etc.).
        subgraph_url: GraphQL endpoint.

    Example:
        >>> async with AaveV3Client() as client:
        ...     reserves = await client.get_reserves()
        ...     for r in reserves[:3]:
        ...         print(f"{r.symbol}: LTV={r.ltv:.0%}, LT={r.liquidation_threshold:.0%}")
    """

    SUBGRAPH_URLS = {
        "mainnet": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
        "arbitrum": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-arbitrum",
        "optimism": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-optimism",
        "polygon": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-polygon",
        "avalanche": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-avalanche",
    }

    def __init__(self, network: str = "mainnet") -> None:
        """Initialize Aave client.

        Args:
            network: Network to connect to.
        """
        self.network = network
        self.subgraph_url = self.SUBGRAPH_URLS.get(
            network, self.SUBGRAPH_URLS["mainnet"]
        )
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AaveV3Client":
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_reserves(self) -> list[ReserveData]:
        """Fetch all active reserve data.

        Returns:
            List of ReserveData for all active reserves.
        """
        query = """
        {
            reserves(where: {isActive: true}) {
                id
                symbol
                name
                decimals
                baseLTVasCollateral
                reserveLiquidationThreshold
                reserveLiquidationBonus
                borrowingEnabled
                usageAsCollateralEnabled
                borrowCap
                supplyCap
                availableLiquidity
                totalCurrentVariableDebt
                totalPrincipalStableDebt
                utilizationRate
                variableBorrowRate
                stableBorrowRate
                liquidityRate
                price {
                    priceInEth
                }
                underlyingAsset
                aToken {
                    id
                }
                isFrozen
            }
        }
        """

        result = await self._execute_query(query)
        reserves_raw = result.get("data", {}).get("reserves", [])

        reserves = []
        for r in reserves_raw:
            try:
                reserve = self._parse_reserve(r)
                reserves.append(reserve)
            except (KeyError, ValueError, TypeError):
                # Skip malformed reserves
                continue

        return reserves

    async def get_reserve(self, symbol: str) -> ReserveData | None:
        """Fetch data for a specific reserve.

        Args:
            symbol: Asset symbol (e.g., "WETH").

        Returns:
            ReserveData or None if not found.
        """
        reserves = await self.get_reserves()
        for reserve in reserves:
            if reserve.symbol.upper() == symbol.upper():
                return reserve
        return None

    async def get_user_position(self, user_address: str) -> UserPositionData:
        """Fetch user's current position.

        Args:
            user_address: Ethereum address.

        Returns:
            UserPositionData with all positions.
        """
        query = """
        query GetUserPosition($user: String!) {
            userReserves(where: {user: $user}) {
                reserve {
                    symbol
                    underlyingAsset
                    decimals
                    baseLTVasCollateral
                    reserveLiquidationThreshold
                    price {
                        priceInEth
                    }
                }
                currentATokenBalance
                currentVariableDebt
                currentStableDebt
                usageAsCollateralEnabledOnUser
            }
        }
        """

        result = await self._execute_query(
            query, {"user": user_address.lower()}
        )
        user_reserves = result.get("data", {}).get("userReserves", [])

        return self._parse_user_position(user_address, user_reserves)

    async def get_liquidation_events(
        self,
        start_time: datetime,
        end_time: datetime,
        asset: str | None = None,
        limit: int = 1000,
    ) -> list[LiquidationEvent]:
        """Fetch historical liquidation events.

        Args:
            start_time: Start of period.
            end_time: End of period.
            asset: Optional filter by collateral asset symbol.
            limit: Maximum number of events.

        Returns:
            List of liquidation events.
        """
        where_clause = f"timestamp_gte: {int(start_time.timestamp())}, timestamp_lte: {int(end_time.timestamp())}"
        if asset:
            where_clause += f', collateralReserve_: {{symbol: "{asset}"}}'

        query = f"""
        {{
            liquidationCalls(
                where: {{{where_clause}}}
                orderBy: timestamp
                orderDirection: asc
                first: {limit}
            ) {{
                id
                timestamp
                collateralReserve {{
                    symbol
                }}
                principalReserve {{
                    symbol
                }}
                collateralAmount
                principalAmount
                liquidator
                user {{
                    id
                }}
            }}
        }}
        """

        result = await self._execute_query(query)
        events_raw = result.get("data", {}).get("liquidationCalls", [])

        events = []
        for e in events_raw:
            try:
                event = LiquidationEvent(
                    timestamp=int(e["timestamp"]),
                    block_number=0,  # Not always available
                    user=e["user"]["id"],
                    liquidator=e["liquidator"],
                    collateral_asset=e["collateralReserve"]["symbol"],
                    debt_asset=e["principalReserve"]["symbol"],
                    collateral_amount=float(e["collateralAmount"]),
                    debt_amount=float(e["principalAmount"]),
                )
                events.append(event)
            except (KeyError, ValueError, TypeError):
                continue

        return events

    async def get_historical_rates(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "hourly",
    ) -> list[dict[str, Any]]:
        """Fetch historical interest rates for a reserve.

        Args:
            symbol: Asset symbol.
            start_time: Start time.
            end_time: End time.
            interval: "hourly" or "daily".

        Returns:
            List of rate records with timestamps.
        """
        # Note: Historical rates may require a different data source
        # This is a placeholder for the API structure
        return []

    async def _execute_query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute GraphQL query against subgraph."""
        if self._session is None:
            raise RuntimeError("Client not initialized. Use 'async with'.")

        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        async with self._session.post(
            self.subgraph_url, json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _parse_reserve(self, raw: dict[str, Any]) -> ReserveData:
        """Parse raw reserve data from subgraph."""
        # Price conversion: priceInEth to USD (assuming ETH = $2000 as placeholder)
        # In production, fetch ETH/USD price from oracle
        eth_price_usd = 2000.0
        price_in_eth = float(raw.get("price", {}).get("priceInEth", 0)) / 1e18
        price_usd = price_in_eth * eth_price_usd

        return ReserveData(
            asset=raw["underlyingAsset"],
            symbol=raw["symbol"],
            decimals=int(raw["decimals"]),
            ltv=float(raw["baseLTVasCollateral"]) / 10000,
            liquidation_threshold=float(raw["reserveLiquidationThreshold"]) / 10000,
            liquidation_bonus=float(raw["reserveLiquidationBonus"]) / 10000 - 1,
            borrow_cap=float(raw["borrowCap"]) if raw.get("borrowCap") else None,
            supply_cap=float(raw["supplyCap"]) if raw.get("supplyCap") else None,
            available_liquidity=float(raw["availableLiquidity"]) / (10 ** int(raw["decimals"])),
            total_debt=(
                float(raw.get("totalCurrentVariableDebt", 0))
                + float(raw.get("totalPrincipalStableDebt", 0))
            ) / (10 ** int(raw["decimals"])),
            utilization_rate=float(raw.get("utilizationRate", 0)) / 1e27,
            variable_borrow_rate=float(raw.get("variableBorrowRate", 0)) / 1e27,
            stable_borrow_rate=float(raw.get("stableBorrowRate", 0)) / 1e27,
            supply_apy=float(raw.get("liquidityRate", 0)) / 1e27,
            price_usd=price_usd,
            is_active=True,
            is_frozen=raw.get("isFrozen", False),
        )

    def _parse_user_position(
        self,
        user_address: str,
        raw_reserves: list[dict[str, Any]],
    ) -> UserPositionData:
        """Parse user position data from subgraph response."""
        collateral_positions = []
        debt_positions = []
        total_collateral_usd = 0.0
        total_debt_usd = 0.0
        weighted_ltv = 0.0
        weighted_lt = 0.0

        eth_price_usd = 2000.0  # Placeholder

        for ur in raw_reserves:
            reserve = ur["reserve"]
            decimals = int(reserve["decimals"])
            price_in_eth = float(reserve.get("price", {}).get("priceInEth", 0)) / 1e18
            price_usd = price_in_eth * eth_price_usd

            # Collateral
            atoken_balance = float(ur.get("currentATokenBalance", 0)) / (10 ** decimals)
            if atoken_balance > 0 and ur.get("usageAsCollateralEnabledOnUser"):
                value = atoken_balance * price_usd
                total_collateral_usd += value
                ltv = float(reserve["baseLTVasCollateral"]) / 10000
                lt = float(reserve["reserveLiquidationThreshold"]) / 10000
                weighted_ltv += value * ltv
                weighted_lt += value * lt

                collateral_positions.append({
                    "symbol": reserve["symbol"],
                    "amount": atoken_balance,
                    "value_usd": value,
                    "ltv": ltv,
                    "liquidation_threshold": lt,
                })

            # Debt
            variable_debt = float(ur.get("currentVariableDebt", 0)) / (10 ** decimals)
            stable_debt = float(ur.get("currentStableDebt", 0)) / (10 ** decimals)
            total_debt = variable_debt + stable_debt

            if total_debt > 0:
                value = total_debt * price_usd
                total_debt_usd += value

                debt_positions.append({
                    "symbol": reserve["symbol"],
                    "amount": total_debt,
                    "value_usd": value,
                    "type": "variable" if variable_debt > stable_debt else "stable",
                })

        # Compute health factor
        if total_debt_usd > 0 and total_collateral_usd > 0:
            health_factor = weighted_lt / total_debt_usd
            avg_ltv = weighted_ltv / total_collateral_usd
        else:
            health_factor = float("inf")
            avg_ltv = 0.0

        available_borrows = max(0, total_collateral_usd * avg_ltv - total_debt_usd)

        return UserPositionData(
            user_address=user_address,
            total_collateral_usd=total_collateral_usd,
            total_debt_usd=total_debt_usd,
            available_borrows_usd=available_borrows,
            health_factor=health_factor,
            ltv=avg_ltv,
            collateral_positions=collateral_positions,
            debt_positions=debt_positions,
        )


def run_sync(coro: Any) -> Any:
    """Run async coroutine synchronously.

    Utility for using async client in sync code.

    Args:
        coro: Coroutine to run.

    Returns:
        Coroutine result.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
