"""
Portfolio state management for DeFi positions.

Tracks positions, values, and provides utilities for portfolio operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PortfolioPosition:
    """A single position in the portfolio.

    Attributes:
        asset: Asset symbol.
        amount: Quantity held.
        is_collateral: True if deposited as collateral.
        is_debt: True if borrowed.
        entry_price: Price at position entry.
        current_price: Current market price.
        timestamp: When position was opened.
    """

    asset: str
    amount: float
    is_collateral: bool = True
    is_debt: bool = False
    entry_price: float = 0.0
    current_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def value(self) -> float:
        """Current USD value."""
        return self.amount * self.current_price

    @property
    def entry_value(self) -> float:
        """Value at entry."""
        return self.amount * self.entry_price

    @property
    def pnl(self) -> float:
        """Profit/loss since entry."""
        return self.value - self.entry_value

    @property
    def pnl_percent(self) -> float:
        """Percentage profit/loss."""
        if self.entry_value == 0:
            return 0.0
        return self.pnl / self.entry_value


@dataclass
class Portfolio:
    """Portfolio of DeFi positions.

    Manages collateral and debt positions with pricing updates.

    Attributes:
        positions: List of positions.
        base_currency: Base currency for value calculations.
        created_at: Portfolio creation timestamp.

    Example:
        >>> portfolio = Portfolio()
        >>> portfolio.add_collateral("WETH", 10.0, price=2000.0)
        >>> portfolio.add_debt("USDC", 10000.0, price=1.0)
        >>> print(f"Health Factor: {portfolio.health_factor:.2f}")
    """

    positions: list[PortfolioPosition] = field(default_factory=list)
    base_currency: str = "USD"
    created_at: datetime = field(default_factory=datetime.now)

    def add_collateral(
        self,
        asset: str,
        amount: float,
        price: float,
        ltv: float = 0.8,
        liquidation_threshold: float = 0.825,
    ) -> None:
        """Add collateral position.

        Args:
            asset: Asset symbol.
            amount: Quantity to deposit.
            price: Current price.
            ltv: Loan-to-value ratio (stored in metadata).
            liquidation_threshold: Liquidation threshold.
        """
        position = PortfolioPosition(
            asset=asset,
            amount=amount,
            is_collateral=True,
            is_debt=False,
            entry_price=price,
            current_price=price,
        )
        self.positions.append(position)

    def add_debt(self, asset: str, amount: float, price: float) -> None:
        """Add debt position.

        Args:
            asset: Asset symbol.
            amount: Quantity borrowed.
            price: Current price.
        """
        position = PortfolioPosition(
            asset=asset,
            amount=amount,
            is_collateral=False,
            is_debt=True,
            entry_price=price,
            current_price=price,
        )
        self.positions.append(position)

    @property
    def collateral_positions(self) -> list[PortfolioPosition]:
        """Get all collateral positions."""
        return [p for p in self.positions if p.is_collateral]

    @property
    def debt_positions(self) -> list[PortfolioPosition]:
        """Get all debt positions."""
        return [p for p in self.positions if p.is_debt]

    @property
    def total_collateral_value(self) -> float:
        """Total USD value of collateral."""
        return sum(p.value for p in self.collateral_positions)

    @property
    def total_debt_value(self) -> float:
        """Total USD value of debt."""
        return sum(p.value for p in self.debt_positions)

    @property
    def net_value(self) -> float:
        """Net portfolio value (collateral - debt)."""
        return self.total_collateral_value - self.total_debt_value

    @property
    def health_factor(self) -> float:
        """Compute health factor.

        Simplified: assumes 0.825 liquidation threshold for all assets.
        """
        if self.total_debt_value == 0:
            return float("inf")
        return (self.total_collateral_value * 0.825) / self.total_debt_value

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update position prices.

        Args:
            prices: Dictionary mapping asset symbols to new prices.
        """
        for position in self.positions:
            if position.asset in prices:
                position.current_price = prices[position.asset]

    def get_weights(self) -> dict[str, float]:
        """Get portfolio weights by asset.

        Returns:
            Dictionary mapping asset to weight (by value).
        """
        total = self.total_collateral_value + self.total_debt_value
        if total == 0:
            return {}

        weights = {}
        for p in self.positions:
            sign = 1 if p.is_collateral else -1
            if p.asset in weights:
                weights[p.asset] += sign * p.value / total
            else:
                weights[p.asset] = sign * p.value / total

        return weights

    def to_weight_array(self, assets: list[str]) -> NDArray[np.floating]:
        """Convert portfolio to weight array.

        Args:
            assets: Ordered list of asset symbols.

        Returns:
            Array of weights in order of assets list.
        """
        weights = self.get_weights()
        return np.array([weights.get(a, 0.0) for a in assets])

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
    ) -> list[tuple[str, float, str]]:
        """Calculate trades needed to reach target weights.

        Args:
            target_weights: Target weight per asset.
            prices: Current prices.

        Returns:
            List of (asset, amount, action) tuples.
            action is "buy", "sell", "deposit", or "withdraw".
        """
        self.update_prices(prices)
        current_weights = self.get_weights()
        total_value = self.total_collateral_value + self.total_debt_value

        trades = []
        for asset, target_w in target_weights.items():
            current_w = current_weights.get(asset, 0.0)
            diff_w = target_w - current_w
            diff_value = diff_w * total_value
            price = prices.get(asset, 1.0)
            amount = abs(diff_value / price)

            if abs(diff_w) < 0.001:
                continue

            if diff_w > 0:
                action = "deposit" if target_w > 0 else "repay"
            else:
                action = "withdraw" if current_w > 0 else "borrow"

            trades.append((asset, amount, action))

        return trades

    def clone(self) -> "Portfolio":
        """Create a copy of the portfolio."""
        new_portfolio = Portfolio(base_currency=self.base_currency)
        for p in self.positions:
            new_portfolio.positions.append(
                PortfolioPosition(
                    asset=p.asset,
                    amount=p.amount,
                    is_collateral=p.is_collateral,
                    is_debt=p.is_debt,
                    entry_price=p.entry_price,
                    current_price=p.current_price,
                    timestamp=p.timestamp,
                )
            )
        return new_portfolio

    def summary(self) -> str:
        """Return portfolio summary string."""
        lines = [
            "Portfolio Summary",
            "=" * 40,
            f"Collateral: ${self.total_collateral_value:,.2f}",
            f"Debt:       ${self.total_debt_value:,.2f}",
            f"Net Value:  ${self.net_value:,.2f}",
            f"Health Factor: {self.health_factor:.2f}",
            "",
            "Positions:",
        ]

        for p in self.positions:
            ptype = "Collateral" if p.is_collateral else "Debt"
            lines.append(
                f"  {p.asset}: {p.amount:.4f} @ ${p.current_price:.2f} "
                f"= ${p.value:,.2f} ({ptype})"
            )

        return "\n".join(lines)
