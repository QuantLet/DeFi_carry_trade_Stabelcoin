"""
Health factor dynamics for DeFi lending positions.

The health factor H is defined as:
    H = Σ(collateral_i × price_i × LT_i) / Σ(debt_j × price_j)

where LT_i is the liquidation threshold for asset i.

Liquidation occurs when H < 1.

The log-health factor X = log(H) follows a spectrally negative Lévy process
due to correlated price movements and sudden market crashes (jumps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from crypto_fht.core.levy_process import LevyParameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Position:
    """A single DeFi position (collateral or debt).

    Attributes:
        asset: Asset identifier (e.g., "ETH", "WBTC", "USDC").
        amount: Quantity of the asset.
        is_collateral: True if this is collateral, False if debt.
        ltv: Loan-to-value ratio (for collateral assets).
        liquidation_threshold: Liquidation threshold (for collateral assets).
        price: Current price in USD (optional, can be set later).
    """

    asset: str
    amount: float
    is_collateral: bool
    ltv: float = 0.0
    liquidation_threshold: float = 0.0
    price: float = 0.0

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f"Amount must be non-negative, got {self.amount}")
        if self.is_collateral:
            if not 0 <= self.ltv <= 1:
                raise ValueError(f"LTV must be in [0, 1], got {self.ltv}")
            if not 0 <= self.liquidation_threshold <= 1:
                raise ValueError(
                    f"Liquidation threshold must be in [0, 1], got {self.liquidation_threshold}"
                )

    @property
    def value(self) -> float:
        """USD value of the position."""
        return self.amount * self.price

    @property
    def effective_collateral_value(self) -> float:
        """Collateral value weighted by liquidation threshold."""
        if not self.is_collateral:
            return 0.0
        return self.value * self.liquidation_threshold


@dataclass
class HealthFactor:
    """Health factor calculator for a portfolio of positions.

    The health factor is the key metric for DeFi lending protocol risk:
        H = Σ(collateral_i × price_i × LT_i) / Σ(debt_j × price_j)

    When H < 1, the position is subject to liquidation.

    Attributes:
        positions: List of positions in the portfolio.
        min_health_factor: Minimum safe health factor (default 1.0).

    Example:
        >>> eth_collateral = Position("ETH", 10.0, True, ltv=0.8, liquidation_threshold=0.825, price=2000)
        >>> usdc_debt = Position("USDC", 10000, False, price=1.0)
        >>> hf = HealthFactor([eth_collateral, usdc_debt])
        >>> hf.current_health_factor
        1.65
    """

    positions: list[Position] = field(default_factory=list)
    min_health_factor: float = 1.0

    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions.append(position)

    @property
    def collateral_positions(self) -> list[Position]:
        """Return all collateral positions."""
        return [p for p in self.positions if p.is_collateral]

    @property
    def debt_positions(self) -> list[Position]:
        """Return all debt positions."""
        return [p for p in self.positions if not p.is_collateral]

    @property
    def total_collateral_value(self) -> float:
        """Total USD value of collateral."""
        return sum(p.value for p in self.collateral_positions)

    @property
    def total_effective_collateral(self) -> float:
        """Total collateral value weighted by liquidation thresholds."""
        return sum(p.effective_collateral_value for p in self.collateral_positions)

    @property
    def total_debt_value(self) -> float:
        """Total USD value of debt."""
        return sum(p.value for p in self.debt_positions)

    @property
    def current_health_factor(self) -> float:
        """Compute current health factor.

        H = Σ(collateral × LT) / Σ(debt)

        Returns:
            Health factor. Returns inf if no debt.
        """
        debt = self.total_debt_value
        if debt == 0:
            return float("inf")
        return self.total_effective_collateral / debt

    @property
    def log_health_factor(self) -> float:
        """Compute log of health factor (for Lévy process modeling)."""
        hf = self.current_health_factor
        if hf <= 0:
            return float("-inf")
        return np.log(hf)

    @property
    def is_healthy(self) -> bool:
        """Check if position is above liquidation threshold."""
        return self.current_health_factor >= self.min_health_factor

    @property
    def distance_to_liquidation(self) -> float:
        """Distance to liquidation in log-space.

        This is x - b where x = log(HF) and b = log(min_HF).
        """
        return self.log_health_factor - np.log(self.min_health_factor)

    def compute_health_factor_with_prices(
        self, prices: dict[str, float]
    ) -> float:
        """Compute health factor with updated prices.

        Args:
            prices: Dictionary mapping asset names to USD prices.

        Returns:
            Health factor with new prices.
        """
        effective_collateral = sum(
            p.amount * prices.get(p.asset, p.price) * p.liquidation_threshold
            for p in self.collateral_positions
        )
        debt = sum(
            p.amount * prices.get(p.asset, p.price) for p in self.debt_positions
        )

        if debt == 0:
            return float("inf")
        return effective_collateral / debt

    def available_to_borrow(self, asset: str, asset_ltv: float, asset_price: float) -> float:
        """Calculate how much more can be borrowed.

        Based on current collateral and existing debt.

        Args:
            asset: Asset to borrow.
            asset_ltv: LTV ratio for the asset.
            asset_price: Price of the asset.

        Returns:
            Maximum additional borrow amount in asset units.
        """
        # Max borrow = Σ(collateral × LTV) - current_debt
        max_borrow_usd = sum(
            p.value * p.ltv for p in self.collateral_positions
        ) - self.total_debt_value

        if max_borrow_usd <= 0 or asset_price <= 0:
            return 0.0

        return max_borrow_usd / asset_price


@dataclass
class HealthFactorDynamics:
    """Model health factor dynamics as a Lévy process.

    Maps portfolio characteristics to Lévy process parameters for
    first-hitting time analysis.

    The log-health factor X_t = log(H_t) follows:
        dX_t = μ dt + σ dW_t - dJ_t

    where the parameters depend on:
    - Portfolio composition
    - Asset volatilities and correlations
    - Jump characteristics (from historical data)

    Attributes:
        health_factor: Current portfolio health factor.
        asset_params: Lévy parameters for each asset.
        correlation_matrix: Asset return correlations.
    """

    health_factor: HealthFactor
    asset_params: dict[str, LevyParameters] = field(default_factory=dict)
    correlation_matrix: NDArray[np.floating] | None = None

    def set_asset_params(self, asset: str, params: LevyParameters) -> None:
        """Set Lévy parameters for an asset."""
        self.asset_params[asset] = params

    def get_portfolio_weights(self) -> dict[str, float]:
        """Compute portfolio weights by USD exposure.

        Returns:
            Dictionary mapping asset names to weights.
        """
        total_exposure = (
            self.health_factor.total_collateral_value
            + self.health_factor.total_debt_value
        )

        if total_exposure == 0:
            return {}

        weights = {}
        for p in self.health_factor.positions:
            if p.asset in weights:
                weights[p.asset] += p.value / total_exposure
            else:
                weights[p.asset] = p.value / total_exposure

        return weights

    def aggregate_levy_params(self) -> LevyParameters:
        """Aggregate Lévy parameters across the portfolio.

        Combines individual asset parameters weighted by exposure.
        This is an approximation that works well for moderate correlations.

        The aggregation:
        - Drift μ: weighted average
        - Volatility σ: portfolio volatility (depends on correlations)
        - Jump intensity λ: combined (sum) for worst-case
        - Jump rate η: weighted average
        - Shift δ: weighted average

        Returns:
            Aggregated LevyParameters for the portfolio.
        """
        weights = self.get_portfolio_weights()

        if not weights or not self.asset_params:
            # Default parameters if no calibration available
            return LevyParameters(mu=0.0, sigma=0.3, lambda_=1.0, eta=5.0, delta=0.02)

        # Weighted drift
        mu = sum(
            weights.get(asset, 0) * params.mu
            for asset, params in self.asset_params.items()
        )

        # Portfolio volatility
        if self.correlation_matrix is not None:
            # Use correlation matrix
            assets = list(self.asset_params.keys())
            n = len(assets)
            sigma_vec = np.array([self.asset_params[a].sigma for a in assets])
            w_vec = np.array([weights.get(a, 0) for a in assets])

            # Portfolio variance: w' Σ w where Σ = diag(σ) @ ρ @ diag(σ)
            cov_matrix = np.diag(sigma_vec) @ self.correlation_matrix[:n, :n] @ np.diag(sigma_vec)
            portfolio_var = w_vec @ cov_matrix @ w_vec
            sigma = np.sqrt(max(portfolio_var, 0))
        else:
            # Assume correlation = 1 (worst case)
            sigma_weighted = sum(
                weights.get(asset, 0) * params.sigma
                for asset, params in self.asset_params.items()
            )
            sigma = sigma_weighted

        # Combined jump intensity (sum for systemic risk)
        lambda_ = sum(
            weights.get(asset, 0) * params.lambda_
            for asset, params in self.asset_params.items()
        )

        # Weighted average jump parameters
        total_weight = sum(weights.get(a, 0) for a in self.asset_params.keys())
        if total_weight > 0:
            eta = sum(
                weights.get(asset, 0) * params.eta
                for asset, params in self.asset_params.items()
            ) / total_weight

            delta = sum(
                weights.get(asset, 0) * params.delta
                for asset, params in self.asset_params.items()
            ) / total_weight
        else:
            eta = 5.0
            delta = 0.02

        return LevyParameters(
            mu=mu,
            sigma=max(sigma, 0.01),
            lambda_=max(lambda_, 0.01),
            eta=max(eta, 0.1),
            delta=max(delta, 0.0),
        )

    def health_factor_at_time(
        self,
        t: float,
        price_changes: dict[str, float],
    ) -> float:
        """Compute hypothetical health factor after price changes.

        Args:
            t: Time horizon (not directly used, for API consistency).
            price_changes: Dictionary of asset -> multiplicative price change.
                          e.g., {"ETH": 0.9} means ETH price drops 10%.

        Returns:
            Health factor after applying price changes.
        """
        new_prices = {}
        for p in self.health_factor.positions:
            change = price_changes.get(p.asset, 1.0)
            new_prices[p.asset] = p.price * change

        return self.health_factor.compute_health_factor_with_prices(new_prices)
