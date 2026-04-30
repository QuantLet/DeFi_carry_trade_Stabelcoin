"""
Aave v3 protocol constraints for portfolio optimization.

Models the key constraints from Aave v3:
- LTV (Loan-to-Value) constraints per asset
- Liquidation threshold constraints
- Borrow and supply caps
- Health factor requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class AssetConstraints:
    """Constraints for a single asset in Aave v3.

    Attributes:
        symbol: Asset symbol.
        ltv: Loan-to-value ratio (0-1).
        liquidation_threshold: Liquidation threshold (0-1).
        liquidation_bonus: Liquidation bonus (0-1).
        borrow_cap: Maximum borrow amount (None = unlimited).
        supply_cap: Maximum supply amount (None = unlimited).
        can_be_collateral: Whether asset can be used as collateral.
        can_be_borrowed: Whether asset can be borrowed.
        price_usd: Current USD price.
    """

    symbol: str
    ltv: float
    liquidation_threshold: float
    liquidation_bonus: float = 0.05
    borrow_cap: float | None = None
    supply_cap: float | None = None
    can_be_collateral: bool = True
    can_be_borrowed: bool = True
    price_usd: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.ltv <= 1:
            raise ValueError(f"LTV must be in [0, 1], got {self.ltv}")
        if not 0 <= self.liquidation_threshold <= 1:
            raise ValueError(f"LT must be in [0, 1], got {self.liquidation_threshold}")
        if self.ltv > self.liquidation_threshold:
            raise ValueError(f"LTV ({self.ltv}) cannot exceed LT ({self.liquidation_threshold})")


@dataclass
class AaveV3Constraints:
    """Complete Aave v3 constraint specification.

    Handles:
    - Per-asset constraints (LTV, LT, caps)
    - Portfolio-level health factor requirement
    - E-mode parameters (optional)

    Attributes:
        assets: Dictionary of asset constraints.
        min_health_factor: Minimum required health factor.
        emode_category: E-mode category ID (None = normal mode).
        emode_ltv: E-mode LTV override.
        emode_liquidation_threshold: E-mode LT override.

    Example:
        >>> constraints = AaveV3Constraints()
        >>> constraints.add_asset(AssetConstraints(
        ...     symbol="WETH",
        ...     ltv=0.80,
        ...     liquidation_threshold=0.825,
        ...     price_usd=2000.0
        ... ))
    """

    assets: dict[str, AssetConstraints] = field(default_factory=dict)
    min_health_factor: float = 1.0
    emode_category: int | None = None
    emode_ltv: float | None = None
    emode_liquidation_threshold: float | None = None

    def add_asset(self, asset: AssetConstraints) -> None:
        """Add asset constraints."""
        self.assets[asset.symbol] = asset

    def get_effective_ltv(self, symbol: str) -> float:
        """Get effective LTV considering E-mode."""
        if self.emode_ltv is not None:
            return self.emode_ltv
        asset = self.assets.get(symbol)
        return asset.ltv if asset else 0.0

    def get_effective_lt(self, symbol: str) -> float:
        """Get effective liquidation threshold considering E-mode."""
        if self.emode_liquidation_threshold is not None:
            return self.emode_liquidation_threshold
        asset = self.assets.get(symbol)
        return asset.liquidation_threshold if asset else 0.0

    def max_borrow_amount(
        self,
        collateral_values: dict[str, float],
        current_debt: float = 0.0,
    ) -> float:
        """Calculate maximum additional borrow amount.

        Args:
            collateral_values: USD value of each collateral asset.
            current_debt: Existing debt in USD.

        Returns:
            Maximum additional borrow in USD.
        """
        total_borrow_power = sum(
            value * self.get_effective_ltv(symbol)
            for symbol, value in collateral_values.items()
            if symbol in self.assets and self.assets[symbol].can_be_collateral
        )
        return max(0, total_borrow_power - current_debt)

    def compute_health_factor(
        self,
        collateral_values: dict[str, float],
        debt_value: float,
    ) -> float:
        """Compute health factor for given position.

        H = Σ(collateral_i × LT_i) / debt

        Args:
            collateral_values: USD value per collateral asset.
            debt_value: Total debt in USD.

        Returns:
            Health factor. Returns inf if no debt.
        """
        if debt_value <= 0:
            return float("inf")

        effective_collateral = sum(
            value * self.get_effective_lt(symbol)
            for symbol, value in collateral_values.items()
            if symbol in self.assets
        )

        return effective_collateral / debt_value

    def is_position_safe(
        self,
        collateral_values: dict[str, float],
        debt_value: float,
    ) -> bool:
        """Check if position satisfies health factor requirement."""
        hf = self.compute_health_factor(collateral_values, debt_value)
        return hf >= self.min_health_factor


def build_constraint_matrices(
    constraints: AaveV3Constraints,
    assets: list[str],
    is_collateral: list[bool],
    is_debt: list[bool],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build linear constraint matrices for optimization.

    Returns A, b such that A @ x <= b represents the constraints.

    The decision variable x is structured as:
    [collateral_amounts..., debt_amounts...]

    Args:
        constraints: Aave constraint specification.
        assets: List of asset symbols in order.
        is_collateral: Which assets are collateral.
        is_debt: Which assets are debt.

    Returns:
        Tuple of (A matrix, b vector) for linear constraints.
    """
    n_assets = len(assets)
    n_vars = 2 * n_assets  # collateral and debt for each

    constraint_rows = []
    constraint_bounds = []

    # Supply cap constraints: collateral_i <= supply_cap_i
    for i, symbol in enumerate(assets):
        if symbol in constraints.assets:
            asset = constraints.assets[symbol]
            if asset.supply_cap is not None:
                row = np.zeros(n_vars)
                row[i] = asset.price_usd
                constraint_rows.append(row)
                constraint_bounds.append(asset.supply_cap * asset.price_usd)

    # Borrow cap constraints: debt_i <= borrow_cap_i
    for i, symbol in enumerate(assets):
        if symbol in constraints.assets:
            asset = constraints.assets[symbol]
            if asset.borrow_cap is not None:
                row = np.zeros(n_vars)
                row[n_assets + i] = asset.price_usd
                constraint_rows.append(row)
                constraint_bounds.append(asset.borrow_cap * asset.price_usd)

    # LTV constraint: total_debt <= total_borrow_power
    # Σ debt_i × price_i <= Σ collateral_j × price_j × LTV_j
    # Rearranged: Σ debt_i × price_i - Σ collateral_j × price_j × LTV_j <= 0
    ltv_row = np.zeros(n_vars)
    for i, symbol in enumerate(assets):
        if symbol in constraints.assets:
            asset = constraints.assets[symbol]
            ltv = constraints.get_effective_ltv(symbol)
            # Collateral contribution (negative in constraint)
            ltv_row[i] = -asset.price_usd * ltv
            # Debt contribution (positive in constraint)
            ltv_row[n_assets + i] = asset.price_usd
    constraint_rows.append(ltv_row)
    constraint_bounds.append(0.0)

    if constraint_rows:
        A = np.array(constraint_rows)
        b = np.array(constraint_bounds)
    else:
        A = np.zeros((1, n_vars))
        b = np.zeros(1)

    return A, b


def health_factor_constraint_gradient(
    constraints: AaveV3Constraints,
    assets: list[str],
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute gradient of health factor constraint.

    For nonlinear optimization, provides gradient of:
    g(x) = min_HF - H(x) <= 0

    Args:
        constraints: Aave constraints.
        assets: Asset list.
        x: Decision variable [collateral..., debt...].

    Returns:
        Gradient vector.
    """
    n_assets = len(assets)
    grad = np.zeros(2 * n_assets)

    # H = Σ(c_i × p_i × LT_i) / Σ(d_j × p_j)
    # ∂H/∂c_i = p_i × LT_i / debt
    # ∂H/∂d_j = -H / debt × p_j

    total_debt = 0.0
    total_collateral = 0.0

    for i, symbol in enumerate(assets):
        if symbol in constraints.assets:
            asset = constraints.assets[symbol]
            total_collateral += x[i] * asset.price_usd * constraints.get_effective_lt(symbol)
            total_debt += x[n_assets + i] * asset.price_usd

    if total_debt < 1e-10:
        return grad

    H = total_collateral / total_debt

    for i, symbol in enumerate(assets):
        if symbol in constraints.assets:
            asset = constraints.assets[symbol]
            # Gradient w.r.t. collateral (positive effect on H)
            grad[i] = -asset.price_usd * constraints.get_effective_lt(symbol) / total_debt
            # Gradient w.r.t. debt (negative effect on H)
            grad[n_assets + i] = H * asset.price_usd / total_debt

    return grad
