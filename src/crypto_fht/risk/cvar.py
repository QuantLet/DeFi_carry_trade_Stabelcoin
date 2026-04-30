"""
Conditional Value-at-Risk (CVaR) computation for liquidation losses.

CVaR (also known as Expected Shortfall) is defined as:
    CVaR_α = E[L | L ≥ VaR_α]

where L is the loss and VaR_α is the α-quantile of the loss distribution.

For liquidation risk:
    L = (debt - recovered_collateral) × I(τ_b ≤ T)

We use the Rockafellar-Uryasev representation:
    CVaR_α = min_ξ { ξ + (1/(1-α)) E[(L - ξ)⁺] }

which enables efficient optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_cvar_from_samples(
    losses: NDArray[np.floating],
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Compute VaR and CVaR from sample losses.

    Args:
        losses: Array of loss values (positive = loss, negative = gain).
        alpha: Confidence level (e.g., 0.95 for 95% CVaR).

    Returns:
        Tuple of (VaR_α, CVaR_α).

    Example:
        >>> losses = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 0.3, 0.4, 0.15, 0.25, 0.35])
        >>> var, cvar = compute_cvar_from_samples(losses, alpha=0.9)
        >>> var  # 90th percentile
        1.0
    """
    if len(losses) == 0:
        return 0.0, 0.0

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    sorted_losses = np.sort(losses)
    n = len(losses)

    # VaR: α-quantile
    var_idx = int(np.ceil(alpha * n)) - 1
    var_idx = max(0, min(var_idx, n - 1))
    var = float(sorted_losses[var_idx])

    # CVaR: expected loss given loss ≥ VaR
    tail_losses = sorted_losses[var_idx:]
    if len(tail_losses) == 0:
        cvar = var
    else:
        cvar = float(np.mean(tail_losses))

    return var, cvar


@dataclass
class CVaRCalculator:
    """CVaR calculator for various loss distributions.

    Supports both sample-based and analytical computation of CVaR.

    Attributes:
        alpha: Confidence level (e.g., 0.95 for 95% CVaR).

    Example:
        >>> calc = CVaRCalculator(alpha=0.95)
        >>> losses = np.random.exponential(1.0, 10000)
        >>> var, cvar = calc.from_samples(losses)
    """

    alpha: float = 0.95

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    def from_samples(
        self, losses: NDArray[np.floating]
    ) -> tuple[float, float]:
        """Compute VaR and CVaR from samples.

        Args:
            losses: Array of loss values.

        Returns:
            Tuple of (VaR, CVaR).
        """
        return compute_cvar_from_samples(losses, self.alpha)

    def from_distribution(
        self,
        loss_cdf: Callable[[float], float],
        loss_pdf: Callable[[float], float],
        lower_bound: float = 0.0,
        upper_bound: float = 1e6,
    ) -> tuple[float, float]:
        """Compute VaR and CVaR from loss distribution functions.

        Uses numerical integration when the distribution is known analytically.

        Args:
            loss_cdf: CDF of loss distribution F(x) = P(L ≤ x).
            loss_pdf: PDF of loss distribution f(x) = F'(x).
            lower_bound: Lower bound for search/integration.
            upper_bound: Upper bound for search/integration.

        Returns:
            Tuple of (VaR, CVaR).
        """
        # Find VaR: smallest x such that F(x) ≥ α
        def cdf_minus_alpha(x: float) -> float:
            return loss_cdf(x) - self.alpha

        try:
            var = brentq(cdf_minus_alpha, lower_bound, upper_bound)
        except ValueError:
            # If bracketing fails, use bisection manually
            var = self._bisection_search(loss_cdf, self.alpha, lower_bound, upper_bound)

        # CVaR = E[L | L ≥ VaR] = ∫_{VaR}^∞ x f(x) dx / (1 - α)
        def integrand(x: float) -> float:
            return x * loss_pdf(x)

        numerator, _ = quad(integrand, var, upper_bound, limit=100)
        cvar = numerator / (1 - self.alpha)

        return float(var), float(cvar)

    def _bisection_search(
        self,
        cdf: Callable[[float], float],
        target: float,
        lo: float,
        hi: float,
        tol: float = 1e-8,
    ) -> float:
        """Bisection search for quantile."""
        for _ in range(100):
            mid = (lo + hi) / 2
            if cdf(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2

    def rockafellar_uryasev(
        self,
        loss_function: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        weights: NDArray[np.floating],
        scenarios: NDArray[np.floating],
    ) -> tuple[float, float]:
        """Compute CVaR via Rockafellar-Uryasev formulation.

        CVaR_α = min_ξ { ξ + (1/(1-α)) E[(L(w) - ξ)⁺] }

        This representation is key for CVaR portfolio optimization.

        Args:
            loss_function: Function L(w, scenario) -> loss for weights w.
            weights: Portfolio weights.
            scenarios: Matrix of scenarios (n_scenarios × n_features).

        Returns:
            Tuple of (optimal ξ = VaR, CVaR).
        """
        n_scenarios = scenarios.shape[0]

        def objective(xi: float) -> float:
            losses = np.array([
                loss_function(np.concatenate([weights, scenarios[i:i+1].flatten()]))
                for i in range(n_scenarios)
            ])
            excess = np.maximum(losses - xi, 0)
            return xi + np.mean(excess) / (1 - self.alpha)

        # Optimize over ξ
        result = minimize_scalar(objective, bounds=(-1e6, 1e6), method="bounded")
        var = result.x
        cvar = result.fun

        return float(var), float(cvar)

    def portfolio_cvar(
        self,
        weights: NDArray[np.floating],
        loss_scenarios: NDArray[np.floating],
    ) -> tuple[float, float]:
        """Compute portfolio CVaR from scenario losses.

        Args:
            weights: Portfolio weights (n_assets,).
            loss_scenarios: Loss scenarios matrix (n_scenarios × n_assets).

        Returns:
            Tuple of (VaR, CVaR) for the portfolio.
        """
        # Portfolio losses: L = w · loss_scenarios
        portfolio_losses = loss_scenarios @ weights
        return self.from_samples(portfolio_losses)


class CVaRLiquidationRisk:
    """CVaR specifically for DeFi liquidation losses.

    Models the loss distribution from liquidation events:
        L = (debt_value - liquidation_recovery) × I(liquidation_occurs)

    where the recovery depends on liquidation penalty and market conditions.

    Attributes:
        alpha: CVaR confidence level.
        liquidation_penalty: Penalty rate (e.g., 0.05 for 5%).
    """

    def __init__(
        self,
        alpha: float = 0.95,
        liquidation_penalty: float = 0.05,
    ) -> None:
        """Initialize liquidation CVaR calculator.

        Args:
            alpha: Confidence level.
            liquidation_penalty: Liquidation penalty as fraction.
        """
        self.alpha = alpha
        self.liquidation_penalty = liquidation_penalty
        self.calculator = CVaRCalculator(alpha=alpha)

    def compute_liquidation_cvar(
        self,
        liquidation_probs: NDArray[np.floating],
        position_values: NDArray[np.floating],
        n_simulations: int = 10000,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float]:
        """Compute CVaR of liquidation losses via Monte Carlo.

        Args:
            liquidation_probs: Probability of liquidation for each position.
            position_values: USD value of each position.
            n_simulations: Number of Monte Carlo simulations.
            rng: Random number generator.

        Returns:
            Tuple of (VaR, CVaR) of total liquidation loss.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_positions = len(liquidation_probs)
        losses = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Simulate which positions get liquidated
            liquidated = rng.random(n_positions) < liquidation_probs

            # Loss = position_value × liquidation_penalty for liquidated positions
            losses[sim] = np.sum(
                position_values[liquidated] * self.liquidation_penalty
            )

        return self.calculator.from_samples(losses)

    def expected_shortfall_contribution(
        self,
        weights: NDArray[np.floating],
        loss_scenarios: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute CVaR contribution of each asset.

        The CVaR contribution shows how much each asset contributes to
        the portfolio's tail risk.

        Args:
            weights: Portfolio weights.
            loss_scenarios: Loss scenarios (n_scenarios × n_assets).

        Returns:
            Array of CVaR contributions per asset.
        """
        n_scenarios, n_assets = loss_scenarios.shape
        portfolio_losses = loss_scenarios @ weights

        # Find scenarios in the tail (above VaR)
        sorted_losses = np.sort(portfolio_losses)
        var_idx = int(np.ceil(self.alpha * n_scenarios))
        var = sorted_losses[var_idx - 1] if var_idx > 0 else sorted_losses[0]

        tail_mask = portfolio_losses >= var

        # Contribution = E[w_i × L_i | portfolio_loss ≥ VaR]
        if tail_mask.sum() == 0:
            return np.zeros(n_assets)

        contributions = np.array([
            np.mean(weights[i] * loss_scenarios[tail_mask, i])
            for i in range(n_assets)
        ])

        return contributions
