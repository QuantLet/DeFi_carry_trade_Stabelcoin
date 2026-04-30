"""
Liquidation risk computation integrating first-hitting time analysis.

Provides high-level APIs for computing liquidation probabilities
for DeFi positions using the underlying Lévy process framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from crypto_fht.core.first_hitting_time import FirstHittingTime
from crypto_fht.core.levy_process import LevyParameters
from crypto_fht.risk.health_factor import HealthFactor, HealthFactorDynamics

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LiquidationRiskCalculator:
    """Calculator for single-position liquidation risk.

    Combines health factor computation with first-hitting time analysis
    to provide liquidation probabilities.

    Attributes:
        params: Lévy process parameters for the position.
        N_stehfest: Number of Gaver-Stehfest terms.
        fht: First-hitting time calculator (created automatically).

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> calc = LiquidationRiskCalculator(params)
        >>> prob = calc.probability(health_factor=1.5, time_horizon=30)
    """

    params: LevyParameters
    N_stehfest: int = 10
    fht: FirstHittingTime = field(init=False)

    def __post_init__(self) -> None:
        self.fht = FirstHittingTime(self.params, N_stehfest=self.N_stehfest)

    def probability(
        self,
        health_factor: float,
        time_horizon: float,
        liquidation_threshold: float = 1.0,
    ) -> float:
        """Compute liquidation probability.

        Args:
            health_factor: Current health factor (typically > 1).
            time_horizon: Time horizon for probability computation.
            liquidation_threshold: Health factor at liquidation (default 1.0).

        Returns:
            P(liquidation within time_horizon).
        """
        return self.fht.from_health_factor(
            health_factor, time_horizon, liquidation_threshold
        )

    def survival_probability(
        self,
        health_factor: float,
        time_horizon: float,
        liquidation_threshold: float = 1.0,
    ) -> float:
        """Compute survival probability (no liquidation).

        Args:
            health_factor: Current health factor.
            time_horizon: Time horizon.
            liquidation_threshold: Liquidation threshold.

        Returns:
            P(no liquidation within time_horizon).
        """
        return self.fht.survival_from_health_factor(
            health_factor, time_horizon, liquidation_threshold
        )

    def term_structure(
        self,
        health_factor: float,
        time_horizons: NDArray[np.floating],
        liquidation_threshold: float = 1.0,
    ) -> NDArray[np.floating]:
        """Compute term structure of liquidation probabilities.

        Args:
            health_factor: Current health factor.
            time_horizons: Array of time horizons.
            liquidation_threshold: Liquidation threshold.

        Returns:
            Array of probabilities for each time horizon.
        """
        x = np.log(health_factor)
        b = np.log(liquidation_threshold)
        return self.fht.liquidation_probability_term_structure(time_horizons, x, b)

    def expected_liquidation_time(
        self,
        health_factor: float,
        liquidation_threshold: float = 1.0,
    ) -> float:
        """Compute expected time to liquidation.

        Args:
            health_factor: Current health factor.
            liquidation_threshold: Liquidation threshold.

        Returns:
            E[time to liquidation]. May be inf if process drifts up.
        """
        x = np.log(health_factor)
        b = np.log(liquidation_threshold)
        return self.fht.expected_hitting_time(x, b)

    def safe_health_factor(
        self,
        target_survival_prob: float,
        time_horizon: float,
        liquidation_threshold: float = 1.0,
        tol: float = 0.01,
    ) -> float:
        """Find health factor that achieves target survival probability.

        Uses binary search to find the minimum health factor such that
        P(survival) >= target_survival_prob.

        Args:
            target_survival_prob: Target survival probability (e.g., 0.99).
            time_horizon: Time horizon for the probability.
            liquidation_threshold: Liquidation threshold.
            tol: Tolerance for binary search.

        Returns:
            Minimum health factor achieving target survival probability.
        """
        # Binary search over health factor
        lo = liquidation_threshold + 0.01
        hi = liquidation_threshold * 10

        for _ in range(50):
            mid = (lo + hi) / 2
            surv = self.survival_probability(mid, time_horizon, liquidation_threshold)

            if surv < target_survival_prob:
                lo = mid
            else:
                hi = mid

            if hi - lo < tol:
                break

        return hi


@dataclass
class PortfolioLiquidationRisk:
    """Liquidation risk for a portfolio of positions.

    Handles multi-asset portfolios by aggregating risk metrics.

    Attributes:
        health_factor_dynamics: Health factor dynamics model.
        base_params: Lévy parameters per asset.
        N_stehfest: Gaver-Stehfest terms.
    """

    health_factor_dynamics: HealthFactorDynamics
    base_params: dict[str, LevyParameters] = field(default_factory=dict)
    N_stehfest: int = 10

    def compute_portfolio_params(self) -> LevyParameters:
        """Compute aggregated Lévy parameters for the portfolio."""
        if self.base_params:
            self.health_factor_dynamics.asset_params = self.base_params
        return self.health_factor_dynamics.aggregate_levy_params()

    def liquidation_probability(
        self,
        time_horizon: float,
        liquidation_threshold: float = 1.0,
    ) -> float:
        """Compute portfolio liquidation probability.

        Args:
            time_horizon: Time horizon.
            liquidation_threshold: Health factor at liquidation.

        Returns:
            P(portfolio liquidation within time_horizon).
        """
        params = self.compute_portfolio_params()
        fht = FirstHittingTime(params, N_stehfest=self.N_stehfest)

        hf = self.health_factor_dynamics.health_factor.current_health_factor
        return fht.from_health_factor(hf, time_horizon, liquidation_threshold)

    def term_structure(
        self,
        time_horizons: NDArray[np.floating],
        liquidation_threshold: float = 1.0,
    ) -> NDArray[np.floating]:
        """Compute term structure for the portfolio."""
        params = self.compute_portfolio_params()
        fht = FirstHittingTime(params, N_stehfest=self.N_stehfest)

        hf = self.health_factor_dynamics.health_factor.current_health_factor
        x = np.log(hf)
        b = np.log(liquidation_threshold)

        return fht.liquidation_probability_term_structure(time_horizons, x, b)

    def risk_contribution(
        self,
        time_horizon: float,
        bump_size: float = 0.01,
    ) -> dict[str, float]:
        """Compute risk contribution of each position.

        Uses finite differences to estimate marginal contribution.

        Args:
            time_horizon: Time horizon for probability.
            bump_size: Size of position bump for finite difference.

        Returns:
            Dictionary mapping position asset to risk contribution.
        """
        base_prob = self.liquidation_probability(time_horizon)
        contributions = {}

        for position in self.health_factor_dynamics.health_factor.positions:
            # Bump position value
            original_amount = position.amount
            position.amount *= (1 + bump_size)

            bumped_prob = self.liquidation_probability(time_horizon)

            # Restore
            position.amount = original_amount

            # Marginal contribution
            contributions[position.asset] = (bumped_prob - base_prob) / bump_size

        return contributions

    def scenario_analysis(
        self,
        price_scenarios: list[dict[str, float]],
        time_horizon: float,
    ) -> list[tuple[dict[str, float], float, float]]:
        """Analyze liquidation risk under different price scenarios.

        Args:
            price_scenarios: List of price change scenarios.
            time_horizon: Time horizon.

        Returns:
            List of (scenario, health_factor, liquidation_prob) tuples.
        """
        results = []

        for scenario in price_scenarios:
            # Compute health factor under scenario
            hf = self.health_factor_dynamics.health_factor_at_time(0, scenario)

            if hf <= 1.0:
                prob = 1.0
            else:
                # Compute probability from this starting point
                params = self.compute_portfolio_params()
                fht = FirstHittingTime(params, N_stehfest=self.N_stehfest)
                prob = fht.from_health_factor(hf, time_horizon, 1.0)

            results.append((scenario, hf, prob))

        return results


def compute_liquidation_surface(
    params: LevyParameters,
    health_factors: NDArray[np.floating],
    time_horizons: NDArray[np.floating],
    N_stehfest: int = 10,
) -> NDArray[np.floating]:
    """Compute liquidation probability surface over (HF, t) grid.

    Useful for visualization and risk dashboards.

    Args:
        params: Lévy parameters.
        health_factors: Array of health factor values.
        time_horizons: Array of time horizons.
        N_stehfest: Gaver-Stehfest terms.

    Returns:
        2D array of shape (len(health_factors), len(time_horizons)).
    """
    from crypto_fht.core.first_hitting_time import compute_liquidation_probability_grid

    return compute_liquidation_probability_grid(
        params, health_factors, time_horizons, N_stehfest
    )


def quick_liquidation_estimate(
    health_factor: float,
    volatility: float,
    time_horizon: float,
    drift: float = 0.0,
) -> float:
    """Quick approximate liquidation probability (no jumps).

    Uses simple diffusion model for rough estimates:
    P(τ ≤ t) ≈ 2 × Φ(-d) where d = (log(HF) - μt) / (σ√t)

    This is a lower bound on true probability (ignores jumps).

    Args:
        health_factor: Current health factor.
        volatility: Annual volatility.
        time_horizon: Time in years (or consistent units).
        drift: Annual drift.

    Returns:
        Approximate liquidation probability.
    """
    from scipy.stats import norm

    if health_factor <= 1.0:
        return 1.0

    x = np.log(health_factor)
    sigma_sqrt_t = volatility * np.sqrt(time_horizon)

    if sigma_sqrt_t < 1e-10:
        return 0.0 if drift >= 0 else 1.0

    # First passage for Brownian motion with drift
    # P(τ_0 ≤ t | X_0 = x) = Φ((-x - μt)/(σ√t)) + exp(2μx/σ²) × Φ((-x + μt)/(σ√t))
    mu_t = drift * time_horizon

    d1 = (-x - mu_t) / sigma_sqrt_t
    d2 = (-x + mu_t) / sigma_sqrt_t

    prob = norm.cdf(d1)
    if drift != 0:
        prob += np.exp(2 * drift * x / volatility**2) * norm.cdf(d2)

    return float(np.clip(prob, 0, 1))
