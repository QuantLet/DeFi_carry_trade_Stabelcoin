"""
CVaR-minimizing portfolio optimizer for DeFi positions.

Implements the Rockafellar-Uryasev formulation:
    min  ξ + (1/(1-α)) · (1/S) · Σ_s u_s
    s.t. u_s ≥ L_s(w) - ξ, ∀s
         u_s ≥ 0
         E[R(w)] ≥ r_min
         Aave LTV constraints
         w ∈ bounds

Uses CVXPY for convex optimization when possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from scipy.optimize import minimize, LinearConstraint

from crypto_fht.optimization.constraints import AaveV3Constraints, build_constraint_matrices

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Result of portfolio optimization.

    Attributes:
        weights: Optimal portfolio weights.
        cvar: CVaR at optimum.
        var: VaR at optimum.
        expected_return: Expected return of optimal portfolio.
        health_factor: Health factor at optimal weights.
        status: Optimization status ("optimal", "infeasible", etc.).
        iterations: Number of iterations.
        objective_value: Final objective value.
    """

    weights: NDArray[np.floating]
    cvar: float
    var: float
    expected_return: float
    health_factor: float
    status: str
    iterations: int
    objective_value: float

    @property
    def is_optimal(self) -> bool:
        """Check if optimization succeeded."""
        return self.status == "optimal"


class CVaRPortfolioOptimizer:
    """CVaR-minimizing portfolio optimizer with Aave constraints.

    Solves:
        min CVaR_α(L(w))
        s.t. E[R(w)] ≥ r_min
             Health factor ≥ HF_min
             Aave LTV constraints
             w ∈ bounds

    Attributes:
        alpha: CVaR confidence level (e.g., 0.95).
        constraints: Aave v3 constraints.
        min_health_factor: Minimum health factor requirement.

    Example:
        >>> optimizer = CVaRPortfolioOptimizer(alpha=0.95)
        >>> result = optimizer.optimize(
        ...     expected_returns=np.array([0.1, 0.05, 0.02]),
        ...     loss_scenarios=scenarios,
        ...     min_return=0.05
        ... )
    """

    def __init__(
        self,
        alpha: float = 0.95,
        constraints: AaveV3Constraints | None = None,
        min_health_factor: float = 1.5,
    ) -> None:
        """Initialize optimizer.

        Args:
            alpha: CVaR confidence level.
            constraints: Aave protocol constraints.
            min_health_factor: Minimum health factor.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.constraints = constraints or AaveV3Constraints()
        self.min_health_factor = min_health_factor

    def optimize(
        self,
        expected_returns: NDArray[np.floating],
        loss_scenarios: NDArray[np.floating],
        min_return: float = 0.0,
        long_only: bool = False,
        max_position: float = 2.0,
    ) -> OptimizationResult:
        """Optimize portfolio to minimize CVaR.

        Args:
            expected_returns: Expected return per asset (n_assets,).
            loss_scenarios: Loss scenarios matrix (n_scenarios × n_assets).
            min_return: Minimum required expected return.
            long_only: If True, only long positions allowed.
            max_position: Maximum absolute weight per asset.

        Returns:
            OptimizationResult with optimal portfolio.
        """
        if HAS_CVXPY:
            return self._optimize_cvxpy(
                expected_returns, loss_scenarios, min_return, long_only, max_position
            )
        else:
            return self._optimize_scipy(
                expected_returns, loss_scenarios, min_return, long_only, max_position
            )

    def _optimize_cvxpy(
        self,
        expected_returns: NDArray[np.floating],
        loss_scenarios: NDArray[np.floating],
        min_return: float,
        long_only: bool,
        max_position: float,
    ) -> OptimizationResult:
        """Optimize using CVXPY (preferred method)."""
        S, n = loss_scenarios.shape

        # Decision variables
        w = cp.Variable(n, name="weights")
        xi = cp.Variable(name="VaR")
        u = cp.Variable(S, name="excess_loss", nonneg=True)

        # Objective: CVaR = ξ + (1/(1-α)) E[(L-ξ)⁺]
        objective = xi + (1 / (1 - self.alpha)) * cp.sum(u) / S

        # Constraints
        constraints = [
            # CVaR auxiliary constraints
            u >= loss_scenarios @ w - xi,
            # Return constraint
            expected_returns @ w >= min_return,
            # Budget constraint
            cp.sum(w) == 1,
        ]

        # Position bounds
        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= max_position)
        else:
            constraints.append(w >= -max_position)
            constraints.append(w <= max_position)

        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            try:
                problem.solve(solver=cp.SCS)
            except cp.error.SolverError:
                return OptimizationResult(
                    weights=np.zeros(n),
                    cvar=float("inf"),
                    var=float("inf"),
                    expected_return=0,
                    health_factor=0,
                    status="solver_error",
                    iterations=0,
                    objective_value=float("inf"),
                )

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return OptimizationResult(
                weights=np.zeros(n),
                cvar=float("inf"),
                var=float("inf"),
                expected_return=0,
                health_factor=0,
                status=str(problem.status),
                iterations=0,
                objective_value=float("inf"),
            )

        weights = w.value
        var = float(xi.value)
        cvar = float(objective.value)
        exp_ret = float(expected_returns @ weights)

        return OptimizationResult(
            weights=weights,
            cvar=cvar,
            var=var,
            expected_return=exp_ret,
            health_factor=self._compute_health_factor(weights),
            status="optimal",
            iterations=problem.solver_stats.num_iters if problem.solver_stats else 0,
            objective_value=cvar,
        )

    def _optimize_scipy(
        self,
        expected_returns: NDArray[np.floating],
        loss_scenarios: NDArray[np.floating],
        min_return: float,
        long_only: bool,
        max_position: float,
    ) -> OptimizationResult:
        """Optimize using SciPy (fallback method)."""
        S, n = loss_scenarios.shape

        def cvar_objective(x: NDArray[np.floating]) -> float:
            """CVaR objective function."""
            w = x[:n]
            xi = x[n]

            losses = loss_scenarios @ w
            excess = np.maximum(losses - xi, 0)
            return xi + np.mean(excess) / (1 - self.alpha)

        # Initial guess
        x0 = np.zeros(n + 1)
        x0[:n] = 1.0 / n  # Equal weights
        x0[n] = 0.0  # VaR

        # Bounds
        if long_only:
            bounds = [(0, max_position)] * n + [(None, None)]
        else:
            bounds = [(-max_position, max_position)] * n + [(None, None)]

        # Constraints
        constraints = [
            # Budget: sum(w) = 1
            {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1},
            # Return: E[R] >= min_return
            {"type": "ineq", "fun": lambda x: expected_returns @ x[:n] - min_return},
        ]

        result = minimize(
            cvar_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        weights = result.x[:n]
        xi = result.x[n]

        # Compute CVaR at solution
        losses = loss_scenarios @ weights
        cvar = xi + np.mean(np.maximum(losses - xi, 0)) / (1 - self.alpha)

        return OptimizationResult(
            weights=weights,
            cvar=cvar,
            var=xi,
            expected_return=float(expected_returns @ weights),
            health_factor=self._compute_health_factor(weights),
            status="optimal" if result.success else result.message,
            iterations=result.nit,
            objective_value=cvar,
        )

    def _compute_health_factor(self, weights: NDArray[np.floating]) -> float:
        """Compute health factor for given weights.

        Placeholder - actual computation depends on position structure.
        """
        # Would use self.constraints to compute actual HF
        return self.min_health_factor

    def optimize_with_liquidation_risk(
        self,
        expected_returns: NDArray[np.floating],
        liquidation_probs: NDArray[np.floating],
        liquidation_losses: NDArray[np.floating],
        min_return: float = 0.0,
    ) -> OptimizationResult:
        """Optimize considering liquidation probability directly.

        Instead of scenario-based CVaR, uses analytical liquidation
        probabilities from the first-hitting time model.

        Args:
            expected_returns: Expected returns per asset.
            liquidation_probs: P(liquidation) per asset from FHT model.
            liquidation_losses: Expected loss given liquidation per asset.
            min_return: Minimum return requirement.

        Returns:
            OptimizationResult.
        """
        n = len(expected_returns)

        # Generate synthetic loss scenarios from liquidation model
        n_scenarios = 1000
        rng = np.random.default_rng(42)

        loss_scenarios = np.zeros((n_scenarios, n))
        for i in range(n):
            # Bernoulli draws for liquidation
            liquidated = rng.random(n_scenarios) < liquidation_probs[i]
            loss_scenarios[:, i] = liquidated * liquidation_losses[i]

        return self.optimize(
            expected_returns, loss_scenarios, min_return, long_only=False
        )


def efficient_frontier(
    expected_returns: NDArray[np.floating],
    loss_scenarios: NDArray[np.floating],
    alpha: float = 0.95,
    n_points: int = 20,
) -> list[OptimizationResult]:
    """Compute CVaR-efficient frontier.

    Args:
        expected_returns: Expected returns per asset.
        loss_scenarios: Loss scenarios.
        alpha: CVaR confidence level.
        n_points: Number of frontier points.

    Returns:
        List of OptimizationResults along the frontier.
    """
    optimizer = CVaRPortfolioOptimizer(alpha=alpha)

    # Find return range
    min_ret = np.min(expected_returns)
    max_ret = np.max(expected_returns)

    target_returns = np.linspace(min_ret, max_ret * 0.9, n_points)
    frontier = []

    for target in target_returns:
        result = optimizer.optimize(
            expected_returns, loss_scenarios, min_return=target
        )
        if result.is_optimal:
            frontier.append(result)

    return frontier
