"""
First-hitting time distributions for spectrally negative Lévy processes.

The first-hitting time (or first-passage time) τ_b is defined as:
    τ_b = inf{t ≥ 0 : X_t ≤ b}

This represents the first time the process X_t crosses below the barrier b,
which in our DeFi context corresponds to the liquidation time when the
log-health factor crosses the liquidation threshold.

Key formula for the Laplace transform of τ_b:
    E[e^{-qτ_b} | X_0 = x] = Z^(q)(x-b) - (q/Φ(q)) W^(q)(x-b)

for x > b (starting above the threshold).

Combined with Gaver-Stehfest inversion, this gives P(τ_b ≤ t).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from crypto_fht.core.laplace_inversion import GaverStehfestInverter
from crypto_fht.core.levy_process import LevyParameters
from crypto_fht.core.scale_function import ScaleFunction
from crypto_fht.core.wiener_hopf import WienerHopfFactorization

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FirstHittingTime:
    """First-hitting time distribution for spectrally negative Lévy processes.

    Computes P(τ_b ≤ t | X_0 = x) where τ_b = inf{t ≥ 0 : X_t ≤ b} is the
    first time the process hits the lower barrier b (liquidation threshold).

    Uses the formula:
        E[e^{-qτ_b} | X_0 = x] = Z^(q)(x-b) - (q/Φ(q)) W^(q)(x-b)

    combined with Gaver-Stehfest inversion to obtain P(τ_b ≤ t).

    In the DeFi context:
    - x = log(current_health_factor)
    - b = log(liquidation_threshold) = log(1) = 0
    - P(τ_b ≤ t) = probability of liquidation within time t

    Attributes:
        params: Lévy process parameters.
        wiener_hopf: Wiener-Hopf factorization.
        scale_function: Scale function calculator.
        inverter: Laplace transform inverter.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> fht = FirstHittingTime(params)
        >>> # Health factor = 1.5, so log(HF) = 0.405
        >>> x = np.log(1.5)
        >>> b = 0.0  # Liquidation at HF = 1
        >>> prob = fht.liquidation_probability(t=30, x=x, b=b)
    """

    def __init__(
        self,
        params: LevyParameters,
        N_stehfest: int = 10,
        cache_size: int = 256,
    ) -> None:
        """Initialize first-hitting time calculator.

        Args:
            params: Lévy process parameters.
            N_stehfest: Number of terms for Gaver-Stehfest inversion (8-12).
            cache_size: Cache size for intermediate computations.
        """
        self.params = params
        self.wiener_hopf = WienerHopfFactorization(params, cache_size=cache_size)
        self.scale_function = ScaleFunction(params, self.wiener_hopf, cache_size=cache_size)
        self.inverter = GaverStehfestInverter(N=N_stehfest)

        # Cache for Laplace transform values
        self._laplace_cache = lru_cache(maxsize=cache_size)(self._compute_laplace_transform)

    def laplace_transform(self, q: float, x: float, b: float) -> float:
        """Compute E[e^{-qτ_b} | X_0 = x], the Laplace transform of hitting time.

        Formula:
            E[e^{-qτ_b}] = Z^(q)(x-b) - (q/Φ(q)) W^(q)(x-b)

        This formula holds for x > b (starting above the threshold).
        If x ≤ b, the process is already at or below the threshold, so τ_b = 0
        and E[e^{-qτ_b}] = 1.

        Args:
            q: Laplace parameter (q > 0).
            x: Starting position (log-health factor).
            b: Lower barrier (liquidation threshold, typically 0 for HF=1).

        Returns:
            Laplace transform value E[e^{-qτ_b}] ∈ [0, 1].
        """
        if x <= b:
            return 1.0  # Already at or below threshold

        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")

        # Use cached computation with rounded arguments
        q_rounded = round(q, 10)
        x_rounded = round(x, 10)
        b_rounded = round(b, 10)
        return self._laplace_cache(q_rounded, x_rounded, b_rounded)

    def _compute_laplace_transform(self, q: float, x: float, b: float) -> float:
        """Internal Laplace transform computation."""
        y = x - b  # Distance to threshold

        # Get Φ(q)
        phi_q = self.wiener_hopf.phi(q)

        # Get W^(q)(y) and Z^(q)(y)
        W_q_y = self.scale_function.W(y, q)
        Z_q_y = self.scale_function.Z(y, q)

        # E[e^{-qτ_b}] = Z^(q)(y) - (q/Φ(q)) W^(q)(y)
        result = Z_q_y - (q / phi_q) * W_q_y

        # Ensure valid Laplace transform value
        return float(np.clip(result, 0.0, 1.0))

    def survival_probability(
        self,
        t: float | NDArray[np.floating],
        x: float,
        b: float,
    ) -> float | NDArray[np.floating]:
        """Compute P(τ_b > t | X_0 = x), the probability of no liquidation by time t.

        This is the survival probability - the probability that the health
        factor stays above the liquidation threshold until time t.

        Uses Gaver-Stehfest inversion of the Laplace transform.

        Args:
            t: Time horizon(s). Must be positive.
            x: Starting position (log-health factor).
            b: Liquidation threshold (typically 0 for HF=1).

        Returns:
            Survival probability P(τ_b > t) ∈ [0, 1].

        Example:
            >>> fht = FirstHittingTime(params)
            >>> # 90-day survival probability with HF = 1.5
            >>> surv = fht.survival_probability(t=90, x=np.log(1.5), b=0)
        """
        if x <= b:
            return 0.0 if isinstance(t, float) else np.zeros_like(t)

        def survival_laplace(q: float) -> float:
            """Laplace transform of survival: L{P(τ > t)}(q) = (1 - E[e^{-qτ}]) / q."""
            L_tau = self.laplace_transform(q, x, b)
            return (1.0 - L_tau) / q

        result = self.inverter.invert(survival_laplace, t)

        # Ensure valid probability
        if isinstance(result, np.ndarray):
            return np.clip(result, 0.0, 1.0)
        else:
            return float(np.clip(result, 0.0, 1.0))

    def liquidation_probability(
        self,
        t: float | NDArray[np.floating],
        x: float,
        b: float,
    ) -> float | NDArray[np.floating]:
        """Compute P(τ_b ≤ t | X_0 = x), the probability of liquidation by time t.

        This is the main risk metric - the probability that the health factor
        drops below the liquidation threshold within time t.

        Args:
            t: Time horizon(s). Must be positive.
            x: Starting position (log-health factor).
            b: Liquidation threshold (typically 0 for HF=1).

        Returns:
            Liquidation probability P(τ_b ≤ t) ∈ [0, 1].

        Example:
            >>> fht = FirstHittingTime(params)
            >>> # 30-day liquidation probability with health factor 1.5
            >>> prob = fht.liquidation_probability(t=30, x=np.log(1.5), b=0)
        """
        survival = self.survival_probability(t, x, b)
        return 1.0 - survival

    def liquidation_probability_term_structure(
        self,
        t_values: NDArray[np.floating],
        x: float,
        b: float,
    ) -> NDArray[np.floating]:
        """Compute term structure of liquidation probabilities.

        Returns P(τ_b ≤ t) for multiple time horizons, useful for
        visualizing how liquidation risk evolves over time.

        Args:
            t_values: Array of time horizons.
            x: Starting position.
            b: Liquidation threshold.

        Returns:
            Array of liquidation probabilities for each time horizon.
        """
        return np.array([
            self.liquidation_probability(float(t), x, b) for t in t_values
        ])

    def expected_hitting_time(self, x: float, b: float, max_time: float = 1000.0) -> float:
        """Compute E[τ_b | X_0 = x], the expected time to liquidation.

        For processes with negative effective drift, E[τ_b] < ∞.
        For processes with positive effective drift, τ_b may be infinite
        with positive probability.

        Uses numerical differentiation: E[τ] = -d/dq E[e^{-qτ}]|_{q=0}

        Args:
            x: Starting position.
            b: Liquidation threshold.
            max_time: Maximum expected time to return (for numerical stability).

        Returns:
            Expected hitting time E[τ_b].
        """
        if x <= b:
            return 0.0

        # Check if process drifts down (finite expected hitting time)
        if not self.params.is_drifting_down():
            # Process may never hit barrier
            # Return estimate based on limiting behavior
            return float("inf")

        # Numerical differentiation: E[τ] = -d/dq E[e^{-qτ}]|_{q=0+}
        epsilon = 1e-6

        L_plus = self.laplace_transform(epsilon, x, b)
        L_2eps = self.laplace_transform(2 * epsilon, x, b)

        # Forward difference approximation
        derivative = (L_2eps - L_plus) / epsilon

        expected_time = -derivative

        return float(np.clip(expected_time, 0.0, max_time))

    def conditional_expected_overshoot(
        self, x: float, b: float, t: float
    ) -> float:
        """Compute E[b - X_{τ_b} | τ_b ≤ t], expected overshoot given liquidation.

        The overshoot is how far below the barrier the process is when it
        first crosses. For DeFi, this relates to the liquidation shortfall.

        For spectrally negative processes, the overshoot distribution depends
        on whether there are jumps.

        Args:
            x: Starting position.
            b: Liquidation threshold.
            t: Time horizon.

        Returns:
            Expected overshoot (non-negative).
        """
        # For processes with jumps, overshoot can be positive
        # For pure diffusion, overshoot is 0 (continuous crossing)
        if self.params.lambda_ == 0:
            return 0.0

        # Approximate via simulation or analytical formula
        # For now, use the expected jump size as a rough estimate
        return self.params.mean_jump_size

    def from_health_factor(
        self,
        health_factor: float,
        t: float,
        liquidation_hf: float = 1.0,
    ) -> float:
        """Convenience method to compute liquidation probability from health factor.

        Converts health factor to log-space and computes probability.

        Args:
            health_factor: Current health factor (> 1 for healthy position).
            t: Time horizon.
            liquidation_hf: Health factor at liquidation (default 1.0).

        Returns:
            Liquidation probability P(HF drops below liquidation_hf by time t).
        """
        if health_factor <= liquidation_hf:
            return 1.0

        x = np.log(health_factor)
        b = np.log(liquidation_hf)
        return self.liquidation_probability(t, x, b)

    def survival_from_health_factor(
        self,
        health_factor: float,
        t: float,
        liquidation_hf: float = 1.0,
    ) -> float:
        """Convenience method to compute survival probability from health factor.

        Args:
            health_factor: Current health factor.
            t: Time horizon.
            liquidation_hf: Health factor at liquidation.

        Returns:
            Survival probability P(HF stays above liquidation_hf until time t).
        """
        return 1.0 - self.from_health_factor(health_factor, t, liquidation_hf)

    def clear_cache(self) -> None:
        """Clear all computation caches."""
        self._laplace_cache.cache_clear()
        self.wiener_hopf.clear_cache()
        self.scale_function.clear_cache()

    def __repr__(self) -> str:
        return (
            f"FirstHittingTime(μ={self.params.mu:.4f}, σ={self.params.sigma:.4f}, "
            f"λ={self.params.lambda_:.4f}, η={self.params.eta:.4f}, δ={self.params.delta:.4f})"
        )


def compute_liquidation_probability_grid(
    params: LevyParameters,
    health_factors: NDArray[np.floating],
    time_horizons: NDArray[np.floating],
    N_stehfest: int = 10,
) -> NDArray[np.floating]:
    """Compute liquidation probabilities over a grid of (HF, t) values.

    Useful for creating heatmaps or surface plots of liquidation risk.

    Args:
        params: Lévy process parameters.
        health_factors: Array of health factor values (> 1).
        time_horizons: Array of time horizons.
        N_stehfest: Gaver-Stehfest terms.

    Returns:
        2D array of shape (len(health_factors), len(time_horizons))
        containing liquidation probabilities.
    """
    fht = FirstHittingTime(params, N_stehfest=N_stehfest)

    n_hf = len(health_factors)
    n_t = len(time_horizons)
    result = np.zeros((n_hf, n_t))

    for i, hf in enumerate(health_factors):
        x = np.log(float(hf))
        for j, t in enumerate(time_horizons):
            result[i, j] = fht.liquidation_probability(float(t), x, b=0.0)

    return result
