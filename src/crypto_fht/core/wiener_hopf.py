"""
Wiener-Hopf factorization for spectrally negative Lévy processes.

For spectrally negative Lévy processes (only downward jumps), the Wiener-Hopf
factorization simplifies considerably. The key object is Φ(q), the right-inverse
of the Laplace exponent ψ.

Φ(q) is defined as the unique positive solution to:
    ψ(Φ(q)) = q

For our process with shifted exponential jumps:
    μΦ + (σ²/2)Φ² + λ(e^{-Φδ} · η/(η+Φ) - 1) = q

The function Φ(q) appears in numerous first-passage formulas for spectrally
negative Lévy processes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq, newton

from crypto_fht.core.levy_process import (
    LevyParameters,
    laplace_exponent,
    laplace_exponent_derivative,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WienerHopfFactorization:
    """Wiener-Hopf factorization for spectrally negative Lévy processes.

    Computes Φ(q), the right-inverse of the Laplace exponent, which is
    fundamental for first-passage problems.

    For spectrally negative processes, Φ(q) is the unique positive root of:
        ψ(Φ) = q

    Properties of Φ(q):
    - Φ(0) = 0
    - Φ(q) > 0 for q > 0
    - Φ is strictly increasing
    - Φ(q) ~ √(2q/σ²) as q → ∞ (if σ > 0)

    Attributes:
        params: Lévy process parameters.
        _cache_size: Number of cached Φ(q) values.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> wh = WienerHopfFactorization(params)
        >>> phi = wh.phi(1.0)
        >>> abs(laplace_exponent(phi, params) - 1.0) < 1e-10
        True
    """

    def __init__(self, params: LevyParameters, cache_size: int = 128) -> None:
        """Initialize Wiener-Hopf factorization.

        Args:
            params: Lévy process parameters.
            cache_size: LRU cache size for Φ(q) computations.
        """
        self.params = params
        self._cache_size = cache_size

        # Create cached version of phi computation
        self._phi_cached = lru_cache(maxsize=cache_size)(self._compute_phi)

    def phi(self, q: float, tol: float = 1e-12) -> float:
        """Compute Φ(q), the right-inverse of the Laplace exponent.

        Solves ψ(Φ) = q for the unique positive root Φ(q).

        Args:
            q: The Laplace transform parameter (q ≥ 0).
            tol: Numerical tolerance for root-finding.

        Returns:
            Φ(q): The unique non-negative root.

        Raises:
            ValueError: If q < 0.
            RuntimeError: If root-finding fails to converge.
        """
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q}")

        if q == 0:
            return 0.0

        # Use cached computation with rounded q for cache efficiency
        q_rounded = round(q, 12)
        return self._phi_cached(q_rounded, tol)

    def _compute_phi(self, q: float, tol: float = 1e-12) -> float:
        """Internal computation of Φ(q) without caching.

        Uses a combination of bracketing (Brent's method) and Newton-Raphson
        for robust and fast convergence.
        """
        # Define the equation to solve: ψ(θ) - q = 0
        def f(theta: float) -> float:
            if theta <= -self.params.eta:
                return float("inf")
            result = laplace_exponent(theta, self.params)
            return float(np.real(result)) - q

        def f_prime(theta: float) -> float:
            if theta <= -self.params.eta:
                return float("inf")
            result = laplace_exponent_derivative(theta, self.params)
            return float(np.real(result))

        # Find upper bound for bracketing
        # For large θ, ψ(θ) ~ (σ²/2)θ², so θ ~ √(2q/σ²)
        if self.params.sigma > 0:
            upper_estimate = max(10.0, 2 * np.sqrt(2 * q / self.params.sigma**2))
        else:
            # Pure jump case: need different estimate
            upper_estimate = max(10.0, q / max(self.params.mu, 0.1))

        # Expand upper bound until we bracket the root
        upper = upper_estimate
        max_expansions = 20
        for _ in range(max_expansions):
            if f(upper) > 0:
                break
            upper *= 2
        else:
            raise RuntimeError(f"Failed to find upper bracket for Φ({q})")

        # Use Brent's method for robust bracketing
        try:
            phi_q = brentq(f, 1e-15, upper, xtol=tol)
        except ValueError as e:
            # Fall back to Newton's method if bracketing fails
            x0 = upper_estimate
            try:
                phi_q = newton(f, x0, fprime=f_prime, tol=tol, maxiter=100)
            except RuntimeError:
                raise RuntimeError(f"Failed to compute Φ({q}): {e}") from e

        return float(phi_q)

    def phi_derivative(self, q: float, tol: float = 1e-10) -> float:
        """Compute Φ'(q), the derivative of the right-inverse.

        By implicit differentiation of ψ(Φ(q)) = q:
            ψ'(Φ(q)) · Φ'(q) = 1
            Φ'(q) = 1 / ψ'(Φ(q))

        Args:
            q: The Laplace parameter (q > 0).
            tol: Tolerance for Φ(q) computation.

        Returns:
            Φ'(q) value.

        Raises:
            ValueError: If q ≤ 0.
        """
        if q <= 0:
            raise ValueError(f"q must be positive for derivative, got {q}")

        phi_q = self.phi(q, tol)
        psi_prime = laplace_exponent_derivative(phi_q, self.params)
        return 1.0 / float(np.real(psi_prime))

    def verify_root(self, q: float, phi_q: float | None = None, tol: float = 1e-10) -> bool:
        """Verify that Φ(q) satisfies ψ(Φ(q)) = q.

        Useful for testing and validation.

        Args:
            q: The Laplace parameter.
            phi_q: Computed Φ(q) value. If None, computes it.
            tol: Tolerance for verification.

        Returns:
            True if |ψ(Φ(q)) - q| < tol.
        """
        if phi_q is None:
            phi_q = self.phi(q)

        psi_phi = laplace_exponent(phi_q, self.params)
        return bool(abs(float(np.real(psi_phi)) - q) < tol)

    def compute_phi_array(
        self, q_values: NDArray[np.floating], tol: float = 1e-12
    ) -> NDArray[np.floating]:
        """Compute Φ(q) for an array of q values.

        Args:
            q_values: Array of non-negative q values.
            tol: Numerical tolerance.

        Returns:
            Array of Φ(q) values with same shape as input.
        """
        return np.array([self.phi(float(q), tol) for q in q_values.flat]).reshape(
            q_values.shape
        )

    def clear_cache(self) -> None:
        """Clear the Φ(q) computation cache."""
        self._phi_cached.cache_clear()

    @property
    def cache_info(self) -> str:
        """Return cache statistics."""
        info = self._phi_cached.cache_info()
        return f"hits={info.hits}, misses={info.misses}, size={info.currsize}/{info.maxsize}"
