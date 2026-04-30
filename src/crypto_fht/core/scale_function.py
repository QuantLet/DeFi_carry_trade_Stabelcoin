"""
Scale functions W^(q)(x) and Z^(q)(x) for spectrally negative Lévy processes.

The q-scale function W^(q)(x) is fundamental for first-passage problems.
It is uniquely characterized by:
- W^(q)(x) = 0 for x < 0
- W^(q) is continuous and strictly increasing on [0, ∞)
- Laplace transform: ∫₀^∞ e^{-θx} W^(q)(x) dx = 1/(ψ(θ) - q) for θ > Φ(q)

The Z^(q)(x) function is defined as:
    Z^(q)(x) = 1 + q ∫₀^x W^(q)(y) dy

For spectrally negative Lévy processes with rational Laplace exponent,
W^(q)(x) can be computed via partial fraction decomposition and series expansion.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad

from crypto_fht.core.levy_process import (
    LevyParameters,
    laplace_exponent,
    laplace_exponent_derivative,
)
from crypto_fht.core.wiener_hopf import WienerHopfFactorization

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ScaleFunction:
    """Scale function W^(q)(x) and Z^(q)(x) for spectrally negative Lévy processes.

    For processes with shifted exponential jumps, the scale function can be
    computed via residue calculus. The Laplace transform identity:

        ∫₀^∞ e^{-θx} W^(q)(x) dx = 1/(ψ(θ) - q)

    leads to the inversion formula using the roots of ψ(θ) = q.

    The computation uses a series expansion based on the poles of 1/(ψ(θ) - q).

    Attributes:
        params: Lévy process parameters.
        wiener_hopf: Wiener-Hopf factorization object.
        n_terms: Number of terms in series expansion.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> wh = WienerHopfFactorization(params)
        >>> sf = ScaleFunction(params, wh)
        >>> W_1 = sf.W(1.0, q=1.0)
        >>> W_1 > 0
        True
    """

    def __init__(
        self,
        params: LevyParameters,
        wiener_hopf: WienerHopfFactorization,
        n_terms: int = 50,
        cache_size: int = 256,
    ) -> None:
        """Initialize scale function calculator.

        Args:
            params: Lévy process parameters.
            wiener_hopf: Pre-computed Wiener-Hopf factorization.
            n_terms: Number of terms for series expansion.
            cache_size: Cache size for W(x, q) values.
        """
        self.params = params
        self.wiener_hopf = wiener_hopf
        self.n_terms = n_terms

        # Create cached versions
        self._W_cached = lru_cache(maxsize=cache_size)(self._compute_W)
        self._Z_cached = lru_cache(maxsize=cache_size)(self._compute_Z)

    def W(self, x: float, q: float, tol: float = 1e-10) -> float:
        """Compute the q-scale function W^(q)(x).

        Uses series expansion based on the residues at poles of 1/(ψ(θ) - q).

        For x ≥ 0:
            W^(q)(x) = e^{Φ(q)x} / ψ'(Φ(q)) + Σ_k (residue contributions)

        where the sum is over other roots of ψ(θ) = q.

        Args:
            x: Spatial argument.
            q: Laplace parameter (q ≥ 0).
            tol: Numerical tolerance.

        Returns:
            W^(q)(x) value.
        """
        if x < 0:
            return 0.0

        if q == 0:
            return self._compute_W_zero(x)

        # Use cached computation with rounded arguments
        x_rounded = round(x, 10)
        q_rounded = round(q, 10)
        return self._W_cached(x_rounded, q_rounded, tol)

    def _compute_W(self, x: float, q: float, tol: float = 1e-10) -> float:
        """Internal computation of W^(q)(x) via series expansion.

        The function 1/(ψ(θ) - q) has poles at the roots of ψ(θ) = q.
        The dominant root is Φ(q) > 0.

        For processes with σ > 0, 1/(ψ(θ) - q) has a simple pole at Φ(q),
        giving the leading term:
            e^{Φ(q)x} / ψ'(Φ(q))
        """
        # Get Φ(q) - the positive root
        phi_q = self.wiener_hopf.phi(q, tol)

        # Compute ψ'(Φ(q)) for the residue
        psi_prime_phi = laplace_exponent_derivative(phi_q, self.params)
        psi_prime_phi_real = float(np.real(psi_prime_phi))

        if abs(psi_prime_phi_real) < 1e-15:
            raise ValueError(f"ψ'(Φ({q})) ≈ 0, cannot compute scale function")

        # Main term from the positive root
        result = np.exp(phi_q * x) / psi_prime_phi_real

        # Add contributions from negative roots
        negative_roots = self._find_negative_roots(q, tol)

        for root in negative_roots:
            psi_prime_root = laplace_exponent_derivative(root, self.params)
            psi_prime_root_real = float(np.real(psi_prime_root))

            if abs(psi_prime_root_real) > 1e-15:
                result += np.exp(root * x) / psi_prime_root_real

        return max(0.0, float(result))

    def _compute_W_zero(self, x: float) -> float:
        """Compute W^(0)(x), the scale function at q = 0.

        For q = 0, ψ(0) = 0, so θ = 0 is a root.
        The behavior depends on whether the process drifts to -∞.

        For μ - λ(δ + 1/η) < 0 (drifting down):
            W^(0)(x) = 1/|μ_eff| for large x (approximately)

        This case is handled via numerical inversion.
        """
        if x <= 0:
            return 0.0

        # For q = 0, use limit q → 0+
        # W^(0)(x) = lim_{q→0} W^(q)(x)
        q_small = 1e-8
        return self._compute_W(x, q_small, tol=1e-10)

    def _find_negative_roots(self, q: float, tol: float = 1e-10) -> list[float]:
        """Find negative roots of ψ(θ) = q.

        For spectrally negative Lévy processes with Gaussian component (σ > 0)
        and compound Poisson jumps, ψ(θ) - q = 0 can have additional negative
        roots besides the positive root Φ(q).

        The equation ψ(θ) - q = 0 with shifted exponential jumps becomes
        transcendental due to the e^{-θδ} term.

        Returns:
            List of negative roots in the valid domain (> -η).
        """
        roots = []

        def f(theta: float) -> float:
            if theta <= -self.params.eta + 1e-10:
                return float("inf")
            psi_val = laplace_exponent(theta, self.params)
            return float(np.real(psi_val)) - q

        # Search for roots in (-η, 0)
        # The function may have at most one negative root in this interval
        # due to the structure of the Laplace exponent

        search_points = np.linspace(-self.params.eta + 0.01, -0.01, 20)

        for i in range(len(search_points) - 1):
            left, right = search_points[i], search_points[i + 1]
            f_left, f_right = f(left), f(right)

            # Check for sign change
            if np.isfinite(f_left) and np.isfinite(f_right) and f_left * f_right < 0:
                from scipy.optimize import brentq

                try:
                    root = brentq(f, left, right, xtol=tol)
                    # Verify it's actually a root
                    if abs(f(root)) < tol * 10:
                        roots.append(root)
                except ValueError:
                    pass

        return roots

    def Z(self, x: float, q: float, tol: float = 1e-10) -> float:
        """Compute Z^(q)(x) = 1 + q ∫₀^x W^(q)(y) dy.

        Z^(q)(x) appears in many first-passage formulas alongside W^(q)(x).

        Args:
            x: Spatial argument.
            q: Laplace parameter (q ≥ 0).
            tol: Numerical tolerance.

        Returns:
            Z^(q)(x) value.
        """
        if x <= 0:
            return 1.0

        if q == 0:
            return 1.0

        # Use cached computation
        x_rounded = round(x, 10)
        q_rounded = round(q, 10)
        return self._Z_cached(x_rounded, q_rounded, tol)

    def _compute_Z(self, x: float, q: float, tol: float = 1e-10) -> float:
        """Internal computation of Z^(q)(x) via numerical integration."""
        # Z^(q)(x) = 1 + q ∫₀^x W^(q)(y) dy
        integral, _ = quad(
            lambda y: self.W(y, q, tol),
            0,
            x,
            limit=100,
            epsabs=tol,
            epsrel=tol,
        )
        return 1.0 + q * integral

    def W_prime(self, x: float, q: float, tol: float = 1e-10) -> float:
        """Compute the derivative W'^(q)(x) = dW^(q)/dx.

        For the series representation:
            W'^(q)(x) = Σ_k (root_k) · (residue_k) · e^{root_k · x}

        Args:
            x: Spatial argument (x > 0).
            q: Laplace parameter (q > 0).
            tol: Numerical tolerance.

        Returns:
            W'^(q)(x) value.
        """
        if x < 0:
            return 0.0

        if x == 0 and self.params.sigma > 0:
            # W'^(q)(0+) = 2/σ² for Lévy processes with Gaussian component
            return 2.0 / self.params.sigma**2

        # Compute via differentiation of series
        phi_q = self.wiener_hopf.phi(q, tol)
        psi_prime_phi = float(np.real(laplace_exponent_derivative(phi_q, self.params)))

        result = phi_q * np.exp(phi_q * x) / psi_prime_phi

        for root in self._find_negative_roots(q, tol):
            psi_prime_root = float(np.real(laplace_exponent_derivative(root, self.params)))
            if abs(psi_prime_root) > 1e-15:
                result += root * np.exp(root * x) / psi_prime_root

        return float(result)

    def W_at_zero(self, q: float) -> float:
        """Compute W^(q)(0).

        For processes with σ > 0: W^(q)(0) = 0
        For processes with σ = 0 (pure jump): W^(q)(0) > 0

        Args:
            q: Laplace parameter (q > 0).

        Returns:
            W^(q)(0) value.
        """
        if self.params.sigma > 0:
            return 0.0
        else:
            # Pure jump case: W^(q)(0) = 1/c where c is the jump activity
            return self.W(1e-10, q)

    def evaluate_array(
        self,
        x_values: NDArray[np.floating],
        q: float,
        tol: float = 1e-10,
    ) -> NDArray[np.floating]:
        """Evaluate W^(q)(x) for an array of x values.

        Args:
            x_values: Array of spatial arguments.
            q: Laplace parameter.
            tol: Numerical tolerance.

        Returns:
            Array of W^(q)(x) values with same shape as input.
        """
        return np.array([self.W(float(x), q, tol) for x in x_values.flat]).reshape(
            x_values.shape
        )

    def clear_cache(self) -> None:
        """Clear computation caches."""
        self._W_cached.cache_clear()
        self._Z_cached.cache_clear()
