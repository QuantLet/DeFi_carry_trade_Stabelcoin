"""
Gaver-Stehfest numerical Laplace transform inversion.

The Gaver-Stehfest algorithm approximates the inverse Laplace transform:

    f(t) ≈ (ln 2 / t) Σ_{k=1}^{N} v_k · F(k · ln 2 / t)

where F(s) = ∫₀^∞ e^{-st} f(t) dt is the Laplace transform and v_k are
the Stehfest weights.

The algorithm is well-suited for smooth, non-oscillatory functions and
provides good accuracy with N = 8-12 terms for most applications.

References:
    - Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace
      transforms. Communications of the ACM, 13(1), 47-49.
    - Kuznetsov, A. (2013). On the convergence of the Gaver-Stehfest
      algorithm. SIAM Journal on Numerical Analysis.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import numpy as np
from mpmath import factorial as mp_factorial
from mpmath import mp, mpf

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GaverStehfestInverter:
    """Gaver-Stehfest numerical Laplace transform inversion.

    Implements the Stehfest algorithm with high-precision arithmetic
    (via mpmath) for computing the weights to handle large intermediate values.

    The algorithm approximates:
        f(t) ≈ (ln 2 / t) Σ_{k=1}^{N} v_k · F(k · ln 2 / t)

    where F is the Laplace transform of f.

    Attributes:
        N: Number of terms (must be even, typically 8-14).
        weights: Precomputed Stehfest weights.

    Example:
        >>> inverter = GaverStehfestInverter(N=10)
        >>> # Invert F(s) = 1/(s+a) which corresponds to f(t) = e^{-at}
        >>> def F(s): return 1 / (s + 2)
        >>> inverter.invert(F, 1.0)  # Should be close to e^{-2}
        0.1353...
    """

    def __init__(self, N: int = 10) -> None:
        """Initialize inverter with N terms.

        Args:
            N: Number of terms (must be even, 4 ≤ N ≤ 18 recommended).

        Raises:
            ValueError: If N is not even or out of recommended range.
        """
        if N % 2 != 0:
            raise ValueError(f"N must be even, got {N}")
        if N < 4 or N > 18:
            raise ValueError(f"N should be between 4 and 18 for stability, got {N}")

        self.N = N
        self.weights = self._compute_weights()
        self._ln2 = np.log(2)

    @lru_cache(maxsize=1)
    def _compute_weights(self) -> list[float]:
        """Compute Stehfest weights using high-precision arithmetic.

        The weights are given by:
            v_k = (-1)^{N/2+k} Σ_{j=⌊(k+1)/2⌋}^{min(k,N/2)}
                  j^{N/2} (2j)! / ((N/2-j)! j! (j-1)! (k-j)! (2j-k)!)

        High precision (mpmath) is essential due to large intermediate values
        that can overflow standard floating-point arithmetic.

        Returns:
            List of N weights as standard Python floats.
        """
        # Use high precision for intermediate calculations
        mp.dps = 50
        N = self.N
        N2 = N // 2

        weights = []
        for k in range(1, N + 1):
            v_k = mpf(0)
            j_min = (k + 1) // 2
            j_max = min(k, N2)

            for j in range(j_min, j_max + 1):
                numerator = mpf(j) ** N2 * mp_factorial(2 * j)
                denominator = (
                    mp_factorial(N2 - j)
                    * mp_factorial(j)
                    * mp_factorial(j - 1)
                    * mp_factorial(k - j)
                    * mp_factorial(2 * j - k)
                )
                v_k += numerator / denominator

            sign = (-1) ** (N2 + k)
            weights.append(float(sign * v_k))

        return weights

    def invert(
        self,
        F: Callable[[float], float],
        t: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Invert Laplace transform F at time t.

        Computes f(t) where F(s) = ∫₀^∞ e^{-st} f(t) dt.

        Args:
            F: Laplace transform function F(s). Must accept positive real s.
            t: Time point(s) at which to evaluate f. Must be positive.

        Returns:
            f(t) approximation. Same shape as input t.

        Raises:
            ValueError: If any t ≤ 0.

        Note:
            The algorithm works best for smooth, non-oscillatory functions.
            For functions with poles far from the real axis or with
            oscillatory behavior, consider Talbot's method instead.
        """
        if isinstance(t, np.ndarray):
            return np.array([self._invert_single(F, float(ti)) for ti in t.flat]).reshape(
                t.shape
            )
        else:
            return self._invert_single(F, t)

    def _invert_single(self, F: Callable[[float], float], t: float) -> float:
        """Invert at a single time point."""
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        factor = self._ln2 / t
        result = 0.0

        for k, v_k in enumerate(self.weights, start=1):
            s = k * factor
            try:
                F_s = F(s)
                if np.isfinite(F_s):
                    result += v_k * F_s
            except (ValueError, ZeroDivisionError, OverflowError):
                # Skip problematic evaluations
                pass

        return factor * result

    def invert_cdf(
        self,
        laplace_cdf: Callable[[float], float],
        t: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Invert to obtain CDF F_T(t) = P(T ≤ t) from Laplace transform.

        For a random variable T with Laplace transform E[e^{-sT}] = L(s),
        the CDF F_T(t) = P(T ≤ t) has Laplace transform L{F_T}(s) = L(s)/s.

        This method inverts L{F_T}(s) = L(s)/s to get P(T ≤ t).

        Args:
            laplace_cdf: Function computing E[e^{-sT}]/s = L(s)/s.
            t: Time point(s).

        Returns:
            CDF value(s) P(T ≤ t), clipped to [0, 1].
        """
        result = self.invert(laplace_cdf, t)

        # Ensure valid probability
        if isinstance(result, np.ndarray):
            return np.clip(result, 0.0, 1.0)
        else:
            return max(0.0, min(1.0, result))

    def invert_survival(
        self,
        laplace_transform: Callable[[float], float],
        t: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Compute survival probability P(T > t) from Laplace transform of T.

        For a non-negative random variable T with E[e^{-sT}] = L(s):
            L{P(T > t)}(s) = (1 - L(s)) / s

        Args:
            laplace_transform: Function computing E[e^{-sT}].
            t: Time point(s).

        Returns:
            Survival probability P(T > t), clipped to [0, 1].
        """

        def survival_laplace(s: float) -> float:
            L_s = laplace_transform(s)
            return (1.0 - L_s) / s

        result = self.invert(survival_laplace, t)

        if isinstance(result, np.ndarray):
            return np.clip(result, 0.0, 1.0)
        else:
            return max(0.0, min(1.0, result))

    @property
    def precision_estimate(self) -> int:
        """Estimate number of correct decimal places for N terms.

        Empirical relationship: accuracy ≈ N/3 decimal places for
        well-behaved functions.

        Returns:
            Estimated decimal places of accuracy.
        """
        return self.N // 3

    def __repr__(self) -> str:
        return f"GaverStehfestInverter(N={self.N})"


class TalbotInverter:
    """Talbot method for Laplace transform inversion (alternative to Gaver-Stehfest).

    The Talbot method uses contour integration with a specially designed
    contour that avoids singularities. It is more robust for oscillatory
    functions but requires complex-valued function evaluations.

    This is provided as an alternative when Gaver-Stehfest produces poor results.

    Attributes:
        M: Number of terms in the Talbot sum.

    Example:
        >>> inverter = TalbotInverter(M=32)
        >>> def F(s): return 1 / (s + 2)
        >>> inverter.invert(F, 1.0)
        0.1353...
    """

    def __init__(self, M: int = 32) -> None:
        """Initialize Talbot inverter.

        Args:
            M: Number of terms (default 32, higher for more precision).
        """
        self.M = M

    def invert(
        self,
        F: Callable[[complex], complex],
        t: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Invert Laplace transform at time t using Talbot's method.

        Args:
            F: Laplace transform function accepting complex argument.
            t: Time point(s) at which to evaluate.

        Returns:
            Inverse transform value(s).
        """
        if isinstance(t, np.ndarray):
            return np.array([self._invert_single(F, float(ti)) for ti in t.flat]).reshape(
                t.shape
            )
        else:
            return self._invert_single(F, t)

    def _invert_single(self, F: Callable[[complex], complex], t: float) -> float:
        """Talbot inversion at a single time point."""
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")

        M = self.M

        # Talbot parameters
        def theta(k: int) -> float:
            return k * np.pi / M

        def sigma(th: float) -> complex:
            if abs(th) < 1e-10:
                return M / t * (0.5 + 0j)
            cot_th = 1 / np.tan(th)
            return M / t * (th * cot_th + 1j * th)

        def sigma_prime(th: float) -> complex:
            if abs(th) < 1e-10:
                return M / t * (0 + 1j)
            cot_th = 1 / np.tan(th)
            return M / t * (cot_th - th / np.sin(th) ** 2 + 1j)

        # Compute sum
        result = 0.5 * np.real(np.exp(sigma(0) * t) * F(sigma(0)))

        for k in range(1, M):
            th = theta(k)
            s = sigma(th)
            sp = sigma_prime(th)
            result += np.real(np.exp(s * t) * F(s) * sp)

        return float(result / M)

    def __repr__(self) -> str:
        return f"TalbotInverter(M={self.M})"


def create_inverter(
    method: str = "gaver_stehfest", **kwargs: int
) -> GaverStehfestInverter | TalbotInverter:
    """Factory function to create a Laplace inverter.

    Args:
        method: Either "gaver_stehfest" or "talbot".
        **kwargs: Parameters passed to the inverter constructor.

    Returns:
        Appropriate inverter instance.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "gaver_stehfest":
        N = kwargs.get("N", 10)
        return GaverStehfestInverter(N=N)
    elif method == "talbot":
        M = kwargs.get("M", 32)
        return TalbotInverter(M=M)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gaver_stehfest' or 'talbot'.")
