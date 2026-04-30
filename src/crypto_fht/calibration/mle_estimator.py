"""
Maximum Likelihood Estimation for spectrally negative Lévy processes.

Estimates the 5 parameters (μ, σ, λ, η, δ) from observed returns using
an EM algorithm that treats jump occurrences as latent variables.

The model:
    X_t = X_0 + μt + σW_t - Σ_{i=1}^{N_t} Y_i
    Y_i = δ + Z_i, Z_i ~ Exp(η)
    N_t ~ Poisson(λt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, norm

from crypto_fht.core.levy_process import LevyParameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CalibrationResult:
    """Result of parameter calibration.

    Attributes:
        params: Estimated LevyParameters.
        log_likelihood: Log-likelihood at optimum.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        n_observations: Number of observations used.
        n_iterations: Number of EM iterations.
        converged: Whether algorithm converged.
        standard_errors: Standard errors for parameters (optional).
    """

    params: LevyParameters
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_iterations: int
    converged: bool
    standard_errors: NDArray[np.floating] | None = None

    def summary(self) -> str:
        """Return formatted summary string."""
        se = self.standard_errors
        se_str = (
            f"  SE: [{se[0]:.4f}, {se[1]:.4f}, {se[2]:.4f}, {se[3]:.4f}, {se[4]:.4f}]"
            if se is not None
            else "  SE: not computed"
        )

        return f"""
Lévy Process Calibration Results
================================
Parameters:
  μ (drift):     {self.params.mu:.6f}
  σ (volatility): {self.params.sigma:.6f}
  λ (intensity): {self.params.lambda_:.6f}
  η (jump rate): {self.params.eta:.6f}
  δ (shift):     {self.params.delta:.6f}

{se_str}

Fit Statistics:
  Log-likelihood: {self.log_likelihood:.4f}
  AIC: {self.aic:.4f}
  BIC: {self.bic:.4f}

  Observations: {self.n_observations}
  Iterations: {self.n_iterations}
  Converged: {self.converged}

Derived Quantities:
  Mean jump size E[Y] = δ + 1/η = {self.params.mean_jump_size:.6f}
  Expected jump rate = λ·E[Y] = {self.params.expected_jump_loss_rate:.6f}
  Effective drift = μ - λ·E[Y] = {self.params.effective_drift:.6f}
"""


class LevyMLEEstimator:
    """MLE estimator for Lévy process with shifted exponential jumps.

    Uses EM algorithm:
    - E-step: Estimate jump indicators and sizes given current parameters
    - M-step: Update parameters given jump estimates

    Attributes:
        dt: Time step between observations (in years or consistent units).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.

    Example:
        >>> returns = np.diff(np.log(prices))  # Log-returns
        >>> estimator = LevyMLEEstimator(dt=1/365)  # Daily data
        >>> result = estimator.fit(returns)
        >>> print(result.summary())
    """

    def __init__(
        self,
        dt: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        """Initialize estimator.

        Args:
            dt: Time step between observations.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance for log-likelihood.
        """
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        returns: NDArray[np.floating],
        initial_params: LevyParameters | None = None,
    ) -> CalibrationResult:
        """Fit Lévy parameters to return data.

        Args:
            returns: Array of log-returns.
            initial_params: Initial parameter guess. If None, uses heuristics.

        Returns:
            CalibrationResult with estimated parameters.
        """
        n = len(returns)
        if n < 10:
            raise ValueError(f"Need at least 10 observations, got {n}")

        # Initialize parameters
        if initial_params is None:
            params = self._initialize_params(returns)
        else:
            params = initial_params

        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E-step: estimate jump probabilities and sizes
            p_jump, expected_jump_size = self._e_step(returns, params)

            # M-step: update parameters
            params = self._m_step(returns, p_jump, expected_jump_size)

            # Compute log-likelihood
            ll = self._log_likelihood(returns, params)

            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                converged = True
                break

            prev_ll = ll

        # Compute information criteria
        k = 5  # Number of parameters
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n)

        # Compute standard errors (optional)
        try:
            se = self._compute_standard_errors(returns, params)
        except Exception:
            se = None

        return CalibrationResult(
            params=params,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            n_observations=n,
            n_iterations=iteration + 1,
            converged=converged,
            standard_errors=se,
        )

    def _initialize_params(self, returns: NDArray[np.floating]) -> LevyParameters:
        """Initialize parameters using moment matching."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Identify potential jumps (large negative returns)
        threshold = mean_return - 2 * std_return
        jump_candidates = returns[returns < threshold]

        if len(jump_candidates) > 0:
            # Estimate jump parameters from tail
            n_jumps = len(jump_candidates)
            lambda_init = n_jumps / (len(returns) * self.dt)
            mean_jump = np.mean(np.abs(jump_candidates))
            delta_init = max(0, np.min(np.abs(jump_candidates)) * 0.5)
            eta_init = 1.0 / max(mean_jump - delta_init, 0.01)
        else:
            lambda_init = 0.5
            eta_init = 5.0
            delta_init = 0.01

        # Adjust drift and volatility for jumps
        mu_init = mean_return / self.dt + lambda_init * (delta_init + 1 / eta_init)
        sigma_init = std_return / np.sqrt(self.dt) * 0.8  # Reduce for jump contribution

        return LevyParameters(
            mu=mu_init,
            sigma=max(sigma_init, 0.01),
            lambda_=max(lambda_init, 0.01),
            eta=max(eta_init, 0.1),
            delta=max(delta_init, 0.0),
        )

    def _e_step(
        self,
        returns: NDArray[np.floating],
        params: LevyParameters,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """E-step: estimate jump probabilities and sizes.

        For each observation x_i, compute:
        - P(jump occurred | x_i, params)
        - E[jump size | x_i, jump occurred, params]
        """
        n = len(returns)
        p_jump = np.zeros(n)
        expected_jump = np.zeros(n)

        # Prior probability of at least one jump in interval dt
        p_jump_prior = 1 - np.exp(-params.lambda_ * self.dt)

        # Distribution parameters
        mean_no_jump = params.mu * self.dt
        std_no_jump = params.sigma * np.sqrt(self.dt)

        for i, x in enumerate(returns):
            # Likelihood under no-jump (pure diffusion)
            ll_no_jump = norm.logpdf(x, mean_no_jump, std_no_jump)

            # Likelihood under jump
            ll_jump = self._jump_observation_likelihood(x, params)

            # Posterior probability of jump (Bayes' rule in log space)
            log_posterior_ratio = (
                ll_jump - ll_no_jump
                + np.log(p_jump_prior + 1e-15)
                - np.log(1 - p_jump_prior + 1e-15)
            )

            # Sigmoid transform
            p_jump[i] = 1 / (1 + np.exp(-np.clip(log_posterior_ratio, -500, 500)))

            # Expected jump size given jump occurred
            # Approximate: use the residual after accounting for diffusion
            residual = mean_no_jump - x
            expected_jump[i] = max(params.delta, residual) if residual > 0 else params.delta

        return p_jump, expected_jump

    def _m_step(
        self,
        returns: NDArray[np.floating],
        p_jump: NDArray[np.floating],
        expected_jump: NDArray[np.floating],
    ) -> LevyParameters:
        """M-step: update parameters given jump estimates."""
        n = len(returns)

        # Estimate λ: expected number of jumps per unit time
        expected_n_jumps = np.sum(p_jump)
        lambda_new = expected_n_jumps / (n * self.dt)

        # Estimate δ and η from jump sizes
        weighted_jumps = p_jump * expected_jump
        total_jump_weight = np.sum(p_jump)

        if total_jump_weight > 1:
            mean_jump_size = np.sum(weighted_jumps) / total_jump_weight

            # For shifted exponential: E[Y] = δ + 1/η
            # Estimate δ as minimum observed jump (with smoothing)
            jump_sizes = expected_jump[p_jump > 0.5]
            if len(jump_sizes) > 0:
                delta_new = max(0, np.percentile(jump_sizes, 10))
            else:
                delta_new = 0.01

            # η from mean: η = 1 / (E[Y] - δ)
            excess_mean = mean_jump_size - delta_new
            eta_new = 1.0 / max(excess_mean, 0.01)
        else:
            delta_new = 0.01
            eta_new = 5.0

        # Estimate μ and σ from "de-jumped" returns
        adjusted_returns = returns + p_jump * expected_jump

        mu_new = np.mean(adjusted_returns) / self.dt
        sigma_new = np.std(adjusted_returns) / np.sqrt(self.dt)

        return LevyParameters(
            mu=mu_new,
            sigma=max(sigma_new, 0.01),
            lambda_=max(lambda_new, 0.01),
            eta=max(eta_new, 0.1),
            delta=max(delta_new, 0.0),
        )

    def _jump_observation_likelihood(
        self,
        x: float,
        params: LevyParameters,
    ) -> float:
        """Log-likelihood of observation under jump scenario.

        For Y = δ + Z with Z ~ Exp(η), and X = μΔt + σW - Y:
        f(x | jump) = ∫ f_BM(x + y) × f_Y(y) dy

        Uses numerical integration or analytical approximation.
        """
        from scipy.integrate import quad

        mean_bm = params.mu * self.dt
        std_bm = params.sigma * np.sqrt(self.dt)

        def integrand(y: float) -> float:
            if y < params.delta:
                return 0.0
            # Diffusion density at x + y
            bm_density = norm.pdf(x + y, mean_bm, std_bm)
            # Jump density: shifted exponential
            z = y - params.delta
            jump_density = params.eta * np.exp(-params.eta * z)
            return bm_density * jump_density

        # Integrate over jump sizes
        upper_limit = params.delta + 10 / params.eta
        integral, _ = quad(integrand, params.delta, upper_limit, limit=50)

        return np.log(max(integral, 1e-300))

    def _log_likelihood(
        self,
        returns: NDArray[np.floating],
        params: LevyParameters,
    ) -> float:
        """Compute total log-likelihood of data."""
        ll = 0.0
        p_jump_prior = 1 - np.exp(-params.lambda_ * self.dt)

        mean_no_jump = params.mu * self.dt
        std_no_jump = params.sigma * np.sqrt(self.dt)

        for x in returns:
            ll_no_jump = norm.logpdf(x, mean_no_jump, std_no_jump)
            ll_jump = self._jump_observation_likelihood(x, params)

            # Mixture log-likelihood
            ll += np.logaddexp(
                np.log(1 - p_jump_prior + 1e-15) + ll_no_jump,
                np.log(p_jump_prior + 1e-15) + ll_jump,
            )

        return ll

    def _compute_standard_errors(
        self,
        returns: NDArray[np.floating],
        params: LevyParameters,
    ) -> NDArray[np.floating]:
        """Compute standard errors via numerical Hessian."""
        def neg_ll(theta: NDArray[np.floating]) -> float:
            p = LevyParameters(
                mu=theta[0],
                sigma=max(theta[1], 0.01),
                lambda_=max(theta[2], 0.01),
                eta=max(theta[3], 0.1),
                delta=max(theta[4], 0.0),
            )
            return -self._log_likelihood(returns, p)

        theta = np.array([
            params.mu, params.sigma, params.lambda_, params.eta, params.delta
        ])

        # Numerical Hessian via finite differences
        eps = 1e-5
        n_params = 5
        hessian = np.zeros((n_params, n_params))

        for i in range(n_params):
            for j in range(i, n_params):
                theta_pp = theta.copy()
                theta_pm = theta.copy()
                theta_mp = theta.copy()
                theta_mm = theta.copy()

                theta_pp[i] += eps
                theta_pp[j] += eps
                theta_pm[i] += eps
                theta_pm[j] -= eps
                theta_mp[i] -= eps
                theta_mp[j] += eps
                theta_mm[i] -= eps
                theta_mm[j] -= eps

                hessian[i, j] = (
                    neg_ll(theta_pp) - neg_ll(theta_pm) - neg_ll(theta_mp) + neg_ll(theta_mm)
                ) / (4 * eps * eps)
                hessian[j, i] = hessian[i, j]

        try:
            cov_matrix = np.linalg.inv(hessian)
            return np.sqrt(np.abs(np.diag(cov_matrix)))
        except np.linalg.LinAlgError:
            return np.full(5, np.nan)


def quick_calibrate(
    returns: NDArray[np.floating],
    dt: float = 1.0,
) -> LevyParameters:
    """Quick calibration using moment matching.

    Faster but less accurate than full MLE.

    Args:
        returns: Array of log-returns.
        dt: Time step.

    Returns:
        Estimated LevyParameters.
    """
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    skew = np.mean(((returns - mean_r) / std_r) ** 3) if std_r > 0 else 0

    # Identify jumps via threshold
    threshold = mean_r - 2 * std_r
    jump_returns = returns[returns < threshold]

    if len(jump_returns) > 0:
        lambda_ = len(jump_returns) / (len(returns) * dt)
        delta = max(0, np.percentile(np.abs(jump_returns), 10))
        mean_excess = np.mean(np.abs(jump_returns)) - delta
        eta = 1.0 / max(mean_excess, 0.01)
    else:
        lambda_ = 0.5
        eta = 5.0
        delta = 0.01

    # Adjust moments
    mu = mean_r / dt + lambda_ * (delta + 1 / eta)
    sigma = std_r / np.sqrt(dt) * 0.7

    return LevyParameters(
        mu=mu,
        sigma=max(sigma, 0.01),
        lambda_=max(lambda_, 0.01),
        eta=max(eta, 0.1),
        delta=max(delta, 0.0),
    )
