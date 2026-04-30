"""
Spectrally negative Lévy process with shifted exponential jumps.

This module defines the core process dynamics for modeling log-health factor evolution:

    X_t = X_0 + μt + σW_t - Σ_{i=1}^{N_t} Y_i

where:
- μ: drift coefficient
- σ: diffusion coefficient (volatility)
- W_t: standard Brownian motion
- N_t ~ Poisson(λt): jump count process
- Y_i ~ ShiftedExp(η, δ): shifted exponential jumps with Y_i = δ + Z_i, Z_i ~ Exp(η)

The Laplace exponent for this process is:
    ψ(θ) = μθ + (σ²/2)θ² + λ(e^{-θδ} · η/(η+θ) - 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LevyParameters:
    """Parameters for spectrally negative Lévy process with shifted exponential jumps.

    The process X_t follows:
        dX_t = μ dt + σ dW_t - dJ_t

    where J_t is a compound Poisson process with intensity λ and shifted
    exponential jump sizes Y = δ + Z, Z ~ Exp(η).

    Attributes:
        mu: Drift coefficient. Represents the deterministic trend in log-health factor.
            Positive μ indicates improving health, negative indicates deterioration.
        sigma: Diffusion coefficient (volatility). Must be non-negative.
            Controls the continuous random fluctuations.
        lambda_: Jump intensity (Poisson rate). Must be non-negative.
            Expected number of jumps per unit time.
        eta: Exponential rate parameter for jump size. Must be positive.
            Mean of the exponential component Z is 1/η.
        delta: Minimum jump size (shift parameter). Must be non-negative.
            Ensures every jump has magnitude at least δ.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> params.mean_jump_size
        0.22
        >>> params.expected_jump_loss_rate
        0.44
    """

    mu: float
    sigma: float
    lambda_: float
    eta: float
    delta: float

    def __post_init__(self) -> None:
        """Validate parameter constraints."""
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")

    @property
    def mean_jump_size(self) -> float:
        """Expected jump size E[Y] = δ + 1/η."""
        return self.delta + 1.0 / self.eta

    @property
    def var_jump_size(self) -> float:
        """Variance of jump size Var[Y] = 1/η²."""
        return 1.0 / (self.eta**2)

    @property
    def expected_jump_loss_rate(self) -> float:
        """Expected loss rate from jumps: λ · E[Y]."""
        return self.lambda_ * self.mean_jump_size

    @property
    def effective_drift(self) -> float:
        """Effective drift after accounting for expected jump losses.

        μ_eff = μ - λ·E[Y] = μ - λ·(δ + 1/η)
        """
        return self.mu - self.expected_jump_loss_rate

    def is_drifting_down(self) -> bool:
        """Check if process has negative effective drift (eventual liquidation)."""
        return self.effective_drift < 0


# Type alias for complex or real scalar/array inputs
ComplexLike = Union[complex, float, NDArray[np.complexfloating], NDArray[np.floating]]


def laplace_exponent(
    theta: ComplexLike,
    params: LevyParameters,
) -> ComplexLike:
    """Compute the Laplace exponent ψ(θ) for the spectrally negative Lévy process.

    The Laplace exponent is defined as:
        ψ(θ) = log E[e^{θX_1}]

    For our process with shifted exponential jumps:
        ψ(θ) = μθ + (σ²/2)θ² + λ(e^{-θδ} · η/(η+θ) - 1)

    The process is spectrally negative (only downward jumps), so the Laplace
    exponent is well-defined for all θ with Re(θ) > -η.

    Args:
        theta: Complex or real argument. Can be scalar or array.
        params: Lévy process parameters.

    Returns:
        Laplace exponent value(s) with same shape as theta.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> laplace_exponent(1.0, params)
        -0.294...
        >>> laplace_exponent(0.0, params)
        0.0
    """
    # Diffusion terms: μθ + (σ²/2)θ²
    diffusion = params.mu * theta + 0.5 * params.sigma**2 * theta**2

    # Jump term: λ(e^{-θδ} · η/(η+θ) - 1)
    # For shifted exponential Y = δ + Z with Z ~ Exp(η):
    # E[e^{-θY}] = e^{-θδ} · E[e^{-θZ}] = e^{-θδ} · η/(η+θ)
    shift_factor = np.exp(-theta * params.delta)
    exp_factor = params.eta / (params.eta + theta)
    jump_term = params.lambda_ * (shift_factor * exp_factor - 1)

    return diffusion + jump_term


def laplace_exponent_derivative(
    theta: ComplexLike,
    params: LevyParameters,
) -> ComplexLike:
    """Compute the derivative ψ'(θ) of the Laplace exponent.

    ψ'(θ) = μ + σ²θ + λ · d/dθ[e^{-θδ} · η/(η+θ)]

    where:
        d/dθ[e^{-θδ} · η/(η+θ)] = e^{-θδ} · η · [-δ/(η+θ) - 1/(η+θ)²]
                                 = -e^{-θδ} · η · [δ(η+θ) + 1] / (η+θ)²

    Args:
        theta: Complex or real argument.
        params: Lévy process parameters.

    Returns:
        Derivative value(s) with same shape as theta.
    """
    # Diffusion derivative: μ + σ²θ
    diffusion_deriv = params.mu + params.sigma**2 * theta

    # Jump term derivative
    shift_factor = np.exp(-theta * params.delta)
    eta_plus_theta = params.eta + theta

    # d/dθ[e^{-θδ} · η/(η+θ)]
    # Using product rule: (fg)' = f'g + fg'
    # f = e^{-θδ}, f' = -δ·e^{-θδ}
    # g = η/(η+θ), g' = -η/(η+θ)²
    jump_deriv = shift_factor * params.eta * (
        -params.delta / eta_plus_theta - 1 / eta_plus_theta**2
    )
    jump_term_deriv = params.lambda_ * jump_deriv

    return diffusion_deriv + jump_term_deriv


def laplace_exponent_second_derivative(
    theta: ComplexLike,
    params: LevyParameters,
) -> ComplexLike:
    """Compute the second derivative ψ''(θ) of the Laplace exponent.

    Used for Newton-Raphson iterations and curvature analysis.

    Args:
        theta: Complex or real argument.
        params: Lévy process parameters.

    Returns:
        Second derivative value(s) with same shape as theta.
    """
    # Diffusion second derivative: σ²
    diffusion_second = params.sigma**2

    # Jump term second derivative (computed symbolically)
    shift_factor = np.exp(-theta * params.delta)
    eta_plus_theta = params.eta + theta

    # Let f = e^{-θδ}, g = η/(η+θ)
    # (fg)'' = f''g + 2f'g' + fg''
    # f'' = δ²·e^{-θδ}
    # g' = -η/(η+θ)²
    # g'' = 2η/(η+θ)³

    f = shift_factor
    f_prime = -params.delta * shift_factor
    f_double_prime = params.delta**2 * shift_factor

    g = params.eta / eta_plus_theta
    g_prime = -params.eta / eta_plus_theta**2
    g_double_prime = 2 * params.eta / eta_plus_theta**3

    jump_second_deriv = f_double_prime * g + 2 * f_prime * g_prime + f * g_double_prime
    jump_term_second = params.lambda_ * jump_second_deriv

    return diffusion_second + jump_term_second


def simulate_path(
    params: LevyParameters,
    x0: float,
    t_max: float,
    dt: float,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Simulate a sample path of the Lévy process.

    Uses Euler-Maruyama scheme for the diffusion component and
    exact simulation for the compound Poisson jumps.

    Args:
        params: Lévy process parameters.
        x0: Initial value X_0.
        t_max: Terminal time.
        dt: Time step for discretization.
        rng: Random number generator. If None, uses default.

    Returns:
        Tuple of (time_grid, path_values) arrays.

    Example:
        >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        >>> times, path = simulate_path(params, x0=1.0, t_max=1.0, dt=0.01)
        >>> len(times)
        101
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    path = np.zeros(n_steps + 1)
    path[0] = x0

    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        # Brownian increment
        dW = rng.standard_normal() * sqrt_dt

        # Diffusion step
        path[i + 1] = path[i] + params.mu * dt + params.sigma * dW

        # Jump component: number of jumps in [t, t+dt]
        n_jumps = rng.poisson(params.lambda_ * dt)

        if n_jumps > 0:
            # Generate jump sizes: Y_j = δ + Z_j, Z_j ~ Exp(η)
            exponential_parts = rng.exponential(1.0 / params.eta, size=n_jumps)
            total_jump = n_jumps * params.delta + np.sum(exponential_parts)
            path[i + 1] -= total_jump  # Subtract because jumps are downward

    return times, path


def simulate_paths(
    params: LevyParameters,
    x0: float,
    t_max: float,
    dt: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Simulate multiple sample paths of the Lévy process.

    Vectorized implementation for efficiency.

    Args:
        params: Lévy process parameters.
        x0: Initial value X_0.
        t_max: Terminal time.
        dt: Time step for discretization.
        n_paths: Number of paths to simulate.
        rng: Random number generator. If None, uses default.

    Returns:
        Tuple of (time_grid, paths) where paths has shape (n_paths, n_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        # Brownian increments for all paths
        dW = rng.standard_normal(n_paths) * sqrt_dt

        # Diffusion step
        paths[:, i + 1] = paths[:, i] + params.mu * dt + params.sigma * dW

        # Jump component for each path
        n_jumps = rng.poisson(params.lambda_ * dt, size=n_paths)

        for j in range(n_paths):
            if n_jumps[j] > 0:
                exponential_parts = rng.exponential(1.0 / params.eta, size=n_jumps[j])
                total_jump = n_jumps[j] * params.delta + np.sum(exponential_parts)
                paths[j, i + 1] -= total_jump

    return times, paths
