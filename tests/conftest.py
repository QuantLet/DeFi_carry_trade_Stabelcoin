"""Pytest fixtures for crypto-fht tests."""

import numpy as np
import pytest

from crypto_fht.core.levy_process import LevyParameters


@pytest.fixture
def sample_levy_params() -> LevyParameters:
    """Standard test parameters for Lévy process."""
    return LevyParameters(
        mu=0.01,
        sigma=0.3,
        lambda_=2.0,
        eta=5.0,
        delta=0.02,
    )


@pytest.fixture
def sample_returns(sample_levy_params: LevyParameters) -> np.ndarray:
    """Generate sample returns for testing."""
    rng = np.random.default_rng(42)
    n = 500
    dt = 1 / 365  # Daily

    # Simulate returns from Lévy process
    returns = np.zeros(n)
    for i in range(n):
        # Diffusion
        dW = rng.standard_normal() * np.sqrt(dt)
        returns[i] = sample_levy_params.mu * dt + sample_levy_params.sigma * dW

        # Jumps
        n_jumps = rng.poisson(sample_levy_params.lambda_ * dt)
        if n_jumps > 0:
            jump_sizes = (
                sample_levy_params.delta
                + rng.exponential(1.0 / sample_levy_params.eta, size=n_jumps)
            )
            returns[i] -= np.sum(jump_sizes)

    return returns


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed random generator for reproducibility."""
    return np.random.default_rng(42)
