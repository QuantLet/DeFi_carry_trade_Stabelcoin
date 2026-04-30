"""Tests for core mathematical components."""

import numpy as np
import pytest

from crypto_fht.core.levy_process import (
    LevyParameters,
    laplace_exponent,
    laplace_exponent_derivative,
    simulate_path,
)
from crypto_fht.core.wiener_hopf import WienerHopfFactorization
from crypto_fht.core.scale_function import ScaleFunction
from crypto_fht.core.laplace_inversion import GaverStehfestInverter
from crypto_fht.core.first_hitting_time import FirstHittingTime


class TestLevyParameters:
    """Tests for LevyParameters dataclass."""

    def test_valid_parameters(self) -> None:
        """Test creation with valid parameters."""
        params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
        assert params.mu == 0.01
        assert params.sigma == 0.3
        assert params.lambda_ == 2.0
        assert params.eta == 5.0
        assert params.delta == 0.02

    def test_invalid_sigma(self) -> None:
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma"):
            LevyParameters(mu=0.01, sigma=-0.1, lambda_=2.0, eta=5.0, delta=0.02)

    def test_invalid_eta(self) -> None:
        """Test that non-positive eta raises error."""
        with pytest.raises(ValueError, match="eta"):
            LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=0.0, delta=0.02)

    def test_mean_jump_size(self, sample_levy_params: LevyParameters) -> None:
        """Test mean jump size calculation."""
        expected = sample_levy_params.delta + 1.0 / sample_levy_params.eta
        assert np.isclose(sample_levy_params.mean_jump_size, expected)

    def test_effective_drift(self, sample_levy_params: LevyParameters) -> None:
        """Test effective drift calculation."""
        expected = (
            sample_levy_params.mu
            - sample_levy_params.lambda_ * sample_levy_params.mean_jump_size
        )
        assert np.isclose(sample_levy_params.effective_drift, expected)


class TestLaplaceExponent:
    """Tests for Laplace exponent computation."""

    def test_zero_at_origin(self, sample_levy_params: LevyParameters) -> None:
        """Test ψ(0) = 0."""
        result = laplace_exponent(0.0, sample_levy_params)
        assert np.isclose(float(np.real(result)), 0.0, atol=1e-10)

    def test_positive_for_large_theta(self, sample_levy_params: LevyParameters) -> None:
        """Test ψ(θ) > 0 for large θ."""
        for theta in [1.0, 5.0, 10.0]:
            result = laplace_exponent(theta, sample_levy_params)
            # For large theta, quadratic term dominates
            assert float(np.real(result)) > -sample_levy_params.lambda_

    def test_array_input(self, sample_levy_params: LevyParameters) -> None:
        """Test with array input."""
        theta = np.array([0.1, 0.5, 1.0, 2.0])
        result = laplace_exponent(theta, sample_levy_params)
        assert result.shape == theta.shape


class TestWienerHopfFactorization:
    """Tests for Wiener-Hopf factorization."""

    def test_phi_positive(self, sample_levy_params: LevyParameters) -> None:
        """Test Φ(q) > 0 for q > 0."""
        wh = WienerHopfFactorization(sample_levy_params)
        for q in [0.1, 1.0, 5.0, 10.0]:
            phi = wh.phi(q)
            assert phi > 0, f"Φ({q}) = {phi} should be positive"

    def test_phi_zero_at_zero(self, sample_levy_params: LevyParameters) -> None:
        """Test Φ(0) = 0."""
        wh = WienerHopfFactorization(sample_levy_params)
        phi = wh.phi(0.0)
        assert np.isclose(phi, 0.0, atol=1e-10)

    def test_phi_is_root(self, sample_levy_params: LevyParameters) -> None:
        """Test ψ(Φ(q)) = q."""
        wh = WienerHopfFactorization(sample_levy_params)
        for q in [0.1, 1.0, 5.0]:
            phi_q = wh.phi(q)
            psi_phi = laplace_exponent(phi_q, sample_levy_params)
            assert np.isclose(float(np.real(psi_phi)), q, rtol=1e-8)

    def test_phi_increasing(self, sample_levy_params: LevyParameters) -> None:
        """Test Φ(q) is increasing in q."""
        wh = WienerHopfFactorization(sample_levy_params)
        q_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        phi_values = [wh.phi(q) for q in q_values]
        for i in range(len(phi_values) - 1):
            assert phi_values[i] < phi_values[i + 1]


class TestScaleFunction:
    """Tests for scale function W^(q)(x)."""

    def test_zero_for_negative_x(self, sample_levy_params: LevyParameters) -> None:
        """Test W^(q)(x) = 0 for x < 0."""
        wh = WienerHopfFactorization(sample_levy_params)
        sf = ScaleFunction(sample_levy_params, wh)
        for x in [-1.0, -0.5, -0.1]:
            assert sf.W(x, q=1.0) == 0.0

    def test_increasing(self, sample_levy_params: LevyParameters) -> None:
        """Test W^(q)(x) is increasing for x > 0."""
        wh = WienerHopfFactorization(sample_levy_params)
        sf = ScaleFunction(sample_levy_params, wh)
        q = 1.0
        x_values = [0.1, 0.5, 1.0, 2.0]
        W_values = [sf.W(x, q) for x in x_values]
        for i in range(len(W_values) - 1):
            assert W_values[i] < W_values[i + 1]

    def test_Z_starts_at_one(self, sample_levy_params: LevyParameters) -> None:
        """Test Z^(q)(0) = 1."""
        wh = WienerHopfFactorization(sample_levy_params)
        sf = ScaleFunction(sample_levy_params, wh)
        Z_0 = sf.Z(0.0, q=1.0)
        assert np.isclose(Z_0, 1.0, atol=1e-6)


class TestGaverStehfestInverter:
    """Tests for Gaver-Stehfest Laplace inversion."""

    def test_exponential_inversion(self) -> None:
        """Test inversion of F(s) = 1/(s+a) -> f(t) = e^{-at}."""
        inverter = GaverStehfestInverter(N=10)
        a = 2.0

        def F(s: float) -> float:
            return 1.0 / (s + a)

        t_values = np.array([0.5, 1.0, 2.0])
        expected = np.exp(-a * t_values)
        computed = inverter.invert(F, t_values)

        np.testing.assert_allclose(computed, expected, rtol=1e-2)

    def test_even_N_required(self) -> None:
        """Test that odd N raises error."""
        with pytest.raises(ValueError, match="even"):
            GaverStehfestInverter(N=9)

    def test_N_range(self) -> None:
        """Test N must be in valid range."""
        with pytest.raises(ValueError):
            GaverStehfestInverter(N=2)
        with pytest.raises(ValueError):
            GaverStehfestInverter(N=20)


class TestFirstHittingTime:
    """Tests for first-hitting time computation."""

    def test_survival_probability_bounds(self, sample_levy_params: LevyParameters) -> None:
        """Test survival probability is in [0, 1]."""
        fht = FirstHittingTime(sample_levy_params)
        x = 0.5  # log(1.65) ≈ 0.5
        b = 0.0

        for t in [1.0, 10.0, 30.0, 100.0]:
            prob = fht.survival_probability(t, x, b)
            assert 0 <= prob <= 1, f"P(τ > {t}) = {prob} out of bounds"

    def test_survival_decreasing(self, sample_levy_params: LevyParameters) -> None:
        """Test survival probability decreases with time."""
        fht = FirstHittingTime(sample_levy_params)
        x = 0.5
        b = 0.0
        t_values = [1.0, 5.0, 10.0, 30.0, 60.0]

        probs = [fht.survival_probability(t, x, b) for t in t_values]

        for i in range(len(probs) - 1):
            # Allow small numerical error
            assert probs[i] >= probs[i + 1] - 0.01

    def test_already_below_threshold(self, sample_levy_params: LevyParameters) -> None:
        """Test that starting below threshold gives probability 1."""
        fht = FirstHittingTime(sample_levy_params)
        x = -0.1  # Below threshold
        b = 0.0

        prob = fht.liquidation_probability(10.0, x, b)
        assert np.isclose(prob, 1.0)

    def test_from_health_factor(self, sample_levy_params: LevyParameters) -> None:
        """Test convenience method from_health_factor."""
        fht = FirstHittingTime(sample_levy_params)

        # Health factor 2.0 should have lower liquidation prob than 1.2
        prob_hf_2 = fht.from_health_factor(2.0, t=30)
        prob_hf_12 = fht.from_health_factor(1.2, t=30)

        assert prob_hf_2 < prob_hf_12


class TestSimulation:
    """Tests for path simulation."""

    def test_simulate_path_shape(
        self, sample_levy_params: LevyParameters, rng: np.random.Generator
    ) -> None:
        """Test simulated path has correct shape."""
        times, path = simulate_path(
            sample_levy_params, x0=0.0, t_max=1.0, dt=0.01, rng=rng
        )
        assert len(times) == len(path)
        assert times[0] == 0.0
        assert np.isclose(times[-1], 1.0)

    def test_simulate_path_starts_at_x0(
        self, sample_levy_params: LevyParameters, rng: np.random.Generator
    ) -> None:
        """Test path starts at initial value."""
        x0 = 1.5
        times, path = simulate_path(
            sample_levy_params, x0=x0, t_max=1.0, dt=0.01, rng=rng
        )
        assert path[0] == x0
