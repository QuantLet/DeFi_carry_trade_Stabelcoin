"""
Wrong-way risk modeling for DeFi lending positions.

Wrong-way risk occurs when exposure increases as credit quality decreases.
In DeFi lending:
- Collateral value drops when the market crashes
- This is exactly when you're most likely to get liquidated
- Liquidation cascades amplify the problem

We model wrong-way risk via correlated jumps:
1. Idiosyncratic jumps: Asset-specific with intensity λ_i
2. Systemic jumps: Shared across all assets with intensity λ_sys

When a systemic jump occurs, all assets experience correlated drops,
creating wrong-way risk exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from crypto_fht.core.levy_process import LevyParameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class JumpCorrelationModel:
    """Model for correlated jumps across assets (wrong-way risk).

    In DeFi, liquidation cascades create correlated downward jumps.
    We model this via:
    1. Idiosyncratic jumps: asset-specific with intensity λ_i
    2. Systemic jumps: shared across all assets with intensity λ_sys

    Total jump intensity for asset i: λ_i + β_i × λ_sys
    where β_i is the asset's sensitivity to systemic events.

    Attributes:
        idiosyncratic_intensities: Jump intensity per asset.
        systemic_intensity: Shared systemic jump intensity.
        systemic_sensitivity: Asset sensitivities to systemic jumps (β_i).
        idiosyncratic_eta: Exponential rate for idiosyncratic jump sizes.
        systemic_eta: Exponential rate for systemic jump sizes.
        systemic_delta: Minimum systemic jump size (shift parameter).
    """

    idiosyncratic_intensities: dict[str, float] = field(default_factory=dict)
    systemic_intensity: float = 0.5
    systemic_sensitivity: dict[str, float] = field(default_factory=dict)
    idiosyncratic_eta: dict[str, float] = field(default_factory=dict)
    systemic_eta: float = 3.0
    systemic_delta: float = 0.05

    def effective_intensity(self, asset: str) -> float:
        """Get effective jump intensity including systemic component.

        λ_eff(i) = λ_i + β_i × λ_sys

        Args:
            asset: Asset identifier.

        Returns:
            Total effective jump intensity.
        """
        idio = self.idiosyncratic_intensities.get(asset, 0.0)
        beta = self.systemic_sensitivity.get(asset, 1.0)
        return idio + beta * self.systemic_intensity

    def sample_jumps(
        self,
        dt: float,
        assets: list[str],
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, float], bool]:
        """Sample jump sizes over interval dt.

        Returns:
            Tuple of (jump_sizes dict, systemic_flag).
        """
        if rng is None:
            rng = np.random.default_rng()

        jump_sizes: dict[str, float] = {asset: 0.0 for asset in assets}
        systemic = False

        # Check for systemic jump
        if rng.poisson(self.systemic_intensity * dt) > 0:
            systemic = True
            # All assets experience correlated jump
            for asset in assets:
                beta = self.systemic_sensitivity.get(asset, 1.0)
                # Systemic jump size: δ_sys + Exp(η_sys), scaled by beta
                systemic_jump = (
                    self.systemic_delta
                    + rng.exponential(1.0 / self.systemic_eta)
                ) * beta
                jump_sizes[asset] += systemic_jump

        # Idiosyncratic jumps
        for asset in assets:
            idio_lambda = self.idiosyncratic_intensities.get(asset, 0.0)
            n_jumps = rng.poisson(idio_lambda * dt)
            if n_jumps > 0:
                eta = self.idiosyncratic_eta.get(asset, 5.0)
                idio_jumps = rng.exponential(1.0 / eta, size=n_jumps)
                jump_sizes[asset] += float(np.sum(idio_jumps))

        return jump_sizes, systemic

    def covariance_contribution(
        self,
        asset_i: str,
        asset_j: str,
    ) -> float:
        """Compute jump covariance contribution between two assets.

        Cov(J_i, J_j) = β_i β_j λ_sys E[Y_sys²]

        For shifted exponential Y_sys = δ + Z, Z ~ Exp(η):
        E[Y²] = δ² + 2δ/η + 2/η²

        Args:
            asset_i: First asset.
            asset_j: Second asset.

        Returns:
            Jump covariance contribution.
        """
        beta_i = self.systemic_sensitivity.get(asset_i, 1.0)
        beta_j = self.systemic_sensitivity.get(asset_j, 1.0)

        # E[Y²] for shifted exponential
        delta = self.systemic_delta
        eta = self.systemic_eta
        e_y_squared = delta**2 + 2 * delta / eta + 2 / eta**2

        return beta_i * beta_j * self.systemic_intensity * e_y_squared


@dataclass
class WrongWayRiskModel:
    """Complete wrong-way risk model for portfolio analysis.

    Integrates jump correlation with continuous correlation to provide
    a full picture of wrong-way risk in DeFi positions.

    Attributes:
        jump_model: Jump correlation model.
        diffusion_correlation: Correlation matrix for diffusion component.
        assets: List of asset identifiers.
    """

    jump_model: JumpCorrelationModel
    diffusion_correlation: NDArray[np.floating] | None = None
    assets: list[str] = field(default_factory=list)

    def compute_effective_params(
        self,
        base_params: dict[str, LevyParameters],
        weights: dict[str, float],
    ) -> LevyParameters:
        """Compute effective Lévy parameters incorporating wrong-way risk.

        Combines base asset parameters with systemic jump effects.

        Args:
            base_params: Base Lévy parameters per asset.
            weights: Portfolio weights.

        Returns:
            Effective LevyParameters for the portfolio.
        """
        if not base_params or not weights:
            return LevyParameters(
                mu=0.0, sigma=0.3, lambda_=1.0, eta=5.0, delta=0.02
            )

        total_weight = sum(abs(w) for w in weights.values())
        if total_weight == 0:
            return LevyParameters(
                mu=0.0, sigma=0.3, lambda_=1.0, eta=5.0, delta=0.02
            )

        # Normalize weights
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted drift
        mu = sum(
            norm_weights.get(asset, 0) * params.mu
            for asset, params in base_params.items()
        )

        # Portfolio volatility with correlations
        sigma = self._compute_portfolio_sigma(base_params, norm_weights)

        # Effective jump intensity including systemic component
        # For wrong-way risk, we use the maximum effective intensity
        lambda_eff = max(
            self.jump_model.effective_intensity(asset)
            for asset in base_params.keys()
        )

        # Weighted average jump parameters
        eta = sum(
            norm_weights.get(asset, 0) * params.eta
            for asset, params in base_params.items()
        )
        delta = sum(
            norm_weights.get(asset, 0) * params.delta
            for asset, params in base_params.items()
        )

        # Add systemic jump contribution to delta (minimum impact)
        delta = max(delta, self.jump_model.systemic_delta)

        return LevyParameters(
            mu=mu,
            sigma=max(sigma, 0.01),
            lambda_=max(lambda_eff, 0.01),
            eta=max(eta, 0.1),
            delta=max(delta, 0.0),
        )

    def _compute_portfolio_sigma(
        self,
        base_params: dict[str, LevyParameters],
        weights: dict[str, float],
    ) -> float:
        """Compute portfolio volatility considering correlations."""
        assets = list(base_params.keys())
        n = len(assets)

        if n == 0:
            return 0.3

        sigma_vec = np.array([base_params[a].sigma for a in assets])
        w_vec = np.array([weights.get(a, 0) for a in assets])

        if self.diffusion_correlation is not None and self.diffusion_correlation.shape[0] >= n:
            corr = self.diffusion_correlation[:n, :n]
        else:
            # Assume high correlation for wrong-way risk
            corr = np.ones((n, n)) * 0.8
            np.fill_diagonal(corr, 1.0)

        # Covariance matrix
        cov = np.diag(sigma_vec) @ corr @ np.diag(sigma_vec)

        # Portfolio variance
        portfolio_var = w_vec @ cov @ w_vec
        return float(np.sqrt(max(portfolio_var, 0)))

    def compute_wrong_way_risk_score(
        self,
        base_params: dict[str, LevyParameters],
        weights: dict[str, float],
    ) -> float:
        """Compute a scalar wrong-way risk score.

        Higher score indicates more exposure to correlated crashes.

        Score components:
        1. Systemic jump exposure: Σ w_i × β_i × λ_sys
        2. Correlation adjustment: average pairwise correlation × volatility

        Args:
            base_params: Base parameters per asset.
            weights: Portfolio weights.

        Returns:
            Wrong-way risk score (non-negative, higher = more risk).
        """
        if not weights:
            return 0.0

        # Systemic exposure
        systemic_exposure = sum(
            abs(weights.get(asset, 0))
            * self.jump_model.systemic_sensitivity.get(asset, 1.0)
            for asset in base_params.keys()
        )
        systemic_exposure *= self.jump_model.systemic_intensity

        # Correlation-weighted volatility
        eff_params = self.compute_effective_params(base_params, weights)
        vol_contribution = eff_params.sigma * systemic_exposure

        # Combined score
        score = systemic_exposure * (1 + vol_contribution)

        return float(score)

    def simulate_correlated_paths(
        self,
        base_params: dict[str, LevyParameters],
        t_max: float,
        dt: float,
        n_paths: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Simulate correlated paths for multiple assets.

        Includes both diffusion and jump correlations.

        Args:
            base_params: Lévy parameters per asset.
            t_max: Terminal time.
            dt: Time step.
            n_paths: Number of paths.
            rng: Random generator.

        Returns:
            Dictionary mapping asset names to path arrays (n_paths × n_steps+1).
        """
        if rng is None:
            rng = np.random.default_rng()

        assets = list(base_params.keys())
        n_assets = len(assets)
        n_steps = int(np.ceil(t_max / dt))

        # Initialize paths
        paths = {asset: np.zeros((n_paths, n_steps + 1)) for asset in assets}

        # Get correlation matrix for diffusion
        if self.diffusion_correlation is not None and self.diffusion_correlation.shape[0] >= n_assets:
            corr = self.diffusion_correlation[:n_assets, :n_assets]
        else:
            corr = np.eye(n_assets)

        # Cholesky decomposition for correlated Brownian motion
        L = np.linalg.cholesky(corr)

        sqrt_dt = np.sqrt(dt)

        for step in range(n_steps):
            # Generate correlated Brownian increments
            Z = rng.standard_normal((n_paths, n_assets))
            dW = (Z @ L.T) * sqrt_dt

            for i, asset in enumerate(assets):
                params = base_params[asset]

                # Diffusion step
                paths[asset][:, step + 1] = (
                    paths[asset][:, step]
                    + params.mu * dt
                    + params.sigma * dW[:, i]
                )

            # Jump component (correlated via systemic jumps)
            for path_idx in range(n_paths):
                jump_sizes, _ = self.jump_model.sample_jumps(dt, assets, rng)
                for asset in assets:
                    paths[asset][path_idx, step + 1] -= jump_sizes[asset]

        return paths

    def estimate_tail_dependence(
        self,
        base_params: dict[str, LevyParameters],
        asset_i: str,
        asset_j: str,
        n_simulations: int = 10000,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Estimate lower tail dependence coefficient.

        λ_L = lim_{u→0} P(Y ≤ F_Y^{-1}(u) | X ≤ F_X^{-1}(u))

        Higher values indicate stronger co-crash tendency (wrong-way risk).

        Args:
            base_params: Lévy parameters.
            asset_i: First asset.
            asset_j: Second asset.
            n_simulations: Number of simulations.
            rng: Random generator.

        Returns:
            Estimated tail dependence coefficient in [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()

        # Simulate returns
        dt = 1.0
        assets = [asset_i, asset_j]
        paths = self.simulate_correlated_paths(
            {a: base_params[a] for a in assets if a in base_params},
            t_max=dt,
            dt=dt,
            n_paths=n_simulations,
            rng=rng,
        )

        returns_i = paths[asset_i][:, -1]
        returns_j = paths[asset_j][:, -1]

        # Estimate tail dependence at various thresholds
        thresholds = [0.05, 0.10, 0.15]
        tail_deps = []

        for u in thresholds:
            q_i = np.quantile(returns_i, u)
            q_j = np.quantile(returns_j, u)

            # P(Y ≤ q_j | X ≤ q_i)
            x_in_tail = returns_i <= q_i
            if x_in_tail.sum() > 0:
                tail_dep = np.mean(returns_j[x_in_tail] <= q_j)
                tail_deps.append(tail_dep)

        if tail_deps:
            return float(np.mean(tail_deps))
        return 0.0
