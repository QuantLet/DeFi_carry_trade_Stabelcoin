"""
Static Matplotlib visualizations for publication-quality figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crypto_fht.core.levy_process import LevyParameters
    from crypto_fht.calibration.mle_estimator import CalibrationResult


def _check_matplotlib() -> None:
    """Check if Matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")


def plot_first_hitting_time_distribution(
    time_horizons: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    health_factor: float = 1.5,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot first-hitting time distribution.

    Args:
        time_horizons: Array of time horizons.
        probabilities: Array of liquidation probabilities.
        health_factor: Health factor used for computation.
        ax: Optional axes to plot on.

    Returns:
        Matplotlib Figure.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(time_horizons, probabilities, "b-", linewidth=2, label="P(τ ≤ t)")
    ax.fill_between(time_horizons, 0, probabilities, alpha=0.2)

    ax.set_xlabel("Time Horizon (days)", fontsize=12)
    ax.set_ylabel("P(Liquidation)", fontsize=12)
    ax.set_title(f"First-Hitting Time Distribution (HF = {health_factor:.2f})", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_scale_function(
    x_values: NDArray[np.floating],
    W_values: NDArray[np.floating],
    q: float,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot q-scale function W^(q)(x).

    Args:
        x_values: Array of x values.
        W_values: Array of W^(q)(x) values.
        q: Laplace parameter.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(x_values, W_values, "b-", linewidth=2)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel(f"W^({q})(x)", fontsize=12)
    ax.set_title(f"Scale Function W^(q)(x) for q = {q}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_levy_paths(
    times: NDArray[np.floating],
    paths: NDArray[np.floating],
    threshold: float = 0.0,
    n_paths_to_show: int = 20,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot sample paths of Lévy process.

    Args:
        times: Time grid.
        paths: Array of paths (n_paths × n_times).
        threshold: Liquidation threshold to mark.
        n_paths_to_show: Number of paths to display.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    n_paths = min(n_paths_to_show, paths.shape[0])

    for i in range(n_paths):
        ax.plot(times, paths[i], alpha=0.5, linewidth=0.8)

    # Threshold line
    ax.axhline(y=threshold, color="r", linestyle="--", linewidth=2, label="Threshold")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("X_t (log health factor)", fontsize=12)
    ax.set_title("Sample Paths of Lévy Process", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_calibration_diagnostics(
    returns: NDArray[np.floating],
    result: "CalibrationResult",
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Create diagnostic plots for calibration results.

    Args:
        returns: Observed returns.
        result: CalibrationResult from calibration.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with multiple subplots.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Return histogram with fitted density
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=50, density=True, alpha=0.7, label="Observed")

    # Theoretical density (normal approximation)
    x = np.linspace(np.min(returns), np.max(returns), 100)
    from scipy.stats import norm
    theoretical = norm.pdf(x, loc=result.params.mu * 1, scale=result.params.sigma)
    ax1.plot(x, theoretical, "r-", linewidth=2, label="Fitted Normal")

    ax1.set_xlabel("Return")
    ax1.set_ylabel("Density")
    ax1.set_title("Return Distribution")
    ax1.legend()

    # 2. Q-Q plot
    ax2 = axes[0, 1]
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")

    # 3. Autocorrelation
    ax3 = axes[1, 0]
    n_lags = min(30, len(returns) // 4)
    acf = np.correlate(returns - np.mean(returns), returns - np.mean(returns), mode="full")
    acf = acf[len(returns)-1:len(returns)+n_lags] / acf[len(returns)-1]
    ax3.bar(range(n_lags+1), acf[:n_lags+1], alpha=0.7)
    ax3.axhline(y=0, color="k", linestyle="-")
    ax3.axhline(y=1.96/np.sqrt(len(returns)), color="r", linestyle="--")
    ax3.axhline(y=-1.96/np.sqrt(len(returns)), color="r", linestyle="--")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")
    ax3.set_title("Autocorrelation")

    # 4. Parameter summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = f"""
    Calibration Results
    ==================
    μ (drift):     {result.params.mu:.6f}
    σ (volatility): {result.params.sigma:.6f}
    λ (intensity): {result.params.lambda_:.6f}
    η (jump rate): {result.params.eta:.6f}
    δ (shift):     {result.params.delta:.6f}

    Log-likelihood: {result.log_likelihood:.2f}
    AIC: {result.aic:.2f}
    BIC: {result.bic:.2f}
    Converged: {result.converged}
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    return fig


def plot_liquidation_heatmap(
    health_factors: NDArray[np.floating],
    time_horizons: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    ax: plt.Axes | None = None,
) -> Figure:
    """Create heatmap of liquidation probabilities.

    Args:
        health_factors: Array of health factors.
        time_horizons: Array of time horizons.
        probabilities: 2D probability array.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        probabilities,
        aspect="auto",
        origin="lower",
        extent=[time_horizons[0], time_horizons[-1], health_factors[0], health_factors[-1]],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(Liquidation)", fontsize=12)

    ax.set_xlabel("Time Horizon (days)", fontsize=12)
    ax.set_ylabel("Health Factor", fontsize=12)
    ax.set_title("Liquidation Probability Heatmap", fontsize=14)

    # Add contour lines
    ax.contour(
        time_horizons,
        health_factors,
        probabilities,
        levels=[0.01, 0.05, 0.1, 0.25, 0.5],
        colors="white",
        linewidths=0.5,
    )

    plt.tight_layout()
    return fig
