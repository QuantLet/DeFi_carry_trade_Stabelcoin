"""
Interactive Plotly visualizations for DeFi risk analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crypto_fht.core.levy_process import LevyParameters
    from crypto_fht.backtest.engine import BacktestResult


def _check_plotly() -> None:
    """Check if Plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly not installed. Install with: pip install plotly")


def plot_liquidation_surface(
    health_factors: NDArray[np.floating],
    time_horizons: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    title: str = "Liquidation Probability Surface",
) -> Any:
    """Create 3D surface plot of liquidation probabilities.

    Args:
        health_factors: Array of health factor values.
        time_horizons: Array of time horizons.
        probabilities: 2D array of probabilities (HF × time).
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    _check_plotly()

    fig = go.Figure(data=[
        go.Surface(
            x=time_horizons,
            y=health_factors,
            z=probabilities,
            colorscale="RdYlGn_r",
            colorbar=dict(title="P(Liquidation)"),
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Time Horizon (days)",
            yaxis_title="Health Factor",
            zaxis_title="Probability",
        ),
        width=800,
        height=600,
    )

    return fig


def plot_health_factor_history(
    timestamps: list[Any],
    health_factors: NDArray[np.floating],
    liquidation_threshold: float = 1.0,
    target_hf: float = 1.5,
    title: str = "Health Factor History",
) -> Any:
    """Plot health factor over time with thresholds.

    Args:
        timestamps: List of timestamps.
        health_factors: Array of health factors.
        liquidation_threshold: Liquidation threshold line.
        target_hf: Target health factor line.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    _check_plotly()

    fig = go.Figure()

    # Health factor line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=health_factors,
        mode="lines",
        name="Health Factor",
        line=dict(color="blue", width=2),
    ))

    # Liquidation threshold
    fig.add_hline(
        y=liquidation_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Liquidation",
    )

    # Target threshold
    fig.add_hline(
        y=target_hf,
        line_dash="dot",
        line_color="orange",
        annotation_text="Target",
    )

    # Color regions
    fig.add_hrect(
        y0=0, y1=liquidation_threshold,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=liquidation_threshold, y1=target_hf,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Health Factor",
        showlegend=True,
        height=400,
    )

    return fig


def plot_portfolio_performance(
    result: "BacktestResult",
    title: str = "Portfolio Performance",
) -> Any:
    """Create multi-panel portfolio performance chart.

    Args:
        result: BacktestResult from backtesting.
        title: Plot title.

    Returns:
        Plotly Figure with subplots.
    """
    _check_plotly()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Value", "Health Factor", "Drawdown"),
        row_heights=[0.4, 0.3, 0.3],
    )

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.portfolio_values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="blue"),
        ),
        row=1, col=1,
    )

    # Health factor
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.health_factors,
            mode="lines",
            name="Health Factor",
            line=dict(color="green"),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=1)

    # Drawdown
    cummax = np.maximum.accumulate(result.portfolio_values)
    drawdown = (cummax - result.portfolio_values) / cummax
    fig.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=-drawdown * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=3, col=1,
    )

    # Mark liquidation events
    for event in result.liquidation_events:
        for row in [1, 2, 3]:
            fig.add_vline(
                x=event["timestamp"],
                line_dash="dash",
                line_color="red",
                row=row, col=1,
            )

    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Health Factor", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    return fig


def plot_efficient_frontier(
    results: list[Any],
    title: str = "CVaR-Efficient Frontier",
) -> Any:
    """Plot CVaR-efficient frontier.

    Args:
        results: List of OptimizationResults.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _check_plotly()

    returns = [r.expected_return for r in results]
    cvars = [r.cvar for r in results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cvars,
        y=returns,
        mode="lines+markers",
        name="Efficient Frontier",
        line=dict(color="blue", width=2),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="CVaR (Risk)",
        yaxis_title="Expected Return",
        height=500,
    )

    return fig


def plot_term_structure(
    time_horizons: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    title: str = "Liquidation Probability Term Structure",
) -> Any:
    """Plot liquidation probability term structure.

    Args:
        time_horizons: Array of time horizons.
        probabilities: Corresponding probabilities.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _check_plotly()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_horizons,
        y=probabilities,
        mode="lines+markers",
        name="P(Liquidation)",
        line=dict(color="red", width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Horizon (days)",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
    )

    return fig
