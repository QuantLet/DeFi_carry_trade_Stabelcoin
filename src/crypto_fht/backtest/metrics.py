"""
Performance metrics for backtesting results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PerformanceMetrics:
    """Performance metrics summary.

    Attributes:
        total_return: Total return over period.
        annualized_return: Annualized return.
        volatility: Annualized volatility.
        sharpe_ratio: Sharpe ratio (assuming 0 risk-free rate).
        sortino_ratio: Sortino ratio.
        max_drawdown: Maximum drawdown.
        calmar_ratio: Calmar ratio (return / max_drawdown).
        var_5pct: 5% Value-at-Risk.
        cvar_5pct: 5% Conditional Value-at-Risk.
        n_liquidations: Number of liquidation events.
        avg_health_factor: Average health factor.
        min_health_factor: Minimum health factor.
        time_below_target_hf: Fraction of time below target HF.
    """

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_5pct: float
    cvar_5pct: float
    n_liquidations: int
    avg_health_factor: float
    min_health_factor: float
    time_below_target_hf: float

    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
Performance Metrics
==================
Returns:
  Total Return:       {self.total_return:.2%}
  Annualized Return:  {self.annualized_return:.2%}
  Volatility:         {self.volatility:.2%}

Risk-Adjusted:
  Sharpe Ratio:       {self.sharpe_ratio:.2f}
  Sortino Ratio:      {self.sortino_ratio:.2f}
  Calmar Ratio:       {self.calmar_ratio:.2f}

Drawdown & Tail Risk:
  Max Drawdown:       {self.max_drawdown:.2%}
  VaR (5%):           {self.var_5pct:.2%}
  CVaR (5%):          {self.cvar_5pct:.2%}

DeFi-Specific:
  Liquidations:       {self.n_liquidations}
  Avg Health Factor:  {self.avg_health_factor:.2f}
  Min Health Factor:  {self.min_health_factor:.2f}
  Time Below Target:  {self.time_below_target_hf:.2%}
"""


def compute_performance_metrics(
    portfolio_values: NDArray[np.floating],
    health_factors: NDArray[np.floating],
    liquidation_events: list[dict[str, Any]],
    periods_per_year: float = 365,
    target_health_factor: float = 1.5,
) -> dict[str, float]:
    """Compute comprehensive performance metrics.

    Args:
        portfolio_values: Array of portfolio values over time.
        health_factors: Array of health factors over time.
        liquidation_events: List of liquidation event dictionaries.
        periods_per_year: Number of periods per year (for annualization).
        target_health_factor: Target HF for time-below calculation.

    Returns:
        Dictionary of metrics.
    """
    # Filter out zeros/nans
    pv = portfolio_values[portfolio_values > 0]
    if len(pv) < 2:
        return _empty_metrics()

    # Returns
    returns = np.diff(pv) / pv[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return _empty_metrics()

    # Basic return metrics
    total_return = (pv[-1] - pv[0]) / pv[0] if pv[0] > 0 else 0
    n_periods = len(pv)
    years = n_periods / periods_per_year

    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0

    # Volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Sharpe ratio (assuming 0 risk-free rate)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0

    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-10
    sortino = mean_return / downside_std * np.sqrt(periods_per_year)

    # Maximum drawdown
    cummax = np.maximum.accumulate(pv)
    drawdowns = (cummax - pv) / cummax
    max_drawdown = np.max(drawdowns)

    # Calmar ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # VaR and CVaR
    sorted_returns = np.sort(returns)
    var_idx = int(0.05 * len(returns))
    var_5 = sorted_returns[var_idx] if var_idx < len(sorted_returns) else sorted_returns[0]
    cvar_5 = np.mean(sorted_returns[:max(1, var_idx)])

    # Health factor metrics
    hf = health_factors[np.isfinite(health_factors)]
    avg_hf = np.mean(hf) if len(hf) > 0 else 0
    min_hf = np.min(hf) if len(hf) > 0 else 0
    time_below = np.mean(hf < target_health_factor) if len(hf) > 0 else 0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
        "var_5pct": float(var_5),
        "cvar_5pct": float(cvar_5),
        "n_liquidations": len(liquidation_events),
        "avg_health_factor": float(avg_hf),
        "min_health_factor": float(min_hf),
        "time_below_target_hf": float(time_below),
    }


def _empty_metrics() -> dict[str, float]:
    """Return empty metrics dictionary."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "var_5pct": 0.0,
        "cvar_5pct": 0.0,
        "n_liquidations": 0,
        "avg_health_factor": 0.0,
        "min_health_factor": 0.0,
        "time_below_target_hf": 0.0,
    }


def compute_rolling_metrics(
    portfolio_values: NDArray[np.floating],
    window: int = 30,
) -> dict[str, NDArray[np.floating]]:
    """Compute rolling performance metrics.

    Args:
        portfolio_values: Portfolio value series.
        window: Rolling window size.

    Returns:
        Dictionary of rolling metric arrays.
    """
    n = len(portfolio_values)
    if n < window:
        return {}

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    rolling_mean = np.zeros(n - window)
    rolling_std = np.zeros(n - window)
    rolling_sharpe = np.zeros(n - window)

    for i in range(n - window):
        window_returns = returns[i : i + window]
        rolling_mean[i] = np.mean(window_returns)
        rolling_std[i] = np.std(window_returns)
        if rolling_std[i] > 0:
            rolling_sharpe[i] = rolling_mean[i] / rolling_std[i] * np.sqrt(365)

    return {
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_sharpe": rolling_sharpe,
    }
