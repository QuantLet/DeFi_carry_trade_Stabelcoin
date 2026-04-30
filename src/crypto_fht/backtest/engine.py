"""
Backtesting engine for DeFi position allocation strategies.

Simulates portfolio evolution under historical scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from crypto_fht.core.levy_process import LevyParameters
from crypto_fht.optimization.portfolio import Portfolio

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class BacktestConfig:
    """Configuration for backtesting.

    Attributes:
        start_date: Start of backtest period.
        end_date: End of backtest period.
        rebalance_frequency: How often to rebalance.
        initial_capital: Starting capital in USD.
        transaction_costs: Cost per trade as fraction.
        slippage: Slippage as fraction.
        min_health_factor: Minimum health factor to maintain.
        target_health_factor: Target health factor for rebalancing.
    """

    start_date: datetime
    end_date: datetime
    rebalance_frequency: timedelta = field(default_factory=lambda: timedelta(days=7))
    initial_capital: float = 100000.0
    transaction_costs: float = 0.001
    slippage: float = 0.0005
    min_health_factor: float = 1.5
    target_health_factor: float = 2.0


@dataclass
class BacktestResult:
    """Results from a backtest run.

    Attributes:
        timestamps: Time points.
        portfolio_values: Portfolio value at each time.
        health_factors: Health factor at each time.
        allocations: Allocation weights at each time.
        trades: List of executed trades.
        liquidation_events: List of liquidation events.
        metrics: Performance metrics dictionary.
    """

    timestamps: list[datetime] = field(default_factory=list)
    portfolio_values: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    health_factors: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    allocations: list[dict[str, float]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    liquidation_events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Backtesting engine for DeFi allocation strategies.

    Simulates portfolio evolution, tracking health factors,
    liquidations, and performance.

    Attributes:
        config: Backtest configuration.
        optimizer: Portfolio optimizer (optional).
        price_data: Historical price data.

    Example:
        >>> config = BacktestConfig(
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 6, 1),
        ...     initial_capital=100000
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(price_data, levy_params)
    """

    def __init__(
        self,
        config: BacktestConfig,
        optimizer: Any | None = None,
    ) -> None:
        """Initialize backtest engine.

        Args:
            config: Backtest configuration.
            optimizer: Portfolio optimizer instance.
        """
        self.config = config
        self.optimizer = optimizer

    def run(
        self,
        price_data: dict[str, NDArray[np.floating]],
        timestamps: list[datetime],
        levy_params: dict[str, LevyParameters] | None = None,
        allocation_strategy: Callable[[Portfolio, dict[str, float]], dict[str, float]] | None = None,
    ) -> BacktestResult:
        """Run backtest simulation.

        Args:
            price_data: Historical prices per asset {asset: prices_array}.
            timestamps: Corresponding timestamps.
            levy_params: Calibrated Lévy parameters per asset.
            allocation_strategy: Function that returns target weights.

        Returns:
            BacktestResult with full history.
        """
        result = BacktestResult()
        result.timestamps = []

        n_periods = len(timestamps)
        result.portfolio_values = np.zeros(n_periods)
        result.health_factors = np.zeros(n_periods)

        # Initialize portfolio
        portfolio = Portfolio()
        capital = self.config.initial_capital

        # Initial allocation (equal weight to collateral, small debt)
        assets = list(price_data.keys())
        initial_prices = {a: price_data[a][0] for a in assets}

        # Start with collateral position
        if len(assets) > 0:
            collateral_asset = assets[0]
            portfolio.add_collateral(
                collateral_asset,
                capital * 0.5 / initial_prices[collateral_asset],
                initial_prices[collateral_asset],
            )

        last_rebalance = timestamps[0]

        for i, ts in enumerate(timestamps):
            result.timestamps.append(ts)

            # Update prices
            current_prices = {a: price_data[a][i] for a in assets}
            portfolio.update_prices(current_prices)

            # Check health factor
            hf = portfolio.health_factor
            result.health_factors[i] = hf

            # Check for liquidation
            if hf < 1.0 and portfolio.total_debt_value > 0:
                liquidation_loss = self._simulate_liquidation(portfolio, current_prices)
                capital -= liquidation_loss

                result.liquidation_events.append({
                    "timestamp": ts,
                    "health_factor": hf,
                    "loss": liquidation_loss,
                })

                # Reset portfolio after liquidation
                portfolio = Portfolio()

            # Check for rebalancing
            if ts - last_rebalance >= self.config.rebalance_frequency:
                if allocation_strategy is not None:
                    target_weights = allocation_strategy(portfolio, current_prices)
                    trades = portfolio.rebalance_to_weights(target_weights, current_prices)

                    # Apply transaction costs
                    for asset, amount, action in trades:
                        cost = amount * current_prices.get(asset, 0) * self.config.transaction_costs
                        capital -= cost

                    result.trades.append({
                        "timestamp": ts,
                        "trades": trades,
                    })

                last_rebalance = ts

            # Record portfolio value
            result.portfolio_values[i] = portfolio.net_value + capital
            result.allocations.append(portfolio.get_weights())

        # Compute metrics
        result.metrics = self._compute_metrics(result)

        return result

    def _simulate_liquidation(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> float:
        """Simulate liquidation and return loss.

        Args:
            portfolio: Current portfolio.
            prices: Current prices.

        Returns:
            Liquidation loss in USD.
        """
        # Liquidation penalty (typically 5-10%)
        penalty_rate = 0.05

        # Loss = debt_covered × penalty
        debt_value = portfolio.total_debt_value
        loss = debt_value * penalty_rate

        return loss

    def _compute_metrics(self, result: BacktestResult) -> dict[str, float]:
        """Compute performance metrics from backtest results."""
        from crypto_fht.backtest.metrics import compute_performance_metrics

        return compute_performance_metrics(
            result.portfolio_values,
            result.health_factors,
            result.liquidation_events,
        )


def run_simple_backtest(
    price_series: NDArray[np.floating],
    initial_capital: float = 100000.0,
    leverage: float = 1.5,
    dt_days: int = 1,
) -> BacktestResult:
    """Run a simple single-asset backtest.

    Convenience function for quick analysis.

    Args:
        price_series: Array of prices.
        initial_capital: Starting capital.
        leverage: Target leverage (collateral/equity).
        dt_days: Days between observations.

    Returns:
        BacktestResult.
    """
    n = len(price_series)
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i * dt_days) for i in range(n)]

    # Simple leveraged position
    portfolio_values = np.zeros(n)
    health_factors = np.zeros(n)

    equity = initial_capital
    collateral_amount = equity * leverage / price_series[0]
    debt_usd = equity * (leverage - 1)

    for i in range(n):
        collateral_value = collateral_amount * price_series[i]
        net_value = collateral_value - debt_usd

        if debt_usd > 0:
            hf = (collateral_value * 0.825) / debt_usd
        else:
            hf = float("inf")

        portfolio_values[i] = max(0, net_value)
        health_factors[i] = hf

        # Check liquidation
        if hf < 1.0:
            portfolio_values[i] = 0
            break

    result = BacktestResult(
        timestamps=timestamps,
        portfolio_values=portfolio_values,
        health_factors=health_factors,
    )

    from crypto_fht.backtest.metrics import compute_performance_metrics
    result.metrics = compute_performance_metrics(
        portfolio_values[:i+1] if hf < 1.0 else portfolio_values,
        health_factors[:i+1] if hf < 1.0 else health_factors,
        [],
    )

    return result
