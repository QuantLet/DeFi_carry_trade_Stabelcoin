"""
Backtesting framework for DeFi allocation strategies.

This module provides:
- Historical simulation engine
- Performance metrics computation
"""

from crypto_fht.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from crypto_fht.backtest.metrics import (
    compute_performance_metrics,
    PerformanceMetrics,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "compute_performance_metrics",
    "PerformanceMetrics",
]
