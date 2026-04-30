"""
Crypto FHT Portfolio - Optimal allocation framework for DeFi lending positions.

This package implements an optimal allocation framework for long-short cryptocurrency
positions on DeFi lending platforms (specifically Aave v3) by analyzing first-hitting
time distributions for log-health processes under spectrally negative Lévy dynamics
with shifted exponential jumps.

Key Features:
- Spectrally negative Lévy process modeling with shifted exponential jumps
- Semi-analytical first-hitting time computation via Laplace transforms
- Gaver-Stehfest numerical inversion for liquidation probabilities
- CVaR optimization with Aave v3 LTV constraints
- Historical calibration via EM algorithm
- Full backtesting framework

Example:
    >>> from crypto_fht.core import LevyParameters, FirstHittingTime
    >>> from crypto_fht.optimization import CVaROptimizer
    >>>
    >>> # Define process parameters
    >>> params = LevyParameters(mu=0.01, sigma=0.3, lambda_=2.0, eta=5.0, delta=0.02)
    >>>
    >>> # Compute liquidation probability
    >>> fht = FirstHittingTime(params)
    >>> prob = fht.liquidation_probability(t=30, x=0.5, b=0.0)
"""

from crypto_fht.core.levy_process import LevyParameters

__version__ = "0.1.0"
__all__ = ["LevyParameters", "__version__"]
