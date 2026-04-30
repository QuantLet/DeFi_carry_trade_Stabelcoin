"""
Portfolio optimization for DeFi positions.

This module provides:
- Aave v3 constraint modeling
- CVaR-minimizing portfolio optimization
- Portfolio state management
"""

from crypto_fht.optimization.constraints import (
    AaveV3Constraints,
    AssetConstraints,
    build_constraint_matrices,
)
from crypto_fht.optimization.cvar_optimizer import (
    CVaRPortfolioOptimizer,
    OptimizationResult,
)
from crypto_fht.optimization.portfolio import (
    Portfolio,
    PortfolioPosition,
)

__all__ = [
    "AaveV3Constraints",
    "AssetConstraints",
    "build_constraint_matrices",
    "CVaRPortfolioOptimizer",
    "OptimizationResult",
    "Portfolio",
    "PortfolioPosition",
]
