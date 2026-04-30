# crypto-fht

Optimal allocation framework for long-short cryptocurrency positions on DeFi lending platforms using spectrally negative Lévy processes with shifted exponential jumps.

## Overview

This framework provides:

- **First-hitting time distributions** for log-health processes under constant-intensity jump-diffusion dynamics
- **Spectrally negative Lévy process** with shifted exponential jumps: Y ~ ShiftedExp(η, δ)
- **Semi-analytical solutions** via Laplace transform methods and Gaver-Stehfest inversion
- **CVaR optimization** subject to Aave v3 collateral constraints
- **Wrong-way risk modeling** via shared jump components

## Mathematical Model

Log-health factor dynamics:
```
X_t = X_0 + μt + σW_t - Σ_{i=1}^{N_t} Y_i
```

Where:
- Y_i = δ + Z_i, Z_i ~ Exp(η) (shifted exponential jumps)
- N_t ~ Poisson(λt) (jump count process)
- δ: minimum jump size (shift parameter)
- η: exponential rate parameter

**Laplace exponent:**
```
ψ(θ) = μθ + (σ²/2)θ² + λ(e^{-θδ} · η/(η+θ) - 1)
```

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from crypto_fht.core.levy_process import LevyParameters
from crypto_fht.core.first_hitting_time import FirstHittingTime

# Define model parameters
params = LevyParameters(
    mu=0.01,      # drift
    sigma=0.3,    # volatility
    lambda_=2.0,  # jump intensity
    eta=5.0,      # exponential rate
    delta=0.02,   # minimum jump size
)

# Compute liquidation probability
fht = FirstHittingTime(params)
prob = fht.from_health_factor(health_factor=1.5, t=30)  # 30-day horizon
print(f"P(liquidation within 30 days | HF=1.5) = {prob:.4f}")
```

## Modules

- `crypto_fht.core` - Mathematical foundations (Lévy process, Wiener-Hopf, scale functions)
- `crypto_fht.risk` - Risk metrics (health factor, CVaR, wrong-way risk)
- `crypto_fht.optimization` - CVaR portfolio optimization with Aave constraints
- `crypto_fht.calibration` - MLE parameter estimation
- `crypto_fht.data` - Aave v3 data client
- `crypto_fht.backtest` - Historical backtesting engine
- `crypto_fht.visualization` - Plotly and Matplotlib visualizations

## Testing

```bash
pytest
```

## License

MIT
