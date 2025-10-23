"""Portfolio construction module.

Strategy Pattern for weight scaling:
- WeightScaler: Abstract base class
- GrossNetScaler: Unified gross/net exposure framework
- DollarNeutralScaler: Convenience wrapper (G=2.0, N=0.0)

Future scalers can be added by inheriting from WeightScaler:
- SoftmaxScaler: Probabilistic scaling
- RiskTargetScaler: Volatility-based scaling
- OptimizationScaler: cvxpy-based optimization
"""
from alpha_canvas.portfolio.base import WeightScaler
from alpha_canvas.portfolio.strategies import (
    GrossNetScaler,
    DollarNeutralScaler,
)

__all__ = [
    'WeightScaler',
    'GrossNetScaler',
    'DollarNeutralScaler',
]

