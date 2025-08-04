"""
Optimization utilities for BOOM Python package.

This module provides various optimization algorithms and wrappers
for parameter estimation in statistical models.
"""

from .base import Optimizer, OptimizationResult
from .line_search import LineSearchOptimizer
from .trust_region import TrustRegionOptimizer
from .nelder_mead import NelderMeadOptimizer
from .bfgs import BfgsOptimizer
from .target_functions import TargetFunction, LogPosteriorTarget

__all__ = [
    'Optimizer',
    'OptimizationResult',
    'LineSearchOptimizer', 
    'TrustRegionOptimizer',
    'NelderMeadOptimizer',
    'BfgsOptimizer',
    'TargetFunction',
    'LogPosteriorTarget'
]