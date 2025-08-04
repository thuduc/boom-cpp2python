"""
Target function utilities for BOOM Python package.

This module provides specialized target functions for optimization
in Bayesian and statistical modeling contexts.
"""

from .base import TargetFunction, LogTargetFunction
from .posterior import LogPosteriorTarget, LogLikelihoodTarget
from .transformations import LogTransform, LogitTransform, TransformedTarget
from .penalized import PenalizedTarget, Ridge, Lasso, ElasticNet

__all__ = [
    'TargetFunction',
    'LogTargetFunction',
    'LogPosteriorTarget',
    'LogLikelihoodTarget',
    'LogTransform',
    'LogitTransform', 
    'TransformedTarget',
    'PenalizedTarget',
    'Ridge',
    'Lasso',
    'ElasticNet'
]