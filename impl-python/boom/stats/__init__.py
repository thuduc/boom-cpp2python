"""
Statistical utilities for BOOM Python package.

This module provides various statistical functions, tests, and utilities
for data analysis and model evaluation.
"""

from .descriptive import DescriptiveStats, compute_summary_stats
from .hypothesis_testing import TTest, ChiSquareTest, KolmogorovSmirnovTest
from .information_criteria import AIC, BIC, compute_ic
from .model_selection import CrossValidator, ModelComparison
from .regression_diagnostics import RegressionDiagnostics, residual_analysis

__all__ = [
    'DescriptiveStats',
    'compute_summary_stats',
    'TTest',
    'ChiSquareTest', 
    'KolmogorovSmirnovTest',
    'AIC',
    'BIC',
    'compute_ic',
    'CrossValidator',
    'ModelComparison',
    'RegressionDiagnostics',
    'residual_analysis'
]