"""Optimization and numerical routines for BOOM."""

from .optimizers import (
    Optimizer, OptimizationResult, NewtonRaphson, BFGS, ConjugateGradient, 
    LevenbergMarquardt, SimulatedAnnealing
)
from .line_search import (
    LineSearch, BacktrackingLineSearch, WolfeLineSearch, StrongWolfeLineSearch,
    ExactLineSearch, AdaptiveLineSearch
)
from .trust_region import (
    TrustRegion, TrustRegionResult, DoglegTrustRegion, CauchyPointTrustRegion,
    SteihaugTrustRegion, ExactTrustRegion
)
from .target_functions import (
    TargetFunction, LogLikelihoodFunction, PosteriorFunction, 
    QuadraticFunction, RosenbrockFunction
)
from .utils import (
    numerical_gradient, numerical_hessian, check_gradient, check_hessian,
    armijo_condition, wolfe_conditions, strong_wolfe_conditions,
    compute_condition_number, is_positive_definite, regularize_hessian,
    backtrack_line_search, generate_test_problems
)

__all__ = [
    # Optimizers
    "Optimizer", "OptimizationResult", "NewtonRaphson", "BFGS", "ConjugateGradient",
    "LevenbergMarquardt", "SimulatedAnnealing",
    
    # Line search
    "LineSearch", "BacktrackingLineSearch", "WolfeLineSearch", "StrongWolfeLineSearch",
    "ExactLineSearch", "AdaptiveLineSearch",
    
    # Trust region
    "TrustRegion", "TrustRegionResult", "DoglegTrustRegion", "CauchyPointTrustRegion",
    "SteihaugTrustRegion", "ExactTrustRegion",
    
    # Target functions
    "TargetFunction", "LogLikelihoodFunction", "PosteriorFunction", 
    "QuadraticFunction", "RosenbrockFunction",
    
    # Utilities
    "numerical_gradient", "numerical_hessian", "check_gradient", "check_hessian",
    "armijo_condition", "wolfe_conditions", "strong_wolfe_conditions",
    "compute_condition_number", "is_positive_definite", "regularize_hessian",
    "backtrack_line_search", "generate_test_problems"
]