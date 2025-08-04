"""
Trust region optimization algorithms.

This module provides trust region based optimization methods.
"""

import numpy as np
from typing import Optional, Callable
from ..linalg import Vector
from .base import Optimizer, OptimizationResult


class TrustRegionOptimizer(Optimizer):
    """
    Trust region optimizer using quadratic model approximation.
    """
    
    def __init__(self, max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6,
                 initial_radius: float = 1.0,
                 max_radius: float = 10.0):
        """Initialize trust region optimizer."""
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self._initial_radius = initial_radius
        self._max_radius = max_radius
    
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """Minimize using trust region method."""
        # Simplified trust region implementation
        self._reset_tracking()
        
        x = x0.copy()
        radius = self._initial_radius
        
        for iteration in range(self._max_iterations):
            self._iteration_count = iteration + 1
            
            # Evaluate function and gradient
            f = self._evaluate_function(objective, x)
            
            if gradient is not None:
                g = self._evaluate_gradient(gradient, x)
            else:
                g = self._numerical_gradient(objective, x)
            
            # Check convergence
            if np.linalg.norm(g.to_numpy()) < self._gradient_tolerance:
                return OptimizationResult(
                    x=x, fun=f, success=True,
                    message="Gradient convergence",
                    nit=self._iteration_count,
                    nfev=self._function_evaluations,
                    gradient=g
                )
            
            # Simple Cauchy point step (steepest descent within trust region)
            g_norm = np.linalg.norm(g.to_numpy())
            if g_norm > 0:
                step_length = min(radius, radius / g_norm)
                step = Vector(-step_length * g.to_numpy() / g_norm)
            else:
                step = Vector(np.zeros(len(x.to_numpy())))
            
            # Try step
            x_new = Vector(x.to_numpy() + step.to_numpy())
            
            if bounds is not None:
                from .base import ConstraintHandler
                x_new = ConstraintHandler.project_bounds(x_new, bounds)
            
            f_new = self._evaluate_function(objective, x_new)
            
            # Update trust region radius based on step quality
            if f_new < f:
                x = x_new
                radius = min(self._max_radius, 1.5 * radius)  # Expand
            else:
                radius *= 0.5  # Contract
                
            if radius < 1e-10:
                break
        
        return OptimizationResult(
            x=x, fun=self._evaluate_function(objective, x),
            success=False, message="Maximum iterations reached",
            nit=self._iteration_count, nfev=self._function_evaluations
        )