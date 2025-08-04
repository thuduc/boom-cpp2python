"""
Line search optimization algorithms.

This module provides line search based optimization methods.
"""

import numpy as np
from typing import Optional, Callable
from ..linalg import Vector
from .base import Optimizer, OptimizationResult, LineSearchMixin


class LineSearchOptimizer(Optimizer, LineSearchMixin):
    """
    Generic line search optimizer using steepest descent direction.
    """
    
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """Minimize using line search with steepest descent."""
        self._reset_tracking()
        
        x = x0.copy()
        
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
            
            # Steepest descent direction
            direction = Vector(-g.to_numpy())
            
            # Line search
            alpha = self._armijo_line_search(objective, x, direction, g)
            
            # Update
            x_new = Vector(x.to_numpy() + alpha * direction.to_numpy())
            
            if bounds is not None:
                from .base import ConstraintHandler
                x_new = ConstraintHandler.project_bounds(x_new, bounds)
            
            x = x_new
        
        return OptimizationResult(
            x=x, fun=self._evaluate_function(objective, x),
            success=False, message="Maximum iterations reached",
            nit=self._iteration_count, nfev=self._function_evaluations
        )