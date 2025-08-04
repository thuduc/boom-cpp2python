"""
Base classes for optimization algorithms.

This module provides the foundational classes for all optimizers
in the BOOM Python package.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Dict, Any, Tuple
from ..linalg import Vector


class OptimizationResult:
    """
    Result of an optimization procedure.
    
    Contains the optimal parameters, function value, and convergence information.
    """
    
    def __init__(self, x: Vector, fun: float, success: bool = True,
                 message: str = "Optimization successful",
                 nit: int = 0, nfev: int = 0, 
                 gradient: Optional[Vector] = None,
                 hessian: Optional[np.ndarray] = None):
        """
        Initialize optimization result.
        
        Args:
            x: Optimal parameters
            fun: Function value at optimum
            success: Whether optimization was successful
            message: Status message
            nit: Number of iterations
            nfev: Number of function evaluations
            gradient: Gradient at optimum
            hessian: Hessian at optimum
        """
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nit = nit
        self.nfev = nfev
        self.gradient = gradient
        self.hessian = hessian
    
    def __str__(self) -> str:
        """String representation of result."""
        return (f"OptimizationResult(success={self.success}, "
                f"fun={self.fun:.6f}, nit={self.nit}, "
                f"message='{self.message}')")


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    Provides common interface for all optimization methods in BOOM.
    """
    
    def __init__(self, max_iterations: int = 1000, 
                 tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6):
        """
        Initialize optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Function value convergence tolerance
            gradient_tolerance: Gradient norm convergence tolerance
        """
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._gradient_tolerance = gradient_tolerance
        
        # Tracking
        self._iteration_count = 0
        self._function_evaluations = 0
        self._gradient_evaluations = 0
        
        # History
        self._function_history = []
        self._parameter_history = []
        self._gradient_history = []
    
    @abstractmethod
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """
        Minimize the objective function.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            gradient: Gradient function (optional)
            hessian: Hessian function (optional)
            bounds: Parameter bounds (optional)
            constraints: Optimization constraints (optional)
            
        Returns:
            OptimizationResult containing optimization results
        """
        pass
    
    def maximize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """
        Maximize the objective function.
        
        Args:
            objective: Function to maximize
            x0: Initial parameter values  
            gradient: Gradient function (optional)
            hessian: Hessian function (optional)
            bounds: Parameter bounds (optional)
            constraints: Optimization constraints (optional)
            
        Returns:
            OptimizationResult containing optimization results
        """
        # Convert to minimization problem
        def neg_objective(x: Vector) -> float:
            return -objective(x)
        
        def neg_gradient(x: Vector) -> Vector:
            if gradient is None:
                return None
            grad = gradient(x)
            return Vector(-grad.to_numpy())
        
        def neg_hessian(x: Vector) -> np.ndarray:
            if hessian is None:
                return None
            return -hessian(x)
        
        result = self.minimize(neg_objective, x0, neg_gradient, neg_hessian, bounds, constraints)
        
        # Flip sign of function value back
        result.fun = -result.fun
        if result.gradient is not None:
            result.gradient = Vector(-result.gradient.to_numpy())
        if result.hessian is not None:
            result.hessian = -result.hessian
        
        return result
    
    def _evaluate_function(self, objective: Callable[[Vector], float], 
                          x: Vector) -> float:
        """Evaluate objective function and track evaluations."""
        self._function_evaluations += 1
        value = objective(x)
        self._function_history.append(value)
        self._parameter_history.append(x.copy())
        return value
    
    def _evaluate_gradient(self, gradient: Callable[[Vector], Vector],
                          x: Vector) -> Vector:
        """Evaluate gradient and track evaluations."""
        self._gradient_evaluations += 1
        grad = gradient(x)
        self._gradient_history.append(grad.copy())
        return grad
    
    def _numerical_gradient(self, objective: Callable[[Vector], float],
                           x: Vector, epsilon: float = 1e-8) -> Vector:
        """Compute numerical gradient using finite differences."""
        x_array = x.to_numpy()
        n = len(x_array)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x_array.copy()
            x_minus = x_array.copy()
            
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = self._evaluate_function(objective, Vector(x_plus))
            f_minus = self._evaluate_function(objective, Vector(x_minus))
            
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return Vector(grad)
    
    def _numerical_hessian(self, objective: Callable[[Vector], float],
                          x: Vector, epsilon: float = 1e-6) -> np.ndarray:
        """Compute numerical Hessian using finite differences."""
        x_array = x.to_numpy()
        n = len(x_array)
        hessian = np.zeros((n, n))
        
        f_x = self._evaluate_function(objective, x)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x_array.copy()
                x_i = x_array.copy()
                x_j = x_array.copy()
                
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                x_i[i] += epsilon
                x_j[j] += epsilon
                
                f_ij = self._evaluate_function(objective, Vector(x_ij))
                f_i = self._evaluate_function(objective, Vector(x_i))
                f_j = self._evaluate_function(objective, Vector(x_j))
                
                hessian[i, j] = (f_ij - f_i - f_j + f_x) / (epsilon * epsilon)
                
                if i != j:
                    hessian[j, i] = hessian[i, j]  # Symmetric
        
        return hessian
    
    def _check_convergence(self, x_new: Vector, x_old: Vector,
                          f_new: float, f_old: float,
                          gradient: Optional[Vector] = None) -> Tuple[bool, str]:
        """
        Check convergence criteria.
        
        Returns:
            Tuple of (converged, message)
        """
        # Function change criterion
        if abs(f_new - f_old) < self._tolerance * (1 + abs(f_old)):
            return True, "Function convergence"
        
        # Parameter change criterion
        x_change = np.linalg.norm(x_new.to_numpy() - x_old.to_numpy())
        x_scale = np.linalg.norm(x_old.to_numpy())
        if x_change < self._tolerance * (1 + x_scale):
            return True, "Parameter convergence"
        
        # Gradient criterion
        if gradient is not None:
            grad_norm = np.linalg.norm(gradient.to_numpy())
            if grad_norm < self._gradient_tolerance:
                return True, "Gradient convergence"
        
        # Maximum iterations
        if self._iteration_count >= self._max_iterations:
            return True, "Maximum iterations reached"
        
        return False, "Continuing optimization"
    
    def _reset_tracking(self) -> None:
        """Reset optimization tracking variables."""
        self._iteration_count = 0
        self._function_evaluations = 0
        self._gradient_evaluations = 0
        self._function_history = []
        self._parameter_history = []
        self._gradient_history = []
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history."""
        return {
            'function_values': self._function_history.copy(),
            'parameters': [x.copy() for x in self._parameter_history],
            'gradients': [g.copy() for g in self._gradient_history],
            'iterations': self._iteration_count,
            'function_evaluations': self._function_evaluations,
            'gradient_evaluations': self._gradient_evaluations
        }
    
    @property
    def max_iterations(self) -> int:
        """Get maximum iterations."""
        return self._max_iterations
    
    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        """Set maximum iterations."""
        if value <= 0:
            raise ValueError("Maximum iterations must be positive")
        self._max_iterations = value
    
    @property
    def tolerance(self) -> float:
        """Get function tolerance."""
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set function tolerance."""
        if value <= 0:
            raise ValueError("Tolerance must be positive")
        self._tolerance = value
    
    @property
    def gradient_tolerance(self) -> float:
        """Get gradient tolerance."""
        return self._gradient_tolerance
    
    @gradient_tolerance.setter
    def gradient_tolerance(self, value: float) -> None:
        """Set gradient tolerance."""
        if value <= 0:
            raise ValueError("Gradient tolerance must be positive")
        self._gradient_tolerance = value


class LineSearchMixin:
    """Mixin class providing line search functionality."""
    
    def _armijo_line_search(self, objective: Callable[[Vector], float],
                           x: Vector, direction: Vector, gradient: Vector,
                           alpha_init: float = 1.0, c1: float = 1e-4,
                           rho: float = 0.5, max_backtracks: int = 50) -> float:
        """
        Armijo line search with backtracking.
        
        Args:
            objective: Objective function
            x: Current point
            direction: Search direction
            gradient: Gradient at current point
            alpha_init: Initial step size
            c1: Armijo constant
            rho: Backtracking factor
            max_backtracks: Maximum backtracking steps
            
        Returns:
            Step size
        """
        f_x = self._evaluate_function(objective, x)
        directional_deriv = np.dot(gradient.to_numpy(), direction.to_numpy())
        
        alpha = alpha_init
        
        for i in range(max_backtracks):
            x_new = Vector(x.to_numpy() + alpha * direction.to_numpy())
            f_new = self._evaluate_function(objective, x_new)
            
            # Armijo condition
            if f_new <= f_x + c1 * alpha * directional_deriv:
                return alpha
            
            alpha *= rho
        
        return alpha  # Return last tried step size
    
    def _wolfe_line_search(self, objective: Callable[[Vector], float],
                          gradient_func: Callable[[Vector], Vector],
                          x: Vector, direction: Vector, gradient: Vector,
                          alpha_init: float = 1.0, c1: float = 1e-4, c2: float = 0.9,
                          max_iterations: int = 20) -> float:
        """
        Wolfe line search.
        
        Args:
            objective: Objective function
            gradient_func: Gradient function
            x: Current point
            direction: Search direction
            gradient: Gradient at current point
            alpha_init: Initial step size
            c1: First Wolfe constant
            c2: Second Wolfe constant
            max_iterations: Maximum iterations
            
        Returns:
            Step size
        """
        # Simplified implementation - full Wolfe search is complex
        # Fall back to Armijo for now
        return self._armijo_line_search(objective, x, direction, gradient, 
                                       alpha_init, c1, max_iterations)


class ConstraintHandler:
    """Helper class for handling optimization constraints."""
    
    @staticmethod
    def project_bounds(x: Vector, bounds: list) -> Vector:
        """Project parameters onto bound constraints."""
        if bounds is None:
            return x
        
        x_array = x.to_numpy()
        
        for i, (lower, upper) in enumerate(bounds):
            if lower is not None:
                x_array[i] = max(x_array[i], lower)
            if upper is not None:
                x_array[i] = min(x_array[i], upper)
        
        return Vector(x_array)
    
    @staticmethod
    def check_bounds(x: Vector, bounds: list) -> bool:
        """Check if parameters satisfy bound constraints."""
        if bounds is None:
            return True
        
        x_array = x.to_numpy()
        
        for i, (lower, upper) in enumerate(bounds):
            if lower is not None and x_array[i] < lower:
                return False
            if upper is not None and x_array[i] > upper:
                return False
        
        return True