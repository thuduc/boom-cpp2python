"""
Base classes for target functions.

This module provides the foundational classes for target functions
used in optimization and sampling algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from ..linalg import Vector


class TargetFunction(ABC):
    """
    Abstract base class for target functions.
    
    Represents a function to be optimized or sampled from,
    with optional gradient and Hessian computation.
    """
    
    def __init__(self, name: str = "TargetFunction"):
        """
        Initialize target function.
        
        Args:
            name: Name of the target function
        """
        self._name = name
        self._evaluation_count = 0
        self._gradient_count = 0
        self._hessian_count = 0
        self._cached_value = None
        self._cached_params = None
        self._cached_gradient = None
        self._cached_gradient_params = None
    
    @abstractmethod
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate the target function.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Function value
        """
        pass
    
    def gradient(self, parameters: Vector) -> Optional[Vector]:
        """
        Compute gradient of target function.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Gradient vector (None if not available)
        """
        self._gradient_count += 1
        
        # Check cache
        if (self._cached_gradient is not None and 
            self._cached_gradient_params is not None and
            np.allclose(parameters.to_numpy(), self._cached_gradient_params.to_numpy())):
            return self._cached_gradient
        
        # Compute gradient
        grad = self._compute_gradient(parameters)
        
        # Cache result
        self._cached_gradient = grad
        self._cached_gradient_params = parameters.copy()
        
        return grad
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """
        Compute gradient (to be overridden by subclasses).
        
        Default implementation uses numerical differentiation.
        """
        return self._numerical_gradient(parameters)
    
    def hessian(self, parameters: Vector) -> Optional[np.ndarray]:
        """
        Compute Hessian of target function.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Hessian matrix (None if not available)
        """
        self._hessian_count += 1
        return self._compute_hessian(parameters)
    
    def _compute_hessian(self, parameters: Vector) -> Optional[np.ndarray]:
        """
        Compute Hessian (to be overridden by subclasses).
        
        Default implementation uses numerical differentiation.
        """
        return self._numerical_hessian(parameters)
    
    def _numerical_gradient(self, parameters: Vector, 
                           epsilon: float = 1e-8) -> Vector:
        """Compute numerical gradient using finite differences."""
        x = parameters.to_numpy()
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = self.evaluate(Vector(x_plus))
            f_minus = self.evaluate(Vector(x_minus))
            
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return Vector(grad)
    
    def _numerical_hessian(self, parameters: Vector,
                          epsilon: float = 1e-6) -> np.ndarray:
        """Compute numerical Hessian using finite differences."""
        x = parameters.to_numpy()
        n = len(x)
        hessian = np.zeros((n, n))
        
        f_x = self.evaluate(parameters)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x.copy()
                x_i = x.copy()
                x_j = x.copy()
                
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                x_i[i] += epsilon
                x_j[j] += epsilon
                
                f_ij = self.evaluate(Vector(x_ij))
                f_i = self.evaluate(Vector(x_i))
                f_j = self.evaluate(Vector(x_j))
                
                hessian[i, j] = (f_ij - f_i - f_j + f_x) / (epsilon * epsilon)
                
                if i != j:
                    hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def __call__(self, parameters: Vector) -> float:
        """Make function callable."""
        return self.evaluate(parameters)
    
    def reset_counters(self) -> None:
        """Reset evaluation counters."""
        self._evaluation_count = 0
        self._gradient_count = 0
        self._hessian_count = 0
    
    def clear_cache(self) -> None:
        """Clear cached values."""
        self._cached_value = None
        self._cached_params = None
        self._cached_gradient = None
        self._cached_gradient_params = None
    
    def get_evaluation_counts(self) -> Dict[str, int]:
        """Get evaluation counts."""
        return {
            'function_evaluations': self._evaluation_count,
            'gradient_evaluations': self._gradient_count, 
            'hessian_evaluations': self._hessian_count
        }
    
    @property
    def name(self) -> str:
        """Get function name."""
        return self._name


class LogTargetFunction(TargetFunction):
    """
    Base class for log-scale target functions.
    
    Used for functions that operate on the log scale,
    such as log posteriors and log likelihoods.
    """
    
    def __init__(self, name: str = "LogTargetFunction"):
        """Initialize log target function."""
        super().__init__(name)
    
    def log_evaluate(self, parameters: Vector) -> float:
        """
        Evaluate function on log scale.
        
        This is the same as evaluate() but makes the intent clear.
        """
        return self.evaluate(parameters)
    
    def probability_evaluate(self, parameters: Vector) -> float:
        """
        Evaluate function on probability scale.
        
        Returns exp(log_evaluate(parameters)).
        """
        log_val = self.log_evaluate(parameters)
        
        # Prevent overflow
        if log_val > 700:  # exp(700) is close to max float
            return np.inf
        elif log_val < -700:  # exp(-700) is close to 0
            return 0.0
        else:
            return np.exp(log_val)


class CompositeTargetFunction(TargetFunction):
    """
    Composite target function that combines multiple target functions.
    
    Useful for creating complex objectives from simpler components.
    """
    
    def __init__(self, functions: list, weights: Optional[list] = None,
                 name: str = "CompositeTarget"):
        """
        Initialize composite target function.
        
        Args:
            functions: List of target functions to combine
            weights: Optional weights for each function (default: equal weights)
            name: Function name
        """
        super().__init__(name)
        
        if not functions:
            raise ValueError("Must provide at least one function")
        
        self._functions = functions
        
        if weights is None:
            self._weights = [1.0] * len(functions)
        else:
            if len(weights) != len(functions):
                raise ValueError("Number of weights must match number of functions")
            self._weights = list(weights)
    
    def evaluate(self, parameters: Vector) -> float:
        """Evaluate composite function as weighted sum."""
        self._evaluation_count += 1
        
        total = 0.0
        for func, weight in zip(self._functions, self._weights):
            total += weight * func.evaluate(parameters)
        
        return total
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """Compute gradient as weighted sum of component gradients."""
        gradients = []
        
        for func in self._functions:
            grad = func.gradient(parameters)
            if grad is None:
                # If any component doesn't have gradient, fall back to numerical
                return self._numerical_gradient(parameters)
            gradients.append(grad)
        
        # Weighted sum of gradients
        total_grad = np.zeros(len(parameters.to_numpy()))
        for grad, weight in zip(gradients, self._weights):
            total_grad += weight * grad.to_numpy()
        
        return Vector(total_grad)
    
    def _compute_hessian(self, parameters: Vector) -> Optional[np.ndarray]:
        """Compute Hessian as weighted sum of component Hessians."""
        hessians = []
        
        for func in self._functions:
            hess = func.hessian(parameters)
            if hess is None:
                # If any component doesn't have Hessian, fall back to numerical
                return self._numerical_hessian(parameters)
            hessians.append(hess)
        
        # Weighted sum of Hessians
        n = len(parameters.to_numpy())
        total_hess = np.zeros((n, n))
        for hess, weight in zip(hessians, self._weights):
            total_hess += weight * hess
        
        return total_hess
    
    @property
    def component_functions(self) -> list:
        """Get component functions."""
        return self._functions.copy()
    
    @property
    def weights(self) -> list:
        """Get component weights."""
        return self._weights.copy()


class ScaledTargetFunction(TargetFunction):
    """
    Target function scaled by a constant factor.
    """
    
    def __init__(self, base_function: TargetFunction, scale: float,
                 name: Optional[str] = None):
        """
        Initialize scaled target function.
        
        Args:
            base_function: Base target function
            scale: Scaling factor
            name: Function name (default: scaled version of base name)
        """
        if name is None:
            name = f"Scaled_{base_function.name}"
        
        super().__init__(name)
        self._base_function = base_function
        self._scale = scale
    
    def evaluate(self, parameters: Vector) -> float:
        """Evaluate scaled function."""
        self._evaluation_count += 1
        return self._scale * self._base_function.evaluate(parameters)
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """Compute scaled gradient."""
        base_grad = self._base_function.gradient(parameters)
        if base_grad is None:
            return None
        
        return Vector(self._scale * base_grad.to_numpy())
    
    def _compute_hessian(self, parameters: Vector) -> Optional[np.ndarray]:
        """Compute scaled Hessian."""
        base_hess = self._base_function.hessian(parameters)
        if base_hess is None:
            return None
        
        return self._scale * base_hess
    
    @property
    def base_function(self) -> TargetFunction:
        """Get base function."""
        return self._base_function
    
    @property
    def scale(self) -> float:
        """Get scaling factor."""
        return self._scale


class ConstantFunction(TargetFunction):
    """
    Constant target function (returns same value for all inputs).
    """
    
    def __init__(self, value: float, name: str = "Constant"):
        """
        Initialize constant function.
        
        Args:
            value: Constant value to return
            name: Function name
        """
        super().__init__(name)
        self._value = value
    
    def evaluate(self, parameters: Vector) -> float:
        """Return constant value."""
        self._evaluation_count += 1
        return self._value
    
    def _compute_gradient(self, parameters: Vector) -> Vector:
        """Return zero gradient."""
        return Vector(np.zeros(len(parameters.to_numpy())))
    
    def _compute_hessian(self, parameters: Vector) -> np.ndarray:
        """Return zero Hessian."""
        n = len(parameters.to_numpy())
        return np.zeros((n, n))