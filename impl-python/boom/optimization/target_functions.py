"""
Target function utilities for optimization.

This module provides target function classes for optimization problems,
particularly for statistical model parameter estimation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict
from ..linalg import Vector
from ..models.base import Model


class TargetFunction(ABC):
    """
    Abstract base class for optimization target functions.
    
    Provides a common interface for functions to be optimized,
    including objective value, gradient, and Hessian computation.
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


class LogPosteriorTarget(TargetFunction):
    """
    Log posterior target function for Bayesian parameter estimation.
    
    Combines log likelihood and log prior for MAP estimation.
    """
    
    def __init__(self, model: Model,
                 log_prior: Optional[Callable[[Vector], float]] = None,
                 name: str = "LogPosterior"):
        """
        Initialize log posterior target.
        
        Args:
            model: Statistical model with log_likelihood method
            log_prior: Log prior function (uniform if None)
            name: Function name
        """
        super().__init__(name)
        self._model = model
        self._log_prior = log_prior
    
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate log posterior.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Log posterior value
        """
        self._evaluation_count += 1
        
        try:
            # Set model parameters
            if hasattr(self._model, 'unvectorize_params'):
                self._model.unvectorize_params(parameters)
            elif hasattr(self._model, 'set_parameters'):
                # Convert Vector to appropriate format for model
                param_dict = self._vector_to_params(parameters)
                self._model.set_parameters(param_dict)
            else:
                raise AttributeError("Model must have unvectorize_params or set_parameters method")
            
            # Compute log likelihood
            log_lik = self._model.log_likelihood()
            
            # Add log prior if specified
            if self._log_prior is not None:
                log_prior = self._log_prior(parameters)
            else:
                log_prior = 0.0  # Uniform prior
            
            return log_lik + log_prior
            
        except Exception as e:
            # Return very negative value for invalid parameters
            return -1e10
    
    def _vector_to_params(self, parameters: Vector) -> Dict[str, Any]:
        """Convert parameter vector to model parameter dictionary."""
        # This is a placeholder - specific models should override
        return {'parameters': parameters.to_numpy()}
    
    def gradient(self, parameters: Vector) -> Optional[Vector]:
        """
        Compute gradient of log posterior.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Gradient vector
        """
        self._gradient_count += 1
        
        # Check if model provides analytical gradient
        if hasattr(self._model, 'log_likelihood_gradient'):
            try:
                if hasattr(self._model, 'unvectorize_params'):
                    self._model.unvectorize_params(parameters)
                
                grad_lik = self._model.log_likelihood_gradient()
                
                # Add prior gradient if available
                if self._log_prior is not None and hasattr(self, '_log_prior_gradient'):
                    grad_prior = self._log_prior_gradient(parameters)
                    grad_lik = Vector(grad_lik.to_numpy() + grad_prior.to_numpy())
                
                return grad_lik
                
            except Exception:
                pass
        
        # Fall back to numerical gradient
        return self._numerical_gradient(parameters)


class LogLikelihoodTarget(TargetFunction):
    """
    Log likelihood target function for maximum likelihood estimation.
    """
    
    def __init__(self, model: Model, name: str = "LogLikelihood"):
        """
        Initialize log likelihood target.
        
        Args:
            model: Statistical model with log_likelihood method
            name: Function name
        """
        super().__init__(name)
        self._model = model
    
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate log likelihood.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Log likelihood value
        """
        self._evaluation_count += 1
        
        try:
            # Set model parameters
            if hasattr(self._model, 'unvectorize_params'):
                self._model.unvectorize_params(parameters)
            elif hasattr(self._model, 'set_parameters'):
                param_dict = self._vector_to_params(parameters)
                self._model.set_parameters(param_dict)
            else:
                raise AttributeError("Model must have unvectorize_params or set_parameters method")
            
            return self._model.log_likelihood()
            
        except Exception:
            return -1e10
    
    def _vector_to_params(self, parameters: Vector) -> Dict[str, Any]:
        """Convert parameter vector to model parameter dictionary."""
        return {'parameters': parameters.to_numpy()}


class QuadraticTarget(TargetFunction):
    """
    Quadratic target function for testing optimizers.
    
    f(x) = 0.5 * (x - center)^T Q (x - center) + offset
    """
    
    def __init__(self, Q: np.ndarray, center: Optional[Vector] = None,
                 offset: float = 0.0, name: str = "Quadratic"):
        """
        Initialize quadratic target function.
        
        Args:
            Q: Positive definite matrix
            center: Center point (zeros if None)
            offset: Constant offset
            name: Function name
        """
        super().__init__(name)
        self._Q = Q
        self._center = center if center is not None else Vector(np.zeros(Q.shape[0]))
        self._offset = offset
        
        # Check that Q is square
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square matrix")
    
    def evaluate(self, parameters: Vector) -> float:
        """Evaluate quadratic function."""
        self._evaluation_count += 1
        
        x = parameters.to_numpy()
        c = self._center.to_numpy()
        diff = x - c
        
        return 0.5 * np.dot(diff, self._Q @ diff) + self._offset
    
    def gradient(self, parameters: Vector) -> Vector:
        """Compute analytical gradient."""
        self._gradient_count += 1
        
        x = parameters.to_numpy()
        c = self._center.to_numpy()
        
        grad = self._Q @ (x - c)
        return Vector(grad)
    
    def hessian(self, parameters: Vector) -> np.ndarray:
        """Compute analytical Hessian."""
        self._hessian_count += 1
        return self._Q.copy()


class RosenbrockTarget(TargetFunction):
    """
    Rosenbrock function for testing optimizers.
    
    f(x, y) = (a - x)² + b(y - x²)²
    Global minimum at (a, a²) with value 0.
    """
    
    def __init__(self, a: float = 1.0, b: float = 100.0, 
                 name: str = "Rosenbrock"):
        """
        Initialize Rosenbrock function.
        
        Args:
            a: Parameter a (default 1.0)
            b: Parameter b (default 100.0)
            name: Function name
        """
        super().__init__(name)
        self._a = a
        self._b = b
    
    def evaluate(self, parameters: Vector) -> float:
        """Evaluate Rosenbrock function."""
        self._evaluation_count += 1
        
        x = parameters.to_numpy()
        if len(x) < 2:
            raise ValueError("Rosenbrock function requires at least 2 parameters")
        
        # Generalized Rosenbrock for n dimensions
        result = 0.0
        for i in range(len(x) - 1):
            result += self._b * (x[i + 1] - x[i]**2)**2 + (self._a - x[i])**2
        
        return result
    
    def gradient(self, parameters: Vector) -> Vector:
        """Compute analytical gradient."""
        self._gradient_count += 1
        
        x = parameters.to_numpy()
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n - 1):
            grad[i] += -2 * (self._a - x[i]) - 4 * self._b * x[i] * (x[i + 1] - x[i]**2)
            grad[i + 1] += 2 * self._b * (x[i + 1] - x[i]**2)
        
        return Vector(grad)
    
    def hessian(self, parameters: Vector) -> np.ndarray:
        """Compute analytical Hessian."""
        self._hessian_count += 1
        
        x = parameters.to_numpy()
        n = len(x)
        H = np.zeros((n, n))
        
        for i in range(n - 1):
            # Diagonal terms
            H[i, i] += 2 + 4 * self._b * (3 * x[i]**2 - x[i + 1])
            H[i + 1, i + 1] += 2 * self._b
            
            # Off-diagonal terms
            H[i, i + 1] = -4 * self._b * x[i]
            H[i + 1, i] = -4 * self._b * x[i]
        
        return H


def create_least_squares_target(residual_func: Callable[[Vector], Vector],
                               name: str = "LeastSquares") -> TargetFunction:
    """
    Create least squares target function.
    
    Args:
        residual_func: Function that computes residuals
        name: Function name
        
    Returns:
        Least squares target function
    """
    
    class LeastSquaresTarget(TargetFunction):
        def __init__(self):
            super().__init__(name)
            self._residual_func = residual_func
        
        def evaluate(self, parameters: Vector) -> float:
            self._evaluation_count += 1
            residuals = self._residual_func(parameters)
            return 0.5 * np.sum(residuals.to_numpy()**2)
    
    return LeastSquaresTarget()