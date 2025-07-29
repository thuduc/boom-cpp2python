"""Target functions for optimization."""
import numpy as np
from typing import Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod
from ..linalg import Vector, Matrix
from ..models.base import Model


class TargetFunction(ABC):
    """Base class for optimization target functions."""
    
    def __init__(self, name: str = ""):
        """Initialize target function.
        
        Args:
            name: Optional name for the function
        """
        self.name = name
        self.n_evaluations = 0
        self.n_gradient_evaluations = 0
        self.n_hessian_evaluations = 0
    
    @abstractmethod
    def evaluate(self, x: Vector) -> float:
        """Evaluate the function at x."""
        pass
    
    def gradient(self, x: Vector) -> Vector:
        """Compute gradient at x (default: numerical differentiation)."""
        self.n_gradient_evaluations += 1
        return self._numerical_gradient(x)
    
    def hessian(self, x: Vector) -> Matrix:
        """Compute Hessian at x (default: numerical differentiation)."""
        self.n_hessian_evaluations += 1
        return self._numerical_hessian(x)
    
    def evaluate_with_gradient(self, x: Vector) -> Tuple[float, Vector]:
        """Evaluate function and gradient simultaneously."""
        f = self.evaluate(x)
        g = self.gradient(x)
        return f, g
    
    def evaluate_with_hessian(self, x: Vector) -> Tuple[float, Vector, Matrix]:
        """Evaluate function, gradient, and Hessian simultaneously."""
        f = self.evaluate(x)
        g = self.gradient(x)
        H = self.hessian(x)
        return f, g, H
    
    def _numerical_gradient(self, x: Vector, h: float = 1e-8) -> Vector:
        """Compute numerical gradient using central differences."""
        n = len(x)
        grad = Vector.zero(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            f_plus = self.evaluate(x_plus)
            f_minus = self.evaluate(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * h)
        
        return grad
    
    def _numerical_hessian(self, x: Vector, h: float = 1e-6) -> Matrix:
        """Compute numerical Hessian using central differences."""
        n = len(x)
        H = Matrix.zero(n, n)
        
        # Diagonal elements (second derivatives)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            f_center = self.evaluate(x)
            f_plus = self.evaluate(x_plus)
            f_minus = self.evaluate(x_minus)
            
            H[i, i] = (f_plus - 2 * f_center + f_minus) / (h * h)
        
        # Off-diagonal elements (mixed derivatives)
        for i in range(n):
            for j in range(i + 1, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                
                f_pp = self.evaluate(x_pp)
                f_pm = self.evaluate(x_pm)
                f_mp = self.evaluate(x_mp)
                f_mm = self.evaluate(x_mm)
                
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                H[j, i] = H[i, j]  # Symmetry
        
        return H
    
    def reset_counters(self):
        """Reset evaluation counters."""
        self.n_evaluations = 0
        self.n_gradient_evaluations = 0
        self.n_hessian_evaluations = 0


class LogLikelihoodFunction(TargetFunction):
    """Log likelihood function for model optimization."""
    
    def __init__(self, model: Model, data: Optional[list] = None, 
                 parameter_names: Optional[list] = None):
        """Initialize log likelihood function.
        
        Args:
            model: Statistical model
            data: Optional data (if not already in model)
            parameter_names: Names of parameters to optimize
        """
        super().__init__(f"LogLikelihood_{model.__class__.__name__}")
        self.model = model
        self.parameter_names = parameter_names or []
        
        if data is not None:
            self.model.clear_data()
            for datum in data:
                self.model.add_data(datum)
    
    def evaluate(self, x: Vector) -> float:
        """Evaluate log likelihood at parameter vector x."""
        self.n_evaluations += 1
        
        # Set model parameters from x
        self._set_parameters(x)
        
        # Return log likelihood
        return self.model.loglike()
    
    def _set_parameters(self, x: Vector):
        """Set model parameters from optimization vector."""
        if self.parameter_names:
            # Use specified parameter names
            for i, param_name in enumerate(self.parameter_names):
                if param_name in self.model._params:
                    self.model._params[param_name].value = x[i]
        else:
            # Default: assume x contains all model parameters in order
            param_idx = 0
            for param_name, param in self.model._params.items():
                if hasattr(param, 'size'):
                    param_size = param.size()
                    if param_size == 1:
                        param.value = x[param_idx]
                        param_idx += 1
                    else:
                        # Vector parameter
                        param.value = Vector(x[param_idx:param_idx + param_size])
                        param_idx += param_size


class PosteriorFunction(TargetFunction):
    """Log posterior function (log likelihood + log prior)."""
    
    def __init__(self, model: Model, log_prior: Callable[[Vector], float],
                 data: Optional[list] = None, parameter_names: Optional[list] = None):
        """Initialize posterior function.
        
        Args:
            model: Statistical model
            log_prior: Function computing log prior density
            data: Optional data
            parameter_names: Names of parameters to optimize
        """
        super().__init__(f"Posterior_{model.__class__.__name__}")
        self.log_likelihood = LogLikelihoodFunction(model, data, parameter_names)
        self.log_prior = log_prior
    
    def evaluate(self, x: Vector) -> float:
        """Evaluate log posterior at parameter vector x."""
        self.n_evaluations += 1
        
        log_like = self.log_likelihood.evaluate(x)
        log_prior = self.log_prior(x)
        
        return log_like + log_prior


class QuadraticFunction(TargetFunction):
    """Quadratic function: f(x) = 0.5 * x^T A x + b^T x + c."""
    
    def __init__(self, A: Matrix, b: Vector, c: float = 0.0):
        """Initialize quadratic function.
        
        Args:
            A: Quadratic coefficient matrix
            b: Linear coefficient vector
            c: Constant term
        """
        super().__init__("Quadratic")
        self.A = Matrix(A)
        self.b = Vector(b)
        self.c = c
    
    def evaluate(self, x: Vector) -> float:
        """Evaluate quadratic function."""
        self.n_evaluations += 1
        x = Vector(x)
        return 0.5 * x.dot(self.A @ x) + self.b.dot(x) + self.c
    
    def gradient(self, x: Vector) -> Vector:
        """Compute gradient: A * x + b."""
        self.n_gradient_evaluations += 1
        x = Vector(x)
        return self.A @ x + self.b
    
    def hessian(self, x: Vector) -> Matrix:
        """Compute Hessian: A."""
        self.n_hessian_evaluations += 1
        return Matrix(self.A)


class RosenbrockFunction(TargetFunction):
    """Rosenbrock function for testing optimization algorithms."""
    
    def __init__(self, a: float = 1.0, b: float = 100.0):
        """Initialize Rosenbrock function.
        
        Args:
            a: First parameter (default 1.0)
            b: Second parameter (default 100.0)
        """
        super().__init__("Rosenbrock")
        self.a = a
        self.b = b
    
    def evaluate(self, x: Vector) -> float:
        """Evaluate Rosenbrock function."""
        self.n_evaluations += 1
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        return (self.a - x1)**2 + self.b * (x2 - x1**2)**2
    
    def gradient(self, x: Vector) -> Vector:
        """Compute gradient analytically."""
        self.n_gradient_evaluations += 1
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        grad = Vector.zero(2)
        grad[0] = -2 * (self.a - x1) - 4 * self.b * x1 * (x2 - x1**2)
        grad[1] = 2 * self.b * (x2 - x1**2)
        return grad
    
    def hessian(self, x: Vector) -> Matrix:
        """Compute Hessian analytically."""
        self.n_hessian_evaluations += 1
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        
        x1, x2 = x[0], x[1]
        H = Matrix.zero(2, 2)
        H[0, 0] = 2 - 4 * self.b * (x2 - 3 * x1**2)
        H[0, 1] = -4 * self.b * x1
        H[1, 0] = -4 * self.b * x1
        H[1, 1] = 2 * self.b
        return H