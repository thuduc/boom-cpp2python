"""
BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer implementation.

This module provides the BFGS quasi-Newton optimization algorithm.
"""

import numpy as np
from typing import Optional, Callable, List
from ..linalg import Vector
from .base import Optimizer, OptimizationResult, LineSearchMixin


class BfgsOptimizer(Optimizer, LineSearchMixin):
    """
    BFGS quasi-Newton optimizer.
    
    Uses BFGS approximation to the inverse Hessian for efficient
    second-order optimization without computing the full Hessian.
    """
    
    def __init__(self, max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6,
                 initial_hessian_scale: float = 1.0):
        """
        Initialize BFGS optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Function value convergence tolerance
            gradient_tolerance: Gradient norm convergence tolerance
            initial_hessian_scale: Scale for initial Hessian approximation
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self._initial_hessian_scale = initial_hessian_scale
    
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """
        Minimize objective function using BFGS algorithm.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            gradient: Gradient function (if None, numerical gradient used)
            hessian: Hessian function (ignored - BFGS approximates)
            bounds: Parameter bounds (optional)
            constraints: Optimization constraints (not supported)
            
        Returns:
            OptimizationResult containing optimization results
        """
        if constraints is not None:
            raise NotImplementedError("Constraints not supported in BFGS optimizer")
        
        self._reset_tracking()
        
        # Initialize
        x = x0.copy()
        n = len(x.to_numpy())
        
        # Initial Hessian approximation (scaled identity)
        B_inv = self._initial_hessian_scale * np.eye(n)
        
        # Evaluate initial function and gradient
        f = self._evaluate_function(objective, x)
        
        if gradient is not None:
            g = self._evaluate_gradient(gradient, x)
        else:
            g = self._numerical_gradient(objective, x)
        
        # Check initial gradient
        if np.linalg.norm(g.to_numpy()) < self._gradient_tolerance:
            return OptimizationResult(
                x=x, fun=f, success=True,
                message="Initial point satisfies gradient tolerance",
                nit=0, nfev=self._function_evaluations,
                gradient=g
            )
        
        # BFGS iterations
        for iteration in range(self._max_iterations):
            self._iteration_count = iteration + 1
            
            # Compute search direction: p = -B_inv * g
            p = Vector(-B_inv @ g.to_numpy())
            
            # Line search
            alpha = self._armijo_line_search(objective, x, p, g)
            
            # Update parameters
            x_new = Vector(x.to_numpy() + alpha * p.to_numpy())
            
            # Apply bounds if specified
            if bounds is not None:
                from .base import ConstraintHandler
                x_new = ConstraintHandler.project_bounds(x_new, bounds)
            
            # Evaluate new function and gradient
            f_new = self._evaluate_function(objective, x_new)
            
            if gradient is not None:
                g_new = self._evaluate_gradient(gradient, x_new)
            else:
                g_new = self._numerical_gradient(objective, x_new)
            
            # Check convergence
            converged, message = self._check_convergence(x_new, x, f_new, f, g_new)
            
            if converged:
                return OptimizationResult(
                    x=x_new, fun=f_new, success=True,
                    message=message, nit=self._iteration_count,
                    nfev=self._function_evaluations, gradient=g_new
                )
            
            # BFGS update of inverse Hessian approximation
            s = x_new.to_numpy() - x.to_numpy()  # step
            y = g_new.to_numpy() - g.to_numpy()  # gradient change
            
            # Check curvature condition
            sy = np.dot(s, y)
            if sy > 1e-10:  # Positive curvature
                # Sherman-Morrison-Woodbury formula for BFGS update
                rho = 1.0 / sy
                
                # B_inv_{k+1} = (I - rho * s * y^T) * B_inv_k * (I - rho * y * s^T) + rho * s * s^T
                I = np.eye(n)
                A = I - rho * np.outer(s, y)
                B_inv = A @ B_inv @ A.T + rho * np.outer(s, s)
            else:
                # Skip update if curvature condition not satisfied
                # This can happen near ill-conditioned regions
                pass
            
            # Update for next iteration
            x = x_new
            f = f_new
            g = g_new
        
        # Maximum iterations reached
        return OptimizationResult(
            x=x, fun=f, success=False,
            message="Maximum iterations reached",
            nit=self._iteration_count, nfev=self._function_evaluations,
            gradient=g
        )
    
    def minimize_with_restarts(self, objective: Callable[[Vector], float],
                              x0: Vector,
                              gradient: Optional[Callable[[Vector], Vector]] = None,
                              n_restarts: int = 3,
                              restart_scale: float = 0.1,
                              bounds: Optional[list] = None) -> OptimizationResult:
        """
        Minimize with multiple random restarts.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            gradient: Gradient function
            n_restarts: Number of random restarts
            restart_scale: Scale for random perturbations
            bounds: Parameter bounds
            
        Returns:
            Best optimization result
        """
        best_result = None
        best_fun = np.inf
        
        # Try initial point
        result = self.minimize(objective, x0, gradient, bounds=bounds)
        if result.success and result.fun < best_fun:
            best_result = result
            best_fun = result.fun
        
        # Try random restarts
        for i in range(n_restarts):
            # Random perturbation
            perturbation = np.random.normal(0, restart_scale, len(x0.to_numpy()))
            x_start = Vector(x0.to_numpy() + perturbation)
            
            # Apply bounds to starting point
            if bounds is not None:
                from .base import ConstraintHandler
                x_start = ConstraintHandler.project_bounds(x_start, bounds)
            
            # Optimize from perturbed starting point
            result = self.minimize(objective, x_start, gradient, bounds=bounds)
            
            if result.success and result.fun < best_fun:
                best_result = result
                best_fun = result.fun
        
        return best_result if best_result is not None else result


class LbfgsOptimizer(Optimizer, LineSearchMixin):
    """
    Limited-memory BFGS (L-BFGS) optimizer.
    
    More memory-efficient version of BFGS that stores only
    a limited number of recent vector pairs.
    """
    
    def __init__(self, max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6,
                 memory_size: int = 10):
        """
        Initialize L-BFGS optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Function value convergence tolerance
            gradient_tolerance: Gradient norm convergence tolerance
            memory_size: Number of vector pairs to store
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self._memory_size = memory_size
    
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """
        Minimize objective function using L-BFGS algorithm.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            gradient: Gradient function (if None, numerical gradient used)
            hessian: Hessian function (ignored)
            bounds: Parameter bounds (optional)
            constraints: Optimization constraints (not supported)
            
        Returns:
            OptimizationResult containing optimization results
        """
        if constraints is not None:
            raise NotImplementedError("Constraints not supported in L-BFGS optimizer")
        
        self._reset_tracking()
        
        # Initialize
        x = x0.copy()
        n = len(x.to_numpy())
        
        # Storage for L-BFGS vectors
        s_vectors = []  # parameter differences
        y_vectors = []  # gradient differences
        rho_values = []  # 1 / (s^T y)
        
        # Evaluate initial function and gradient
        f = self._evaluate_function(objective, x)
        
        if gradient is not None:
            g = self._evaluate_gradient(gradient, x)
        else:
            g = self._numerical_gradient(objective, x)
        
        # Check initial gradient
        if np.linalg.norm(g.to_numpy()) < self._gradient_tolerance:
            return OptimizationResult(
                x=x, fun=f, success=True,
                message="Initial point satisfies gradient tolerance",
                nit=0, nfev=self._function_evaluations,
                gradient=g
            )
        
        # L-BFGS iterations
        for iteration in range(self._max_iterations):
            self._iteration_count = iteration + 1
            
            # Compute search direction using two-loop recursion
            q = g.to_numpy().copy()
            
            # First loop (backward)
            alphas = []
            for i in range(len(s_vectors) - 1, -1, -1):
                alpha = rho_values[i] * np.dot(s_vectors[i], q)
                q -= alpha * y_vectors[i]
                alphas.append(alpha)
            
            alphas.reverse()  # Match order with s_vectors
            
            # Initial Hessian approximation (scaled identity)
            if len(y_vectors) > 0:
                # Use scaling based on most recent vectors
                gamma = np.dot(s_vectors[-1], y_vectors[-1]) / np.dot(y_vectors[-1], y_vectors[-1])
                r = gamma * q
            else:
                r = q
            
            # Second loop (forward)
            for i in range(len(s_vectors)):
                beta = rho_values[i] * np.dot(y_vectors[i], r)
                r += s_vectors[i] * (alphas[i] - beta)
            
            # Search direction
            p = Vector(-r)
            
            # Line search
            alpha = self._armijo_line_search(objective, x, p, g)
            
            # Update parameters
            x_new = Vector(x.to_numpy() + alpha * p.to_numpy())
            
            # Apply bounds if specified
            if bounds is not None:
                from .base import ConstraintHandler
                x_new = ConstraintHandler.project_bounds(x_new, bounds)
            
            # Evaluate new function and gradient
            f_new = self._evaluate_function(objective, x_new)
            
            if gradient is not None:
                g_new = self._evaluate_gradient(gradient, x_new)
            else:
                g_new = self._numerical_gradient(objective, x_new)
            
            # Check convergence
            converged, message = self._check_convergence(x_new, x, f_new, f, g_new)
            
            if converged:
                return OptimizationResult(
                    x=x_new, fun=f_new, success=True,
                    message=message, nit=self._iteration_count,
                    nfev=self._function_evaluations, gradient=g_new
                )
            
            # Update L-BFGS storage
            s = x_new.to_numpy() - x.to_numpy()
            y = g_new.to_numpy() - g.to_numpy()
            
            # Check curvature condition
            sy = np.dot(s, y)
            if sy > 1e-10:
                # Add new vector pair
                s_vectors.append(s)
                y_vectors.append(y)
                rho_values.append(1.0 / sy)
                
                # Remove oldest if memory exceeded
                if len(s_vectors) > self._memory_size:
                    s_vectors.pop(0)
                    y_vectors.pop(0)
                    rho_values.pop(0)
            
            # Update for next iteration
            x = x_new
            f = f_new
            g = g_new
        
        # Maximum iterations reached
        return OptimizationResult(
            x=x, fun=f, success=False,
            message="Maximum iterations reached",
            nit=self._iteration_count, nfev=self._function_evaluations,
            gradient=g
        )