"""Line search algorithms for optimization."""
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from ..linalg import Vector
from .target_functions import TargetFunction


class LineSearch(ABC):
    """Base class for line search algorithms."""
    
    def __init__(self, max_iterations: int = 50):
        """Initialize line search.
        
        Args:
            max_iterations: Maximum number of line search iterations
        """
        self.max_iterations = max_iterations
        self.n_evaluations = 0
    
    @abstractmethod
    def search(self, target_function: TargetFunction, x: Vector, p: Vector,
               f: float, grad: Vector) -> float:
        """Perform line search.
        
        Args:
            target_function: Function to minimize
            x: Current point
            p: Search direction
            f: Function value at x
            grad: Gradient at x
            
        Returns:
            Step size alpha
        """
        pass
    
    def reset_counters(self):
        """Reset evaluation counters."""
        self.n_evaluations = 0


class BacktrackingLineSearch(LineSearch):
    """Backtracking line search with Armijo condition."""
    
    def __init__(self, max_iterations: int = 50, c1: float = 1e-4, 
                 rho: float = 0.5, alpha_init: float = 1.0):
        """Initialize backtracking line search.
        
        Args:
            max_iterations: Maximum iterations
            c1: Armijo parameter (sufficient decrease)
            rho: Backtracking factor (0 < rho < 1)
            alpha_init: Initial step size
        """
        super().__init__(max_iterations)
        self.c1 = c1
        self.rho = rho
        self.alpha_init = alpha_init
    
    def search(self, target_function: TargetFunction, x: Vector, p: Vector,
               f: float, grad: Vector) -> float:
        """Perform backtracking line search."""
        self.reset_counters()
        
        alpha = self.alpha_init
        directional_derivative = grad.dot(p)
        
        # If p is not a descent direction, return small step
        if directional_derivative >= 0:
            return 1e-8
        
        for _ in range(self.max_iterations):
            x_new = x + alpha * p
            f_new = target_function.evaluate(x_new)
            self.n_evaluations += 1
            
            # Armijo condition
            if f_new <= f + self.c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= self.rho
        
        return alpha


class WolfeLineSearch(LineSearch):
    """Line search satisfying Wolfe conditions."""
    
    def __init__(self, max_iterations: int = 50, c1: float = 1e-4, 
                 c2: float = 0.9, alpha_max: float = 10.0):
        """Initialize Wolfe line search.
        
        Args:
            max_iterations: Maximum iterations
            c1: Armijo parameter (sufficient decrease)
            c2: Curvature parameter
            alpha_max: Maximum step size
        """
        super().__init__(max_iterations)
        self.c1 = c1
        self.c2 = c2
        self.alpha_max = alpha_max
    
    def search(self, target_function: TargetFunction, x: Vector, p: Vector,
               f: float, grad: Vector) -> float:
        """Perform Wolfe line search using zoom procedure."""
        self.reset_counters()
        
        alpha_0 = 0.0
        alpha_1 = 1.0
        f_0 = f
        grad_0 = grad.dot(p)
        
        # If p is not a descent direction, return small step
        if grad_0 >= 0:
            return 1e-8
        
        for i in range(self.max_iterations):
            x_1 = x + alpha_1 * p
            f_1 = target_function.evaluate(x_1)
            self.n_evaluations += 1
            
            # Check Armijo condition
            if f_1 > f_0 + self.c1 * alpha_1 * grad_0 or (i > 0 and f_1 >= f_prev):
                return self._zoom(target_function, x, p, f, grad, alpha_prev, alpha_1)
            
            # Compute gradient at new point
            grad_1 = target_function.gradient(x_1).dot(p)
            
            # Check curvature condition
            if abs(grad_1) <= -self.c2 * grad_0:
                return alpha_1
            
            # Check if we've gone too far
            if grad_1 >= 0:
                return self._zoom(target_function, x, p, f, grad, alpha_1, alpha_prev)
            
            # Update for next iteration
            alpha_prev = alpha_1
            f_prev = f_1
            alpha_1 = min(2 * alpha_1, self.alpha_max)
        
        return alpha_1
    
    def _zoom(self, target_function: TargetFunction, x: Vector, p: Vector,
              f: float, grad: Vector, alpha_lo: float, alpha_hi: float) -> float:
        """Zoom procedure for Wolfe line search."""
        grad_0 = grad.dot(p)
        
        for _ in range(self.max_iterations // 2):
            # Interpolate
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            
            x_j = x + alpha_j * p
            f_j = target_function.evaluate(x_j)
            self.n_evaluations += 1
            
            # Check Armijo condition
            if f_j > f + self.c1 * alpha_j * grad_0 or f_j >= target_function.evaluate(x + alpha_lo * p):
                alpha_hi = alpha_j
            else:
                grad_j = target_function.gradient(x_j).dot(p)
                
                # Check curvature condition
                if abs(grad_j) <= -self.c2 * grad_0:
                    return alpha_j
                
                if grad_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                
                alpha_lo = alpha_j
        
        return 0.5 * (alpha_lo + alpha_hi)


class StrongWolfeLineSearch(WolfeLineSearch):
    """Line search satisfying strong Wolfe conditions."""
    
    def __init__(self, max_iterations: int = 50, c1: float = 1e-4, 
                 c2: float = 0.1, alpha_max: float = 10.0):
        """Initialize strong Wolfe line search.
        
        Note: Uses smaller c2 for strong Wolfe conditions.
        """
        super().__init__(max_iterations, c1, c2, alpha_max)


class ExactLineSearch(LineSearch):
    """Exact line search for quadratic functions."""
    
    def search(self, target_function: TargetFunction, x: Vector, p: Vector,
               f: float, grad: Vector) -> float:
        """Perform exact line search.
        
        For quadratic functions f(x) = 0.5 x^T A x + b^T x + c,
        the exact step size is: alpha = -g^T p / (p^T A p)
        """
        self.reset_counters()
        
        # Try to get Hessian for exact solution
        try:
            hessian = target_function.hessian(x)
            self.n_evaluations += 1
            
            # Exact step for quadratic: alpha = -g^T p / (p^T H p)
            numerator = -grad.dot(p)
            denominator = p.dot(hessian @ p)
            
            if abs(denominator) > 1e-12:
                alpha = numerator / denominator
                # Ensure positive step in descent direction
                if alpha > 0 and grad.dot(p) < 0:
                    return alpha
                elif alpha < 0 and grad.dot(p) > 0:
                    return -alpha
        
        except:
            pass
        
        # Fallback to backtracking
        backtrack = BacktrackingLineSearch()
        return backtrack.search(target_function, x, p, f, grad)


class AdaptiveLineSearch(LineSearch):
    """Adaptive line search that switches strategies."""
    
    def __init__(self, max_iterations: int = 50):
        """Initialize adaptive line search."""
        super().__init__(max_iterations)
        self.wolfe_search = WolfeLineSearch(max_iterations // 2)
        self.backtrack_search = BacktrackingLineSearch(max_iterations // 2)
        self.use_wolfe = True
    
    def search(self, target_function: TargetFunction, x: Vector, p: Vector,
               f: float, grad: Vector) -> float:
        """Perform adaptive line search."""
        self.reset_counters()
        
        if self.use_wolfe:
            try:
                alpha = self.wolfe_search.search(target_function, x, p, f, grad)
                self.n_evaluations += self.wolfe_search.n_evaluations
                
                # If Wolfe search succeeded, continue using it
                if alpha > 1e-12:
                    return alpha
                else:
                    # Switch to backtracking for future searches
                    self.use_wolfe = False
            except:
                self.use_wolfe = False
        
        # Use backtracking line search
        alpha = self.backtrack_search.search(target_function, x, p, f, grad)
        self.n_evaluations += self.backtrack_search.n_evaluations
        return alpha