"""Optimization algorithms for BOOM."""
import numpy as np
from typing import Optional, Tuple, Callable, NamedTuple
from abc import ABC, abstractmethod
from ..linalg import Vector, Matrix, SpdMatrix
from .target_functions import TargetFunction


class OptimizationResult(NamedTuple):
    """Result from optimization."""
    x: Vector                    # Optimal point
    f: float                    # Function value at optimum
    grad: Vector                # Gradient at optimum
    n_iterations: int          # Number of iterations
    n_function_evaluations: int # Number of function evaluations
    success: bool              # Whether optimization succeeded
    message: str               # Status message


class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6):
        """Initialize optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for function value
            gradient_tolerance: Convergence tolerance for gradient norm
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.gradient_tolerance = gradient_tolerance
        
        # Tracking
        self.iteration_history = []
        self.function_history = []
        self.gradient_history = []
    
    @abstractmethod
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize the target function."""
        pass
    
    def _check_convergence(self, f_current: float, f_previous: float,
                          grad: Vector, iteration: int) -> Tuple[bool, str]:
        """Check convergence criteria."""
        # Function value convergence
        if iteration > 0:
            f_change = abs(f_current - f_previous)
            if f_change < self.tolerance:
                return True, f"Function converged (change: {f_change:.2e})"
        
        # Gradient norm convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < self.gradient_tolerance:
            return True, f"Gradient converged (norm: {grad_norm:.2e})"
        
        # Max iterations
        if iteration >= self.max_iterations:
            return True, f"Maximum iterations reached ({self.max_iterations})"
        
        return False, ""
    
    def reset_history(self):
        """Reset optimization history."""
        self.iteration_history = []
        self.function_history = []
        self.gradient_history = []


class NewtonRaphson(Optimizer):
    """Newton-Raphson optimization algorithm."""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-8,
                 gradient_tolerance: float = 1e-8, regularization: float = 1e-8):
        """Initialize Newton-Raphson optimizer.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Function tolerance
            gradient_tolerance: Gradient tolerance
            regularization: Hessian regularization parameter
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self.regularization = regularization
    
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize using Newton-Raphson method."""
        self.reset_history()
        target_function.reset_counters()
        
        x = Vector(initial_point)
        f_previous = float('inf')
        
        for iteration in range(self.max_iterations):
            # Evaluate function, gradient, and Hessian
            f, grad, hessian = target_function.evaluate_with_hessian(x)
            
            # Store history
            self.iteration_history.append(iteration)
            self.function_history.append(f)
            self.gradient_history.append(np.linalg.norm(grad))
            
            # Check convergence
            converged, message = self._check_convergence(f, f_previous, grad, iteration)
            if converged:
                return OptimizationResult(
                    x=x, f=f, grad=grad, n_iterations=iteration,
                    n_function_evaluations=target_function.n_evaluations,
                    success=True, message=message
                )
            
            # Regularize Hessian if needed
            try:
                # Add regularization to ensure positive definiteness
                H_reg = Matrix(hessian + self.regularization * np.eye(len(x)))
                H_inv = SpdMatrix(H_reg).inv()
            except np.linalg.LinAlgError:
                return OptimizationResult(
                    x=x, f=f, grad=grad, n_iterations=iteration,
                    n_function_evaluations=target_function.n_evaluations,
                    success=False, message="Singular Hessian matrix"
                )
            
            # Newton step
            step = H_inv @ grad
            x = x - step
            
            f_previous = f
        
        return OptimizationResult(
            x=x, f=f, grad=grad, n_iterations=self.max_iterations,
            n_function_evaluations=target_function.n_evaluations,
            success=False, message="Maximum iterations reached"
        )


class BFGS(Optimizer):
    """BFGS quasi-Newton optimization algorithm."""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6, line_search_max_iter: int = 20):
        """Initialize BFGS optimizer.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Function tolerance
            gradient_tolerance: Gradient tolerance
            line_search_max_iter: Maximum line search iterations
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self.line_search_max_iter = line_search_max_iter
    
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize using BFGS method."""
        self.reset_history()
        target_function.reset_counters()
        
        x = Vector(initial_point)
        n = len(x)
        
        # Initialize inverse Hessian approximation
        H_inv = Matrix(np.eye(n))
        
        f, grad = target_function.evaluate_with_gradient(x)
        f_previous = float('inf')
        
        for iteration in range(self.max_iterations):
            # Store history
            self.iteration_history.append(iteration)
            self.function_history.append(f)
            self.gradient_history.append(np.linalg.norm(grad))
            
            # Check convergence
            converged, message = self._check_convergence(f, f_previous, grad, iteration)
            if converged:
                return OptimizationResult(
                    x=x, f=f, grad=grad, n_iterations=iteration,
                    n_function_evaluations=target_function.n_evaluations,
                    success=True, message=message
                )
            
            # Compute search direction
            p = -(H_inv @ grad)
            
            # Line search
            alpha = self._line_search(target_function, x, p, f, grad)
            
            # Update
            x_new = x + alpha * p
            f_new, grad_new = target_function.evaluate_with_gradient(x_new)
            
            # BFGS update
            s = x_new - x
            y = grad_new - grad
            
            if s.dot(y) > 1e-10:  # Curvature condition
                rho = 1.0 / s.dot(y)
                I = Matrix(np.eye(n))
                
                # Sherman-Morrison formula
                V = I - rho * np.outer(s, y)
                H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)
                H_inv = Matrix(H_inv)
            
            x = x_new
            f = f_new
            grad = grad_new
            f_previous = f
        
        return OptimizationResult(
            x=x, f=f, grad=grad, n_iterations=self.max_iterations,
            n_function_evaluations=target_function.n_evaluations,
            success=False, message="Maximum iterations reached"
        )
    
    def _line_search(self, target_function: TargetFunction, x: Vector, 
                     p: Vector, f: float, grad: Vector) -> float:
        """Simple backtracking line search."""
        alpha = 1.0
        c1 = 1e-4  # Armijo parameter
        
        for _ in range(self.line_search_max_iter):
            x_new = x + alpha * p
            f_new = target_function.evaluate(x_new)
            
            # Armijo condition
            if f_new <= f + c1 * alpha * grad.dot(p):
                return alpha
            
            alpha *= 0.5
        
        return alpha


class ConjugateGradient(Optimizer):
    """Conjugate Gradient optimization algorithm."""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6, restart_frequency: int = 100):
        """Initialize Conjugate Gradient optimizer.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Function tolerance
            gradient_tolerance: Gradient tolerance
            restart_frequency: Frequency to restart CG
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self.restart_frequency = restart_frequency
    
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize using Conjugate Gradient method."""
        self.reset_history()
        target_function.reset_counters()
        
        x = Vector(initial_point)
        f, grad = target_function.evaluate_with_gradient(x)
        
        # Initialize search direction
        p = -grad
        f_previous = float('inf')
        
        for iteration in range(self.max_iterations):
            # Store history
            self.iteration_history.append(iteration)
            self.function_history.append(f)
            self.gradient_history.append(np.linalg.norm(grad))
            
            # Check convergence
            converged, message = self._check_convergence(f, f_previous, grad, iteration)
            if converged:
                return OptimizationResult(
                    x=x, f=f, grad=grad, n_iterations=iteration,
                    n_function_evaluations=target_function.n_evaluations,
                    success=True, message=message
                )
            
            # Line search
            alpha = self._line_search(target_function, x, p, f, grad)
            
            # Update position
            x_new = x + alpha * p
            f_new, grad_new = target_function.evaluate_with_gradient(x_new)
            
            # Compute beta (Polak-Ribière formula)
            if iteration % self.restart_frequency == 0:
                # Restart: steepest descent direction
                beta = 0.0
            else:
                numerator = grad_new.dot(grad_new - grad)
                denominator = grad.dot(grad)
                beta = max(0.0, numerator / denominator) if denominator > 1e-10 else 0.0
            
            # Update search direction
            p = -grad_new + beta * p
            
            x = x_new
            f = f_new
            grad = grad_new
            f_previous = f
        
        return OptimizationResult(
            x=x, f=f, grad=grad, n_iterations=self.max_iterations,
            n_function_evaluations=target_function.n_evaluations,
            success=False, message="Maximum iterations reached"
        )
    
    def _line_search(self, target_function: TargetFunction, x: Vector, 
                     p: Vector, f: float, grad: Vector) -> float:
        """Simple backtracking line search."""
        alpha = 1.0
        c1 = 1e-4
        
        for _ in range(20):
            x_new = x + alpha * p
            f_new = target_function.evaluate(x_new)
            
            if f_new <= f + c1 * alpha * grad.dot(p):
                return alpha
            
            alpha *= 0.5
        
        return alpha


class LevenbergMarquardt(Optimizer):
    """Levenberg-Marquardt algorithm for nonlinear least squares."""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-8,
                 gradient_tolerance: float = 1e-8, lambda_init: float = 1e-3):
        """Initialize Levenberg-Marquardt optimizer.
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Function tolerance
            gradient_tolerance: Gradient tolerance
            lambda_init: Initial damping parameter
        """
        super().__init__(max_iterations, tolerance, gradient_tolerance)
        self.lambda_init = lambda_init
    
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize using Levenberg-Marquardt method."""
        self.reset_history()
        target_function.reset_counters()
        
        x = Vector(initial_point)
        lambda_param = self.lambda_init
        
        f, grad, hessian = target_function.evaluate_with_hessian(x)
        f_previous = float('inf')
        
        for iteration in range(self.max_iterations):
            # Store history
            self.iteration_history.append(iteration)
            self.function_history.append(f)
            self.gradient_history.append(np.linalg.norm(grad))
            
            # Check convergence
            converged, message = self._check_convergence(f, f_previous, grad, iteration)
            if converged:
                return OptimizationResult(
                    x=x, f=f, grad=grad, n_iterations=iteration,
                    n_function_evaluations=target_function.n_evaluations,
                    success=True, message=message
                )
            
            # Levenberg-Marquardt step
            try:
                # Damped Hessian
                H_damped = Matrix(hessian + lambda_param * np.eye(len(x)))
                H_inv = SpdMatrix(H_damped).inv()
                step = H_inv @ grad
                
                x_trial = x - step
                f_trial = target_function.evaluate(x_trial)
                
                if f_trial < f:
                    # Accept step, decrease damping
                    x = x_trial
                    f, grad, hessian = target_function.evaluate_with_hessian(x)
                    lambda_param = max(lambda_param / 10, 1e-10)
                else:
                    # Reject step, increase damping
                    lambda_param = min(lambda_param * 10, 1e10)
                
            except np.linalg.LinAlgError:
                lambda_param = min(lambda_param * 10, 1e10)
            
            f_previous = f
        
        return OptimizationResult(
            x=x, f=f, grad=grad, n_iterations=self.max_iterations,
            n_function_evaluations=target_function.n_evaluations,
            success=False, message="Maximum iterations reached"
        )


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing global optimization algorithm."""
    
    def __init__(self, max_iterations: int = 10000, initial_temperature: float = 1.0,
                 cooling_rate: float = 0.99, min_temperature: float = 1e-8,
                 step_size: float = 1.0):
        """Initialize Simulated Annealing optimizer.
        
        Args:
            max_iterations: Maximum iterations
            initial_temperature: Initial temperature
            cooling_rate: Temperature cooling rate (< 1)
            min_temperature: Minimum temperature
            step_size: Step size for proposals
        """
        super().__init__(max_iterations, tolerance=1e-12, gradient_tolerance=float('inf'))
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.step_size = step_size
    
    def optimize(self, target_function: TargetFunction, 
                 initial_point: Vector) -> OptimizationResult:
        """Optimize using Simulated Annealing."""
        self.reset_history()
        target_function.reset_counters()
        
        from ...distributions import rng
        
        x = Vector(initial_point)
        f = target_function.evaluate(x)
        
        best_x = x.copy()
        best_f = f
        
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Store history
            self.iteration_history.append(iteration)
            self.function_history.append(f)
            self.gradient_history.append(0.0)  # SA doesn't use gradients
            
            # Generate proposal
            proposal = x + self.step_size * rng.rnorm_vec(len(x))
            f_proposal = target_function.evaluate(proposal)
            
            # Acceptance criterion
            if f_proposal < f or rng.runif() < np.exp(-(f_proposal - f) / temperature):
                x = proposal
                f = f_proposal
                
                # Update best
                if f < best_f:
                    best_x = x.copy()
                    best_f = f
            
            # Cool down
            temperature = max(temperature * self.cooling_rate, self.min_temperature)
            
            # Early stopping if temperature is too low
            if temperature <= self.min_temperature:
                break
        
        # Use numerical gradient for final result
        grad = target_function.gradient(best_x)
        
        return OptimizationResult(
            x=best_x, f=best_f, grad=grad, n_iterations=iteration,
            n_function_evaluations=target_function.n_evaluations,
            success=True, message=f"Completed with final temperature {temperature:.2e}"
        )