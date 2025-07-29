"""Optimization utilities and helper functions."""
import numpy as np
from typing import Callable, Optional, Tuple, List
from ..linalg import Vector, Matrix


def numerical_gradient(f: Callable[[Vector], float], x: Vector, 
                      h: float = 1e-8) -> Vector:
    """Compute numerical gradient using central differences.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute gradient
        h: Step size for finite differences
        
    Returns:
        Gradient vector
    """
    n = len(x)
    grad = Vector.zero(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def numerical_hessian(f: Callable[[Vector], float], x: Vector,
                     h: float = 1e-6) -> Matrix:
    """Compute numerical Hessian using central differences.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute Hessian
        h: Step size for finite differences
        
    Returns:
        Hessian matrix
    """
    n = len(x)
    H = Matrix.zero(n, n)
    
    # Diagonal elements (second derivatives)
    f_center = f(x)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
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
            
            f_pp = f(x_pp)
            f_pm = f(x_pm)
            f_mp = f(x_mp)
            f_mm = f(x_mm)
            
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
            H[j, i] = H[i, j]  # Symmetry
    
    return H


def check_gradient(f: Callable[[Vector], float], 
                  grad_f: Callable[[Vector], Vector],
                  x: Vector, h: float = 1e-8, tol: float = 1e-6) -> bool:
    """Check analytical gradient against numerical gradient.
    
    Args:
        f: Function
        grad_f: Analytical gradient function
        x: Point to check at
        h: Step size for numerical gradient
        tol: Tolerance for comparison
        
    Returns:
        True if gradients match within tolerance
    """
    analytical_grad = grad_f(x)
    numerical_grad = numerical_gradient(f, x, h)
    
    error = np.linalg.norm(analytical_grad - numerical_grad)
    grad_norm = np.linalg.norm(analytical_grad)
    
    if grad_norm > 1e-15:
        relative_error = error / grad_norm
        return relative_error < tol
    else:
        return error < tol


def check_hessian(grad_f: Callable[[Vector], Vector],
                 hess_f: Callable[[Vector], Matrix],
                 x: Vector, h: float = 1e-6, tol: float = 1e-4) -> bool:
    """Check analytical Hessian against numerical Hessian.
    
    Args:
        grad_f: Gradient function
        hess_f: Analytical Hessian function
        x: Point to check at
        h: Step size for numerical Hessian
        tol: Tolerance for comparison
        
    Returns:
        True if Hessians match within tolerance
    """
    def f_wrapper(y):
        return 0.0  # Not used in this context
    
    # Compute numerical Hessian from gradient
    n = len(x)
    numerical_hess = Matrix.zero(n, n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad_plus = grad_f(x_plus)
        grad_minus = grad_f(x_minus)
        
        numerical_hess[:, i] = (grad_plus - grad_minus) / (2 * h)
    
    analytical_hess = hess_f(x)
    
    error = np.linalg.norm(analytical_hess - numerical_hess)
    hess_norm = np.linalg.norm(analytical_hess)
    
    if hess_norm > 1e-15:
        relative_error = error / hess_norm
        return relative_error < tol
    else:
        return error < tol


def armijo_condition(f: Callable[[Vector], float], x: Vector, p: Vector,
                    alpha: float, f_x: float, grad_x: Vector, 
                    c1: float = 1e-4) -> bool:
    """Check Armijo condition for line search.
    
    Args:
        f: Function
        x: Current point
        p: Search direction
        alpha: Step size
        f_x: Function value at x
        grad_x: Gradient at x
        c1: Armijo parameter
        
    Returns:
        True if Armijo condition is satisfied
    """
    f_new = f(x + alpha * p)
    return f_new <= f_x + c1 * alpha * grad_x.dot(p)


def wolfe_conditions(f: Callable[[Vector], float],
                    grad_f: Callable[[Vector], Vector],
                    x: Vector, p: Vector, alpha: float,
                    f_x: float, grad_x: Vector,
                    c1: float = 1e-4, c2: float = 0.9) -> Tuple[bool, bool]:
    """Check Wolfe conditions for line search.
    
    Args:
        f: Function
        grad_f: Gradient function
        x: Current point
        p: Search direction
        alpha: Step size
        f_x: Function value at x
        grad_x: Gradient at x
        c1: Armijo parameter
        c2: Curvature parameter
        
    Returns:
        Tuple of (armijo_satisfied, curvature_satisfied)
    """
    x_new = x + alpha * p
    f_new = f(x_new)
    grad_new = grad_f(x_new)
    
    # Armijo condition
    armijo = f_new <= f_x + c1 * alpha * grad_x.dot(p)
    
    # Curvature condition
    curvature = grad_new.dot(p) >= c2 * grad_x.dot(p)
    
    return armijo, curvature


def strong_wolfe_conditions(f: Callable[[Vector], float],
                           grad_f: Callable[[Vector], Vector],
                           x: Vector, p: Vector, alpha: float,
                           f_x: float, grad_x: Vector,
                           c1: float = 1e-4, c2: float = 0.1) -> Tuple[bool, bool]:
    """Check strong Wolfe conditions for line search.
    
    Args:
        f: Function
        grad_f: Gradient function  
        x: Current point
        p: Search direction
        alpha: Step size
        f_x: Function value at x
        grad_x: Gradient at x
        c1: Armijo parameter (sufficient decrease)
        c2: Curvature parameter (less than weak Wolfe)
        
    Returns:
        Tuple of (armijo_satisfied, strong_curvature_satisfied)
    """
    x_new = x + alpha * p
    f_new = f(x_new)
    grad_new = grad_f(x_new)
    
    # Armijo condition
    armijo = f_new <= f_x + c1 * alpha * grad_x.dot(p)
    
    # Strong curvature condition
    strong_curvature = abs(grad_new.dot(p)) <= c2 * abs(grad_x.dot(p))
    
    return armijo, strong_curvature


def compute_condition_number(A: Matrix) -> float:
    """Compute condition number of matrix.
    
    Args:
        A: Matrix
        
    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    try:
        singular_values = np.linalg.svd(A, compute_uv=False)
        return singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-15 else np.inf
    except np.linalg.LinAlgError:
        return np.inf


def is_positive_definite(A: Matrix, tol: float = 1e-12) -> bool:
    """Check if matrix is positive definite.
    
    Args:
        A: Matrix to check
        tol: Tolerance for eigenvalue positivity
        
    Returns:
        True if matrix is positive definite
    """
    try:
        eigenvals = np.linalg.eigvals(A)
        return np.all(eigenvals > tol)
    except np.linalg.LinAlgError:
        return False


def regularize_hessian(H: Matrix, lambda_reg: float = 1e-8) -> Matrix:
    """Regularize Hessian to ensure positive definiteness.
    
    Args:
        H: Hessian matrix
        lambda_reg: Regularization parameter
        
    Returns:
        Regularized Hessian
    """
    try:
        eigenvals, eigenvecs = np.linalg.eigh(H)
        
        # Ensure all eigenvalues are positive
        eigenvals_reg = np.maximum(eigenvals, lambda_reg)
        
        # Reconstruct matrix
        H_reg = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
        return Matrix(H_reg)
        
    except np.linalg.LinAlgError:
        # Fallback: add regularization to diagonal
        return Matrix(H + lambda_reg * np.eye(H.shape[0]))


def backtrack_line_search(f: Callable[[Vector], float], x: Vector, p: Vector,
                         f_x: float, grad_x: Vector, alpha_init: float = 1.0,
                         c1: float = 1e-4, rho: float = 0.5, 
                         max_iter: int = 50) -> float:
    """Simple backtracking line search.
    
    Args:
        f: Function to minimize
        x: Current point
        p: Search direction
        f_x: Function value at x
        grad_x: Gradient at x
        alpha_init: Initial step size
        c1: Armijo parameter
        rho: Backtracking factor
        max_iter: Maximum iterations
        
    Returns:
        Step size satisfying Armijo condition
    """
    alpha = alpha_init
    directional_derivative = grad_x.dot(p)
    
    # If not a descent direction, return small step
    if directional_derivative >= 0:
        return 1e-8
    
    for _ in range(max_iter):
        if armijo_condition(f, x, p, alpha, f_x, grad_x, c1):
            return alpha
        alpha *= rho
    
    return alpha


def generate_test_problems() -> List[Tuple[str, Callable, Callable, Vector, Vector]]:
    """Generate standard test problems for optimization.
    
    Returns:
        List of (name, function, gradient, initial_point, optimal_point) tuples
    """
    problems = []
    
    # Rosenbrock function
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        grad = Vector.zero(2)
        grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        grad[1] = 200 * (x[1] - x[0]**2)
        return grad
    
    problems.append((
        "Rosenbrock", rosenbrock, rosenbrock_grad,
        Vector([-1.2, 1.0]), Vector([1.0, 1.0])
    ))
    
    # Quadratic function
    def quadratic(x):
        A = Matrix([[2.0, 0.5], [0.5, 1.0]])
        b = Vector([1.0, -1.0])
        return 0.5 * x.dot(A @ x) + b.dot(x)
    
    def quadratic_grad(x):
        A = Matrix([[2.0, 0.5], [0.5, 1.0]])
        b = Vector([1.0, -1.0])
        return A @ x + b
    
    problems.append((
        "Quadratic", quadratic, quadratic_grad,
        Vector([0.0, 0.0]), Vector([-0.571, 0.857])
    ))
    
    # Beale function
    def beale(x):
        term1 = (1.5 - x[0] + x[0] * x[1])**2
        term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
        term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
        return term1 + term2 + term3
    
    def beale_grad(x):
        grad = Vector.zero(2)
        
        term1 = 1.5 - x[0] + x[0] * x[1]
        term2 = 2.25 - x[0] + x[0] * x[1]**2
        term3 = 2.625 - x[0] + x[0] * x[1]**3
        
        grad[0] = 2 * term1 * (-1 + x[1]) + 2 * term2 * (-1 + x[1]**2) + 2 * term3 * (-1 + x[1]**3)
        grad[1] = 2 * term1 * x[0] + 2 * term2 * x[0] * 2 * x[1] + 2 * term3 * x[0] * 3 * x[1]**2
        
        return grad
    
    problems.append((
        "Beale", beale, beale_grad,
        Vector([1.0, 1.0]), Vector([3.0, 0.5])
    ))
    
    return problems