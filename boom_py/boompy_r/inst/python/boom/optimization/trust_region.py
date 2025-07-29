"""Trust region algorithms for optimization."""
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from ..linalg import Vector, Matrix, SpdMatrix
from .target_functions import TargetFunction


class TrustRegionResult:
    """Result from trust region step."""
    
    def __init__(self, step: Vector, predicted_reduction: float, 
                 actual_reduction: float, rho: float, accepted: bool):
        """Initialize trust region result.
        
        Args:
            step: Computed step
            predicted_reduction: Predicted function reduction
            actual_reduction: Actual function reduction
            rho: Ratio of actual to predicted reduction
            accepted: Whether step was accepted
        """
        self.step = step
        self.predicted_reduction = predicted_reduction
        self.actual_reduction = actual_reduction
        self.rho = rho
        self.accepted = accepted


class TrustRegion(ABC):
    """Base class for trust region algorithms."""
    
    def __init__(self, initial_radius: float = 1.0, max_radius: float = 100.0,
                 eta1: float = 0.25, eta2: float = 0.75, 
                 gamma1: float = 0.25, gamma2: float = 2.0):
        """Initialize trust region method.
        
        Args:
            initial_radius: Initial trust region radius
            max_radius: Maximum trust region radius
            eta1: Threshold for shrinking radius
            eta2: Threshold for expanding radius
            gamma1: Factor for shrinking radius
            gamma2: Factor for expanding radius
        """
        self.radius = initial_radius
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        # History
        self.radius_history = [initial_radius]
        self.step_history = []
        self.rho_history = []
    
    @abstractmethod
    def solve_subproblem(self, grad: Vector, hessian: Matrix) -> Vector:
        """Solve trust region subproblem.
        
        Minimize: m(p) = f + g^T p + 0.5 p^T H p
        Subject to: ||p|| <= radius
        
        Args:
            grad: Gradient at current point
            hessian: Hessian at current point
            
        Returns:
            Step vector p
        """
        pass
    
    def compute_step(self, target_function: TargetFunction, x: Vector) -> TrustRegionResult:
        """Compute trust region step."""
        # Evaluate function, gradient, and Hessian
        f, grad, hessian = target_function.evaluate_with_hessian(x)
        
        # Solve trust region subproblem
        step = self.solve_subproblem(grad, hessian)
        
        # Compute predicted reduction
        predicted_reduction = -(grad.dot(step) + 0.5 * step.dot(hessian @ step))
        
        # Evaluate at new point
        x_new = x + step
        f_new = target_function.evaluate(x_new)
        
        # Compute actual reduction
        actual_reduction = f - f_new
        
        # Compute ratio
        rho = actual_reduction / predicted_reduction if abs(predicted_reduction) > 1e-15 else 0.0
        
        # Update trust region radius
        if rho < self.eta1:
            # Poor agreement: shrink radius
            self.radius = self.gamma1 * np.linalg.norm(step)
            accepted = False
        else:
            # Good agreement: accept step
            accepted = True
            if rho > self.eta2 and np.linalg.norm(step) > 0.8 * self.radius:
                # Very good agreement and step touches boundary: expand radius
                self.radius = min(self.gamma2 * self.radius, self.max_radius)
        
        # Store history
        self.radius_history.append(self.radius)
        self.step_history.append(step)
        self.rho_history.append(rho)
        
        return TrustRegionResult(step, predicted_reduction, actual_reduction, rho, accepted)
    
    def reset(self):
        """Reset trust region to initial state."""
        self.radius = self.initial_radius
        self.radius_history = [self.initial_radius]
        self.step_history = []
        self.rho_history = []


class CauchyPointTrustRegion(TrustRegion):
    """Trust region using Cauchy point method."""
    
    def solve_subproblem(self, grad: Vector, hessian: Matrix) -> Vector:
        """Solve subproblem using Cauchy point.
        
        The Cauchy point is the minimizer of the quadratic model
        along the steepest descent direction within the trust region.
        """
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            return Vector.zero(len(grad))
        
        # Steepest descent direction
        p_u = -grad / grad_norm
        
        # Compute step length along steepest descent
        p_u_T_H_p_u = p_u.dot(hessian @ p_u)
        
        if p_u_T_H_p_u <= 0:
            # Negative curvature: go to boundary
            tau = self.radius
        else:
            # Positive curvature: minimize quadratic
            tau = min(grad_norm**2 / p_u_T_H_p_u, self.radius)
        
        return tau * p_u


class DoglegTrustRegion(TrustRegion):
    """Dogleg trust region method."""
    
    def solve_subproblem(self, grad: Vector, hessian: Matrix) -> Vector:
        """Solve subproblem using dogleg method.
        
        The dogleg method interpolates between the Cauchy point
        and the Newton step based on the trust region radius.
        """
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            return Vector.zero(len(grad))
        
        # Try Newton step first
        try:
            newton_step = -SpdMatrix(hessian).inv() @ grad
            newton_norm = np.linalg.norm(newton_step)
            
            # If Newton step is within trust region, use it
            if newton_norm <= self.radius:
                return newton_step
        except np.linalg.LinAlgError:
            # Hessian is not positive definite
            newton_step = None
            newton_norm = float('inf')
        
        # Compute Cauchy point
        p_u = -grad / grad_norm
        p_u_T_H_p_u = p_u.dot(hessian @ p_u)
        
        if p_u_T_H_p_u <= 0:
            # Negative curvature: Cauchy point is at boundary
            tau_c = self.radius
        else:
            # Positive curvature
            tau_c = min(grad_norm**2 / p_u_T_H_p_u, self.radius)
        
        cauchy_point = tau_c * p_u
        cauchy_norm = np.linalg.norm(cauchy_point)
        
        # If Cauchy point is at boundary or Newton step failed, use Cauchy point
        if cauchy_norm >= self.radius or newton_step is None:
            return cauchy_point * (self.radius / cauchy_norm)
        
        # Dogleg path: interpolate between Cauchy point and Newton step
        # p(tau) = p_c + tau * (p_n - p_c) for tau in [0, 1]
        
        diff = newton_step - cauchy_point
        a = diff.dot(diff)
        b = 2 * cauchy_point.dot(diff)
        c = cauchy_point.dot(cauchy_point) - self.radius**2
        
        # Solve ||p_c + tau * (p_n - p_c)||^2 = radius^2
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0 or a < 1e-15:
            # Should not happen, but fallback to Cauchy point
            return cauchy_point
        
        tau = (-b + np.sqrt(discriminant)) / (2 * a)
        tau = max(0.0, min(1.0, tau))
        
        return cauchy_point + tau * diff


class SteihaugTrustRegion(TrustRegion):
    """Trust region using Steihaug-Toint conjugate gradient method."""
    
    def __init__(self, max_cg_iter: int = 100, **kwargs):
        """Initialize Steihaug trust region.
        
        Args:
            max_cg_iter: Maximum CG iterations
        """
        super().__init__(**kwargs)
        self.max_cg_iter = max_cg_iter
    
    def solve_subproblem(self, grad: Vector, hessian: Matrix) -> Vector:
        """Solve subproblem using truncated conjugate gradient."""
        n = len(grad)
        p = Vector.zero(n)
        r = grad.copy()  # Residual
        d = -r.copy()    # Search direction
        
        for j in range(min(self.max_cg_iter, n)):
            # Check if we've found the solution
            if np.linalg.norm(r) < 1e-15:
                break
            
            # Compute step length
            Hd = hessian @ d
            d_T_Hd = d.dot(Hd)
            
            # Check for negative curvature
            if d_T_Hd <= 0:
                # Find intersection with trust region boundary
                p_norm_sq = p.dot(p)
                p_T_d = p.dot(d)
                d_norm_sq = d.dot(d)
                
                # Solve ||p + tau * d||^2 = radius^2
                a = d_norm_sq
                b = 2 * p_T_d
                c = p_norm_sq - self.radius**2
                
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    tau = (-b + np.sqrt(discriminant)) / (2 * a)
                    return p + tau * d
                else:
                    return p
            
            # Standard CG step
            alpha = r.dot(r) / d_T_Hd
            p_new = p + alpha * d
            
            # Check trust region constraint
            if np.linalg.norm(p_new) >= self.radius:
                # Find intersection with boundary
                p_norm_sq = p.dot(p)
                p_T_d = p.dot(d)
                d_norm_sq = d.dot(d)
                
                # Solve ||p + tau * d||^2 = radius^2
                a = d_norm_sq
                b = 2 * p_T_d
                c = p_norm_sq - self.radius**2
                
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    tau = (-b + np.sqrt(discriminant)) / (2 * a)
                    return p + tau * d
                else:
                    return p_new
            
            # Update
            r_new = r + alpha * Hd
            beta = r_new.dot(r_new) / r.dot(r)
            d = -r_new + beta * d
            
            p = p_new
            r = r_new
        
        return p


class ExactTrustRegion(TrustRegion):
    """Trust region with exact subproblem solution."""
    
    def solve_subproblem(self, grad: Vector, hessian: Matrix) -> Vector:
        """Solve trust region subproblem exactly using eigendecomposition."""
        try:
            # Eigendecomposition of Hessian
            eigenvals, eigenvecs = np.linalg.eigh(hessian)
            Q = Matrix(eigenvecs)
            Lambda = np.diag(eigenvals)
            
            # Transform gradient
            g_tilde = Q.T @ grad
            
            # Find optimal lambda (Lagrange multiplier)
            lambda_opt = self._find_optimal_lambda(eigenvals, g_tilde)
            
            # Compute step in eigenvector space
            p_tilde = Vector.zero(len(grad))
            for i in range(len(eigenvals)):
                if eigenvals[i] + lambda_opt > 1e-15:
                    p_tilde[i] = -g_tilde[i] / (eigenvals[i] + lambda_opt)
            
            # Transform back to original space
            step = Q @ p_tilde
            
            return step
            
        except np.linalg.LinAlgError:
            # Fallback to dogleg method
            dogleg = DoglegTrustRegion(self.radius, self.max_radius,
                                     self.eta1, self.eta2, self.gamma1, self.gamma2)
            return dogleg.solve_subproblem(grad, hessian)
    
    def _find_optimal_lambda(self, eigenvals: np.ndarray, g_tilde: Vector) -> float:
        """Find optimal Lagrange multiplier using Newton's method."""
        
        def phi(lam):
            """Trust region constraint function."""
            result = 0.0
            for i in range(len(eigenvals)):
                denom = eigenvals[i] + lam
                if denom > 1e-15:
                    result += (g_tilde[i] / denom)**2
            return np.sqrt(result) - self.radius
        
        def phi_prime(lam):
            """Derivative of phi."""
            result = 0.0
            phi_val = 0.0
            for i in range(len(eigenvals)):
                denom = eigenvals[i] + lam
                if denom > 1e-15:
                    term = g_tilde[i] / denom
                    phi_val += term**2
                    result -= 2 * term**2 / denom
            
            if phi_val > 1e-15:
                return result / (2 * np.sqrt(phi_val))
            else:
                return 0.0
        
        # Initial guess
        lambda_min = max(0.0, -min(eigenvals))
        lam = lambda_min + 0.1
        
        # Newton's method
        for _ in range(50):
            phi_val = phi(lam)
            phi_prime_val = phi_prime(lam)
            
            if abs(phi_val) < 1e-12:
                break
            
            if abs(phi_prime_val) < 1e-15:
                break
            
            lam_new = lam - phi_val / phi_prime_val
            lam = max(lambda_min, lam_new)
        
        return lam