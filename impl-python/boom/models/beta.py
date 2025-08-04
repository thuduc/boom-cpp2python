"""Beta model implementation."""

from typing import List, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import RNG
from ..distributions.rmath import dbeta, rbeta, lgamma_func
from .base import Model, Data, LoglikeModel
from .params import UnivParams
from .data import DoubleData


class BetaSuf:
    """Sufficient statistics for Beta distribution.
    
    Sufficient statistics are:
    - n: number of observations
    - sum_log_x: sum of log observations
    - sum_log_1_minus_x: sum of log(1-x) observations
    """
    
    def __init__(self):
        """Initialize Beta sufficient statistics."""
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        self._sum_log_x = 0.0
        self._sum_log_1_minus_x = 0.0
    
    def update(self, data: Union[Data, float, List[float]]):
        """Update with new data.
        
        Args:
            data: Data to incorporate. Can be:
                - Single float value in (0, 1)
                - List of float values in (0, 1)
                - Data object with value() method
                - List of Data objects
        """
        if isinstance(data, list):
            for item in data:
                self.update(item)
        elif isinstance(data, (int, float)):
            x = float(data)
            if x <= 0 or x >= 1:
                raise ValueError("Beta observations must be in (0, 1)")
            self._n += 1
            self._sum_log_x += np.log(x)
            self._sum_log_1_minus_x += np.log(1 - x)
        elif hasattr(data, 'value'):
            # Assume it's a Data object with value() method
            self.update(data.value())
        else:
            raise TypeError(f"Cannot update BetaSuf with data of type {type(data)}")
    
    def combine(self, other: 'BetaSuf') -> 'BetaSuf':
        """Combine with another BetaSuf."""
        if not isinstance(other, BetaSuf):
            raise TypeError("Can only combine with another BetaSuf")
        
        result = BetaSuf()
        result._n = self._n + other._n
        result._sum_log_x = self._sum_log_x + other._sum_log_x
        result._sum_log_1_minus_x = self._sum_log_1_minus_x + other._sum_log_1_minus_x
        return result
    
    def n(self) -> int:
        """Get sample size."""
        return self._n
    
    def sum_log_x(self) -> float:
        """Get sum of log observations."""
        return self._sum_log_x
    
    def sum_log_1_minus_x(self) -> float:
        """Get sum of log(1-x) observations."""
        return self._sum_log_1_minus_x
    
    def mean_log_x(self) -> float:
        """Get mean of log observations."""
        if self._n == 0:
            return 0.0
        return self._sum_log_x / self._n
    
    def mean_log_1_minus_x(self) -> float:
        """Get mean of log(1-x) observations."""
        if self._n == 0:
            return 0.0
        return self._sum_log_1_minus_x / self._n
    
    def clone(self) -> 'BetaSuf':
        """Create copy."""
        result = BetaSuf()
        result._n = self._n
        result._sum_log_x = self._sum_log_x
        result._sum_log_1_minus_x = self._sum_log_1_minus_x
        return result
    
    def __str__(self) -> str:
        """String representation."""
        return f"BetaSuf(n={self._n}, mean_log_x={self.mean_log_x():.3f}, mean_log_1_minus_x={self.mean_log_1_minus_x():.3f})"


class BetaModel(LoglikeModel):
    """Beta model with unknown shape parameters.
    
    This model assumes observations are Beta random variables:
    X ~ Beta(alpha, beta) with both parameters unknown.
    
    The Beta distribution has support on (0, 1) with:
    - mean = alpha / (alpha + beta)
    - variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Beta model.
        
        Args:
            alpha: Initial alpha parameter (first shape parameter).
            beta: Initial beta parameter (second shape parameter).
        """
        super().__init__()
        
        # Validate parameters
        if alpha <= 0:
            raise ValueError("Alpha parameter must be positive")
        if beta <= 0:
            raise ValueError("Beta parameter must be positive")
        
        # Create parameter objects
        self._alpha_param = UnivParams(alpha)
        self._beta_param = UnivParams(beta)
        
        # Set up parameter management
        self.set_parameter('alpha', self._alpha_param)
        self.set_parameter('beta', self._beta_param)
        
        # Initialize sufficient statistics
        self._suf = BetaSuf()
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def alpha(self) -> float:
        """Get alpha parameter (first shape parameter)."""
        return self._alpha_param.value()
    
    def set_alpha(self, alpha: float):
        """Set alpha parameter."""
        if alpha <= 0:
            raise ValueError("Alpha parameter must be positive")
        self._alpha_param.set_value(alpha)
        self._notify_observers()
    
    def beta_param(self) -> float:
        """Get beta parameter (second shape parameter)."""
        return self._beta_param.value()
    
    def set_beta_param(self, beta: float):
        """Set beta parameter."""
        if beta <= 0:
            raise ValueError("Beta parameter must be positive")
        self._beta_param.set_value(beta)
        self._notify_observers()
    
    def mean(self) -> float:
        """Get distribution mean."""
        alpha_val = self.alpha()
        beta_val = self.beta_param()
        return alpha_val / (alpha_val + beta_val)
    
    def variance(self) -> float:
        """Get distribution variance."""
        alpha_val = self.alpha()
        beta_val = self.beta_param()
        sum_params = alpha_val + beta_val
        return (alpha_val * beta_val) / (sum_params * sum_params * (sum_params + 1))
    
    def set_params(self, alpha: float, beta: float):
        """Set both parameters simultaneously."""
        self.set_alpha(alpha)
        self.set_beta_param(beta)
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, float, List[float]]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be:
                - DoubleData object
                - float value in (0, 1)
                - list of float values in (0, 1)
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, (int, float)):
            x = float(data)
            if x <= 0 or x >= 1:
                raise ValueError("Beta observations must be in (0, 1)")
            data_obj = DoubleData(x)
        elif isinstance(data, DoubleData):
            x = data.value()
            if x <= 0 or x >= 1:
                raise ValueError("Beta observations must be in (0, 1)")
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update(data_obj.value())
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> BetaSuf:
        """Get sufficient statistics."""
        return self._suf
    
    # ============================================================================
    # Model Interface Implementation
    # ============================================================================
    
    def log_likelihood(self, data: Optional[List[Data]] = None) -> float:
        """Compute log likelihood.
        
        Args:
            data: Data to compute likelihood for. If None, uses model's data.
            
        Returns:
            Log likelihood value.
        """
        if data is not None:
            # Compute likelihood for provided data
            log_lik = 0.0
            alpha_val = self.alpha()
            beta_val = self.beta_param()
            
            for data_point in data:
                if isinstance(data_point, DoubleData):
                    x = data_point.value()
                    log_lik += dbeta(x, alpha_val, beta_val, log=True)
                else:
                    raise TypeError(f"Expected DoubleData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            n = self._suf.n()
            sum_log_x = self._suf.sum_log_x()
            sum_log_1_minus_x = self._suf.sum_log_1_minus_x()
            alpha_val = self.alpha()
            beta_val = self.beta_param()
            
            # Beta log likelihood:
            # n * (log(Gamma(alpha + beta)) - log(Gamma(alpha)) - log(Gamma(beta))) +
            # (alpha - 1) * sum_log_x + (beta - 1) * sum_log_1_minus_x
            log_beta_function = lgamma_func(alpha_val) + lgamma_func(beta_val) - lgamma_func(alpha_val + beta_val)
            
            return (-n * log_beta_function +
                   (alpha_val - 1) * sum_log_x +
                   (beta_val - 1) * sum_log_1_minus_x)
    
    def simulate_data(self, n: Optional[int] = None) -> List[DoubleData]:
        """Simulate data from the model.
        
        Args:
            n: Number of data points to simulate. If None, uses current sample size.
            
        Returns:
            List of simulated DoubleData objects.
        """
        if n is None:
            n = len(self._data) if self._data else 1
        
        if n <= 0:
            return []
        
        alpha_val = self.alpha()
        beta_val = self.beta_param()
        rng = RNG()
        
        simulated = []
        for _ in range(n):
            x = rbeta(alpha_val, beta_val, rng)
            simulated.append(DoubleData(x))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form: [log(alpha), log(beta)]."""
        # Use log transformation for unconstrained optimization
        return Vector([np.log(self.alpha()), np.log(self.beta_param())])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form: [log(alpha), log(beta)]."""
        if len(theta) != 2:
            raise ValueError("Parameter vector must have exactly 2 elements")
        
        log_alpha = theta[0]
        log_beta = theta[1]
        
        alpha_val = np.exp(log_alpha)
        beta_val = np.exp(log_beta)
        
        self.set_params(alpha_val, beta_val)
    
    # ============================================================================
    # Maximum Likelihood Estimation
    # ============================================================================
    
    def mle(self, max_iterations: int = 100, tolerance: float = 1e-8):
        """Compute maximum likelihood estimates using Newton-Raphson.
        
        Args:
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
        """
        if self._suf.n() == 0:
            return  # No data to estimate from
        
        n = self._suf.n()
        mean_log_x = self._suf.mean_log_x()
        mean_log_1_minus_x = self._suf.mean_log_1_minus_x()
        
        # Method of moments initial estimates
        # Compute sample mean and variance from log statistics
        # This is an approximation - ideally we'd compute from raw data
        alpha_est = self.alpha()
        beta_est = self.beta_param()
        
        # Newton-Raphson iteration
        from scipy.special import digamma, polygamma
        
        for iteration in range(max_iterations):
            # Current parameter values
            psi_alpha = digamma(alpha_est)
            psi_beta = digamma(beta_est)
            psi_sum = digamma(alpha_est + beta_est)
            
            tri_alpha = polygamma(1, alpha_est)
            tri_beta = polygamma(1, beta_est)
            tri_sum = polygamma(1, alpha_est + beta_est)
            
            # Score functions (gradients)
            score_alpha = n * (psi_sum - psi_alpha) + n * mean_log_x
            score_beta = n * (psi_sum - psi_beta) + n * mean_log_1_minus_x
            
            # Hessian matrix elements
            h_alpha_alpha = n * (tri_sum - tri_alpha)
            h_beta_beta = n * (tri_sum - tri_beta)
            h_alpha_beta = n * tri_sum
            
            # Determinant of Hessian
            det_h = h_alpha_alpha * h_beta_beta - h_alpha_beta * h_alpha_beta
            
            if abs(det_h) < tolerance:
                break
            
            # Newton-Raphson updates
            delta_alpha = (h_beta_beta * score_alpha - h_alpha_beta * score_beta) / det_h
            delta_beta = (h_alpha_alpha * score_beta - h_alpha_beta * score_alpha) / det_h
            
            alpha_new = alpha_est + delta_alpha
            beta_new = beta_est + delta_beta
            
            # Ensure parameters remain positive
            if alpha_new <= 0:
                alpha_new = alpha_est / 2
            if beta_new <= 0:
                beta_new = beta_est / 2
            
            # Check convergence
            if abs(delta_alpha) < tolerance and abs(delta_beta) < tolerance:
                alpha_est = alpha_new
                beta_est = beta_new
                break
            
            alpha_est = alpha_new
            beta_est = beta_new
        
        self.set_params(alpha_est, beta_est)
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'BetaModel':
        """Create a deep copy of the model."""
        cloned = BetaModel(self.alpha(), self.beta_param())
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        return (f"BetaModel(alpha={self.alpha():.3f}, beta={self.beta_param():.3f}, "
                f"mean={self.mean():.3f}, data_points={len(self._data)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)