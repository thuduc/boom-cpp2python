"""Gamma model implementation."""

from typing import List, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import RNG
from ..distributions.rmath import dgamma, rgamma, lgamma_func
from .base import Model, Data, LoglikeModel
from .params import UnivParams
from .data import DoubleData


class GammaSuf:
    """Sufficient statistics for Gamma distribution.
    
    Sufficient statistics are:
    - n: number of observations
    - sum_x: sum of observations
    - sum_log_x: sum of log observations
    """
    
    def __init__(self):
        """Initialize Gamma sufficient statistics."""
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        self._sum_x = 0.0
        self._sum_log_x = 0.0
    
    def update(self, data: Union[Data, float, List[float]]):
        """Update with new data.
        
        Args:
            data: Data to incorporate. Can be:
                - Single positive float value
                - List of positive float values
                - Data object with value() method
                - List of Data objects
        """
        if isinstance(data, list):
            for item in data:
                self.update(item)
        elif isinstance(data, (int, float)):
            x = float(data)
            if x <= 0:
                raise ValueError("Gamma observations must be positive")
            self._n += 1
            self._sum_x += x
            self._sum_log_x += np.log(x)
        elif hasattr(data, 'value'):
            # Assume it's a Data object with value() method
            self.update(data.value())
        else:
            raise TypeError(f"Cannot update GammaSuf with data of type {type(data)}")
    
    def combine(self, other: 'GammaSuf') -> 'GammaSuf':
        """Combine with another GammaSuf."""
        if not isinstance(other, GammaSuf):
            raise TypeError("Can only combine with another GammaSuf")
        
        result = GammaSuf()
        result._n = self._n + other._n
        result._sum_x = self._sum_x + other._sum_x
        result._sum_log_x = self._sum_log_x + other._sum_log_x
        return result
    
    def n(self) -> int:
        """Get sample size."""
        return self._n
    
    def sum_x(self) -> float:
        """Get sum of observations."""
        return self._sum_x
    
    def sum_log_x(self) -> float:
        """Get sum of log observations."""
        return self._sum_log_x
    
    def mean(self) -> float:
        """Get sample mean."""
        if self._n == 0:
            return 0.0
        return self._sum_x / self._n
    
    def mean_log(self) -> float:
        """Get mean of log observations."""
        if self._n == 0:
            return 0.0
        return self._sum_log_x / self._n
    
    def clone(self) -> 'GammaSuf':
        """Create copy."""
        result = GammaSuf()
        result._n = self._n
        result._sum_x = self._sum_x
        result._sum_log_x = self._sum_log_x
        return result
    
    def __str__(self) -> str:
        """String representation."""
        return f"GammaSuf(n={self._n}, mean={self.mean():.3f}, mean_log={self.mean_log():.3f})"


class GammaModel(LoglikeModel):
    """Gamma model with unknown shape and rate parameters.
    
    This model assumes observations are Gamma random variables:
    X ~ Gamma(shape, rate) with both parameters unknown.
    
    Note: We use the shape-rate parameterization where:
    - shape = alpha (shape parameter)
    - rate = beta (rate parameter, inverse of scale)
    - mean = shape/rate
    - variance = shape/rate^2
    """
    
    def __init__(self, shape: float = 1.0, rate: float = 1.0):
        """Initialize Gamma model.
        
        Args:
            shape: Initial shape parameter (alpha).
            rate: Initial rate parameter (beta).
        """
        super().__init__()
        
        # Validate parameters
        if shape <= 0:
            raise ValueError("Shape parameter must be positive")
        if rate <= 0:
            raise ValueError("Rate parameter must be positive")
        
        # Create parameter objects
        self._shape_param = UnivParams(shape)
        self._rate_param = UnivParams(rate)
        
        # Set up parameter management
        self.set_parameter('shape', self._shape_param)
        self.set_parameter('rate', self._rate_param)
        
        # Initialize sufficient statistics
        self._suf = GammaSuf()
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def shape(self) -> float:
        """Get shape parameter (alpha)."""
        return self._shape_param.value()
    
    def set_shape(self, shape: float):
        """Set shape parameter (alpha)."""
        if shape <= 0:
            raise ValueError("Shape parameter must be positive")
        self._shape_param.set_value(shape)
        self._notify_observers()
    
    def rate(self) -> float:
        """Get rate parameter (beta)."""
        return self._rate_param.value()
    
    def set_rate(self, rate: float):
        """Set rate parameter (beta)."""
        if rate <= 0:
            raise ValueError("Rate parameter must be positive")
        self._rate_param.set_value(rate)
        self._notify_observers()
    
    def scale(self) -> float:
        """Get scale parameter (1/rate)."""
        return 1.0 / self.rate()
    
    def set_scale(self, scale: float):
        """Set scale parameter (1/rate)."""
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")
        self.set_rate(1.0 / scale)
    
    def mean(self) -> float:
        """Get distribution mean."""
        return self.shape() / self.rate()
    
    def variance(self) -> float:
        """Get distribution variance."""
        rate_val = self.rate()
        return self.shape() / (rate_val * rate_val)
    
    def set_params(self, shape: float, rate: float):
        """Set both parameters simultaneously."""
        self.set_shape(shape)
        self.set_rate(rate)
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, float, List[float]]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be:
                - DoubleData object
                - positive float value
                - list of positive float values
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, (int, float)):
            if data <= 0:
                raise ValueError("Gamma observations must be positive")
            data_obj = DoubleData(float(data))
        elif isinstance(data, DoubleData):
            if data.value() <= 0:
                raise ValueError("Gamma observations must be positive")
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update(data_obj.value())
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> GammaSuf:
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
            shape_val = self.shape()
            rate_val = self.rate()
            
            for data_point in data:
                if isinstance(data_point, DoubleData):
                    x = data_point.value()
                    log_lik += dgamma(x, shape_val, rate_val, log=True)
                else:
                    raise TypeError(f"Expected DoubleData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            n = self._suf.n()
            sum_x = self._suf.sum_x()
            sum_log_x = self._suf.sum_log_x()
            shape_val = self.shape()
            rate_val = self.rate()
            
            # Gamma log likelihood:
            # n * (shape * log(rate) - log(Gamma(shape))) + (shape-1) * sum_log_x - rate * sum_x
            return (n * (shape_val * np.log(rate_val) - lgamma_func(shape_val)) +
                   (shape_val - 1) * sum_log_x - rate_val * sum_x)
    
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
        
        shape_val = self.shape()
        rate_val = self.rate()
        rng = RNG()
        
        simulated = []
        for _ in range(n):
            x = rgamma(shape_val, rate_val, rng)
            simulated.append(DoubleData(x))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form: [log(shape), log(rate)]."""
        # Use log transformation for unconstrained optimization
        return Vector([np.log(self.shape()), np.log(self.rate())])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form: [log(shape), log(rate)]."""
        if len(theta) != 2:
            raise ValueError("Parameter vector must have exactly 2 elements")
        
        log_shape = theta[0]
        log_rate = theta[1]
        
        shape_val = np.exp(log_shape)
        rate_val = np.exp(log_rate)
        
        self.set_params(shape_val, rate_val)
    
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
        mean_x = self._suf.mean()
        mean_log_x = self._suf.mean_log()
        
        # Method of moments initial estimate
        log_mean_x = np.log(mean_x)
        s = log_mean_x - mean_log_x
        
        if s <= 0:
            # Fallback to simple estimates
            self.set_params(1.0, 1.0 / mean_x)
            return
        
        # Initial estimate for shape
        shape_est = (3 - s + np.sqrt((s - 3)**2 + 24*s)) / (12 * s)
        
        # Newton-Raphson iteration for shape parameter
        from scipy.special import digamma, polygamma
        
        for _ in range(max_iterations):
            # Digamma and trigamma functions
            psi_shape = digamma(shape_est)
            tri_shape = polygamma(1, shape_est)
            
            # Score function (derivative of log-likelihood)
            score = n * (np.log(mean_x) - psi_shape) + n * mean_log_x - n * log_mean_x
            
            # Fisher information (negative second derivative)
            fisher_info = n * tri_shape
            
            if abs(fisher_info) < tolerance:
                break
            
            # Newton-Raphson update
            delta = score / fisher_info
            shape_new = shape_est + delta
            
            if shape_new <= 0:
                shape_new = shape_est / 2
            
            if abs(delta) < tolerance:
                shape_est = shape_new
                break
            
            shape_est = shape_new
        
        # MLE for rate given shape
        rate_est = shape_est / mean_x
        
        self.set_params(shape_est, rate_est)
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'GammaModel':
        """Create a deep copy of the model."""
        cloned = GammaModel(self.shape(), self.rate())
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        return (f"GammaModel(shape={self.shape():.3f}, rate={self.rate():.3f}, "
                f"mean={self.mean():.3f}, data_points={len(self._data)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)