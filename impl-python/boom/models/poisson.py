"""Poisson model implementation."""

from typing import List, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import RNG
from ..distributions.rmath import dpois, rpois, dgamma, rgamma, lgamma_func
from .base import Model, Data, LoglikeModel, ConjugateModel
from .params import UnivParams
from .data import DoubleData


class PoissonSuf:
    """Sufficient statistics for Poisson distribution.
    
    Sufficient statistics are:
    - n: number of observations
    - sum: sum of observations
    """
    
    def __init__(self):
        """Initialize Poisson sufficient statistics."""
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        self._sum = 0.0
    
    def update(self, data: Union[Data, float, List[float]]):
        """Update with new data.
        
        Args:
            data: Data to incorporate. Can be:
                - Single int/float value
                - List of int/float values
                - Data object with value() method
                - List of Data objects
        """
        if isinstance(data, list):
            for item in data:
                self.update(item)
        elif isinstance(data, (int, float)):
            x = float(data)
            if x < 0:
                raise ValueError("Poisson observations must be non-negative")
            self._n += 1
            self._sum += x
        elif hasattr(data, 'value'):
            # Assume it's a Data object with value() method
            self.update(data.value())
        else:
            raise TypeError(f"Cannot update PoissonSuf with data of type {type(data)}")
    
    def combine(self, other: 'PoissonSuf') -> 'PoissonSuf':
        """Combine with another PoissonSuf."""
        if not isinstance(other, PoissonSuf):
            raise TypeError("Can only combine with another PoissonSuf")
        
        result = PoissonSuf()
        result._n = self._n + other._n
        result._sum = self._sum + other._sum
        return result
    
    def n(self) -> int:
        """Get sample size."""
        return self._n
    
    def sum(self) -> float:
        """Get sum of observations."""
        return self._sum
    
    def mean(self) -> float:
        """Get sample mean."""
        if self._n == 0:
            return 0.0
        return self._sum / self._n
    
    def clone(self) -> 'PoissonSuf':
        """Create copy."""
        result = PoissonSuf()
        result._n = self._n
        result._sum = self._sum
        return result
    
    def __str__(self) -> str:
        """String representation."""
        return f"PoissonSuf(n={self._n}, sum={self._sum}, mean={self.mean():.3f})"


class PoissonModel(ConjugateModel):
    """Poisson model with unknown rate parameter.
    
    This model assumes observations are Poisson random variables:
    X ~ Poisson(lambda) with unknown lambda.
    
    The model uses a conjugate Gamma prior:
    lambda ~ Gamma(alpha, beta)
    
    This gives the gamma-Poisson conjugate family.
    """
    
    def __init__(self, lam: float = 1.0):
        """Initialize Poisson model.
        
        Args:
            lam: Initial rate parameter (lambda).
        """
        super().__init__()
        
        # Validate parameter
        if lam <= 0:
            raise ValueError("Rate parameter must be positive")
        
        # Create parameter objects
        self._lambda_param = UnivParams(lam)
        
        # Set up parameter management
        self.set_parameter('lambda', self._lambda_param)
        
        # Initialize sufficient statistics
        self._suf = PoissonSuf()
        
        # Prior parameters (default to non-informative: Gamma(0.001, 0.001))
        self._alpha = 0.001
        self._beta = 0.001
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def lam(self) -> float:
        """Get rate parameter (lambda)."""
        return self._lambda_param.value()
    
    def set_lam(self, lam: float):
        """Set rate parameter (lambda)."""
        if lam <= 0:
            raise ValueError("Rate parameter must be positive")
        self._lambda_param.set_value(lam)
        self._notify_observers()
    
    # Alias for consistency with mathematical notation
    def lambda_(self) -> float:
        """Get rate parameter (alias for lam)."""
        return self.lam()
    
    def set_lambda(self, lam: float):
        """Set rate parameter (alias for set_lam)."""
        self.set_lam(lam)
    
    # ============================================================================
    # Prior Management
    # ============================================================================
    
    def set_conjugate_prior(self, alpha: float, beta: float):
        """Set conjugate Gamma prior.
        
        Args:
            alpha: Shape parameter of Gamma prior.
            beta: Rate parameter of Gamma prior.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Gamma prior parameters must be positive")
        self._alpha = alpha
        self._beta = beta
    
    def log_prior(self) -> float:
        """Compute log prior probability under Gamma prior."""
        lam_val = self.lam()
        return dgamma(lam_val, self._alpha, self._beta, log=True)
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, int, float, List]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be:
                - DoubleData object
                - int or float value
                - list of int/float values
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, (int, float)):
            if data < 0:
                raise ValueError("Poisson observations must be non-negative")
            data_obj = DoubleData(float(data))
        elif isinstance(data, DoubleData):
            if data.value() < 0:
                raise ValueError("Poisson observations must be non-negative")
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update(data_obj.value())
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> PoissonSuf:
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
            lam_val = self.lam()
            
            for data_point in data:
                if isinstance(data_point, DoubleData):
                    x = int(data_point.value())
                    log_lik += dpois(x, lam_val, log=True)
                else:
                    raise TypeError(f"Expected DoubleData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            n = self._suf.n()
            sum_x = self._suf.sum()
            lam_val = self.lam()
            
            # Log likelihood = sum_x * log(lambda) - n * lambda - sum(log(x_i!))
            # We omit the factorial terms as they don't depend on lambda
            return sum_x * np.log(lam_val) - n * lam_val
    
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
        
        lam_val = self.lam()
        rng = RNG()
        
        simulated = []
        for _ in range(n):
            x = rpois(lam_val, rng)
            simulated.append(DoubleData(float(x)))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form: [log(lambda)]."""
        # Use log transformation for unconstrained optimization
        lam_val = self.lam()
        return Vector([np.log(lam_val)])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form: [log(lambda)]."""
        if len(theta) != 1:
            raise ValueError("Parameter vector must have exactly 1 element")
        
        log_lam = theta[0]
        lam_val = np.exp(log_lam)
        
        self.set_lam(lam_val)
    
    # ============================================================================
    # Maximum Likelihood Estimation
    # ============================================================================
    
    def mle(self):
        """Compute maximum likelihood estimate."""
        if self._suf.n() == 0:
            return  # No data to estimate from
        
        # MLE for lambda is simply the sample mean
        lam_mle = self._suf.mean()
        self.set_lam(lam_mle)
    
    # ============================================================================
    # Conjugate Model Implementation
    # ============================================================================
    
    def posterior_mode(self) -> float:
        """Compute posterior mode under conjugate Gamma prior.
        
        Returns:
            Posterior mode of lambda.
        """
        n = self._suf.n()
        sum_x = self._suf.sum()
        
        # Posterior parameters
        alpha_post = self._alpha + sum_x
        beta_post = self._beta + n
        
        # Mode of Gamma(alpha, beta) is (alpha-1)/beta for alpha > 1
        if alpha_post > 1:
            lam_mode = (alpha_post - 1) / beta_post
        else:
            # Use mean if mode is not well-defined
            lam_mode = alpha_post / beta_post
        
        return lam_mode
    
    def posterior_mean(self) -> float:
        """Compute posterior mean under conjugate Gamma prior.
        
        Returns:
            Posterior mean of lambda.
        """
        n = self._suf.n()
        sum_x = self._suf.sum()
        
        # Posterior parameters
        alpha_post = self._alpha + sum_x
        beta_post = self._beta + n
        
        # Mean of Gamma(alpha, beta) is alpha/beta
        lam_mean = alpha_post / beta_post
        
        return lam_mean
    
    def posterior_variance(self) -> float:
        """Compute posterior variance under conjugate Gamma prior.
        
        Returns:
            Posterior variance of lambda.
        """
        n = self._suf.n()
        sum_x = self._suf.sum()
        
        # Posterior parameters
        alpha_post = self._alpha + sum_x
        beta_post = self._beta + n
        
        # Variance of Gamma(alpha, beta) is alpha/beta^2
        lam_var = alpha_post / (beta_post * beta_post)
        
        return lam_var
    
    def sample_posterior(self, n: int = 1) -> List[float]:
        """Sample from posterior distribution.
        
        Args:
            n: Number of samples.
            
        Returns:
            List of lambda samples.
        """
        n_data = self._suf.n()
        sum_x = self._suf.sum()
        
        # Posterior parameters
        alpha_post = self._alpha + sum_x
        beta_post = self._beta + n_data
        
        rng = RNG()
        samples = []
        
        for _ in range(n):
            # rgamma expects shape and scale, but beta_post is rate
            # So scale = 1/rate = 1/beta_post
            lam_sample = rgamma(alpha_post, 1.0 / beta_post, rng)
            samples.append(lam_sample)
        
        return samples
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'PoissonModel':
        """Create a deep copy of the model."""
        cloned = PoissonModel(self.lam())
        cloned.set_conjugate_prior(self._alpha, self._beta)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        return (f"PoissonModel(lambda={self.lam():.3f}, "
                f"data_points={len(self._data)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)