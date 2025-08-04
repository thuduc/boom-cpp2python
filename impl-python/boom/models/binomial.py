"""Binomial model implementation."""

from typing import List, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import RNG
from ..distributions.rmath import dbinom, rbinom, dbeta, rbeta, lgamma_func
from .base import Model, Data, LoglikeModel, ConjugateModel
from .params import UnivParams
from .sufstat import BinomialSuf
from .data import BinomialData


class BinomialModel(ConjugateModel):
    """Binomial model with unknown success probability.
    
    This model assumes observations are binomial random variables:
    X ~ Binomial(n, p) with known n and unknown p.
    
    The model uses a conjugate Beta prior:
    p ~ Beta(alpha, beta)
    
    This gives the beta-binomial conjugate family.
    """
    
    def __init__(self, p: float = 0.5, n: int = 1):
        """Initialize Binomial model.
        
        Args:
            p: Initial success probability parameter.
            n: Number of trials (can be different for each observation).
        """
        super().__init__()
        
        # Validate parameters
        if not 0 <= p <= 1:
            raise ValueError("Success probability must be in [0, 1]")
        if n < 0:
            raise ValueError("Number of trials must be non-negative")
        
        # Create parameter objects
        self._p_param = UnivParams(p)
        
        # Set up parameter management
        self.set_parameter('p', self._p_param)
        
        # Default number of trials (can be overridden per observation)
        self._n = n
        
        # Initialize sufficient statistics
        self._suf = BinomialSuf()
        
        # Prior parameters (default to uniform: Beta(1, 1))
        self._alpha = 1.0
        self._beta = 1.0
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def p(self) -> float:
        """Get success probability parameter."""
        return self._p_param.value()
    
    def set_p(self, p: float):
        """Set success probability parameter."""
        if not 0 <= p <= 1:
            raise ValueError("Success probability must be in [0, 1]")
        self._p_param.set_value(p)
        self._notify_observers()
    
    def n(self) -> int:
        """Get default number of trials."""
        return self._n
    
    def set_n(self, n: int):
        """Set default number of trials."""
        if n < 0:
            raise ValueError("Number of trials must be non-negative")
        self._n = n
    
    # ============================================================================
    # Prior Management
    # ============================================================================
    
    def set_conjugate_prior(self, alpha: float, beta: float):
        """Set conjugate Beta prior.
        
        Args:
            alpha: Alpha parameter of Beta prior (pseudo-successes).
            beta: Beta parameter of Beta prior (pseudo-failures).
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self._alpha = alpha
        self._beta = beta
    
    def log_prior(self) -> float:
        """Compute log prior probability under Beta prior."""
        p_val = self.p()
        return dbeta(p_val, self._alpha, self._beta, log=True)
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, tuple, dict, List]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be:
                - BinomialData object
                - tuple: (n_trials, n_successes)
                - dict: {'trials': n, 'successes': k}
                - list of any of the above
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, tuple) and len(data) == 2:
            trials, successes = data
            data_obj = BinomialData(trials, successes)
        elif isinstance(data, dict):
            trials = data.get('trials', self._n)
            successes = data.get('successes', 0)
            data_obj = BinomialData(trials, successes)
        elif isinstance(data, BinomialData):
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update((data_obj.trials(), data_obj.successes()))
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> BinomialSuf:
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
            p_val = self.p()
            
            for data_point in data:
                if isinstance(data_point, BinomialData):
                    n_trials = data_point.trials()
                    n_successes = data_point.successes()
                    log_lik += dbinom(n_successes, n_trials, p_val, log=True)
                else:
                    raise TypeError(f"Expected BinomialData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            total_trials = self._suf.n()
            total_successes = self._suf.successes()
            p_val = self.p()
            
            # Log likelihood = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
            # We omit the binomial coefficient as it doesn't depend on p
            return (total_successes * np.log(p_val) + 
                   (total_trials - total_successes) * np.log(1 - p_val))
    
    def simulate_data(self, n: Optional[int] = None, trials_per_obs: Optional[int] = None) -> List[BinomialData]:
        """Simulate data from the model.
        
        Args:
            n: Number of observations to simulate. If None, uses current sample size.
            trials_per_obs: Number of trials per observation. If None, uses model default.
            
        Returns:
            List of simulated BinomialData objects.
        """
        if n is None:
            n = len(self._data) if self._data else 1
        
        if n <= 0:
            return []
        
        if trials_per_obs is None:
            trials_per_obs = self._n
        
        p_val = self.p()
        rng = RNG()
        
        simulated = []
        for _ in range(n):
            n_successes = rbinom(trials_per_obs, p_val, rng)
            simulated.append(BinomialData(trials_per_obs, n_successes))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form: [logit(p)]."""
        # Use logit transformation for unconstrained optimization
        p_val = self.p()
        # Handle boundary cases
        if p_val <= 0:
            logit_p = -10.0  # Very negative
        elif p_val >= 1:
            logit_p = 10.0   # Very positive
        else:
            logit_p = np.log(p_val / (1 - p_val))
        
        return Vector([logit_p])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form: [logit(p)]."""
        if len(theta) != 1:
            raise ValueError("Parameter vector must have exactly 1 element")
        
        logit_p = theta[0]
        # Inverse logit transformation
        p_val = 1.0 / (1.0 + np.exp(-logit_p))
        
        self.set_p(p_val)
    
    # ============================================================================
    # Maximum Likelihood Estimation
    # ============================================================================
    
    def mle(self):
        """Compute maximum likelihood estimate."""
        if self._suf.n() == 0:
            return  # No data to estimate from
        
        # MLE for p is simply the success rate
        p_mle = self._suf.success_rate()
        self.set_p(p_mle)
    
    # ============================================================================
    # Conjugate Model Implementation
    # ============================================================================
    
    def posterior_mode(self) -> float:
        """Compute posterior mode under conjugate Beta prior.
        
        Returns:
            Posterior mode of p.
        """
        total_successes = self._suf.successes()
        total_failures = self._suf.failures()
        
        # Posterior parameters
        alpha_post = self._alpha + total_successes
        beta_post = self._beta + total_failures
        
        # Mode of Beta(alpha, beta) is (alpha-1)/(alpha+beta-2) for alpha,beta > 1
        if alpha_post > 1 and beta_post > 1:
            p_mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
        else:
            # Use mean if mode is not well-defined
            p_mode = alpha_post / (alpha_post + beta_post)
        
        return p_mode
    
    def posterior_mean(self) -> float:
        """Compute posterior mean under conjugate Beta prior.
        
        Returns:
            Posterior mean of p.
        """
        total_successes = self._suf.successes()
        total_failures = self._suf.failures()
        
        # Posterior parameters
        alpha_post = self._alpha + total_successes
        beta_post = self._beta + total_failures
        
        # Mean of Beta(alpha, beta) is alpha/(alpha+beta)
        p_mean = alpha_post / (alpha_post + beta_post)
        
        return p_mean
    
    def posterior_variance(self) -> float:
        """Compute posterior variance under conjugate Beta prior.
        
        Returns:
            Posterior variance of p.
        """
        total_successes = self._suf.successes()
        total_failures = self._suf.failures()
        
        # Posterior parameters
        alpha_post = self._alpha + total_successes
        beta_post = self._beta + total_failures
        
        # Variance of Beta(alpha, beta) is alpha*beta/((alpha+beta)^2*(alpha+beta+1))
        sum_post = alpha_post + beta_post
        p_var = (alpha_post * beta_post) / (sum_post * sum_post * (sum_post + 1))
        
        return p_var
    
    def sample_posterior(self, n: int = 1) -> List[float]:
        """Sample from posterior distribution.
        
        Args:
            n: Number of samples.
            
        Returns:
            List of p samples.
        """
        total_successes = self._suf.successes()
        total_failures = self._suf.failures()
        
        # Posterior parameters
        alpha_post = self._alpha + total_successes
        beta_post = self._beta + total_failures
        
        rng = RNG()
        samples = []
        
        for _ in range(n):
            p_sample = rbeta(alpha_post, beta_post, rng)
            samples.append(p_sample)
        
        return samples
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'BinomialModel':
        """Create a deep copy of the model."""
        cloned = BinomialModel(self.p(), self.n())
        cloned.set_conjugate_prior(self._alpha, self._beta)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        return (f"BinomialModel(p={self.p():.3f}, n={self.n()}, "
                f"data_points={len(self._data)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)