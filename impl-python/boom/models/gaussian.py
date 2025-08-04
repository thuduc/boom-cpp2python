"""Gaussian (Normal) model implementation."""

from typing import List, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import rnorm, dnorm
from .base import Model, Data, LoglikeModel, ConjugateModel
from .params import UnivParams
from .sufstat import GaussianSuf
from .data import DoubleData


class GaussianModel(ConjugateModel):
    """Gaussian (normal) model with unknown mean and variance.
    
    This model assumes observations are independently drawn from
    N(mu, sigma^2) with both mu and sigma^2 unknown.
    
    The model uses conjugate priors:
    - mu | sigma^2 ~ N(mu0, sigma^2/kappa0)  
    - sigma^2 ~ InverseGamma(alpha0, beta0)
    
    This gives the normal-inverse-gamma conjugate family.
    """
    
    def __init__(self, mu: float = 0.0, sigma_sq: float = 1.0):
        """Initialize Gaussian model.
        
        Args:
            mu: Initial mean parameter.
            sigma_sq: Initial variance parameter.
        """
        super().__init__()
        
        # Create parameter objects
        self._mu_param = UnivParams(mu)
        self._sigma_sq_param = UnivParams(sigma_sq)
        
        # Set up parameter management
        self.set_parameter('mu', self._mu_param)
        self.set_parameter('sigma_sq', self._sigma_sq_param)
        
        # Initialize sufficient statistics
        self._suf = GaussianSuf()
        
        # Prior parameters (default to non-informative)
        self._mu0 = 0.0          # Prior mean for mu
        self._kappa0 = 0.0       # Prior precision for mu (0 = non-informative)
        self._alpha0 = 0.0       # Prior shape for sigma^2 (0 = non-informative)
        self._beta0 = 0.0        # Prior rate for sigma^2 (0 = non-informative)
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def mu(self) -> float:
        """Get mean parameter."""
        return self._mu_param.value()
    
    def set_mu(self, mu: float):
        """Set mean parameter."""
        self._mu_param.set_value(mu)
        self._notify_observers()
    
    def sigma_sq(self) -> float:
        """Get variance parameter."""
        return self._sigma_sq_param.value()
    
    def set_sigma_sq(self, sigma_sq: float):
        """Set variance parameter."""
        if sigma_sq <= 0:
            raise ValueError("Variance must be positive")
        self._sigma_sq_param.set_value(sigma_sq)
        self._notify_observers()
    
    def sigma(self) -> float:
        """Get standard deviation parameter."""
        return np.sqrt(self.sigma_sq())
    
    def set_sigma(self, sigma: float):
        """Set standard deviation parameter."""
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
        self.set_sigma_sq(sigma * sigma)
    
    def set_params(self, mu: float, sigma_sq: float):
        """Set both parameters simultaneously."""
        self.set_mu(mu)
        self.set_sigma_sq(sigma_sq)
    
    # ============================================================================
    # Prior Management
    # ============================================================================
    
    def set_conjugate_prior(self, mu0: float, kappa0: float, 
                           alpha0: float, beta0: float):
        """Set conjugate normal-inverse-gamma prior.
        
        Args:
            mu0: Prior mean for mu.
            kappa0: Prior precision for mu (set to 0 for non-informative).
            alpha0: Prior shape for sigma^2 (set to 0 for non-informative).
            beta0: Prior rate for sigma^2 (set to 0 for non-informative).
        """
        self._mu0 = mu0
        self._kappa0 = max(0.0, kappa0)
        self._alpha0 = max(0.0, alpha0)  
        self._beta0 = max(0.0, beta0)
    
    def log_prior(self) -> float:
        """Compute log prior probability under conjugate prior."""
        log_prior_val = 0.0
        
        mu_val = self.mu()
        sigma_sq_val = self.sigma_sq()
        
        # Prior for mu | sigma^2: N(mu0, sigma^2/kappa0)
        if self._kappa0 > 0:
            prior_var_mu = sigma_sq_val / self._kappa0
            log_prior_val += dnorm(mu_val, self._mu0, np.sqrt(prior_var_mu), log=True)
        
        # Prior for sigma^2: InverseGamma(alpha0, beta0)
        if self._alpha0 > 0 and self._beta0 > 0:
            # InverseGamma log pdf: alpha*log(beta) - loggamma(alpha) - (alpha+1)*log(x) - beta/x
            from ..distributions.rmath import lgamma_func
            log_prior_val += (self._alpha0 * np.log(self._beta0) - 
                             lgamma_func(self._alpha0) - 
                             (self._alpha0 + 1) * np.log(sigma_sq_val) - 
                             self._beta0 / sigma_sq_val)
        
        return log_prior_val
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, float, List[float]]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be DoubleData, float, or list of floats.
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, (int, float)):
            data_obj = DoubleData(float(data))
        elif isinstance(data, DoubleData):
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update(data_obj.value())
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> GaussianSuf:
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
            mu_val = self.mu()
            sigma_val = self.sigma()
            
            for data_point in data:
                if isinstance(data_point, DoubleData):
                    x = data_point.value()
                    log_lik += dnorm(x, mu_val, sigma_val, log=True)
                else:
                    raise TypeError(f"Expected DoubleData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            n = self._suf.n()
            mu_val = self.mu()
            sigma_sq_val = self.sigma_sq()
            
            # Log likelihood = -n/2 * log(2*pi) - n/2 * log(sigma^2) - SS/(2*sigma^2)
            # where SS = sum((x_i - mu)^2) = sumsq - 2*mu*sum + n*mu^2
            sum_sq_deviations = (self._suf.sumsq() - 
                               2 * mu_val * self._suf.sum() + 
                               n * mu_val * mu_val)
            
            return (-0.5 * n * np.log(2 * np.pi) - 
                    0.5 * n * np.log(sigma_sq_val) -
                    0.5 * sum_sq_deviations / sigma_sq_val)
    
    def simulate_data(self, n: Optional[int] = None) -> List[DoubleData]:
        """Simulate data from the model.
        
        Args:
            n: Number of data points to simulate. If None, uses current sample size.
            
        Returns:
            List of simulated DoubleData objects.
        """
        if n is None:
            n = self.sample_size()
        
        if n <= 0:
            return []
        
        mu_val = self.mu()
        sigma_val = self.sigma()
        
        simulated = []
        for _ in range(n):
            x = rnorm(mu_val, sigma_val)
            simulated.append(DoubleData(x))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form: [mu, log(sigma^2)]."""
        # Use log(sigma^2) for unconstrained optimization
        return Vector([self.mu(), np.log(self.sigma_sq())])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form: [mu, log(sigma^2)]."""
        if len(theta) != 2:
            raise ValueError("Parameter vector must have exactly 2 elements")
        
        mu_val = theta[0]
        log_sigma_sq = theta[1]
        sigma_sq_val = np.exp(log_sigma_sq)
        
        self.set_params(mu_val, sigma_sq_val)
    
    # ============================================================================
    # Maximum Likelihood Estimation
    # ============================================================================
    
    def mle(self):
        """Compute maximum likelihood estimates."""
        if self._suf.n() == 0:
            return  # No data to estimate from
        
        # MLE for mu
        mu_mle = self._suf.mean()
        
        # MLE for sigma^2 (using n denominator, not n-1)
        n = self._suf.n()
        sigma_sq_mle = self._suf.variance(sample=False)  # Population variance
        
        self.set_params(mu_mle, sigma_sq_mle)
    
    # ============================================================================
    # Conjugate Model Implementation
    # ============================================================================
    
    def posterior_mode(self) -> tuple:
        """Compute posterior mode under conjugate prior.
        
        Returns:
            Tuple of (mu_mode, sigma_sq_mode).
        """
        if self._kappa0 <= 0 or self._alpha0 <= 0 or self._beta0 <= 0:
            # Non-informative prior, use MLE
            self.mle()
            return (self.mu(), self.sigma_sq())
        
        n = self._suf.n()
        if n == 0:
            return (self._mu0, self._beta0 / (self._alpha0 + 1))
        
        # Posterior hyperparameters
        kappa_n = self._kappa0 + n
        alpha_n = self._alpha0 + n / 2
        
        sample_mean = self._suf.mean()
        mu_n = (self._kappa0 * self._mu0 + n * sample_mean) / kappa_n
        
        sum_sq_deviations = (self._suf.sumsq() - 
                           2 * sample_mean * self._suf.sum() + 
                           n * sample_mean * sample_mean)
        
        beta_n = (self._beta0 + 0.5 * sum_sq_deviations + 
                 0.5 * self._kappa0 * n * (sample_mean - self._mu0)**2 / kappa_n)
        
        # Mode of normal-inverse-gamma
        mu_mode = mu_n
        sigma_sq_mode = beta_n / (alpha_n + 1)
        
        return (mu_mode, sigma_sq_mode)
    
    def posterior_mean(self) -> tuple:
        """Compute posterior mean under conjugate prior.
        
        Returns:
            Tuple of (mu_mean, sigma_sq_mean).
        """
        if self._kappa0 <= 0 or self._alpha0 <= 0 or self._beta0 <= 0:
            # Non-informative prior, use MLE
            self.mle()
            return (self.mu(), self.sigma_sq())
        
        n = self._suf.n()
        if n == 0:
            mu_mean = self._mu0
            sigma_sq_mean = self._beta0 / (self._alpha0 - 1) if self._alpha0 > 1 else np.inf
            return (mu_mean, sigma_sq_mean)
        
        # Posterior hyperparameters
        kappa_n = self._kappa0 + n
        alpha_n = self._alpha0 + n / 2
        
        sample_mean = self._suf.mean()
        mu_n = (self._kappa0 * self._mu0 + n * sample_mean) / kappa_n
        
        sum_sq_deviations = (self._suf.sumsq() - 
                           2 * sample_mean * self._suf.sum() + 
                           n * sample_mean * sample_mean)
        
        beta_n = (self._beta0 + 0.5 * sum_sq_deviations + 
                 0.5 * self._kappa0 * n * (sample_mean - self._mu0)**2 / kappa_n)
        
        # Mean of normal-inverse-gamma
        mu_mean = mu_n
        sigma_sq_mean = beta_n / (alpha_n - 1) if alpha_n > 1 else np.inf
        
        return (mu_mean, sigma_sq_mean)
    
    def sample_posterior(self, n: int = 1) -> List[tuple]:
        """Sample from posterior distribution.
        
        Args:
            n: Number of samples.
            
        Returns:
            List of (mu, sigma_sq) tuples.
        """
        if self._kappa0 <= 0 or self._alpha0 <= 0 or self._beta0 <= 0:
            raise NotImplementedError("Posterior sampling requires informative conjugate prior")
        
        from ..distributions import RNG
        from ..distributions.custom import InverseGamma
        
        rng = RNG()
        
        # Compute posterior hyperparameters
        n_data = self._suf.n()
        kappa_n = self._kappa0 + n_data
        alpha_n = self._alpha0 + n_data / 2
        
        if n_data > 0:
            sample_mean = self._suf.mean()
            mu_n = (self._kappa0 * self._mu0 + n_data * sample_mean) / kappa_n
            
            sum_sq_deviations = (self._suf.sumsq() - 
                               2 * sample_mean * self._suf.sum() + 
                               n_data * sample_mean * sample_mean)
            
            beta_n = (self._beta0 + 0.5 * sum_sq_deviations + 
                     0.5 * self._kappa0 * n_data * (sample_mean - self._mu0)**2 / kappa_n)
        else:
            mu_n = self._mu0
            beta_n = self._beta0
        
        # Sample from posterior
        samples = []
        inv_gamma = InverseGamma(alpha_n, beta_n)
        
        for _ in range(n):
            # Sample sigma^2 from inverse gamma
            sigma_sq_sample = inv_gamma.rvs(rng)
            
            # Sample mu from normal given sigma^2
            mu_var = sigma_sq_sample / kappa_n
            mu_sample = rnorm(mu_n, np.sqrt(mu_var), rng)
            
            samples.append((mu_sample, sigma_sq_sample))
        
        return samples
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'GaussianModel':
        """Create a deep copy of the model."""
        cloned = GaussianModel(self.mu(), self.sigma_sq())
        cloned.set_conjugate_prior(self._mu0, self._kappa0, self._alpha0, self._beta0)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        return (f"GaussianModel(mu={self.mu():.3f}, sigma={self.sigma():.3f}, "
                f"n={self.sample_size()})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)