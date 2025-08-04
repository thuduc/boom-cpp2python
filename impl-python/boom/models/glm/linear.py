"""Linear regression model with conjugate prior support."""

import numpy as np
from typing import List, Optional, Union, Tuple
from scipy import stats

from .base import GlmModel, GlmData
from boom.linalg import Vector, SpdMatrix
from boom.distributions import RNG
from boom.distributions.rmath import dnorm, rnorm
from boom.models.base import ConjugateModel


class LinearRegressionModel(GlmModel, ConjugateModel):
    """Linear regression model with normal-inverse-gamma conjugate prior.
    
    Model: y_i ~ Normal(X_i^T * beta, sigma^2)
    Prior: beta | sigma^2 ~ MVN(beta_0, sigma^2 * V_0)
           sigma^2 ~ InvGamma(a_0, b_0)
    """
    
    def __init__(self, xdim: int, sigma_sq: float = 1.0):
        """Initialize linear regression model.
        
        Args:
            xdim: Dimension of predictor space
            sigma_sq: Error variance
        """
        super().__init__(xdim)
        if sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive")
        self._sigma_sq = float(sigma_sq)
        
        # Prior parameters (non-informative by default)
        self._beta_prior_mean = Vector(np.zeros(xdim))
        self._beta_prior_precision = SpdMatrix(1e-6 * np.eye(xdim))  # Very diffuse
        self._sigma_prior_a = 0.001  # Shape parameter (non-informative)
        self._sigma_prior_b = 0.001  # Rate parameter (non-informative)
        
        self._has_conjugate_prior = False
    
    def sigma_sq(self) -> float:
        """Get error variance."""
        return self._sigma_sq
    
    def sigma(self) -> float:
        """Get error standard deviation."""
        return np.sqrt(self._sigma_sq)
    
    def set_sigma_sq(self, sigma_sq: float):
        """Set error variance."""
        if sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive")
        self._sigma_sq = float(sigma_sq)
        self._notify_observers()
    
    def set_sigma(self, sigma: float):
        """Set error standard deviation."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.set_sigma_sq(sigma * sigma)
    
    def set_conjugate_prior(self, beta_mean: Union[Vector, np.ndarray, List[float]],
                           beta_precision: Union[SpdMatrix, np.ndarray],
                           sigma_a: float, sigma_b: float):
        """Set conjugate normal-inverse-gamma prior.
        
        Args:
            beta_mean: Prior mean for beta
            beta_precision: Prior precision matrix for beta (inverse covariance)
            sigma_a: Shape parameter for inverse-gamma prior on sigma^2
            sigma_b: Rate parameter for inverse-gamma prior on sigma^2
        """
        if isinstance(beta_mean, (np.ndarray, list)):
            self._beta_prior_mean = Vector(beta_mean)
        elif isinstance(beta_mean, Vector):
            self._beta_prior_mean = beta_mean.copy()
        else:
            raise ValueError(f"beta_mean must be Vector, ndarray, or list, got {type(beta_mean)}")
        
        if len(self._beta_prior_mean) != self._xdim:
            raise ValueError(f"beta_mean dimension {len(self._beta_prior_mean)} doesn't match model dimension {self._xdim}")
        
        if isinstance(beta_precision, np.ndarray):
            self._beta_prior_precision = SpdMatrix(beta_precision)
        elif isinstance(beta_precision, SpdMatrix):
            self._beta_prior_precision = beta_precision.copy()
        else:
            raise ValueError(f"beta_precision must be SpdMatrix or ndarray, got {type(beta_precision)}")
        
        if self._beta_prior_precision.nrow() != self._xdim:
            raise ValueError(f"beta_precision dimension doesn't match model dimension")
        
        if sigma_a <= 0 or sigma_b <= 0:
            raise ValueError("sigma_a and sigma_b must be positive")
        
        self._sigma_prior_a = float(sigma_a)
        self._sigma_prior_b = float(sigma_b)
        self._has_conjugate_prior = True
    
    def mean_function(self, linear_pred: float) -> float:
        """Identity link: mean = linear predictor."""
        return linear_pred
    
    def variance_function(self, mean: float) -> float:
        """Constant variance."""
        return self._sigma_sq
    
    def log_likelihood(self, data: Optional[List[GlmData]] = None) -> float:
        """Compute log likelihood."""
        if data is None:
            data = self._data
        
        log_lik = 0.0
        for data_point in data:
            y = data_point.y()
            x = data_point.x()
            mu = self.linear_predictor(x)
            log_lik += dnorm(y, mu, self.sigma(), log=True)
        
        return log_lik
    
    def log_prior(self) -> float:
        """Compute log prior density."""
        if not self._has_conjugate_prior:
            return 0.0
        
        # Log prior for beta given sigma^2
        beta_diff = self._beta - self._beta_prior_mean
        beta_diff_array = beta_diff.to_numpy()
        precision_array = self._beta_prior_precision.to_numpy()
        
        beta_log_prior = (-0.5 * self._xdim * np.log(2 * np.pi * self._sigma_sq) +
                         0.5 * np.log(np.linalg.det(precision_array)) -
                         0.5 / self._sigma_sq * np.dot(beta_diff_array, precision_array @ beta_diff_array))
        
        # Log prior for sigma^2 (inverse gamma)
        from scipy.special import loggamma
        sigma_log_prior = (self._sigma_prior_a * np.log(self._sigma_prior_b) -
                          loggamma(self._sigma_prior_a) -
                          (self._sigma_prior_a + 1) * np.log(self._sigma_sq) -
                          self._sigma_prior_b / self._sigma_sq)
        
        return beta_log_prior + sigma_log_prior
    
    def mle(self, max_iterations: int = 100, tolerance: float = 1e-8):
        """Compute maximum likelihood estimates using sufficient statistics."""
        suf = self.suf()
        if suf.n() == 0:
            return
        
        # OLS estimate for beta
        beta_hat = suf.beta_hat()
        self.set_beta(beta_hat)
        
        # MLE for sigma^2
        sigma_sq_hat = suf.sse() / suf.n()
        self.set_sigma_sq(max(sigma_sq_hat, 1e-10))  # Avoid zero variance
    
    def posterior_mode(self) -> Tuple[Vector, float]:
        """Compute posterior mode (MAP estimate)."""
        if not self._has_conjugate_prior:
            self.mle()
            return self.beta().copy(), self._sigma_sq
        
        suf = self.suf()
        if suf.n() == 0:
            return self._beta_prior_mean.copy(), self._sigma_prior_b / (self._sigma_prior_a + 1)
        
        # Posterior parameters
        precision_post = self._beta_prior_precision + suf.xtx() / self._sigma_sq
        mean_post_unnorm = (self._beta_prior_precision.to_numpy() @ self._beta_prior_mean.to_numpy() + 
                           suf.xty().to_numpy() / self._sigma_sq)
        
        try:
            precision_post_inv = np.linalg.inv(precision_post.to_numpy())
            beta_post_mode = Vector(precision_post_inv @ mean_post_unnorm)
        except np.linalg.LinAlgError:
            beta_post_mode = Vector(np.linalg.pinv(precision_post.to_numpy()) @ mean_post_unnorm)
        
        # Posterior mode for sigma^2
        a_post = self._sigma_prior_a + suf.n() / 2
        b_post = self._sigma_prior_b + 0.5 * suf.sse()
        sigma_sq_post_mode = b_post / (a_post + 1)
        
        return beta_post_mode, sigma_sq_post_mode
    
    def posterior_mean(self) -> Tuple[Vector, float]:
        """Compute posterior mean."""
        if not self._has_conjugate_prior:
            self.mle()
            return self.beta().copy(), self._sigma_sq
        
        suf = self.suf()
        if suf.n() == 0:
            return self._beta_prior_mean.copy(), self._sigma_prior_b / (self._sigma_prior_a - 1)
        
        # For conjugate prior, we need to iterate since beta depends on sigma^2
        # Use current sigma^2 estimate for beta posterior
        precision_post = self._beta_prior_precision + suf.xtx() / self._sigma_sq
        mean_post_unnorm = (self._beta_prior_precision.to_numpy() @ self._beta_prior_mean.to_numpy() + 
                           suf.xty().to_numpy() / self._sigma_sq)
        
        try:
            precision_post_inv = np.linalg.inv(precision_post.to_numpy())
            beta_post_mean = Vector(precision_post_inv @ mean_post_unnorm)
        except np.linalg.LinAlgError:
            beta_post_mean = Vector(np.linalg.pinv(precision_post.to_numpy()) @ mean_post_unnorm)
        
        # Posterior mean for sigma^2
        a_post = self._sigma_prior_a + suf.n() / 2
        b_post = self._sigma_prior_b + 0.5 * suf.sse()
        if a_post > 1:
            sigma_sq_post_mean = b_post / (a_post - 1)
        else:
            sigma_sq_post_mean = self._sigma_sq  # Fall back to current value
        
        return beta_post_mean, sigma_sq_post_mean
    
    def sample_posterior(self, n: int = 1, rng: Optional[RNG] = None) -> List[Tuple[Vector, float]]:
        """Sample from posterior distribution."""
        if rng is None:
            rng = RNG()
        
        if not self._has_conjugate_prior:
            # Without conjugate prior, return MLE (could implement importance sampling)
            self.mle()
            return [(self.beta().copy(), self._sigma_sq)] * n
        
        suf = self.suf()
        samples = []
        
        for _ in range(n):
            # Sample sigma^2 from posterior inverse gamma
            a_post = self._sigma_prior_a + suf.n() / 2
            b_post = self._sigma_prior_b + 0.5 * suf.sse()
            
            # Sample from inverse gamma by sampling from gamma and taking reciprocal
            gamma_sample = stats.gamma.rvs(a_post, scale=1/b_post, random_state=rng._rng)
            sigma_sq_sample = 1.0 / gamma_sample
            
            # Sample beta given sigma^2 from posterior normal
            precision_post = self._beta_prior_precision + suf.xtx() / sigma_sq_sample
            mean_post_unnorm = (self._beta_prior_precision.to_numpy() @ self._beta_prior_mean.to_numpy() + 
                               suf.xty().to_numpy() / sigma_sq_sample)
            
            try:
                precision_post_inv = np.linalg.inv(precision_post.to_numpy())
                beta_post_mean = precision_post_inv @ mean_post_unnorm
                beta_post_cov = sigma_sq_sample * precision_post_inv
                
                beta_sample = stats.multivariate_normal.rvs(
                    mean=beta_post_mean, cov=beta_post_cov, random_state=rng._rng)
                beta_sample = Vector(beta_sample)
            except np.linalg.LinAlgError:
                # Fall back to prior if numerical issues
                beta_sample = self._beta_prior_mean.copy()
            
            samples.append((beta_sample, sigma_sq_sample))
        
        return samples
    
    def simulate_data(self, n: int, X: Union[np.ndarray, List[List[float]]],
                     rng: Optional[RNG] = None) -> List[GlmData]:
        """Simulate data from the model.
        
        Args:
            n: Number of observations to simulate
            X: Design matrix (n x p)
            rng: Random number generator
            
        Returns:
            List of simulated data points
        """
        if rng is None:
            rng = RNG()
        
        X_array = np.array(X)
        if X_array.shape[0] != n:
            raise ValueError(f"X has {X_array.shape[0]} rows, expected {n}")
        if X_array.shape[1] != self._xdim:
            raise ValueError(f"X has {X_array.shape[1]} columns, expected {self._xdim}")
        
        simulated_data = []
        beta_array = self._beta.to_numpy()
        sigma = self.sigma()
        
        for i in range(n):
            x_i = Vector(X_array[i])
            mu_i = np.dot(beta_array, X_array[i])
            y_i = rnorm(mu_i, sigma, rng)
            simulated_data.append(GlmData(y_i, x_i))
        
        return simulated_data
    
    def r_squared(self) -> float:
        """Compute R-squared coefficient of determination."""
        return self.suf().r_squared()
    
    def clone(self) -> 'LinearRegressionModel':
        """Create a copy of this model."""
        cloned = LinearRegressionModel(self._xdim, self._sigma_sq)
        cloned.set_beta(self._beta)
        
        if self._has_conjugate_prior:
            cloned.set_conjugate_prior(
                self._beta_prior_mean, self._beta_prior_precision,
                self._sigma_prior_a, self._sigma_prior_b)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"LinearRegressionModel(xdim={self._xdim}, beta={self._beta.to_numpy()}, "
                f"sigma={self.sigma():.3f}, data_points={len(self._data)})")