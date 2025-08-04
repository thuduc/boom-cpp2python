"""Custom distributions not readily available in SciPy."""

from typing import Optional
import numpy as np
import scipy.stats as stats
import scipy.special as special
from .rng import RNG, GlobalRng


class TriangularDistribution:
    """Triangular distribution implementation."""
    
    def __init__(self, left: float, mode: float, right: float):
        """Initialize triangular distribution.
        
        Args:
            left: Lower bound.
            mode: Mode (peak) of the distribution.
            right: Upper bound.
        """
        if not (left <= mode <= right):
            raise ValueError("Must have left <= mode <= right")
        
        self.left = left
        self.mode = mode
        self.right = right
    
    def pdf(self, x: float, log: bool = False) -> float:
        """Probability density function."""
        if x < self.left or x > self.right:
            density = 0.0
        elif x <= self.mode:
            density = 2 * (x - self.left) / ((self.right - self.left) * (self.mode - self.left))
        else:
            density = 2 * (self.right - x) / ((self.right - self.left) * (self.right - self.mode))
        
        if log:
            return np.log(density) if density > 0 else -np.inf
        else:
            return density
    
    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        if x < self.left:
            return 0.0
        elif x <= self.mode:
            return ((x - self.left)**2) / ((self.right - self.left) * (self.mode - self.left))
        elif x < self.right:
            return 1.0 - ((self.right - x)**2) / ((self.right - self.left) * (self.right - self.mode))
        else:
            return 1.0
    
    def quantile(self, p: float) -> float:
        """Quantile function."""
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        
        if p == 0:
            return self.left
        elif p == 1:
            return self.right
        
        # Check which side of the mode
        p_mode = (self.mode - self.left) / (self.right - self.left)
        
        if p <= p_mode:
            # Left side
            return self.left + np.sqrt(p * (self.right - self.left) * (self.mode - self.left))
        else:
            # Right side
            return self.right - np.sqrt((1 - p) * (self.right - self.left) * (self.right - self.mode))
    
    def rvs(self, rng: Optional[RNG] = None) -> float:
        """Generate random sample."""
        if rng is None:
            rng = GlobalRng.rng
        u = rng()
        return self.quantile(u)


class TruncatedNormal:
    """Truncated normal distribution."""
    
    def __init__(self, mu: float, sigma: float, lower: float = -np.inf, upper: float = np.inf):
        """Initialize truncated normal.
        
        Args:
            mu: Mean of untruncated normal.
            sigma: Standard deviation of untruncated normal.
            lower: Lower truncation point.
            upper: Upper truncation point.
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if lower >= upper:
            raise ValueError("lower must be less than upper")
        
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        
        # Standardized bounds
        self.alpha = (lower - mu) / sigma if np.isfinite(lower) else -np.inf
        self.beta = (upper - mu) / sigma if np.isfinite(upper) else np.inf
        
        # Normalizing constant
        self.Z = stats.norm.cdf(self.beta) - stats.norm.cdf(self.alpha)
    
    def pdf(self, x: float, log: bool = False) -> float:
        """Probability density function."""
        if x < self.lower or x > self.upper:
            density = 0.0
        else:
            std_x = (x - self.mu) / self.sigma
            if log:
                return stats.norm.logpdf(std_x) - np.log(self.sigma) - np.log(self.Z)
            else:
                density = stats.norm.pdf(std_x) / (self.sigma * self.Z)
        
        if log and density == 0.0:
            return -np.inf
        return density if not log else np.log(density)
    
    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        if x <= self.lower:
            return 0.0
        elif x >= self.upper:
            return 1.0
        else:
            std_x = (x - self.mu) / self.sigma
            return (stats.norm.cdf(std_x) - stats.norm.cdf(self.alpha)) / self.Z
    
    def quantile(self, p: float) -> float:
        """Quantile function."""
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        
        if p == 0:
            return self.lower
        elif p == 1:
            return self.upper
        
        # Inverse CDF
        Phi_alpha = stats.norm.cdf(self.alpha)
        q_std = stats.norm.ppf(Phi_alpha + p * self.Z)
        return self.mu + self.sigma * q_std
    
    def rvs(self, rng: Optional[RNG] = None) -> float:
        """Generate random sample using inverse CDF method."""
        if rng is None:
            rng = GlobalRng.rng
        u = rng()
        return self.quantile(u)


class InverseGamma:
    """Inverse gamma distribution."""
    
    def __init__(self, alpha: float, beta: float):
        """Initialize inverse gamma distribution.
        
        Args:
            alpha: Shape parameter.
            beta: Scale parameter.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        
        self.alpha = alpha
        self.beta = beta
    
    def pdf(self, x: float, log: bool = False) -> float:
        """Probability density function."""
        if x <= 0:
            if log:
                return -np.inf
            else:
                return 0.0
        
        if log:
            return (self.alpha * np.log(self.beta) - 
                   special.loggamma(self.alpha) - 
                   (self.alpha + 1) * np.log(x) - 
                   self.beta / x)
        else:
            return (self.beta**self.alpha / special.gamma(self.alpha) * 
                   x**(-self.alpha - 1) * np.exp(-self.beta / x))
    
    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        if x <= 0:
            return 0.0
        
        # Use relationship to gamma distribution
        return stats.gamma.sf(self.beta / x, a=self.alpha)
    
    def quantile(self, p: float) -> float:
        """Quantile function."""
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        
        if p == 0:
            return 0.0
        elif p == 1:
            return np.inf
        
        # Use relationship to gamma distribution
        gamma_q = stats.gamma.ppf(1 - p, a=self.alpha)
        return self.beta / gamma_q
    
    def rvs(self, rng: Optional[RNG] = None) -> float:
        """Generate random sample."""
        if rng is None:
            rng = GlobalRng.rng
        
        # Sample from gamma and invert
        gamma_sample = rng.gamma(self.alpha, 1.0)
        return self.beta / gamma_sample
    
    def mean(self) -> float:
        """Mean of the distribution."""
        if self.alpha <= 1:
            return np.inf
        return self.beta / (self.alpha - 1)
    
    def var(self) -> float:
        """Variance of the distribution."""
        if self.alpha <= 2:
            return np.inf
        return (self.beta**2) / ((self.alpha - 1)**2 * (self.alpha - 2))


class Dirichlet:
    """Dirichlet distribution."""
    
    def __init__(self, alpha: np.ndarray):
        """Initialize Dirichlet distribution.
        
        Args:
            alpha: Concentration parameters (must be positive).
        """
        self.alpha = np.asarray(alpha, dtype=float)
        if np.any(self.alpha <= 0):
            raise ValueError("All alpha parameters must be positive")
        
        self.k = len(self.alpha)
        self.alpha_sum = np.sum(self.alpha)
    
    def pdf(self, x: np.ndarray, log: bool = False) -> float:
        """Probability density function."""
        x = np.asarray(x)
        
        if len(x) != self.k:
            raise ValueError(f"x must have length {self.k}")
        
        if not np.allclose(np.sum(x), 1.0) or np.any(x <= 0):
            if log:
                return -np.inf
            else:
                return 0.0
        
        if log:
            return (special.loggamma(self.alpha_sum) - 
                   np.sum(special.loggamma(self.alpha)) + 
                   np.sum((self.alpha - 1) * np.log(x)))
        else:
            return (special.gamma(self.alpha_sum) / 
                   np.prod(special.gamma(self.alpha)) * 
                   np.prod(x**(self.alpha - 1)))
    
    def rvs(self, rng: Optional[RNG] = None) -> np.ndarray:
        """Generate random sample."""
        if rng is None:
            rng = GlobalRng.rng
        
        # Sample from gamma distributions
        samples = np.array([rng.gamma(a, 1.0) for a in self.alpha])
        return samples / np.sum(samples)
    
    def mean(self) -> np.ndarray:
        """Mean of the distribution."""
        return self.alpha / self.alpha_sum
    
    def var(self) -> np.ndarray:
        """Variance of the distribution."""
        mean = self.mean()
        return mean * (1 - mean) / (self.alpha_sum + 1)


# Triangle distribution convenience functions
def dtriangle(x: float, left: float, mode: float, right: float, log: bool = False) -> float:
    """Triangular density function."""
    dist = TriangularDistribution(left, mode, right)
    return dist.pdf(x, log=log)


def ptriangle(x: float, left: float, mode: float, right: float) -> float:
    """Triangular cumulative distribution function."""
    dist = TriangularDistribution(left, mode, right)
    return dist.cdf(x)


def qtriangle(p: float, left: float, mode: float, right: float) -> float:
    """Triangular quantile function."""
    dist = TriangularDistribution(left, mode, right)
    return dist.quantile(p)


def rtriangle(left: float, mode: float, right: float, rng: Optional[RNG] = None) -> float:
    """Generate triangular random number."""
    dist = TriangularDistribution(left, mode, right)
    return dist.rvs(rng=rng)


# Export all
__all__ = [
    'TriangularDistribution', 'TruncatedNormal', 'InverseGamma', 'Dirichlet',
    'dtriangle', 'ptriangle', 'qtriangle', 'rtriangle'
]