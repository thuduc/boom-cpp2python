"""Discrete probability distributions for BOOM."""
import numpy as np
from scipy import stats
from typing import Union, Optional
from ..math.special_functions import lgamma, lchoose


class DiscreteDistribution:
    """Base class for discrete probability distributions."""
    
    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability mass function."""
        raise NotImplementedError
    
    def logpmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Log probability mass function."""
        return np.log(self.pmf(k))
    
    def cdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        raise NotImplementedError
    
    def quantile(self, p: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Quantile function (inverse CDF)."""
        raise NotImplementedError
    
    def mean(self) -> float:
        """Mean of the distribution."""
        raise NotImplementedError
    
    def variance(self) -> float:
        """Variance of the distribution."""
        raise NotImplementedError
    
    def sample(self, size: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        """Sample from the distribution."""
        raise NotImplementedError


class Binomial(DiscreteDistribution):
    """Binomial distribution."""
    
    def __init__(self, n: int, p: float):
        """Initialize Binomial distribution.
        
        Args:
            n: Number of trials (must be non-negative integer)
            p: Success probability (must be in [0, 1])
        """
        if n < 0 or not isinstance(n, (int, np.integer)):
            raise ValueError("n must be non-negative integer")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.n = n
        self.p = p
    
    def pmf(self, k):
        return stats.binom.pmf(k, n=self.n, p=self.p)
    
    def logpmf(self, k):
        return stats.binom.logpmf(k, n=self.n, p=self.p)
    
    def cdf(self, k):
        return stats.binom.cdf(k, n=self.n, p=self.p)
    
    def quantile(self, p):
        return stats.binom.ppf(p, n=self.n, p=self.p)
    
    def mean(self):
        return self.n * self.p
    
    def variance(self):
        return self.n * self.p * (1 - self.p)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.binomial(self.n, self.p, size=size)


class Poisson(DiscreteDistribution):
    """Poisson distribution."""
    
    def __init__(self, lam: float):
        """Initialize Poisson distribution.
        
        Args:
            lam: Rate parameter (lambda, must be positive)
        """
        if lam <= 0:
            raise ValueError("Lambda must be positive")
        self.lam = lam
    
    def pmf(self, k):
        return stats.poisson.pmf(k, mu=self.lam)
    
    def logpmf(self, k):
        return stats.poisson.logpmf(k, mu=self.lam)
    
    def cdf(self, k):
        return stats.poisson.cdf(k, mu=self.lam)
    
    def quantile(self, p):
        return stats.poisson.ppf(p, mu=self.lam)
    
    def mean(self):
        return self.lam
    
    def variance(self):
        return self.lam
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.poisson(self.lam, size=size)


class NegativeBinomial(DiscreteDistribution):
    """Negative Binomial distribution (number of failures before r successes)."""
    
    def __init__(self, r: float, p: float):
        """Initialize Negative Binomial distribution.
        
        Args:
            r: Number of successes (must be positive)
            p: Success probability (must be in (0, 1])
        """
        if r <= 0:
            raise ValueError("r must be positive")
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.r = r
        self.p = p
    
    def pmf(self, k):
        return stats.nbinom.pmf(k, n=self.r, p=self.p)
    
    def logpmf(self, k):
        return stats.nbinom.logpmf(k, n=self.r, p=self.p)
    
    def cdf(self, k):
        return stats.nbinom.cdf(k, n=self.r, p=self.p)
    
    def quantile(self, p):
        return stats.nbinom.ppf(p, n=self.r, p=self.p)
    
    def mean(self):
        return self.r * (1 - self.p) / self.p
    
    def variance(self):
        return self.r * (1 - self.p) / (self.p ** 2)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.negative_binomial(self.r, self.p, size=size)


class Geometric(DiscreteDistribution):
    """Geometric distribution (number of failures before first success)."""
    
    def __init__(self, p: float):
        """Initialize Geometric distribution.
        
        Args:
            p: Success probability (must be in (0, 1])
        """
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        self.p = p
    
    def pmf(self, k):
        return stats.geom.pmf(k, p=self.p)
    
    def logpmf(self, k):
        return stats.geom.logpmf(k, p=self.p)
    
    def cdf(self, k):
        return stats.geom.cdf(k, p=self.p)
    
    def quantile(self, p):
        return stats.geom.ppf(p, p=self.p)
    
    def mean(self):
        return (1 - self.p) / self.p
    
    def variance(self):
        return (1 - self.p) / (self.p ** 2)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        # np.random.geometric returns trials until first success (including the success)
        # scipy.stats.geom uses same parameterization
        return rng.geometric(self.p, size=size) - 1  # Convert to failures before success


class Hypergeometric(DiscreteDistribution):
    """Hypergeometric distribution."""
    
    def __init__(self, N: int, K: int, n: int):
        """Initialize Hypergeometric distribution.
        
        Args:
            N: Population size (must be positive integer)
            K: Number of successes in population (must be in [0, N])
            n: Number of draws (must be in [0, N])
        """
        if N <= 0 or not isinstance(N, (int, np.integer)):
            raise ValueError("N must be positive integer")
        if not (0 <= K <= N) or not isinstance(K, (int, np.integer)):
            raise ValueError("K must be integer in [0, N]")
        if not (0 <= n <= N) or not isinstance(n, (int, np.integer)):
            raise ValueError("n must be integer in [0, N]")
        self.N = N
        self.K = K
        self.n = n
    
    def pmf(self, k):
        return stats.hypergeom.pmf(k, M=self.N, n=self.K, N=self.n)
    
    def logpmf(self, k):
        return stats.hypergeom.logpmf(k, M=self.N, n=self.K, N=self.n)
    
    def cdf(self, k):
        return stats.hypergeom.cdf(k, M=self.N, n=self.K, N=self.n)
    
    def quantile(self, p):
        return stats.hypergeom.ppf(p, M=self.N, n=self.K, N=self.n)
    
    def mean(self):
        return self.n * self.K / self.N
    
    def variance(self):
        p = self.K / self.N
        return self.n * p * (1 - p) * (self.N - self.n) / (self.N - 1)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.hypergeometric(self.K, self.N - self.K, self.n, size=size)


class DiscreteUniform(DiscreteDistribution):
    """Discrete uniform distribution."""
    
    def __init__(self, low: int, high: int):
        """Initialize Discrete Uniform distribution.
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive, must be >= low)
        """
        if not isinstance(low, (int, np.integer)) or not isinstance(high, (int, np.integer)):
            raise ValueError("Bounds must be integers")
        if high < low:
            raise ValueError("Upper bound must be >= lower bound")
        self.low = low
        self.high = high
        self.n_values = high - low + 1
    
    def pmf(self, k):
        k = np.asarray(k)
        return np.where((k >= self.low) & (k <= self.high), 
                       1.0 / self.n_values, 0.0)
    
    def logpmf(self, k):
        k = np.asarray(k)
        return np.where((k >= self.low) & (k <= self.high), 
                       -np.log(self.n_values), -np.inf)
    
    def cdf(self, k):
        k = np.asarray(k)
        return np.where(k < self.low, 0.0,
                       np.where(k >= self.high, 1.0,
                               (k - self.low + 1) / self.n_values))
    
    def quantile(self, p):
        p = np.asarray(p)
        return np.floor(self.low + p * self.n_values).astype(int)
    
    def mean(self):
        return (self.low + self.high) / 2
    
    def variance(self):
        return (self.n_values ** 2 - 1) / 12
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.randint(self.low, self.high + 1, size=size)