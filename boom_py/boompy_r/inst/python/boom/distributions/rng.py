"""Random number generation for BOOM."""
import numpy as np
from typing import Optional, Union, Tuple
from scipy import stats


class GlobalRng:
    """Global random number generator matching BOOM's RNG interface."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the RNG.
        
        Args:
            seed: Random seed. If None, uses numpy's default initialization.
        """
        self._rng = np.random.RandomState(seed)
    
    def seed(self, seed: Optional[int] = None):
        """Set the random seed.
        
        Args:
            seed: Random seed. If None, uses system time or other entropy source.
        """
        if seed is None:
            # Use numpy's entropy-based seed
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(seed)
    
    # Uniform distribution
    def runif(self, lo: float = 0.0, hi: float = 1.0) -> float:
        """Generate uniform random number."""
        return self._rng.uniform(lo, hi)
    
    def runif_vec(self, n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
        """Generate vector of uniform random numbers."""
        return self._rng.uniform(lo, hi, size=n)
    
    # Normal distribution  
    def rnorm(self, mean: float = 0.0, sd: float = 1.0) -> float:
        """Generate normal random number."""
        return self._rng.normal(mean, sd)
    
    def rnorm_vec(self, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        """Generate vector of normal random numbers."""
        return self._rng.normal(mean, sd, size=n)
    
    def rnorm_mt(self, mean: float, var: float) -> float:
        """Generate normal random number (mean/variance parameterization)."""
        return self._rng.normal(mean, np.sqrt(var))
    
    # Exponential distribution
    def rexp(self, lam: float = 1.0) -> float:
        """Generate exponential random number."""
        return self._rng.exponential(1.0 / lam)
    
    def rexp_vec(self, n: int, lam: float = 1.0) -> np.ndarray:
        """Generate vector of exponential random numbers."""
        return self._rng.exponential(1.0 / lam, size=n)
    
    # Gamma distribution
    def rgamma(self, shape: float, scale: float = 1.0) -> float:
        """Generate gamma random number."""
        return self._rng.gamma(shape, scale)
    
    def rgamma_vec(self, n: int, shape: float, scale: float = 1.0) -> np.ndarray:
        """Generate vector of gamma random numbers."""
        return self._rng.gamma(shape, scale, size=n)
    
    # Beta distribution
    def rbeta(self, a: float, b: float) -> float:
        """Generate beta random number."""
        return self._rng.beta(a, b)
    
    def rbeta_vec(self, n: int, a: float, b: float) -> np.ndarray:
        """Generate vector of beta random numbers."""
        return self._rng.beta(a, b, size=n)
    
    # Chi-square distribution
    def rchisq(self, df: float) -> float:
        """Generate chi-square random number."""
        return self._rng.chisquare(df)
    
    def rchisq_vec(self, n: int, df: float) -> np.ndarray:
        """Generate vector of chi-square random numbers."""
        return self._rng.chisquare(df, size=n)
    
    # Binomial distribution
    def rbinom(self, n: int, p: float) -> int:
        """Generate binomial random number."""
        return int(self._rng.binomial(n, p))
    
    def rbinom_vec(self, size: int, n: int, p: float) -> np.ndarray:
        """Generate vector of binomial random numbers."""
        return self._rng.binomial(n, p, size=size)
    
    # Poisson distribution
    def rpois(self, lam: float) -> int:
        """Generate Poisson random number."""
        return int(self._rng.poisson(lam))
    
    def rpois_vec(self, n: int, lam: float) -> np.ndarray:
        """Generate vector of Poisson random numbers."""
        return self._rng.poisson(lam, size=n)
    
    # Multinomial distribution
    def rmulti(self, n: int, probs: np.ndarray) -> np.ndarray:
        """Generate multinomial random vector."""
        return self._rng.multinomial(n, probs)
    
    # Student t distribution
    def rt(self, df: float) -> float:
        """Generate Student t random number."""
        return float(self._rng.standard_t(df))
    
    def rt_vec(self, n: int, df: float) -> np.ndarray:
        """Generate vector of Student t random numbers."""
        return self._rng.standard_t(df, size=n)
    
    # F distribution
    def rf(self, df1: float, df2: float) -> float:
        """Generate F random number."""
        return self._rng.f(df1, df2)
    
    # Multivariate normal
    def rmvn(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate multivariate normal random vector."""
        return self._rng.multivariate_normal(mean, cov)
    
    def rmvn_L(self, mean: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Generate multivariate normal using Cholesky factor L where Cov = LL'."""
        z = self._rng.standard_normal(size=len(mean))
        return mean + L @ z
    
    # Dirichlet distribution
    def rdirichlet(self, alpha: np.ndarray) -> np.ndarray:
        """Generate Dirichlet random vector."""
        return self._rng.dirichlet(alpha)
    
    # Wishart distribution
    def rwish(self, df: int, scale: np.ndarray) -> np.ndarray:
        """Generate Wishart random matrix."""
        # Use Bartlett decomposition
        n = scale.shape[0]
        L = np.linalg.cholesky(scale)
        A = np.zeros((n, n))
        
        # Fill diagonal with chi-square variates
        for i in range(n):
            A[i, i] = np.sqrt(self.rchisq(df - i))
        
        # Fill lower triangle with standard normals
        for i in range(1, n):
            for j in range(i):
                A[i, j] = self.rnorm()
        
        # W = L * A * A' * L'
        LA = L @ A
        return LA @ LA.T
    
    # Utility functions
    def random_sample(self, population: Union[list, np.ndarray], k: int, 
                     replace: bool = False) -> np.ndarray:
        """Random sample from population."""
        return self._rng.choice(population, size=k, replace=replace)
    
    def shuffle(self, x: Union[list, np.ndarray]) -> None:
        """Shuffle array in-place."""
        self._rng.shuffle(x)
    
    def permutation(self, n: Union[int, np.ndarray]) -> np.ndarray:
        """Random permutation."""
        return self._rng.permutation(n)


# Module-level function to seed the global RNG
def seed_rng(seed: Optional[int] = None):
    """Seed the global random number generator.
    
    Args:
        seed: Random seed. If None, uses system entropy.
    """
    from . import rng
    rng.seed(seed)