"""Random Number Generator - wrapper around NumPy RandomState."""

from typing import Optional
import numpy as np


class RNG:
    """A random number generator that wraps NumPy RandomState with BOOM-like interface.
    
    This class provides a BOOM-compatible interface to NumPy's random number generation,
    mimicking the C++ RNG class behavior.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize RNG.
        
        Args:
            seed: Random seed. If None, uses system entropy.
        """
        if seed is None:
            # Use numpy's default random seed behavior
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(seed)
    
    def seed(self, seed: Optional[int] = None) -> int:
        """Seed the random number generator.
        
        Args:
            seed: Random seed. If None, generates a random seed.
            
        Returns:
            The seed that was used.
        """
        if seed is None:
            # Generate a random seed
            seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)
        
        self._rng.seed(seed)
        return int(seed)
    
    def __call__(self) -> float:
        """Generate a uniform random number in [0, 1).
        
        Returns:
            Random float in [0, 1).
        """
        return self._rng.uniform(0.0, 1.0)
    
    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate uniform random number.
        
        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            
        Returns:
            Random float in [low, high).
        """
        return self._rng.uniform(low, high)
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Generate normal random number.
        
        Args:
            mu: Mean.
            sigma: Standard deviation.
            
        Returns:
            Random float from normal distribution.
        """
        return self._rng.normal(mu, sigma)
    
    def gamma(self, shape: float, scale: float = 1.0) -> float:
        """Generate gamma random number.
        
        Args:
            shape: Shape parameter (alpha).
            scale: Scale parameter (1/beta).
            
        Returns:
            Random float from gamma distribution.
        """
        return self._rng.gamma(shape, scale)
    
    def beta(self, alpha: float, beta: float) -> float:
        """Generate beta random number.
        
        Args:
            alpha: First shape parameter.
            beta: Second shape parameter.
            
        Returns:
            Random float from beta distribution.
        """
        return self._rng.beta(alpha, beta)
    
    def binomial(self, n: int, p: float) -> int:
        """Generate binomial random number.
        
        Args:
            n: Number of trials.
            p: Probability of success.
            
        Returns:
            Random integer from binomial distribution.
        """
        return self._rng.binomial(n, p)
    
    def poisson(self, lam: float) -> int:
        """Generate Poisson random number.
        
        Args:
            lam: Rate parameter.
            
        Returns:
            Random integer from Poisson distribution.
        """
        return self._rng.poisson(lam)
    
    def exponential(self, scale: float = 1.0) -> float:
        """Generate exponential random number.
        
        Args:
            scale: Scale parameter (1/rate).
            
        Returns:
            Random float from exponential distribution.
        """
        return self._rng.exponential(scale)
    
    def chi_square(self, df: float) -> float:
        """Generate chi-square random number.
        
        Args:
            df: Degrees of freedom.
            
        Returns:
            Random float from chi-square distribution.
        """
        return self._rng.chisquare(df)
    
    def student_t(self, df: float) -> float:
        """Generate Student's t random number.
        
        Args:
            df: Degrees of freedom.
            
        Returns:
            Random float from Student's t distribution.
        """
        return self._rng.standard_t(df)
    
    def choice(self, a, size=None, replace=True, p=None):
        """Generate random sample from array.
        
        Args:
            a: Array-like or int (if int, equivalent to np.arange(a)).
            size: Output shape.
            replace: Whether to sample with replacement.
            p: Probabilities for each element.
            
        Returns:
            Random sample(s).
        """
        return self._rng.choice(a, size=size, replace=replace, p=p)
    
    def multinomial(self, n: int, pvals) -> np.ndarray:
        """Generate multinomial random sample.
        
        Args:
            n: Number of trials.
            pvals: Probabilities for each outcome.
            
        Returns:
            Array of counts for each outcome.
        """
        return self._rng.multinomial(n, pvals)
    
    def multivariate_normal(self, mean, cov) -> np.ndarray:
        """Generate multivariate normal random sample.
        
        Args:
            mean: Mean vector.
            cov: Covariance matrix.
            
        Returns:
            Random vector from multivariate normal distribution.
        """
        return self._rng.multivariate_normal(mean, cov)
    
    def get_state(self):
        """Get the internal state of the generator."""
        return self._rng.get_state()
    
    def set_state(self, state):
        """Set the internal state of the generator."""
        self._rng.set_state(state)
    
    @property
    def generator(self) -> np.random.RandomState:
        """Access to underlying NumPy RandomState."""
        return self._rng


class GlobalRng:
    """Global random number generator singleton.
    
    This mimics the C++ GlobalRng behavior by providing a single
    global RNG instance that can be accessed throughout the application.
    """
    
    # Class variable - single instance shared across all uses
    rng = RNG()
    
    @classmethod
    def seed(cls, seed: Optional[int] = None) -> int:
        """Seed the global RNG.
        
        Args:
            seed: Random seed. If None, generates a random seed.
            
        Returns:
            The seed that was used.
        """
        return cls.rng.seed(seed)
    
    @classmethod
    def get_rng(cls) -> RNG:
        """Get the global RNG instance."""
        return cls.rng
    
    @classmethod
    def set_rng(cls, rng: RNG):
        """Set a new global RNG instance."""
        cls.rng = rng


def seed_rng(rng: Optional[RNG] = None) -> int:
    """Generate a random seed and seed the given RNG.
    
    Args:
        rng: RNG to seed. If None, uses GlobalRng.rng.
        
    Returns:
        The seed that was used.
    """
    if rng is None:
        rng = GlobalRng.rng
    
    # Generate random seed from system entropy
    seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
    return rng.seed(seed)