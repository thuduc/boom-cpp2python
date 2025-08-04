"""Sufficient statistics classes for BOOM models."""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
import numpy as np
from ..linalg import Vector, Matrix
from .base import Data


class Sufstat(Data):
    """Abstract base class for sufficient statistics.
    
    Sufficient statistics inherit from Data because they can be viewed as
    data for hierarchical models, and because of the duality between
    sufficient statistics and model parameters.
    """
    
    def __init__(self):
        """Initialize sufficient statistics."""
        super().__init__()
    
    @abstractmethod
    def clear(self):
        """Reset sufficient statistics to initial state."""
        pass
    
    @abstractmethod
    def update(self, data: Union[Data, List[Data]]):
        """Update sufficient statistics with new data.
        
        Args:
            data: Data point(s) to incorporate into sufficient statistics.
        """
        pass
    
    @abstractmethod
    def combine(self, other: 'Sufstat') -> 'Sufstat':
        """Combine with another sufficient statistic.
        
        Args:
            other: Another sufficient statistic of the same type.
            
        Returns:
            Combined sufficient statistic.
        """
        pass
    
    @abstractmethod
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert sufficient statistics to vector form.
        
        Args:
            minimal: If True, use minimal representation.
            
        Returns:
            Sufficient statistics as vector.
        """
        pass
    
    @abstractmethod
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set sufficient statistics from vector form.
        
        Args:
            v: Vector containing sufficient statistics.
            minimal: If True, interpret as minimal representation.
        """
        pass
    
    @abstractmethod
    def clone(self) -> 'Sufstat':
        """Create deep copy of sufficient statistics."""
        pass


class GaussianSuf(Sufstat):
    """Sufficient statistics for Gaussian (normal) distribution.
    
    Sufficient statistics are:
    - n: sample size
    - sum: sum of observations
    - sumsq: sum of squared observations (or sum of x*x.T for multivariate)
    """
    
    def __init__(self):
        """Initialize Gaussian sufficient statistics."""
        super().__init__()
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        self._sum = 0.0
        self._sumsq = 0.0
    
    def update(self, data: Union[Data, List[Data], float, List[float]]):
        """Update with new data.
        
        Args:
            data: Data to incorporate. Can be:
                - Single float value
                - List of float values 
                - Data object with value() method
                - List of Data objects
        """
        if isinstance(data, list):
            for item in data:
                self.update(item)
        elif isinstance(data, (int, float)):
            x = float(data)
            self._n += 1
            self._sum += x
            self._sumsq += x * x
        elif hasattr(data, 'value'):
            # Assume it's a Data object with value() method
            self.update(data.value())
        else:
            raise TypeError(f"Cannot update GaussianSuf with data of type {type(data)}")
    
    def combine(self, other: 'GaussianSuf') -> 'GaussianSuf':
        """Combine with another GaussianSuf."""
        if not isinstance(other, GaussianSuf):
            raise TypeError("Can only combine with another GaussianSuf")
        
        result = GaussianSuf()
        result._n = self._n + other._n
        result._sum = self._sum + other._sum
        result._sumsq = self._sumsq + other._sumsq
        return result
    
    def n(self) -> int:
        """Get sample size."""
        return self._n
    
    def sum(self) -> float:
        """Get sum of observations."""
        return self._sum
    
    def sumsq(self) -> float:
        """Get sum of squared observations.""" 
        return self._sumsq
    
    def mean(self) -> float:
        """Get sample mean."""
        if self._n == 0:
            return 0.0
        return self._sum / self._n
    
    def variance(self, sample: bool = True) -> float:
        """Get sample variance.
        
        Args:
            sample: If True, use sample variance (divide by n-1).
                   If False, use population variance (divide by n).
        """
        if self._n <= (1 if sample else 0):
            return 0.0
        
        mean_val = self.mean()
        sum_sq_deviations = self._sumsq - self._n * mean_val * mean_val
        
        divisor = self._n - 1 if sample else self._n
        return sum_sq_deviations / divisor
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector: [n, sum, sumsq]."""
        return Vector([float(self._n), self._sum, self._sumsq])
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector [n, sum, sumsq]."""
        if len(v) != 3:
            raise ValueError("Vector must have exactly 3 elements for GaussianSuf")
        
        self._n = int(v[0])
        self._sum = v[1]
        self._sumsq = v[2]
    
    def clone(self) -> 'GaussianSuf':
        """Create copy."""
        result = GaussianSuf()
        result._n = self._n
        result._sum = self._sum
        result._sumsq = self._sumsq
        return result
    
    def __str__(self) -> str:
        """String representation."""
        return f"GaussianSuf(n={self._n}, mean={self.mean():.3f}, var={self.variance():.3f})"


class MultivariateGaussianSuf(Sufstat):
    """Sufficient statistics for multivariate Gaussian distribution.
    
    Sufficient statistics are:
    - n: sample size
    - sum: sum vector
    - sumsq: sum of outer products (x * x.T)
    """
    
    def __init__(self, dim: int = 0):
        """Initialize multivariate Gaussian sufficient statistics.
        
        Args:
            dim: Dimension of the observations. If 0, dimension will be
                inferred from first data point.
        """
        super().__init__()
        self._dim = dim
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        if self._dim > 0:
            self._sum = Vector(self._dim, 0.0)
            self._sumsq = Matrix((self._dim, self._dim), fill_value=0.0)
        else:
            self._sum = None
            self._sumsq = None
    
    def update(self, data: Union[Data, List[Data], Vector, List[float], np.ndarray]):
        """Update with new data.
        
        Args:
            data: Data to incorporate. Can be:
                - Vector
                - List of numbers
                - numpy array
                - Data object with value() method
                - List of any of the above
        """
        if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], (int, float)):
            # List of data objects/vectors
            for item in data:
                self.update(item)
        else:
            # Single data point
            if isinstance(data, Vector):
                x = data
            elif isinstance(data, (list, np.ndarray)):
                x = Vector(data)
            elif hasattr(data, 'value'):
                x = Vector(data.value())
            else:
                raise TypeError(f"Cannot update MultivariateGaussianSuf with data of type {type(data)}")
            
            # Initialize dimension if needed
            if self._dim == 0:
                self._dim = len(x)
                self._sum = Vector(self._dim, 0.0)
                self._sumsq = Matrix((self._dim, self._dim), fill_value=0.0)
            elif len(x) != self._dim:
                raise ValueError(f"Data dimension {len(x)} doesn't match expected {self._dim}")
            
            # Update statistics
            self._n += 1
            self._sum += x
            self._sumsq.add_outer(x, x)  # Add x * x.T
    
    def combine(self, other: 'MultivariateGaussianSuf') -> 'MultivariateGaussianSuf':
        """Combine with another MultivariateGaussianSuf."""
        if not isinstance(other, MultivariateGaussianSuf):
            raise TypeError("Can only combine with another MultivariateGaussianSuf")
        
        if self._dim != other._dim:
            raise ValueError("Cannot combine sufficient statistics of different dimensions")
        
        result = MultivariateGaussianSuf(self._dim)
        result._n = self._n + other._n
        
        if self._n > 0 and other._n > 0:
            result._sum = self._sum + other._sum
            result._sumsq = self._sumsq + other._sumsq
        elif self._n > 0:
            result._sum = self._sum.copy()
            result._sumsq = self._sumsq.copy()
        elif other._n > 0:
            result._sum = other._sum.copy()
            result._sumsq = other._sumsq.copy()
        
        return result
    
    def n(self) -> int:
        """Get sample size."""
        return self._n
    
    def dim(self) -> int:
        """Get dimension."""
        return self._dim
    
    def sum(self) -> Optional[Vector]:
        """Get sum vector."""
        if self._n == 0:
            return None
        return self._sum.copy() if self._sum is not None else None
    
    def sumsq(self) -> Optional[Matrix]:
        """Get sum of outer products matrix."""
        if self._n == 0:
            return None
        return self._sumsq.copy() if self._sumsq is not None else None
    
    def mean(self) -> Optional[Vector]:
        """Get sample mean vector."""
        if self._n == 0 or self._sum is None:
            return None
        return self._sum / self._n
    
    def covariance(self, sample: bool = True) -> Optional[Matrix]:
        """Get sample covariance matrix.
        
        Args:
            sample: If True, use sample covariance (divide by n-1).
                   If False, use population covariance (divide by n).
        """
        if self._n <= (1 if sample else 0) or self._sum is None or self._sumsq is None:
            return None
        
        mean_vec = self.mean()
        # Covariance = (1/divisor) * (sumsq - n * mean * mean.T)
        mean_outer_array = mean_vec.outer()  # Returns numpy array
        mean_outer = Matrix(mean_outer_array)
        cov = self._sumsq - (self._n * mean_outer)
        
        divisor = self._n - 1 if sample else self._n
        return cov / divisor
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector."""
        if self._sum is None or self._sumsq is None:
            return Vector([float(self._n)])
        
        result = [float(self._n)]
        result.extend(self._sum.to_numpy())
        
        # Vectorize sumsq matrix
        if minimal:
            # Store only upper triangle of symmetric matrix
            for i in range(self._dim):
                for j in range(i, self._dim):
                    result.append(self._sumsq(i, j))
        else:
            # Store full matrix
            sumsq_vec = self._sumsq.vectorize()
            result.extend(sumsq_vec.to_numpy())
        
        return Vector(result)
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector."""
        if len(v) < 1:
            raise ValueError("Vector too short")
        
        self._n = int(v[0])
        
        if self._n == 0:
            self.clear()
            return
        
        if self._dim == 0:
            raise ValueError("Cannot unvectorize without knowing dimension")
        
        # Extract sum vector
        sum_start = 1
        sum_end = sum_start + self._dim
        if len(v) < sum_end:
            raise ValueError("Vector too short for sum vector")
        
        self._sum = Vector(v[sum_start:sum_end])
        
        # Extract sumsq matrix
        if minimal:
            # Upper triangle format
            expected_sumsq_size = self._dim * (self._dim + 1) // 2
            if len(v) < sum_end + expected_sumsq_size:
                raise ValueError("Vector too short for sumsq matrix")
            
            self._sumsq = Matrix((self._dim, self._dim), fill_value=0.0)
            idx = sum_end
            for i in range(self._dim):
                for j in range(i, self._dim):
                    val = v[idx]
                    self._sumsq[i, j] = val
                    if i != j:
                        self._sumsq[j, i] = val
                    idx += 1
        else:
            # Full matrix format
            expected_sumsq_size = self._dim * self._dim
            if len(v) < sum_end + expected_sumsq_size:
                raise ValueError("Vector too short for sumsq matrix")
            
            sumsq_vec = Vector(v[sum_end:sum_end + expected_sumsq_size])
            self._sumsq = Matrix((self._dim, self._dim))
            self._sumsq.unvectorize(sumsq_vec)
    
    def clone(self) -> 'MultivariateGaussianSuf':
        """Create copy."""
        result = MultivariateGaussianSuf(self._dim)
        result._n = self._n
        
        if self._sum is not None:
            result._sum = self._sum.copy()
        if self._sumsq is not None:
            result._sumsq = self._sumsq.copy()
        
        return result
    
    def __str__(self) -> str:
        """String representation."""
        mean = self.mean()
        mean_str = f"[{', '.join(f'{x:.3f}' for x in mean)}]" if mean else "None"
        return f"MultivariateGaussianSuf(n={self._n}, dim={self._dim}, mean={mean_str})"


class BinomialSuf(Sufstat):
    """Sufficient statistics for binomial distribution.
    
    Sufficient statistics are:
    - n: number of trials
    - successes: number of successes
    """
    
    def __init__(self):
        """Initialize binomial sufficient statistics."""
        super().__init__()
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        self._successes = 0
    
    def update(self, data: Union[Data, tuple, dict]):
        """Update with new data.
        
        Args:
            data: Data in one of these formats:
                - tuple: (n_trials, n_successes)
                - dict: {'trials': n, 'successes': k}
                - Data object with trials() and successes() methods
        """
        if isinstance(data, tuple) and len(data) == 2:
            trials, successes = data
            self._n += trials
            self._successes += successes
        elif isinstance(data, dict):
            trials = data.get('trials', 0)
            successes = data.get('successes', 0)
            self._n += trials
            self._successes += successes
        elif hasattr(data, 'trials') and hasattr(data, 'successes'):
            self._n += data.trials()
            self._successes += data.successes()
        else:
            raise TypeError(f"Cannot update BinomialSuf with data of type {type(data)}")
    
    def combine(self, other: 'BinomialSuf') -> 'BinomialSuf':
        """Combine with another BinomialSuf."""
        if not isinstance(other, BinomialSuf):
            raise TypeError("Can only combine with another BinomialSuf")
        
        result = BinomialSuf()
        result._n = self._n + other._n
        result._successes = self._successes + other._successes
        return result
    
    def n(self) -> int:
        """Get total number of trials."""
        return self._n
    
    def successes(self) -> int:
        """Get total number of successes."""
        return self._successes
    
    def failures(self) -> int:
        """Get total number of failures."""
        return self._n - self._successes
    
    def success_rate(self) -> float:
        """Get empirical success rate."""
        if self._n == 0:
            return 0.0
        return self._successes / self._n
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector: [n, successes]."""
        return Vector([float(self._n), float(self._successes)])
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector [n, successes]."""
        if len(v) != 2:
            raise ValueError("Vector must have exactly 2 elements for BinomialSuf")
        
        self._n = int(v[0])
        self._successes = int(v[1])
    
    def clone(self) -> 'BinomialSuf':
        """Create copy."""
        result = BinomialSuf()
        result._n = self._n
        result._successes = self._successes
        return result
    
    def __str__(self) -> str:
        """String representation."""
        rate = self.success_rate()
        return f"BinomialSuf(n={self._n}, successes={self._successes}, rate={rate:.3f})"