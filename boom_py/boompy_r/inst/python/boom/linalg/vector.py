"""Vector implementation for BOOM."""
import numpy as np
from typing import Union, List, Optional, Any
import numbers


class Vector(np.ndarray):
    """Vector class that extends numpy array with BOOM-specific functionality."""
    
    def __new__(cls, data: Union[List[float], np.ndarray, int, float] = None):
        """Create a new Vector instance.
        
        Args:
            data: Can be a list, numpy array, scalar, or size (creates zero vector)
        """
        if data is None:
            obj = np.array([], dtype=np.float64).view(cls)
        elif isinstance(data, numbers.Integral) and not isinstance(data, bool):
            # If data is an integer, create a zero vector of that size
            obj = np.zeros(data, dtype=np.float64).view(cls)
        elif isinstance(data, numbers.Real):
            # If data is a single float, create a 1-element vector
            obj = np.array([data], dtype=np.float64).view(cls)
        else:
            # Otherwise, convert to array
            obj = np.asarray(data, dtype=np.float64).view(cls)
            if obj.ndim != 1:
                raise ValueError("Vector must be 1-dimensional")
        return obj
    
    def __array_finalize__(self, obj):
        """Called whenever the array is created."""
        pass
    
    @classmethod
    def zero(cls, size: int) -> 'Vector':
        """Create a zero vector of given size."""
        return cls(np.zeros(size))
    
    @classmethod
    def ones(cls, size: int) -> 'Vector':
        """Create a vector of ones of given size."""
        return cls(np.ones(size))
    
    @classmethod
    def seq(cls, from_: float, to: float, by: float = 1.0) -> 'Vector':
        """Create a sequence vector."""
        return cls(np.arange(from_, to + by/2, by))
    
    def inc(self) -> int:
        """Returns the stride of the vector (always 1 for contiguous arrays)."""
        return 1
    
    def is_all_finite(self) -> bool:
        """Check if all elements are finite (not nan or inf)."""
        return np.all(np.isfinite(self))
    
    def all_positive(self) -> bool:
        """Check if all elements are positive."""
        return np.all(self > 0)
    
    def all_non_negative(self) -> bool:
        """Check if all elements are non-negative."""
        return np.all(self >= 0)
    
    def min(self) -> float:
        """Return the minimum element."""
        if self.size == 0:
            return np.inf
        return float(np.ndarray.min(self))
    
    def max(self) -> float:
        """Return the maximum element."""
        if self.size == 0:
            return -np.inf
        return float(np.ndarray.max(self))
    
    def imax(self) -> int:
        """Return the index of the maximum element."""
        if self.size == 0:
            return -1
        return int(np.argmax(self))
    
    def imin(self) -> int:
        """Return the index of the minimum element."""
        if self.size == 0:
            return -1
        return int(np.argmin(self))
    
    def sum(self) -> float:
        """Return the sum of all elements."""
        return float(np.ndarray.sum(self))
    
    def prod(self) -> float:
        """Return the product of all elements."""
        return float(np.ndarray.prod(self))
    
    def mean(self) -> float:
        """Return the mean of all elements."""
        if self.size == 0:
            return np.nan
        return float(np.ndarray.mean(self))
    
    def var(self, ddof: int = 1) -> float:
        """Return the variance of all elements."""
        if self.size <= ddof:
            return np.nan
        return float(np.ndarray.var(self, ddof=ddof))
    
    def sd(self, ddof: int = 1) -> float:
        """Return the standard deviation of all elements."""
        return np.sqrt(self.var(ddof=ddof))
    
    def abs_norm(self) -> float:
        """Return the L1 norm (sum of absolute values)."""
        return float(np.ndarray.sum(np.abs(self)))
    
    def normsq(self) -> float:
        """Return the squared L2 norm."""
        return np.dot(self, self)
    
    def norm(self) -> float:
        """Return the L2 norm."""
        return np.linalg.norm(self)
    
    def normalize_prob(self) -> 'Vector':
        """Normalize to sum to 1 (for probability vectors)."""
        total = self.sum()
        if total == 0:
            raise ValueError("Cannot normalize a zero vector")
        return self / total
    
    def cumsum(self) -> 'Vector':
        """Return cumulative sum."""
        return Vector(np.cumsum(self))
    
    def sort(self) -> 'Vector':
        """Return a sorted copy of the vector."""
        return Vector(np.sort(self.view(np.ndarray)))
    
    def permute(self, indices: Union[List[int], np.ndarray]) -> 'Vector':
        """Return a permuted copy of the vector."""
        return Vector(self[indices])
    
    def dot(self, other: 'Vector') -> float:
        """Dot product with another vector."""
        return np.dot(self, other)
    
    def outer(self, other: 'Vector') -> np.ndarray:
        """Outer product with another vector."""
        return np.outer(self, other)
    
    # Arithmetic operations that return Vector type
    def __add__(self, other):
        result = np.add(self, other)
        if result.ndim == 1:
            return Vector(result)
        return result
    
    def __sub__(self, other):
        result = np.subtract(self, other)
        if result.ndim == 1:
            return Vector(result)
        return result
    
    def __mul__(self, other):
        result = np.multiply(self, other)
        if result.ndim == 1:
            return Vector(result)
        return result
    
    def __truediv__(self, other):
        result = np.divide(self, other)
        if result.ndim == 1:
            return Vector(result)
        return result
    
    def __neg__(self):
        return Vector(np.negative(self))
    
    def __repr__(self):
        return f"Vector({np.array_repr(self)})"