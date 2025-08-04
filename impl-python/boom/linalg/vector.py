"""Vector class - a wrapper around numpy arrays with BOOM-like interface."""

from typing import Union, List, Optional, Iterator, Tuple
import numpy as np
from numpy.typing import ArrayLike
import re


class Vector:
    """A vector class that wraps numpy arrays and provides BOOM Vector interface."""
    
    def __init__(self, data: Union[int, ArrayLike, str, 'Vector'] = 0, 
                 fill_value: float = 0.0, sep: Optional[str] = None):
        """Initialize a Vector.
        
        Args:
            data: Can be:
                - int: create a vector of that size filled with fill_value
                - array-like: create vector from the data
                - str: parse string as space/comma separated values
                - Vector: copy constructor
            fill_value: value to fill vector with if data is an int
            sep: separator for string parsing (if None, uses space/comma)
        """
        if isinstance(data, int):
            # Create vector of given size
            self.data = np.full(data, fill_value, dtype=np.float64)
        elif isinstance(data, str):
            # Parse string
            if sep is None:
                # Split by whitespace and/or commas
                values = re.split(r'[,\s]+', data.strip())
            else:
                values = data.split(sep)
            self.data = np.array([float(v) for v in values if v], dtype=np.float64)
        elif isinstance(data, Vector):
            # Copy constructor
            self.data = data.data.copy()
        else:
            # Create from array-like
            self.data = np.array(data, dtype=np.float64)
            if self.data.ndim != 1:
                raise ValueError("Vector must be 1-dimensional")
    
    # Size and access methods
    def __len__(self) -> int:
        """Return the length of the vector."""
        return len(self.data)
    
    def size(self) -> int:
        """Return the size of the vector."""
        return len(self.data)
    
    def length(self) -> int:
        """Return the length of the vector (same as size)."""
        return len(self.data)
    
    def stride(self) -> int:
        """Return the stride (always 1 for vectors)."""
        return 1
    
    def __getitem__(self, index: Union[int, slice]) -> Union[float, 'Vector']:
        """Get element(s) by index."""
        result = self.data[index]
        if isinstance(index, slice):
            return Vector(result)
        return float(result)
    
    def __setitem__(self, index: Union[int, slice], value: Union[float, ArrayLike]):
        """Set element(s) by index."""
        self.data[index] = value
    
    def __call__(self, index: int) -> float:
        """Get element by index (1-based for C++ compatibility)."""
        if not 0 <= index < len(self.data):
            raise IndexError(f"Index {index} out of range [0, {len(self.data)})")
        return float(self.data[index])
    
    # Comparison operators
    def __eq__(self, other: object) -> bool:
        """Check equality with another Vector."""
        if not isinstance(other, Vector):
            return False
        return np.array_equal(self.data, other.data)
    
    def all_finite(self) -> bool:
        """Check if all elements are finite."""
        return bool(np.all(np.isfinite(self.data)))
    
    # Initialization methods
    def set_to_zero(self):
        """Set all elements to zero."""
        self.data.fill(0.0)
    
    def zero(self) -> 'Vector':
        """Return a same-sized vector filled with zeros."""
        return Vector(np.zeros_like(self.data))
    
    def one(self) -> 'Vector':
        """Return a same-sized vector filled with ones."""
        return Vector(np.ones_like(self.data))
    
    def randomize(self, rng: Optional[np.random.RandomState] = None) -> 'Vector':
        """Fill with uniform random numbers in [0, 1)."""
        if rng is None:
            rng = np.random
        self.data = rng.uniform(0, 1, size=len(self.data))
        return self
    
    def randomize_gaussian(self, mean: float = 0.0, sd: float = 1.0, 
                          rng: Optional[np.random.RandomState] = None) -> 'Vector':
        """Fill with Gaussian random numbers."""
        if rng is None:
            rng = np.random
        self.data = rng.normal(mean, sd, size=len(self.data))
        return self
    
    # Arithmetic operators (in-place)
    def __iadd__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """In-place addition."""
        if isinstance(other, Vector):
            self.data += other.data
        else:
            self.data += other
        return self
    
    def __isub__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """In-place subtraction."""
        if isinstance(other, Vector):
            self.data -= other.data
        else:
            self.data -= other
        return self
    
    def __imul__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """In-place multiplication."""
        if isinstance(other, Vector):
            self.data *= other.data
        else:
            self.data *= other
        return self
    
    def __itruediv__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """In-place division."""
        if isinstance(other, Vector):
            self.data /= other.data
        else:
            self.data /= other
        return self
    
    # Arithmetic operators (new objects)
    def __add__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """Addition."""
        result = Vector(self)
        result += other
        return result
    
    def __radd__(self, other: Union[float, ArrayLike]) -> 'Vector':
        """Right addition."""
        return self + other
    
    def __sub__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """Subtraction."""
        result = Vector(self)
        result -= other
        return result
    
    def __rsub__(self, other: Union[float, ArrayLike]) -> 'Vector':
        """Right subtraction."""
        return Vector(other) - self
    
    def __mul__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """Multiplication."""
        result = Vector(self)
        result *= other
        return result
    
    def __rmul__(self, other: Union[float, ArrayLike]) -> 'Vector':
        """Right multiplication."""
        return self * other
    
    def __truediv__(self, other: Union[float, 'Vector', ArrayLike]) -> 'Vector':
        """Division."""
        result = Vector(self)
        result /= other
        return result
    
    def __rtruediv__(self, other: Union[float, ArrayLike]) -> 'Vector':
        """Right division."""
        return Vector(other) / self
    
    def __neg__(self) -> 'Vector':
        """Negation."""
        return Vector(-self.data)
    
    # Linear algebra operations
    def axpy(self, x: 'Vector', w: float) -> 'Vector':
        """Add w*x to this vector (self += w*x)."""
        self.data += w * x.data
        return self
    
    def dot(self, y: 'Vector') -> float:
        """Compute dot product with another vector."""
        return float(np.dot(self.data, y.data))
    
    def affdot(self, y: 'Vector') -> float:
        """Affine dot product: dim(y) == dim(self)-1."""
        if len(y) != len(self) - 1:
            raise ValueError(f"Affine dot requires dim(y)={len(self)-1}, got {len(y)}")
        return float(self.data[0] + np.dot(self.data[1:], y.data))
    
    def outer(self, y: Optional['Vector'] = None, a: float = 1.0) -> np.ndarray:
        """Compute outer product.
        
        If y is None, returns x @ x.T
        Otherwise returns a * x @ y.T
        """
        if y is None:
            return np.outer(self.data, self.data)
        return a * np.outer(self.data, y.data)
    
    # Normalization methods
    def normalize_prob(self) -> 'Vector':
        """Normalize to sum to 1 (probability normalization)."""
        total = self.sum()
        if total != 0:
            self.data /= total
        return self
    
    def normalize_logprob(self) -> 'Vector':
        """Normalize log probabilities using log-sum-exp trick."""
        max_val = self.max()
        exp_vals = np.exp(self.data - max_val)
        self.data = exp_vals / np.sum(exp_vals)
        return self
    
    def normalize_L2(self) -> 'Vector':
        """Normalize to unit L2 norm."""
        norm = self.norm()
        if norm != 0:
            self.data /= norm
        return self
    
    # Norms and statistics
    def normsq(self) -> float:
        """Return squared L2 norm."""
        return float(np.dot(self.data, self.data))
    
    def norm(self) -> float:
        """Return L2 norm."""
        return float(np.linalg.norm(self.data))
    
    def abs_norm(self) -> float:
        """Return L1 norm (sum of absolute values)."""
        return float(np.sum(np.abs(self.data)))
    
    def max_abs(self) -> float:
        """Return maximum absolute value (-1 if empty)."""
        if len(self.data) == 0:
            return -1.0
        return float(np.max(np.abs(self.data)))
    
    def min(self) -> float:
        """Return minimum value."""
        return float(np.min(self.data))
    
    def max(self) -> float:
        """Return maximum value."""
        return float(np.max(self.data))
    
    def imax(self) -> int:
        """Return index of maximum element."""
        return int(np.argmax(self.data))
    
    def imin(self) -> int:
        """Return index of minimum element."""
        return int(np.argmin(self.data))
    
    def sum(self) -> float:
        """Return sum of elements."""
        return float(np.sum(self.data))
    
    def prod(self) -> float:
        """Return product of elements."""
        return float(np.prod(self.data))
    
    # Modification methods
    def sort(self) -> 'Vector':
        """Sort elements in ascending order (in-place)."""
        self.data.sort()
        return self
    
    def concat(self, v: Union['Vector', ArrayLike]) -> 'Vector':
        """Concatenate another vector to this one."""
        if isinstance(v, Vector):
            self.data = np.concatenate([self.data, v.data])
        else:
            self.data = np.concatenate([self.data, np.array(v)])
        return self
    
    def push_back(self, x: float) -> 'Vector':
        """Append a value to the end."""
        self.data = np.append(self.data, x)
        return self
    
    # String representation
    def __str__(self) -> str:
        """String representation."""
        return str(self.data)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Vector({self.data})"
    
    # Iterator support
    def __iter__(self) -> Iterator[float]:
        """Iterate over elements."""
        return iter(self.data)
    
    # Conversion methods
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()
    
    @property
    def T(self) -> np.ndarray:
        """Return transpose (for compatibility)."""
        return self.data
    
    # Utility methods
    def copy(self) -> 'Vector':
        """Create a copy of this vector."""
        return Vector(self)
    
    def swap(self, other: 'Vector') -> 'Vector':
        """Swap contents with another vector."""
        self.data, other.data = other.data, self.data
        return self