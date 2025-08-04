"""Parameter classes for BOOM models."""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple, Any
import numpy as np
from ..linalg import Vector, Matrix
from .base import Data


class Params(Data):
    """Abstract base class for model parameters.
    
    Parameters in BOOM inherit from Data so that parameters from one level
    of a hierarchical model can be viewed as data for the next level.
    """
    
    def __init__(self):
        """Initialize parameter object."""
        super().__init__()
    
    @abstractmethod
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert parameter to vector representation.
        
        Args:
            minimal: If True, use minimal representation (e.g., upper triangle
                    for symmetric matrices). If False, use full representation.
                    
        Returns:
            Parameter as vector.
        """
        pass
    
    @abstractmethod
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set parameter from vector representation.
        
        Args:
            v: Vector containing parameter values.
            minimal: If True, interpret as minimal representation.
        """
        pass
    
    @abstractmethod
    def size(self, minimal: bool = True) -> int:
        """Get size of parameter when vectorized.
        
        Args:
            minimal: If True, return minimal representation size.
            
        Returns:
            Number of elements in vectorized form.
        """
        pass
    
    @abstractmethod
    def clone(self) -> 'Params':
        """Create deep copy of parameter."""
        pass


class UnivParams(Params):
    """Univariate parameter class.
    
    This represents a single scalar parameter.
    """
    
    def __init__(self, value: float = 0.0):
        """Initialize univariate parameter.
        
        Args:
            value: Initial parameter value.
        """
        super().__init__()
        self._value = float(value)
    
    def value(self) -> float:
        """Get parameter value."""
        return self._value
    
    def set_value(self, value: float):
        """Set parameter value."""
        self._value = float(value)
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector (single element)."""
        return Vector([self._value])
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector."""
        if len(v) != 1:
            raise ValueError("Vector must have exactly one element")
        self._value = float(v[0])
    
    def size(self, minimal: bool = True) -> int:
        """Size is always 1 for univariate parameter."""
        return 1
    
    def clone(self) -> 'UnivParams':
        """Create copy."""
        result = UnivParams(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __float__(self) -> float:
        """Convert to float."""
        return self._value
    
    def __str__(self) -> str:
        """String representation."""
        return f"UnivParams({self._value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return str(self)


class VectorParams(Params):
    """Vector parameter class.
    
    This represents a parameter that is a vector.
    """
    
    def __init__(self, value: Union[Vector, List[float], np.ndarray] = None, dim: int = 0):
        """Initialize vector parameter.
        
        Args:
            value: Initial parameter value.
            dim: Dimension if value is None.
        """
        super().__init__()
        
        if value is None:
            if dim <= 0:
                raise ValueError("Must specify positive dimension if value is None")
            self._value = Vector(dim, 0.0)
        elif isinstance(value, Vector):
            self._value = value.copy()
        else:
            self._value = Vector(value)
    
    def value(self) -> Vector:
        """Get parameter value."""
        return self._value.copy()
    
    def set_value(self, value: Union[Vector, List[float], np.ndarray]):
        """Set parameter value."""
        if isinstance(value, Vector):
            self._value = value.copy()
        else:
            self._value = Vector(value)
    
    def dim(self) -> int:
        """Get dimension of vector."""
        return len(self._value)
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector (returns copy of internal vector)."""
        return self._value.copy()
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector."""
        self._value = v.copy()
    
    def size(self, minimal: bool = True) -> int:
        """Size equals vector dimension."""
        return len(self._value)
    
    def clone(self) -> 'VectorParams':
        """Create copy."""
        result = VectorParams(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __len__(self) -> int:
        """Length of vector."""
        return len(self._value)
    
    def __getitem__(self, index: int) -> float:
        """Get element."""
        return self._value[index]
    
    def __setitem__(self, index: int, value: float):
        """Set element."""
        self._value[index] = value
    
    def __str__(self) -> str:
        """String representation."""
        return f"VectorParams({self._value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return str(self)


class MatrixParams(Params):
    """Matrix parameter class.
    
    This represents a parameter that is a matrix.
    """
    
    def __init__(self, value: Union[Matrix, List[List[float]], np.ndarray] = None, 
                 nrow: int = 0, ncol: int = 0):
        """Initialize matrix parameter.
        
        Args:
            value: Initial parameter value.
            nrow: Number of rows if value is None.
            ncol: Number of columns if value is None.
        """
        super().__init__()
        
        if value is None:
            if nrow <= 0 or ncol <= 0:
                raise ValueError("Must specify positive dimensions if value is None")
            self._value = Matrix((nrow, ncol), fill_value=0.0)
        elif isinstance(value, Matrix):
            self._value = value.copy()
        else:
            self._value = Matrix(value)
    
    def value(self) -> Matrix:
        """Get parameter value."""
        return self._value.copy()
    
    def set_value(self, value: Union[Matrix, List[List[float]], np.ndarray]):
        """Set parameter value."""
        if isinstance(value, Matrix):
            self._value = value.copy()
        else:
            self._value = Matrix(value)
    
    def nrow(self) -> int:
        """Get number of rows."""
        return self._value.nrow()
    
    def ncol(self) -> int:
        """Get number of columns."""
        return self._value.ncol()
    
    def shape(self) -> Tuple[int, int]:
        """Get matrix shape."""
        return self._value.shape()
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector.
        
        Args:
            minimal: If True and matrix is symmetric, return upper triangle only.
                    Otherwise return full vectorized form.
        """
        if minimal and self._value.is_square() and self._value.is_sym():
            # Return upper triangle only for symmetric matrices
            result = []
            for i in range(self.nrow()):
                for j in range(i, self.ncol()):
                    result.append(self._value(i, j))
            return Vector(result)
        else:
            # Return full vectorized form (column-major order)
            result = []
            for j in range(self.ncol()):
                for i in range(self.nrow()):
                    result.append(self._value(i, j))
            return Vector(result)
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector.
        
        Args:
            v: Vector containing parameter values.
            minimal: If True, interpret as upper triangle for symmetric matrices.
        """
        if minimal and self._value.is_square():
            # Try to reconstruct symmetric matrix from upper triangle
            n = self.nrow()
            expected_size = n * (n + 1) // 2
            
            if len(v) == expected_size:
                # Reconstruct symmetric matrix
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        val = v[idx]
                        self._value[i, j] = val
                        if i != j:
                            self._value[j, i] = val
                        idx += 1
                return
        
        # Full reconstruction (column-major order)
        expected_size = self.nrow() * self.ncol()
        if len(v) != expected_size:
            raise ValueError(f"Vector size {len(v)} doesn't match matrix size {expected_size}")
        
        idx = 0
        for j in range(self.ncol()):
            for i in range(self.nrow()):
                self._value[i, j] = v[idx]
                idx += 1
    
    def size(self, minimal: bool = True) -> int:
        """Size when vectorized."""
        if minimal and self._value.is_square() and self._value.is_sym():
            n = self.nrow()
            return n * (n + 1) // 2  # Upper triangle
        else:
            return self.nrow() * self.ncol()  # Full matrix
    
    def clone(self) -> 'MatrixParams':
        """Create copy."""
        result = MatrixParams(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __getitem__(self, index: Tuple[int, int]) -> float:
        """Get element."""
        return self._value[index]
    
    def __setitem__(self, index: Tuple[int, int], value: float):
        """Set element."""
        self._value[index] = value
    
    def __str__(self) -> str:
        """String representation."""
        return f"MatrixParams({self._value.shape()})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return str(self)


class SpdMatrixParams(MatrixParams):
    """Symmetric positive definite matrix parameter class.
    
    This represents a parameter that is a symmetric positive definite matrix.
    """
    
    def __init__(self, value: Union[Matrix, List[List[float]], np.ndarray] = None, 
                 dim: int = 0):
        """Initialize SPD matrix parameter.
        
        Args:
            value: Initial parameter value (should be SPD).
            dim: Dimension if value is None.
        """
        if value is None:
            if dim <= 0:
                raise ValueError("Must specify positive dimension if value is None")
            # Initialize as identity matrix
            identity = Matrix((dim, dim), fill_value=0.0)
            identity.set_diag(1.0)
            super().__init__(identity)
        else:
            super().__init__(value)
            
        # Verify it's square
        if not self._value.is_square():
            raise ValueError("SPD matrix must be square")
    
    def is_spd(self) -> bool:
        """Check if matrix is symmetric positive definite.
        
        This is a basic check using symmetry and positive definiteness.
        """
        if not self._value.is_sym():
            return False
        
        try:
            # Try Cholesky decomposition
            eigenvals = np.linalg.eigvals(self._value.to_numpy())
            return np.all(eigenvals > 0)
        except np.linalg.LinAlgError:
            return False
    
    def set_value(self, value: Union[Matrix, List[List[float]], np.ndarray]):
        """Set parameter value with SPD check."""
        super().set_value(value)
        
        if not self._value.is_square():
            raise ValueError("SPD matrix must be square")
        
        # Optional: Add SPD verification
        # if not self.is_spd():
        #     raise ValueError("Matrix must be symmetric positive definite")
    
    def vectorize(self, minimal: bool = True) -> Vector:
        """Convert to vector using upper triangle (always minimal for SPD)."""
        return super().vectorize(minimal=True)  # Always use minimal for SPD
    
    def unvectorize(self, v: Vector, minimal: bool = True):
        """Set from vector using upper triangle."""
        super().unvectorize(v, minimal=True)  # Always use minimal for SPD
    
    def size(self, minimal: bool = True) -> int:
        """Size when vectorized (always minimal for SPD)."""
        return super().size(minimal=True)  # Always use minimal for SPD
    
    def clone(self) -> 'SpdMatrixParams':
        """Create copy."""
        result = SpdMatrixParams(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __str__(self) -> str:
        """String representation."""
        return f"SpdMatrixParams({self._value.shape()})"


# Convenience functions for creating common parameter types

def create_univariate_param(value: float = 0.0) -> UnivParams:
    """Create a univariate parameter."""
    return UnivParams(value)


def create_vector_param(value: Union[Vector, List[float], np.ndarray] = None, 
                       dim: int = 0) -> VectorParams:
    """Create a vector parameter."""
    return VectorParams(value, dim)


def create_matrix_param(value: Union[Matrix, List[List[float]], np.ndarray] = None,
                       nrow: int = 0, ncol: int = 0) -> MatrixParams:
    """Create a matrix parameter.""" 
    return MatrixParams(value, nrow, ncol)


def create_spd_matrix_param(value: Union[Matrix, List[List[float]], np.ndarray] = None,
                           dim: int = 0) -> SpdMatrixParams:
    """Create an SPD matrix parameter."""
    return SpdMatrixParams(value, dim)