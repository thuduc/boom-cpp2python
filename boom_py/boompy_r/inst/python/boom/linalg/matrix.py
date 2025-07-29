"""Matrix implementation for BOOM."""
import numpy as np
from typing import Union, List, Tuple, Optional
from .vector import Vector


class Matrix(np.ndarray):
    """Matrix class that extends numpy array with BOOM-specific functionality."""
    
    def __new__(cls, data: Union[List[List[float]], np.ndarray, Tuple[int, int]] = None, 
                nrow: Optional[int] = None, ncol: Optional[int] = None):
        """Create a new Matrix instance.
        
        Args:
            data: Can be a 2D list, numpy array, or tuple of (nrow, ncol) for zero matrix
            nrow: Number of rows (if data is 1D)
            ncol: Number of columns (if data is 1D)
        """
        if data is None:
            if nrow is not None and ncol is not None:
                obj = np.zeros((nrow, ncol), dtype=np.float64).view(cls)
            else:
                obj = np.array([[]], dtype=np.float64).view(cls)
        elif isinstance(data, tuple) and len(data) == 2:
            # Create zero matrix of specified size
            obj = np.zeros(data, dtype=np.float64).view(cls)
        else:
            obj = np.asarray(data, dtype=np.float64)
            if obj.ndim == 1:
                if nrow is not None and ncol is not None:
                    obj = obj.reshape(nrow, ncol)
                else:
                    raise ValueError("Matrix must be 2-dimensional. For 1D data, provide nrow and ncol.")
            elif obj.ndim != 2:
                raise ValueError("Matrix must be 2-dimensional")
            obj = obj.view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        """Called whenever the array is created."""
        pass
    
    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        """Create an identity matrix."""
        return cls(np.eye(n))
    
    @classmethod
    def zero(cls, nrow: int, ncol: int) -> 'Matrix':
        """Create a zero matrix."""
        return cls(np.zeros((nrow, ncol)))
    
    @classmethod
    def ones(cls, nrow: int, ncol: int) -> 'Matrix':
        """Create a matrix of ones."""
        return cls(np.ones((nrow, ncol)))
    
    def nrow(self) -> int:
        """Return number of rows."""
        return self.shape[0]
    
    def ncol(self) -> int:
        """Return number of columns."""
        return self.shape[1] if self.ndim > 1 else 1
    
    def row(self, i: int) -> Vector:
        """Return the i-th row as a Vector."""
        return Vector(self[i, :])
    
    def col(self, j: int) -> Vector:
        """Return the j-th column as a Vector."""
        return Vector(self[:, j])
    
    def set_row(self, i: int, v: Union[Vector, np.ndarray, List[float]]):
        """Set the i-th row."""
        self[i, :] = v
    
    def set_col(self, j: int, v: Union[Vector, np.ndarray, List[float]]):
        """Set the j-th column."""
        self[:, j] = v
    
    def transpose(self) -> 'Matrix':
        """Return the transpose of the matrix."""
        return Matrix(self.T)
    
    def t(self) -> 'Matrix':
        """Shorthand for transpose."""
        return self.transpose()
    
    def inv(self) -> 'Matrix':
        """Return the inverse of the matrix."""
        return Matrix(np.linalg.inv(self))
    
    def det(self) -> float:
        """Return the determinant."""
        return np.linalg.det(self)
    
    def trace(self) -> float:
        """Return the trace (sum of diagonal)."""
        return float(np.ndarray.trace(self))
    
    def diag(self) -> Vector:
        """Return the diagonal as a Vector."""
        return Vector(np.diag(self))
    
    def set_diag(self, v: Union[Vector, np.ndarray, List[float], float]):
        """Set the diagonal elements."""
        if np.isscalar(v):
            np.fill_diagonal(self, v)
        else:
            np.fill_diagonal(self, v)
    
    def is_symmetric(self, tol: float = 1e-8) -> bool:
        """Check if matrix is symmetric."""
        if self.nrow() != self.ncol():
            return False
        return np.allclose(self, self.T, atol=tol)
    
    def is_pos_def(self) -> bool:
        """Check if matrix is positive definite."""
        if not self.is_symmetric():
            return False
        try:
            np.linalg.cholesky(self)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def singval(self) -> Tuple[Vector, 'Matrix', 'Matrix']:
        """Compute singular value decomposition."""
        u, s, vt = np.linalg.svd(self)
        return Vector(s), Matrix(u), Matrix(vt.T)
    
    def rank(self, tol: float = 1e-8) -> int:
        """Return the rank of the matrix."""
        return np.linalg.matrix_rank(self, tol=tol)
    
    def sum(self) -> float:
        """Return the sum of all elements."""
        return float(np.ndarray.sum(self))
    
    def colsums(self) -> Vector:
        """Return column sums."""
        return Vector(np.ndarray.sum(self, axis=0))
    
    def rowsums(self) -> Vector:
        """Return row sums."""
        return Vector(np.ndarray.sum(self, axis=1))
    
    def norm(self, ord=None) -> float:
        """Return matrix norm."""
        return np.linalg.norm(self, ord=ord)
    
    def solve(self, b: Union[Vector, 'Matrix']) -> Union[Vector, 'Matrix']:
        """Solve Ax = b for x."""
        result = np.linalg.solve(self, b)
        if result.ndim == 1:
            return Vector(result)
        return Matrix(result)
    
    def chol(self, lower: bool = True) -> 'Matrix':
        """Return Cholesky decomposition."""
        L = np.linalg.cholesky(self)
        if not lower:
            L = L.T
        return Matrix(L)
    
    def eig(self) -> Tuple[Vector, 'Matrix']:
        """Return eigenvalues and eigenvectors."""
        w, v = np.linalg.eig(self)
        return Vector(w), Matrix(v)
    
    def kronecker(self, other: 'Matrix') -> 'Matrix':
        """Return Kronecker product with another matrix."""
        return Matrix(np.kron(self, other))
    
    # Arithmetic operations that return Matrix type
    def __add__(self, other):
        result = np.add(self, other)
        if result.ndim == 2:
            return Matrix(result)
        return result
    
    def __sub__(self, other):
        result = np.subtract(self, other)
        if result.ndim == 2:
            return Matrix(result)
        return result
    
    def __mul__(self, other):
        result = np.multiply(self, other)
        if result.ndim == 2:
            return Matrix(result)
        return result
    
    def __truediv__(self, other):
        result = np.divide(self, other)
        if result.ndim == 2:
            return Matrix(result)
        return result
    
    def __matmul__(self, other):
        result = np.matmul(self, other)
        if result.ndim == 2:
            return Matrix(result)
        elif result.ndim == 1:
            return Vector(result)
        return result
    
    def __neg__(self):
        return Matrix(np.negative(self))
    
    def __repr__(self):
        return f"Matrix({np.array_repr(self)})"