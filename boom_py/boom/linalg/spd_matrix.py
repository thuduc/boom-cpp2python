"""Symmetric Positive Definite Matrix implementation for BOOM."""
import numpy as np
from typing import Union, List, Optional
from .matrix import Matrix
from .vector import Vector


class SpdMatrix(Matrix):
    """Symmetric Positive Definite Matrix class."""
    
    def __new__(cls, data: Union[List[List[float]], np.ndarray, int] = None):
        """Create a new SpdMatrix instance.
        
        Args:
            data: Can be a 2D array/list or an integer (creates identity matrix of that size)
        """
        if isinstance(data, int):
            # Create identity matrix of given size
            obj = np.eye(data, dtype=np.float64).view(cls)
        else:
            # Convert to array first to avoid recursion
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError("SpdMatrix must be 2-dimensional")
            obj = arr.view(cls)
            # Verify symmetric
            if not np.allclose(obj, obj.T, atol=1e-8):
                # Make it symmetric by averaging with transpose
                obj = ((obj + obj.T) / 2).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        """Called whenever the array is created."""
        # Ensure matrix is square
        if obj is not None and obj.ndim == 2:
            if obj.shape[0] != obj.shape[1]:
                raise ValueError("SpdMatrix must be square")
    
    @classmethod
    def identity(cls, n: int) -> 'SpdMatrix':
        """Create an identity matrix."""
        return cls(n)
    
    @classmethod
    def from_correlation(cls, corr: Matrix, sd: Vector) -> 'SpdMatrix':
        """Create covariance matrix from correlation matrix and standard deviations."""
        D = np.diag(sd)
        cov = D @ corr @ D
        return cls(cov)
    
    def chol(self, lower: bool = True) -> Matrix:
        """Return Cholesky decomposition.
        
        Args:
            lower: If True, return lower triangular L such that A = LL'.
                   If False, return upper triangular U such that A = U'U.
        """
        try:
            L = np.linalg.cholesky(self)
            if not lower:
                return Matrix(L.T)
            return Matrix(L)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not positive definite")
    
    def inv(self) -> 'SpdMatrix':
        """Return the inverse of the matrix."""
        return SpdMatrix(np.linalg.inv(self))
    
    def solve(self, b: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """Solve Ax = b for x using Cholesky decomposition."""
        try:
            # Use Cholesky decomposition for better numerical stability
            L = self.chol()
            # Solve L y = b
            y = np.linalg.solve(L, b)
            # Solve L' x = y
            x = np.linalg.solve(L.T, y)
            if x.ndim == 1:
                return Vector(x)
            return Matrix(x)
        except:
            # Fall back to standard solve
            return super().solve(b)
    
    def logdet(self) -> float:
        """Return log determinant (more stable than log(det))."""
        try:
            L = self.chol()
            return 2 * np.sum(np.log(np.diag(L)))
        except:
            # Fall back to eigenvalue method
            eigvals = np.linalg.eigvalsh(self)
            return np.sum(np.log(eigvals))
    
    def is_pos_def(self) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(self)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def add_to_diag(self, x: float) -> 'SpdMatrix':
        """Add a scalar to the diagonal."""
        result = self.copy()
        np.fill_diagonal(result, np.diag(result) + x)
        return SpdMatrix(result)
    
    def scale_rows(self, v: Vector) -> 'SpdMatrix':
        """Scale rows and columns by vector v (returns vSv where v is diagonal)."""
        D = np.diag(v)
        return SpdMatrix(D @ self @ D)
    
    def quad_form(self, x: Vector) -> float:
        """Compute quadratic form x'Ax."""
        return float(x @ self @ x)
    
    def triple_product(self, B: Matrix) -> 'SpdMatrix':
        """Compute B'AB."""
        return SpdMatrix(B.T @ self @ B)
    
    def sandwich(self, B: Matrix) -> 'SpdMatrix':
        """Compute BAB'."""
        return SpdMatrix(B @ self @ B.T)
    
    def condition_number(self) -> float:
        """Return condition number."""
        eigvals = np.linalg.eigvalsh(self)
        return eigvals[-1] / eigvals[0]
    
    def to_correlation(self) -> 'SpdMatrix':
        """Convert covariance matrix to correlation matrix."""
        D_inv = np.diag(1.0 / np.sqrt(np.diag(self)))
        corr = D_inv @ self @ D_inv
        return SpdMatrix(corr)
    
    def var(self) -> Vector:
        """Return diagonal (variances)."""
        return Vector(np.diag(self))
    
    def sd(self) -> Vector:
        """Return standard deviations (sqrt of diagonal)."""
        return Vector(np.sqrt(np.diag(self)))
    
    # Override parent methods to ensure SpdMatrix type is preserved
    def __add__(self, other):
        result = Matrix.__add__(self, other)
        if isinstance(result, np.ndarray) and result.ndim == 2:
            if np.allclose(result, result.T, atol=1e-8):
                return result.view(SpdMatrix)
        return result
    
    def __sub__(self, other):
        result = Matrix.__sub__(self, other)
        if isinstance(result, np.ndarray) and result.ndim == 2:
            if np.allclose(result, result.T, atol=1e-8):
                return result.view(SpdMatrix)
        return result
    
    def __mul__(self, other):
        if np.isscalar(other):
            result = Matrix.__mul__(self, other)
            return result.view(SpdMatrix)
        return Matrix.__mul__(self, other)
    
    def __truediv__(self, other):
        if np.isscalar(other):
            result = Matrix.__truediv__(self, other)
            return result.view(SpdMatrix)
        return Matrix.__truediv__(self, other)
    
    def __repr__(self):
        return f"SpdMatrix({np.array_repr(self)})"