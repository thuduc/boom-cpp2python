"""Matrix class - a wrapper around numpy 2D arrays with BOOM-like interface."""

from typing import Union, List, Optional, Tuple, Iterator
import numpy as np
from numpy.typing import ArrayLike
import re

from .vector import Vector


class Matrix:
    """A matrix class that wraps numpy 2D arrays and provides BOOM Matrix interface."""
    
    def __init__(self, data: Union[int, ArrayLike, str, 'Matrix', Tuple[int, int]] = None,
                 ncol: Optional[int] = None, fill_value: float = 0.0, 
                 byrow: bool = False, row_delim: str = "|"):
        """Initialize a Matrix.
        
        Args:
            data: Can be:
                - None: create empty matrix
                - (nrow, ncol) tuple: create matrix of that size filled with fill_value
                - int with ncol: create matrix with data rows and ncol columns
                - array-like: create matrix from the data
                - str: parse string as space-separated values with row_delim
                - Matrix: copy constructor
                - list of Vectors: stack as rows or columns based on byrow
            ncol: number of columns (used when data is an int)
            fill_value: value to fill matrix with if creating from dimensions
            byrow: if True, fill/interpret data by rows, else by columns
            row_delim: delimiter for rows when parsing strings
        """
        if data is None:
            # Empty matrix
            self.data = np.array([], dtype=np.float64).reshape(0, 0)
            
        elif isinstance(data, tuple) and len(data) == 2:
            # Create from dimensions
            nrow, ncol = data
            self.data = np.full((nrow, ncol), fill_value, dtype=np.float64)
            
        elif isinstance(data, int) and ncol is not None:
            # Create from dimensions (alternative syntax)
            self.data = np.full((data, ncol), fill_value, dtype=np.float64)
            
        elif isinstance(data, str):
            # Parse string
            rows = [r.strip() for r in data.split(row_delim) if r.strip()]
            matrix_data = []
            for row in rows:
                values = [float(v) for v in row.split() if v]
                matrix_data.append(values)
            self.data = np.array(matrix_data, dtype=np.float64)
            
        elif isinstance(data, Matrix):
            # Copy constructor
            self.data = data.data.copy()
            
        elif isinstance(data, list) and data and isinstance(data[0], Vector):
            # List of Vectors
            if byrow:
                # Stack vectors as rows
                self.data = np.vstack([v.data for v in data])
            else:
                # Stack vectors as columns
                self.data = np.column_stack([v.data for v in data])
                
        elif isinstance(data, list) and data and isinstance(data[0], list):
            # List of lists (rows)
            self.data = np.array(data, dtype=np.float64)
            
        else:
            # Create from array-like
            self.data = np.array(data, dtype=np.float64)
            if self.data.ndim == 1 and ncol is not None:
                # Reshape 1D array into matrix
                nrow = len(self.data) // ncol
                if len(self.data) != nrow * ncol:
                    raise ValueError(f"Cannot reshape {len(self.data)} elements into {nrow}x{ncol} matrix")
                if byrow:
                    self.data = self.data.reshape(nrow, ncol)
                else:
                    self.data = self.data.reshape(ncol, nrow).T
            elif self.data.ndim != 2:
                raise ValueError("Matrix must be 2-dimensional")
    
    # Size and shape methods
    def size(self) -> int:
        """Return total number of elements."""
        return self.data.size
    
    def nrow(self) -> int:
        """Return number of rows."""
        return self.data.shape[0]
    
    def ncol(self) -> int:
        """Return number of columns."""
        return self.data.shape[1]
    
    def shape(self) -> Tuple[int, int]:
        """Return (nrow, ncol) tuple."""
        return self.data.shape
    
    # Also provide as property for compatibility
    @property
    def shape_property(self) -> Tuple[int, int]:
        """Return (nrow, ncol) tuple as property."""
        return self.data.shape
    
    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.nrow() == self.ncol()
    
    def is_sym(self, tol: float = 1e-4) -> bool:
        """Check if matrix is symmetric within tolerance."""
        if not self.is_square():
            return False
        return np.allclose(self.data, self.data.T, rtol=tol, atol=tol)
    
    def same_dim(self, other: 'Matrix') -> bool:
        """Check if matrices have same dimensions."""
        return self.shape() == other.shape()
    
    # Element access
    def __call__(self, row: int, col: int) -> float:
        """Get element at (row, col)."""
        return float(self.data[row, col])
    
    def __getitem__(self, index: Union[int, Tuple[int, int], slice]) -> Union[float, Vector, 'Matrix']:
        """Get element(s) by index."""
        if isinstance(index, tuple) and len(index) == 2:
            # Single element or submatrix
            result = self.data[index]
            if np.isscalar(result):
                return float(result)
            elif result.ndim == 1:
                return Vector(result)
            else:
                return Matrix(result)
        elif isinstance(index, int):
            # Get row
            return Vector(self.data[index])
        else:
            # Slice rows
            return Matrix(self.data[index])
    
    def __setitem__(self, index: Union[Tuple[int, int], int], value: Union[float, ArrayLike]):
        """Set element(s) by index."""
        self.data[index] = value
    
    # Row and column access
    def row(self, i: int) -> Vector:
        """Get row i as a Vector."""
        return Vector(self.data[i])
    
    def col(self, j: int) -> Vector:
        """Get column j as a Vector."""
        return Vector(self.data[:, j])
    
    def column(self, j: int) -> Vector:
        """Get column j as a Vector (alias for col)."""
        return self.col(j)
    
    def set_row(self, i: int, v: Union[Vector, ArrayLike, float]):
        """Set row i."""
        if isinstance(v, Vector):
            self.data[i, :] = v.data
        else:
            self.data[i, :] = v
    
    def set_col(self, j: int, v: Union[Vector, ArrayLike, float]):
        """Set column j."""
        if isinstance(v, Vector):
            self.data[:, j] = v.data
        else:
            self.data[:, j] = v
    
    def set_rc(self, i: int, x: float):
        """Set row and column i to x."""
        self.data[i, :] = x
        self.data[:, i] = x
    
    # Diagonal access
    def diag(self) -> Vector:
        """Get main diagonal as Vector."""
        return Vector(np.diag(self.data))
    
    def set_diag(self, v: Union[Vector, ArrayLike, float], zero_offdiag: bool = True) -> 'Matrix':
        """Set diagonal elements."""
        if zero_offdiag:
            self.data.fill(0.0)
        if isinstance(v, Vector):
            np.fill_diagonal(self.data, v.data)
        else:
            np.fill_diagonal(self.data, v)
        return self
    
    # Comparison and properties
    def __eq__(self, other: object) -> bool:
        """Check equality with another Matrix."""
        if not isinstance(other, Matrix):
            return False
        return np.array_equal(self.data, other.data)
    
    def all_finite(self) -> bool:
        """Check if all elements are finite."""
        return bool(np.all(np.isfinite(self.data)))
    
    # Initialization methods
    def randomize(self, rng: Optional[np.random.RandomState] = None) -> 'Matrix':
        """Fill with uniform random numbers in [0, 1)."""
        if rng is None:
            rng = np.random
        self.data = rng.uniform(0, 1, size=self.shape())
        return self
    
    def randomize_gaussian(self, mean: float = 0.0, sd: float = 1.0,
                          rng: Optional[np.random.RandomState] = None) -> 'Matrix':
        """Fill with Gaussian random numbers."""
        if rng is None:
            rng = np.random
        self.data = rng.normal(mean, sd, size=self.shape())
        return self
    
    # Arithmetic operators (in-place)
    def __iadd__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """In-place addition."""
        if isinstance(other, Matrix):
            self.data += other.data
        else:
            self.data += other
        return self
    
    def __isub__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """In-place subtraction."""
        if isinstance(other, Matrix):
            self.data -= other.data
        else:
            self.data -= other
        return self
    
    def __imul__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """In-place multiplication (element-wise or scalar)."""
        if isinstance(other, Matrix):
            self.data *= other.data
        else:
            self.data *= other
        return self
    
    def __itruediv__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """In-place division."""
        if isinstance(other, Matrix):
            self.data /= other.data
        else:
            self.data /= other
        return self
    
    # Arithmetic operators (new objects)
    def __add__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """Addition."""
        result = Matrix(self)
        result += other
        return result
    
    def __radd__(self, other: Union[float]) -> 'Matrix':
        """Right addition."""
        return self + other
    
    def __sub__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """Subtraction."""
        result = Matrix(self)
        result -= other
        return result
    
    def __rsub__(self, other: Union[float]) -> 'Matrix':
        """Right subtraction."""
        return Matrix((self.nrow(), self.ncol()), fill_value=other) - self
    
    def __mul__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """Multiplication (element-wise or scalar)."""
        result = Matrix(self)
        result *= other
        return result
    
    def __rmul__(self, other: Union[float]) -> 'Matrix':
        """Right multiplication."""
        return self * other
    
    def __truediv__(self, other: Union[float, 'Matrix']) -> 'Matrix':
        """Division."""
        result = Matrix(self)
        result /= other
        return result
    
    def __rtruediv__(self, other: Union[float]) -> 'Matrix':
        """Right division."""
        return Matrix((self.nrow(), self.ncol()), fill_value=other) / self
    
    def __neg__(self) -> 'Matrix':
        """Negation."""
        return Matrix(-self.data)
    
    # Matrix multiplication
    def mult(self, other: Union['Matrix', Vector], scal: float = 1.0) -> Union['Matrix', Vector]:
        """Matrix multiplication: scal * self @ other."""
        if isinstance(other, Matrix):
            return Matrix(scal * np.dot(self.data, other.data))
        elif isinstance(other, Vector):
            return Vector(scal * np.dot(self.data, other.data))
        else:
            raise TypeError("Can only multiply with Matrix or Vector")
    
    def Tmult(self, other: Union['Matrix', Vector], scal: float = 1.0) -> Union['Matrix', Vector]:
        """Transposed multiplication: scal * self.T @ other."""
        if isinstance(other, Matrix):
            return Matrix(scal * np.dot(self.data.T, other.data))
        elif isinstance(other, Vector):
            return Vector(scal * np.dot(self.data.T, other.data))
        else:
            raise TypeError("Can only multiply with Matrix or Vector")
    
    def multT(self, other: 'Matrix', scal: float = 1.0) -> 'Matrix':
        """Multiply by transpose: scal * self @ other.T."""
        return Matrix(scal * np.dot(self.data, other.data.T))
    
    def __matmul__(self, other: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        """Matrix multiplication operator @."""
        return self.mult(other)
    
    # Linear algebra operations
    def transpose(self) -> 'Matrix':
        """Return transpose."""
        return Matrix(self.data.T)
    
    def transpose_inplace_square(self) -> 'Matrix':
        """Transpose in place (only for square matrices)."""
        if not self.is_square():
            raise ValueError("Can only transpose square matrices in place")
        self.data = self.data.T
        return self
    
    def inv(self) -> 'Matrix':
        """Return inverse."""
        return Matrix(np.linalg.inv(self.data))
    
    def solve(self, b: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        """Solve Ax = b for x."""
        if isinstance(b, Matrix):
            return Matrix(np.linalg.solve(self.data, b.data))
        elif isinstance(b, Vector):
            return Vector(np.linalg.solve(self.data, b.data))
        else:
            raise TypeError("Can only solve with Matrix or Vector")
    
    def det(self) -> float:
        """Return determinant."""
        return float(np.linalg.det(self.data))
    
    def logdet(self) -> float:
        """Return log of absolute determinant."""
        sign, logdet = np.linalg.slogdet(self.data)
        return float(logdet)
    
    def trace(self) -> float:
        """Return trace (sum of diagonal)."""
        return float(np.trace(self.data))
    
    def inner(self, weights: Optional[Vector] = None) -> np.ndarray:
        """Return X.T @ X or X.T @ diag(weights) @ X."""
        if weights is None:
            return np.dot(self.data.T, self.data)
        else:
            # X.T @ diag(weights) @ X
            weighted = self.data * weights.data[:, np.newaxis]
            return np.dot(self.data.T, weighted)
    
    def outer(self) -> np.ndarray:
        """Return X @ X.T."""
        return np.dot(self.data, self.data.T)
    
    def singular_values(self) -> Vector:
        """Return singular values (largest to smallest)."""
        return Vector(np.linalg.svd(self.data, compute_uv=False))
    
    def condition_number(self) -> float:
        """Return condition number (ratio of largest to smallest singular value)."""
        sv = self.singular_values()
        if len(sv) == 0:
            return np.inf
        min_sv = sv.min()
        if min_sv == 0:
            return np.inf
        return float(sv.max() / min_sv)
    
    def rank(self, prop: float = 1e-12) -> int:
        """Return rank (number of singular values >= prop * largest)."""
        sv = self.singular_values()
        if len(sv) == 0:
            return 0
        threshold = prop * sv.max()
        return int(np.sum(sv.data >= threshold))
    
    # Modification methods
    def resize(self, nrow: int, ncol: int) -> 'Matrix':
        """Resize matrix (invalidates elements if shape changes)."""
        self.data = np.resize(self.data, (nrow, ncol))
        return self
    
    def rbind(self, other: Union['Matrix', Vector]) -> 'Matrix':
        """Row bind - append rows."""
        if isinstance(other, Matrix):
            self.data = np.vstack([self.data, other.data])
        elif isinstance(other, Vector):
            self.data = np.vstack([self.data, other.data.reshape(1, -1)])
        else:
            raise TypeError("Can only bind Matrix or Vector")
        return self
    
    def cbind(self, other: Union['Matrix', Vector]) -> 'Matrix':
        """Column bind - append columns."""
        if isinstance(other, Matrix):
            self.data = np.hstack([self.data, other.data])
        elif isinstance(other, Vector):
            self.data = np.hstack([self.data, other.data.reshape(-1, 1)])
        else:
            raise TypeError("Can only bind Matrix or Vector")
        return self
    
    def add_outer(self, x: Vector, y: Vector, w: float = 1.0) -> 'Matrix':
        """Add outer product: self += w * x @ y.T."""
        self.data += w * np.outer(x.data, y.data)
        return self
    
    # Utility methods
    def Id(self) -> 'Matrix':
        """Return identity matrix of same size (square matrices only)."""
        if not self.is_square():
            raise ValueError("Identity matrix only defined for square matrices")
        return Matrix(np.eye(self.nrow()))
    
    def copy(self) -> 'Matrix':
        """Create a copy."""
        return Matrix(self)
    
    def swap(self, other: 'Matrix') -> 'Matrix':
        """Swap contents with another matrix."""
        self.data, other.data = other.data, self.data
        return self
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()
    
    @property
    def T(self) -> 'Matrix':
        """Return transpose (property)."""
        return self.transpose()
    
    # String representation
    def __str__(self) -> str:
        """String representation."""
        return str(self.data)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Matrix({self.data})"
    
    # Iterator support
    def __iter__(self) -> Iterator[Vector]:
        """Iterate over rows."""
        for i in range(self.nrow()):
            yield self.row(i)


class SpdMatrix(Matrix):
    """Symmetric positive definite matrix - a special case of Matrix.
    
    This class enforces that the matrix is square and provides methods
    specific to symmetric positive definite matrices like Cholesky decomposition.
    """
    
    def __init__(self, data: Union[ArrayLike, str, 'Matrix', Tuple[int, int]] = None,
                 fill_value: float = 0.0, check_spd: bool = False):
        """Initialize a symmetric positive definite matrix.
        
        Args:
            data: Matrix data (must result in square matrix)
            fill_value: Fill value for initialization
            check_spd: Whether to check positive definiteness
        """
        super().__init__(data, fill_value=fill_value)
        
        if not self.is_square():
            raise ValueError("SpdMatrix must be square")
        
        if check_spd and self.nrow() > 0:
            self._check_positive_definite()
    
    def _check_positive_definite(self):
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(self.data)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not positive definite")
    
    def cholesky(self) -> 'Matrix':
        """Compute Cholesky decomposition."""
        try:
            L = np.linalg.cholesky(self.data)
            return Matrix(L)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not positive definite")
    
    def solve(self, b: Union[Vector, 'Matrix']) -> Union[Vector, 'Matrix']:
        """Solve Ax = b using Cholesky decomposition."""
        if isinstance(b, Vector):
            x = np.linalg.solve(self.data, b.data)
            return Vector(x)
        elif isinstance(b, Matrix):
            x = np.linalg.solve(self.data, b.data)
            return Matrix(x)
        else:
            raise TypeError("b must be Vector or Matrix")
    
    def inv(self) -> 'SpdMatrix':
        """Compute matrix inverse."""
        inv_data = np.linalg.inv(self.data)
        return SpdMatrix(inv_data)
    
    def logdet(self) -> float:
        """Compute log determinant."""
        return np.linalg.slogdet(self.data)[1]
    
    def __add__(self, other: Union['SpdMatrix', 'Matrix', float]) -> 'SpdMatrix':
        """Add matrices."""
        if isinstance(other, (SpdMatrix, Matrix)):
            return SpdMatrix(self.data + other.data)
        else:
            return SpdMatrix(self.data + other)
    
    def __radd__(self, other: Union[float, int]) -> 'SpdMatrix':
        """Right add (for scalars)."""
        return SpdMatrix(other + self.data)
    
    def __sub__(self, other: Union['SpdMatrix', 'Matrix', float]) -> 'SpdMatrix':
        """Subtract matrices."""
        if isinstance(other, (SpdMatrix, Matrix)):
            return SpdMatrix(self.data - other.data)
        else:
            return SpdMatrix(self.data - other)
    
    def __mul__(self, other: Union[float, int]) -> 'SpdMatrix':
        """Scalar multiplication."""
        return SpdMatrix(self.data * other)
    
    def __rmul__(self, other: Union[float, int]) -> 'SpdMatrix':
        """Right scalar multiplication."""
        return SpdMatrix(other * self.data)
    
    def __truediv__(self, other: Union[float, int]) -> 'SpdMatrix':
        """Scalar division."""
        return SpdMatrix(self.data / other)
    
    def copy(self) -> 'SpdMatrix':
        """Create a copy."""
        return SpdMatrix(self.data)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SpdMatrix({self.data})"