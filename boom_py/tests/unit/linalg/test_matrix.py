"""Tests for Matrix class."""
import pytest
import numpy as np
from boom.linalg import Matrix, Vector


class TestMatrix:
    """Test suite for Matrix class."""
    
    def test_construction(self):
        """Test various ways to construct a Matrix."""
        # From 2D list
        m1 = Matrix([[1, 2], [3, 4]])
        assert m1.nrow() == 2
        assert m1.ncol() == 2
        assert np.array_equal(m1, [[1, 2], [3, 4]])
        
        # From numpy array
        m2 = Matrix(np.array([[5, 6], [7, 8]]))
        assert m2.shape == (2, 2)
        
        # From size tuple
        m3 = Matrix((3, 4))
        assert m3.shape == (3, 4)
        assert np.all(m3 == 0)
        
        # From 1D array with reshape
        m4 = Matrix([1, 2, 3, 4, 5, 6], nrow=2, ncol=3)
        assert m4.shape == (2, 3)
        assert np.array_equal(m4, [[1, 2, 3], [4, 5, 6]])
        
        # Using nrow, ncol for zero matrix
        m5 = Matrix(nrow=2, ncol=3)
        assert m5.shape == (2, 3)
        assert np.all(m5 == 0)
    
    def test_class_methods(self):
        """Test class construction methods."""
        # Identity matrix
        m1 = Matrix.identity(3)
        assert np.array_equal(m1, np.eye(3))
        
        # Zero matrix
        m2 = Matrix.zero(2, 3)
        assert m2.shape == (2, 3)
        assert np.all(m2 == 0)
        
        # Ones matrix
        m3 = Matrix.ones(2, 2)
        assert np.all(m3 == 1)
    
    def test_row_col_access(self):
        """Test row and column access."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        
        # Get row
        row0 = m.row(0)
        assert isinstance(row0, Vector)
        assert np.array_equal(row0, [1, 2, 3])
        
        # Get column
        col1 = m.col(1)
        assert isinstance(col1, Vector)
        assert np.array_equal(col1, [2, 5])
        
        # Set row
        m.set_row(0, [7, 8, 9])
        assert np.array_equal(m[0, :], [7, 8, 9])
        
        # Set column
        m.set_col(2, [10, 11])
        assert np.array_equal(m[:, 2], [10, 11])
    
    def test_transpose(self):
        """Test transpose operations."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        
        mt = m.transpose()
        assert mt.shape == (3, 2)
        assert np.array_equal(mt, [[1, 4], [2, 5], [3, 6]])
        
        # Test shorthand
        mt2 = m.t()
        assert np.array_equal(mt, mt2)
    
    def test_linear_algebra(self):
        """Test linear algebra operations."""
        # Inverse
        m = Matrix([[2, 1], [1, 2]])
        m_inv = m.inv()
        identity = m @ m_inv
        assert np.allclose(identity, np.eye(2))
        
        # Determinant
        det = m.det()
        assert np.isclose(det, 3)
        
        # Trace
        trace = m.trace()
        assert trace == 4
        
        # Solve linear system
        b = Vector([5, 4])
        x = m.solve(b)
        assert isinstance(x, Vector)
        assert np.allclose(m @ x, b)
    
    def test_diagonal(self):
        """Test diagonal operations."""
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Get diagonal
        diag = m.diag()
        assert isinstance(diag, Vector)
        assert np.array_equal(diag, [1, 5, 9])
        
        # Set diagonal
        m.set_diag([10, 11, 12])
        assert np.array_equal(m.diag(), [10, 11, 12])
        
        # Set diagonal with scalar
        m.set_diag(0)
        assert np.array_equal(m.diag(), [0, 0, 0])
    
    def test_properties(self):
        """Test matrix properties."""
        # Symmetric matrix
        m1 = Matrix([[1, 2], [2, 1]])
        assert m1.is_symmetric()
        
        m2 = Matrix([[1, 2], [3, 1]])
        assert not m2.is_symmetric()
        
        # Positive definite
        m3 = Matrix([[2, 1], [1, 2]])
        assert m3.is_pos_def()
        
        m4 = Matrix([[1, 2], [2, 1]])
        assert not m4.is_pos_def()
    
    def test_decompositions(self):
        """Test matrix decompositions."""
        # Cholesky
        m = Matrix([[4, 2], [2, 3]])
        L = m.chol()
        assert np.allclose(L @ L.T, m)
        
        # SVD
        m2 = Matrix([[1, 2], [3, 4], [5, 6]])
        s, u, v = m2.singval()
        assert isinstance(s, Vector)
        # Reconstruct matrix
        S = Matrix.zero(u.shape[1], v.shape[0])
        S.set_diag(s)
        reconstructed = u @ S @ v.T
        assert np.allclose(reconstructed, m2)
        
        # Eigenvalues
        m3 = Matrix([[1, 2], [2, 1]])
        eigvals, eigvecs = m3.eig()
        assert isinstance(eigvals, Vector)
        # Verify Av = λv for first eigenvalue/vector
        assert np.allclose(m3 @ eigvecs[:, 0], eigvals[0] * eigvecs[:, 0])
    
    def test_rank(self):
        """Test rank calculation."""
        # Full rank
        m1 = Matrix([[1, 0], [0, 1]])
        assert m1.rank() == 2
        
        # Rank deficient
        m2 = Matrix([[1, 2], [2, 4]])
        assert m2.rank() == 1
    
    def test_sums(self):
        """Test sum operations."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        
        assert m.sum() == 21
        
        colsums = m.colsums()
        assert isinstance(colsums, Vector)
        assert np.array_equal(colsums, [5, 7, 9])
        
        rowsums = m.rowsums()
        assert isinstance(rowsums, Vector)
        assert np.array_equal(rowsums, [6, 15])
    
    def test_norms(self):
        """Test matrix norms."""
        m = Matrix([[1, 2], [3, 4]])
        
        # Frobenius norm
        frob_norm = m.norm()
        assert np.isclose(frob_norm, np.sqrt(1+4+9+16))
        
        # 1-norm
        norm1 = m.norm(1)
        assert norm1 == 6  # max column sum
        
        # inf-norm
        norm_inf = m.norm(np.inf)
        assert norm_inf == 7  # max row sum
    
    def test_kronecker(self):
        """Test Kronecker product."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[0, 5], [6, 7]])
        
        kron = m1.kronecker(m2)
        expected = Matrix([[0, 5, 0, 10],
                          [6, 7, 12, 14],
                          [0, 15, 0, 20],
                          [18, 21, 24, 28]])
        assert np.array_equal(kron, expected)
    
    def test_arithmetic(self):
        """Test arithmetic operations."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        
        # Addition
        m_add = m1 + m2
        assert isinstance(m_add, Matrix)
        assert np.array_equal(m_add, [[6, 8], [10, 12]])
        
        # Subtraction
        m_sub = m2 - m1
        assert isinstance(m_sub, Matrix)
        assert np.array_equal(m_sub, [[4, 4], [4, 4]])
        
        # Scalar multiplication
        m_mul = m1 * 2
        assert isinstance(m_mul, Matrix)
        assert np.array_equal(m_mul, [[2, 4], [6, 8]])
        
        # Matrix multiplication
        m_matmul = m1 @ m2
        assert isinstance(m_matmul, Matrix)
        assert np.array_equal(m_matmul, [[19, 22], [43, 50]])
        
        # Matrix-vector multiplication
        v = Vector([1, 2])
        mv = m1 @ v
        assert isinstance(mv, Vector)
        assert np.array_equal(mv, [5, 11])
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Non-2D data should raise error
        with pytest.raises(ValueError):
            Matrix([1, 2, 3])  # 1D without reshape info
        
        # Inverse of singular matrix
        m_singular = Matrix([[1, 2], [2, 4]])
        with pytest.raises(np.linalg.LinAlgError):
            m_singular.inv()
    
    def test_representation(self):
        """Test string representation."""
        m = Matrix([[1, 2], [3, 4]])
        repr_str = repr(m)
        assert "Matrix" in repr_str