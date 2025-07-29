"""Tests for SpdMatrix class."""
import pytest
import numpy as np
from boom.linalg import SpdMatrix, Matrix, Vector


class TestSpdMatrix:
    """Test suite for SpdMatrix class."""
    
    def test_construction(self):
        """Test various ways to construct an SpdMatrix."""
        # From symmetric array
        m1 = SpdMatrix([[2, 1], [1, 2]])
        assert m1.shape == (2, 2)
        assert m1.is_symmetric()
        assert m1.is_pos_def()
        
        # From integer (creates identity)
        m2 = SpdMatrix(3)
        assert m2.shape == (3, 3)
        assert np.array_equal(m2, np.eye(3))
        
        # From non-symmetric array (symmetrized)
        m3 = SpdMatrix([[2, 1], [0, 2]])
        assert m3.is_symmetric()
        assert np.array_equal(m3, [[2, 0.5], [0.5, 2]])
    
    def test_class_methods(self):
        """Test class construction methods."""
        # Identity
        m1 = SpdMatrix.identity(4)
        assert np.array_equal(m1, np.eye(4))
        
        # From correlation and standard deviations
        corr = Matrix([[1, 0.5], [0.5, 1]])
        sd = Vector([2, 3])
        cov = SpdMatrix.from_correlation(corr, sd)
        expected = Matrix([[4, 3], [3, 9]])
        assert np.allclose(cov, expected)
    
    def test_cholesky(self):
        """Test Cholesky decomposition."""
        m = SpdMatrix([[4, 2], [2, 3]])
        
        # Lower triangular
        L = m.chol(lower=True)
        assert np.allclose(L @ L.T, m)
        assert np.all(np.triu(L, 1) == 0)  # Upper triangle is zero
        
        # Upper triangular
        U = m.chol(lower=False)
        assert np.allclose(U.T @ U, m)
        assert np.all(np.tril(U, -1) == 0)  # Lower triangle is zero
        
        # Non-positive definite
        m_bad = SpdMatrix([[1, 2], [2, 1]])
        with pytest.raises(ValueError):
            m_bad.chol()
    
    def test_inverse(self):
        """Test matrix inversion."""
        m = SpdMatrix([[2, 1], [1, 2]])
        m_inv = m.inv()
        
        assert isinstance(m_inv, SpdMatrix)
        identity = m @ m_inv
        assert np.allclose(identity, np.eye(2))
    
    def test_solve(self):
        """Test solving linear systems."""
        m = SpdMatrix([[4, 2], [2, 3]])
        b = Vector([6, 5])
        
        x = m.solve(b)
        assert isinstance(x, Vector)
        assert np.allclose(m @ x, b)
        
        # Multiple right-hand sides
        B = Matrix([[6, 1], [5, 2]])
        X = m.solve(B)
        assert isinstance(X, Matrix)
        assert np.allclose(m @ X, B)
    
    def test_logdet(self):
        """Test log determinant."""
        m = SpdMatrix([[4, 2], [2, 3]])
        
        logdet = m.logdet()
        det = m.det()
        assert np.isclose(logdet, np.log(det))
        
        # Large matrix (where det might overflow)
        n = 10
        m_large = SpdMatrix.identity(n).add_to_diag(1)  # 2*I
        logdet_large = m_large.logdet()
        assert np.isclose(logdet_large, n * np.log(2))
    
    def test_is_pos_def(self):
        """Test positive definite check."""
        # Positive definite
        m1 = SpdMatrix([[2, 1], [1, 2]])
        assert m1.is_pos_def()
        
        # Not positive definite
        m2 = SpdMatrix([[1, 2], [2, 1]])
        assert not m2.is_pos_def()
        
        # Barely positive definite
        m3 = SpdMatrix([[1, 0.99], [0.99, 1]])
        assert m3.is_pos_def()
    
    def test_diagonal_operations(self):
        """Test diagonal manipulation."""
        m = SpdMatrix([[2, 1], [1, 2]])
        
        # Add to diagonal
        m2 = m.add_to_diag(0.5)
        assert np.array_equal(m2.diag(), [2.5, 2.5])
        assert np.array_equal(m2[0, 1], 1)  # Off-diagonal unchanged
        
        # Get variances
        var = m.var()
        assert isinstance(var, Vector)
        assert np.array_equal(var, [2, 2])
        
        # Get standard deviations
        sd = m.sd()
        assert isinstance(sd, Vector)
        assert np.allclose(sd, [np.sqrt(2), np.sqrt(2)])
    
    def test_scale_rows(self):
        """Test row/column scaling."""
        m = SpdMatrix([[4, 2], [2, 1]])
        v = Vector([2, 3])
        
        scaled = m.scale_rows(v)
        # Result should be diag(v) @ m @ diag(v)
        expected = SpdMatrix([[16, 12], [12, 9]])
        assert np.allclose(scaled, expected)
    
    def test_quadratic_forms(self):
        """Test quadratic form computations."""
        m = SpdMatrix([[2, 1], [1, 2]])
        x = Vector([1, 2])
        
        # x'Ax
        quad = m.quad_form(x)
        # x'Ax = [1,2] @ [[2,1],[1,2]] @ [1,2] = [1,2] @ [4,5] = 14
        assert np.isclose(quad, 14)
    
    def test_triple_products(self):
        """Test triple product operations."""
        A = SpdMatrix([[4, 2], [2, 3]])
        B = Matrix([[1, 0], [1, 1]])
        
        # B'AB
        triple = A.triple_product(B)
        assert isinstance(triple, SpdMatrix)
        expected = B.T @ A @ B
        assert np.allclose(triple, expected)
        
        # BAB'
        sandwich = A.sandwich(B)
        assert isinstance(sandwich, SpdMatrix)
        expected = B @ A @ B.T
        assert np.allclose(sandwich, expected)
    
    def test_condition_number(self):
        """Test condition number calculation."""
        # Well-conditioned matrix
        m1 = SpdMatrix([[2, 0.1], [0.1, 2]])
        cond1 = m1.condition_number()
        assert cond1 < 2
        
        # Ill-conditioned matrix
        m2 = SpdMatrix([[1, 0.99], [0.99, 1]])
        cond2 = m2.condition_number()
        assert cond2 > 100
    
    def test_correlation_conversion(self):
        """Test conversion to correlation matrix."""
        # Covariance matrix
        cov = SpdMatrix([[4, 3], [3, 9]])
        
        corr = cov.to_correlation()
        assert isinstance(corr, SpdMatrix)
        assert np.allclose(corr.diag(), [1, 1])
        assert np.isclose(corr[0, 1], 0.5)  # 3/(2*3) = 0.5
    
    def test_arithmetic(self):
        """Test arithmetic operations preserve SpdMatrix type."""
        m1 = SpdMatrix([[2, 1], [1, 2]])
        m2 = SpdMatrix([[1, 0.5], [0.5, 1]])
        
        # Addition
        m_add = m1 + m2
        assert isinstance(m_add, SpdMatrix)
        assert np.array_equal(m_add, [[3, 1.5], [1.5, 3]])
        
        # Subtraction (result might not be positive definite)
        m_sub = m1 - m2
        assert isinstance(m_sub, Matrix) or isinstance(m_sub, SpdMatrix)
        
        # Scalar multiplication
        m_mul = m1 * 2
        assert isinstance(m_mul, SpdMatrix)
        assert np.array_equal(m_mul, [[4, 2], [2, 4]])
        
        # Scalar division
        m_div = m1 / 2
        assert isinstance(m_div, SpdMatrix)
        assert np.array_equal(m_div, [[1, 0.5], [0.5, 1]])
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Non-square matrix
        with pytest.raises(ValueError):
            SpdMatrix([[1, 2, 3], [4, 5, 6]])
        
        # 1x1 matrix
        m1 = SpdMatrix([[4]])
        assert m1.shape == (1, 1)
        assert m1.is_pos_def()
        assert m1.chol()[0, 0] == 2
    
    def test_representation(self):
        """Test string representation."""
        m = SpdMatrix(2)
        repr_str = repr(m)
        assert "SpdMatrix" in repr_str