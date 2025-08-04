"""Tests for parameter classes."""

import pytest
import numpy as np
from boom.models.params import *
from boom.linalg import Vector, Matrix


class TestUnivParams:
    """Test UnivParams class."""
    
    def test_construction(self):
        """Test construction."""
        # Default construction
        param = UnivParams()
        assert param.value() == 0.0
        
        # With initial value
        param = UnivParams(3.14)
        assert param.value() == 3.14
    
    def test_value_access(self):
        """Test value access and modification."""
        param = UnivParams(1.0)
        
        assert param.value() == 1.0
        
        param.set_value(2.5)
        assert param.value() == 2.5
        
        # Float conversion
        assert float(param) == 2.5
    
    def test_vectorization(self):
        """Test vectorization."""
        param = UnivParams(3.0)
        
        # Vectorize
        vec = param.vectorize()
        assert len(vec) == 1
        assert vec[0] == 3.0
        
        # Size
        assert param.size() == 1
        assert param.size(minimal=False) == 1
        
        # Unvectorize
        param.unvectorize(Vector([5.0]))
        assert param.value() == 5.0
    
    def test_unvectorize_error(self):
        """Test unvectorize with wrong size."""
        param = UnivParams()
        
        with pytest.raises(ValueError):
            param.unvectorize(Vector([1.0, 2.0]))
    
    def test_missing_status(self):
        """Test missing status inheritance."""
        param = UnivParams(1.0)
        assert not param.is_missing()
        
        param.set_missing(True)
        assert param.is_missing()
    
    def test_clone(self):
        """Test cloning."""
        param = UnivParams(2.5)
        param.set_missing(True)
        
        cloned = param.clone()
        assert cloned.value() == 2.5
        assert cloned.is_missing()
        assert cloned is not param
    
    def test_str_representation(self):
        """Test string representation."""
        param = UnivParams(1.23)
        s = str(param)
        assert "UnivParams" in s
        assert "1.23" in s


class TestVectorParams:
    """Test VectorParams class."""
    
    def test_construction(self):
        """Test construction."""
        # From dimension
        param = VectorParams(dim=3)
        assert param.dim() == 3
        assert len(param) == 3
        
        # From list
        param = VectorParams([1, 2, 3])
        assert param.dim() == 3
        assert param[0] == 1.0
        assert param[2] == 3.0
        
        # From Vector
        vec = Vector([4, 5, 6])
        param = VectorParams(vec)
        assert param.dim() == 3
        assert param[1] == 5.0
    
    def test_construction_errors(self):
        """Test construction errors."""
        with pytest.raises(ValueError):
            VectorParams()  # No value or dim
        
        with pytest.raises(ValueError):
            VectorParams(dim=0)  # Invalid dim
    
    def test_value_access(self):
        """Test value access and modification."""
        param = VectorParams([1, 2, 3])
        
        # Get value
        val = param.value()
        assert isinstance(val, Vector)
        assert list(val) == [1.0, 2.0, 3.0]
        
        # Set value
        param.set_value([4, 5, 6])
        assert list(param.value()) == [4.0, 5.0, 6.0]
        
        # Element access
        assert param[0] == 4.0
        param[1] = 10.0
        assert param[1] == 10.0
    
    def test_vectorization(self):
        """Test vectorization."""
        param = VectorParams([1, 2, 3])
        
        # Vectorize
        vec = param.vectorize()
        assert len(vec) == 3
        assert list(vec) == [1.0, 2.0, 3.0]
        
        # Size
        assert param.size() == 3
        
        # Unvectorize
        param.unvectorize(Vector([4, 5, 6]))
        assert list(param.value()) == [4.0, 5.0, 6.0]
    
    def test_clone(self):
        """Test cloning."""
        param = VectorParams([1, 2, 3])
        param.set_missing(True)
        
        cloned = param.clone()
        assert list(cloned.value()) == [1.0, 2.0, 3.0]
        assert cloned.is_missing()
        assert cloned is not param
        
        # Modifications should be independent
        cloned[0] = 99
        assert param[0] == 1.0


class TestMatrixParams:
    """Test MatrixParams class."""
    
    def test_construction(self):
        """Test construction."""
        # From dimensions
        param = MatrixParams(nrow=2, ncol=3)
        assert param.nrow() == 2
        assert param.ncol() == 3
        assert param.shape() == (2, 3)
        
        # From list of lists
        param = MatrixParams([[1, 2], [3, 4]])
        assert param.nrow() == 2
        assert param.ncol() == 2
        assert param[0, 0] == 1.0
        assert param[1, 1] == 4.0
        
        # From Matrix
        mat = Matrix([[5, 6], [7, 8]])
        param = MatrixParams(mat)
        assert param[0, 1] == 6.0
        assert param[1, 0] == 7.0
    
    def test_construction_errors(self):
        """Test construction errors."""
        with pytest.raises(ValueError):
            MatrixParams()  # No value or dimensions
        
        with pytest.raises(ValueError):
            MatrixParams(nrow=0, ncol=2)  # Invalid dimensions
    
    def test_value_access(self):
        """Test value access and modification."""
        param = MatrixParams([[1, 2], [3, 4]])
        
        # Get value
        val = param.value()
        assert isinstance(val, Matrix)
        assert val[0, 0] == 1.0
        assert val[1, 1] == 4.0
        
        # Set value
        param.set_value([[5, 6], [7, 8]])
        assert param[0, 0] == 5.0
        assert param[1, 1] == 8.0
        
        # Element access
        param[0, 1] = 99
        assert param[0, 1] == 99.0
    
    def test_vectorization_regular(self):
        """Test vectorization for regular matrices."""
        param = MatrixParams([[1, 2], [3, 4]])
        
        # Vectorize (column-major order)
        vec = param.vectorize(minimal=False)
        assert len(vec) == 4
        assert list(vec) == [1.0, 3.0, 2.0, 4.0]  # Column-major
        
        # Size
        assert param.size(minimal=False) == 4
        
        # Unvectorize
        param.unvectorize(Vector([5, 6, 7, 8]), minimal=False)
        assert param[0, 0] == 5.0  # (0,0)
        assert param[1, 0] == 6.0  # (1,0)
        assert param[0, 1] == 7.0  # (0,1)  
        assert param[1, 1] == 8.0  # (1,1)
    
    def test_vectorization_symmetric(self):
        """Test vectorization for symmetric matrices."""
        # Create symmetric matrix
        param = MatrixParams([[1, 2], [2, 3]])
        
        # Minimal vectorization (upper triangle)
        vec = param.vectorize(minimal=True)
        assert len(vec) == 3  # Upper triangle
        assert list(vec) == [1.0, 2.0, 3.0]
        
        # Size
        assert param.size(minimal=True) == 3
        
        # Unvectorize
        param2 = MatrixParams(nrow=2, ncol=2)
        param2.unvectorize(Vector([4, 5, 6]), minimal=True)
        assert param2[0, 0] == 4.0
        assert param2[0, 1] == 5.0
        assert param2[1, 0] == 5.0  # Should be symmetric
        assert param2[1, 1] == 6.0
    
    def test_clone(self):
        """Test cloning."""
        param = MatrixParams([[1, 2], [3, 4]])
        param.set_missing(True)
        
        cloned = param.clone()
        assert cloned[0, 0] == 1.0
        assert cloned[1, 1] == 4.0
        assert cloned.is_missing()
        assert cloned is not param
        
        # Modifications should be independent
        cloned[0, 0] = 99
        assert param[0, 0] == 1.0


class TestSpdMatrixParams:
    """Test SpdMatrixParams class."""
    
    def test_construction(self):
        """Test construction."""
        # From dimension (creates identity)
        param = SpdMatrixParams(dim=3)
        assert param.nrow() == 3
        assert param.ncol() == 3
        
        # Should be identity matrix
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert param[i, j] == expected
        
        # From matrix
        spd_matrix = [[2, 1], [1, 2]]  # SPD matrix
        param = SpdMatrixParams(spd_matrix)
        assert param[0, 0] == 2.0
        assert param[0, 1] == 1.0
        assert param[1, 0] == 1.0
        assert param[1, 1] == 2.0
    
    def test_construction_non_square_error(self):
        """Test error for non-square matrix."""
        with pytest.raises(ValueError):
            SpdMatrixParams([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
    
    def test_spd_check(self):
        """Test SPD checking."""
        # Identity is SPD
        param = SpdMatrixParams(dim=2)
        assert param.is_spd()
        
        # Create a known SPD matrix
        param = SpdMatrixParams([[2, 1], [1, 2]])
        assert param.is_spd()
        
        # Non-symmetric matrix
        param = SpdMatrixParams([[1, 2], [3, 4]])
        assert not param.is_spd()
    
    def test_vectorization_always_minimal(self):
        """Test that SPD matrices always use minimal vectorization."""
        param = SpdMatrixParams([[2, 1], [1, 3]])
        
        # Should always use minimal (upper triangle)
        vec_minimal = param.vectorize(minimal=True)
        vec_full = param.vectorize(minimal=False)
        
        # Both should be the same for SPD
        assert len(vec_minimal) == len(vec_full) == 3
        assert list(vec_minimal) == list(vec_full)
        
        # Size should always be minimal
        assert param.size(minimal=True) == param.size(minimal=False) == 3
    
    def test_clone(self):
        """Test cloning."""
        param = SpdMatrixParams([[2, 1], [1, 2]])
        cloned = param.clone()
        
        assert isinstance(cloned, SpdMatrixParams)
        assert cloned[0, 0] == 2.0
        assert cloned[0, 1] == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_univariate_param(self):
        """Test create_univariate_param."""
        param = create_univariate_param(5.0)
        assert isinstance(param, UnivParams)
        assert param.value() == 5.0
    
    def test_create_vector_param(self):
        """Test create_vector_param."""
        param = create_vector_param([1, 2, 3])
        assert isinstance(param, VectorParams)
        assert param.dim() == 3
        
        param = create_vector_param(dim=4)
        assert param.dim() == 4
    
    def test_create_matrix_param(self):
        """Test create_matrix_param."""
        param = create_matrix_param([[1, 2], [3, 4]])
        assert isinstance(param, MatrixParams)
        assert param.shape() == (2, 2)
        
        param = create_matrix_param(nrow=2, ncol=3)
        assert param.shape() == (2, 3)
    
    def test_create_spd_matrix_param(self):
        """Test create_spd_matrix_param."""
        param = create_spd_matrix_param(dim=3)
        assert isinstance(param, SpdMatrixParams)
        assert param.shape() == (3, 3)
        
        param = create_spd_matrix_param([[2, 1], [1, 2]])
        assert param.shape() == (2, 2)