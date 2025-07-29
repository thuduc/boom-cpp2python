"""Tests for Vector class."""
import pytest
import numpy as np
from boom.linalg import Vector


class TestVector:
    """Test suite for Vector class."""
    
    def test_construction(self):
        """Test various ways to construct a Vector."""
        # From list
        v1 = Vector([1, 2, 3])
        assert len(v1) == 3
        assert np.array_equal(v1, [1, 2, 3])
        
        # From numpy array
        v2 = Vector(np.array([4, 5, 6]))
        assert len(v2) == 3
        assert np.array_equal(v2, [4, 5, 6])
        
        # From scalar
        v3 = Vector(5.0)
        assert len(v3) == 1
        assert v3[0] == 5.0
        
        # From size (creates zero vector)
        v4 = Vector(4)
        assert len(v4) == 4
        assert np.array_equal(v4, [0, 0, 0, 0])
        
        # Empty vector
        v5 = Vector()
        assert len(v5) == 0
        
    def test_class_methods(self):
        """Test class construction methods."""
        # Zero vector
        v1 = Vector.zero(5)
        assert len(v1) == 5
        assert np.all(v1 == 0)
        
        # Ones vector
        v2 = Vector.ones(3)
        assert len(v2) == 3
        assert np.all(v2 == 1)
        
        # Sequence
        v3 = Vector.seq(1, 5, 1)
        assert np.array_equal(v3, [1, 2, 3, 4, 5])
        
        v4 = Vector.seq(0, 1, 0.25)
        assert np.array_equal(v4, [0, 0.25, 0.5, 0.75, 1.0])
    
    def test_properties(self):
        """Test vector properties."""
        v = Vector([1, 2, 3, 4, 5])
        
        assert v.inc() == 1
        assert v.is_all_finite()
        assert v.all_positive()
        assert v.all_non_negative()
        
        # Test with negative values
        v2 = Vector([-1, 0, 1])
        assert not v2.all_positive()
        assert not v2.all_non_negative()
        
        # Test with inf/nan
        v3 = Vector([1, np.inf])
        assert not v3.is_all_finite()
        
        v4 = Vector([1, np.nan])
        assert not v4.is_all_finite()
    
    def test_statistics(self):
        """Test statistical methods."""
        v = Vector([1, 2, 3, 4, 5])
        
        assert v.min() == 1
        assert v.max() == 5
        assert v.imin() == 0
        assert v.imax() == 4
        assert v.sum() == 15
        assert v.mean() == 3
        assert v.prod() == 120
        assert np.isclose(v.var(), 2.5)
        assert np.isclose(v.sd(), np.sqrt(2.5))
        
        # Empty vector
        v_empty = Vector()
        assert v_empty.min() == np.inf
        assert v_empty.max() == -np.inf
        assert v_empty.imin() == -1
        assert v_empty.imax() == -1
        assert np.isnan(v_empty.mean())
    
    def test_norms(self):
        """Test norm calculations."""
        v = Vector([3, 4])
        
        assert v.abs_norm() == 7
        assert v.normsq() == 25
        assert v.norm() == 5
    
    def test_operations(self):
        """Test vector operations."""
        v = Vector([1, 2, 3])
        
        # Normalize
        v_norm = v.normalize_prob()
        assert np.isclose(v_norm.sum(), 1.0)
        assert np.allclose(v_norm, [1/6, 2/6, 3/6])
        
        # Cumsum
        v_cumsum = v.cumsum()
        assert np.array_equal(v_cumsum, [1, 3, 6])
        
        # Sort
        v2 = Vector([3, 1, 4, 1, 5])
        v_sorted = v2.sort()
        assert np.array_equal(v_sorted, [1, 1, 3, 4, 5])
        
        # Permute
        v_perm = v.permute([2, 0, 1])
        assert np.array_equal(v_perm, [3, 1, 2])
    
    def test_dot_outer(self):
        """Test dot and outer products."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        
        # Dot product
        dot = v1.dot(v2)
        assert dot == 32  # 1*4 + 2*5 + 3*6
        
        # Outer product
        outer = v1.outer(v2)
        expected = np.array([[4, 5, 6],
                            [8, 10, 12],
                            [12, 15, 18]])
        assert np.array_equal(outer, expected)
    
    def test_arithmetic(self):
        """Test arithmetic operations."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        
        # Addition
        v_add = v1 + v2
        assert isinstance(v_add, Vector)
        assert np.array_equal(v_add, [5, 7, 9])
        
        # Subtraction
        v_sub = v2 - v1
        assert isinstance(v_sub, Vector)
        assert np.array_equal(v_sub, [3, 3, 3])
        
        # Multiplication
        v_mul = v1 * 2
        assert isinstance(v_mul, Vector)
        assert np.array_equal(v_mul, [2, 4, 6])
        
        # Division
        v_div = v2 / 2
        assert isinstance(v_div, Vector)
        assert np.array_equal(v_div, [2, 2.5, 3])
        
        # Negation
        v_neg = -v1
        assert isinstance(v_neg, Vector)
        assert np.array_equal(v_neg, [-1, -2, -3])
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Cannot normalize zero vector
        v_zero = Vector([0, 0, 0])
        with pytest.raises(ValueError):
            v_zero.normalize_prob()
        
        # Non-1D data should raise error
        with pytest.raises(ValueError):
            Vector([[1, 2], [3, 4]])
    
    def test_representation(self):
        """Test string representation."""
        v = Vector([1, 2, 3])
        repr_str = repr(v)
        assert "Vector" in repr_str
        assert "[1" in repr_str or "1." in repr_str