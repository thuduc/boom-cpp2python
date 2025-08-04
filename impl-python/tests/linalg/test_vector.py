"""Comprehensive tests for Vector class."""

import pytest
import numpy as np
from boom.linalg import Vector


class TestVectorConstruction:
    """Test Vector construction methods."""
    
    def test_empty_vector(self):
        """Test creating empty vector."""
        v = Vector()
        assert len(v) == 0
        assert v.size() == 0
    
    def test_sized_vector(self):
        """Test creating vector of specific size."""
        v = Vector(5, 3.14)
        assert len(v) == 5
        assert all(x == 3.14 for x in v)
    
    def test_from_list(self):
        """Test creating vector from list."""
        v = Vector([1, 2, 3, 4])
        assert len(v) == 4
        assert list(v) == [1.0, 2.0, 3.0, 4.0]
    
    def test_from_numpy(self):
        """Test creating vector from numpy array."""
        arr = np.array([1.5, 2.5, 3.5])
        v = Vector(arr)
        assert len(v) == 3
        assert list(v) == [1.5, 2.5, 3.5]
    
    def test_from_string_spaces(self):
        """Test creating vector from space-separated string."""
        v = Vector("1.0 2.0 3.0 4.0")
        assert list(v) == [1.0, 2.0, 3.0, 4.0]
    
    def test_from_string_commas(self):
        """Test creating vector from comma-separated string."""
        v = Vector("1.0,2.0,3.0,4.0")
        assert list(v) == [1.0, 2.0, 3.0, 4.0]
    
    def test_from_string_mixed(self):
        """Test creating vector from mixed separator string."""
        v = Vector("1.0, 2.0  3.0,4.0")
        assert list(v) == [1.0, 2.0, 3.0, 4.0]
    
    def test_from_string_custom_sep(self):
        """Test creating vector with custom separator."""
        v = Vector("1.0;2.0;3.0", sep=";")
        assert list(v) == [1.0, 2.0, 3.0]
    
    def test_copy_constructor(self):
        """Test copy constructor."""
        v1 = Vector([1, 2, 3])
        v2 = Vector(v1)
        assert v1 == v2
        assert v1 is not v2
        v2[0] = 999
        assert v1[0] == 1  # Original unchanged


class TestVectorAccess:
    """Test element access methods."""
    
    def test_indexing(self):
        """Test basic indexing."""
        v = Vector([1, 2, 3, 4])
        assert v[0] == 1.0
        assert v[3] == 4.0
        assert v[-1] == 4.0
    
    def test_slicing(self):
        """Test slicing."""
        v = Vector([1, 2, 3, 4, 5])
        v_slice = v[1:4]
        assert isinstance(v_slice, Vector)
        assert list(v_slice) == [2.0, 3.0, 4.0]
    
    def test_call_operator(self):
        """Test call operator for element access."""
        v = Vector([10, 20, 30])
        assert v(0) == 10.0
        assert v(2) == 30.0
    
    def test_call_operator_bounds(self):
        """Test call operator bounds checking."""
        v = Vector([1, 2, 3])
        with pytest.raises(IndexError):
            v(3)
        with pytest.raises(IndexError):
            v(-1)
    
    def test_setitem(self):
        """Test setting elements."""
        v = Vector([1, 2, 3])
        v[1] = 99
        assert v[1] == 99
        assert list(v) == [1.0, 99.0, 3.0]
    
    def test_properties(self):
        """Test size/length/stride properties."""
        v = Vector([1, 2, 3, 4])
        assert v.size() == 4
        assert v.length() == 4
        assert v.stride() == 1
        assert len(v) == 4


class TestVectorArithmetic:
    """Test arithmetic operations."""
    
    def test_add_scalar(self):
        """Test adding scalar to vector."""
        v = Vector([1, 2, 3])
        v2 = v + 10
        assert list(v2) == [11.0, 12.0, 13.0]
        assert list(v) == [1.0, 2.0, 3.0]  # Original unchanged
    
    def test_add_vector(self):
        """Test adding two vectors."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([10, 20, 30])
        v3 = v1 + v2
        assert list(v3) == [11.0, 22.0, 33.0]
    
    def test_iadd_scalar(self):
        """Test in-place scalar addition."""
        v = Vector([1, 2, 3])
        v += 5
        assert list(v) == [6.0, 7.0, 8.0]
    
    def test_iadd_vector(self):
        """Test in-place vector addition."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([10, 20, 30])
        v1 += v2
        assert list(v1) == [11.0, 22.0, 33.0]
    
    def test_subtract_scalar(self):
        """Test subtracting scalar from vector."""
        v = Vector([10, 20, 30])
        v2 = v - 5
        assert list(v2) == [5.0, 15.0, 25.0]
    
    def test_subtract_vector(self):
        """Test subtracting vectors."""
        v1 = Vector([10, 20, 30])
        v2 = Vector([1, 2, 3])
        v3 = v1 - v2
        assert list(v3) == [9.0, 18.0, 27.0]
    
    def test_multiply_scalar(self):
        """Test multiplying by scalar."""
        v = Vector([1, 2, 3])
        v2 = v * 5
        assert list(v2) == [5.0, 10.0, 15.0]
    
    def test_multiply_vector(self):
        """Test element-wise multiplication."""
        v1 = Vector([2, 3, 4])
        v2 = Vector([10, 20, 30])
        v3 = v1 * v2
        assert list(v3) == [20.0, 60.0, 120.0]
    
    def test_divide_scalar(self):
        """Test dividing by scalar."""
        v = Vector([10, 20, 30])
        v2 = v / 5
        assert list(v2) == [2.0, 4.0, 6.0]
    
    def test_divide_vector(self):
        """Test element-wise division."""
        v1 = Vector([20, 60, 120])
        v2 = Vector([2, 3, 4])
        v3 = v1 / v2
        assert list(v3) == [10.0, 20.0, 30.0]
    
    def test_negation(self):
        """Test negation operator."""
        v = Vector([1, -2, 3])
        v2 = -v
        assert list(v2) == [-1.0, 2.0, -3.0]
    
    def test_right_operators(self):
        """Test right-hand side operators."""
        v = Vector([1, 2, 3])
        v2 = 10 + v
        assert list(v2) == [11.0, 12.0, 13.0]
        v3 = 2 * v
        assert list(v3) == [2.0, 4.0, 6.0]


class TestLinearAlgebra:
    """Test linear algebra operations."""
    
    def test_dot_product(self):
        """Test dot product."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        assert v1.dot(v2) == 32.0  # 1*4 + 2*5 + 3*6
    
    def test_axpy(self):
        """Test axpy operation (y += a*x)."""
        y = Vector([1, 2, 3])
        x = Vector([10, 20, 30])
        y.axpy(x, 0.5)
        assert list(y) == [6.0, 12.0, 18.0]
    
    def test_affdot(self):
        """Test affine dot product."""
        v1 = Vector([2, 3, 4])  # First element is constant
        v2 = Vector([5, 6])     # One dimension less
        result = v1.affdot(v2)
        assert result == 2 + 3*5 + 4*6  # 2 + 15 + 24 = 41
    
    def test_affdot_wrong_size(self):
        """Test affine dot with wrong size."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])  # Should be size 2
        with pytest.raises(ValueError):
            v1.affdot(v2)
    
    def test_outer_product_self(self):
        """Test outer product with self."""
        v = Vector([1, 2, 3])
        outer = v.outer()
        expected = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        np.testing.assert_array_equal(outer, expected)
    
    def test_outer_product_other(self):
        """Test outer product with another vector."""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4, 5])
        outer = v1.outer(v2)
        expected = np.array([[3, 4, 5], [6, 8, 10]])
        np.testing.assert_array_equal(outer, expected)
    
    def test_outer_product_scaled(self):
        """Test scaled outer product."""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        outer = v1.outer(v2, a=2.0)
        expected = np.array([[6, 8], [12, 16]])
        np.testing.assert_array_equal(outer, expected)


class TestNormalization:
    """Test normalization methods."""
    
    def test_normalize_prob(self):
        """Test probability normalization."""
        v = Vector([1, 2, 3, 4])
        v.normalize_prob()
        assert abs(v.sum() - 1.0) < 1e-10
        assert list(v) == [0.1, 0.2, 0.3, 0.4]
    
    def test_normalize_prob_zero(self):
        """Test probability normalization with zero sum."""
        v = Vector([0, 0, 0])
        v.normalize_prob()
        assert list(v) == [0.0, 0.0, 0.0]
    
    def test_normalize_logprob(self):
        """Test log probability normalization."""
        v = Vector([1, 2, 3])  # log probs
        v.normalize_logprob()
        assert abs(v.sum() - 1.0) < 1e-10
        # Check relative proportions
        assert v[2] > v[1] > v[0]
    
    def test_normalize_L2(self):
        """Test L2 normalization."""
        v = Vector([3, 4])
        v.normalize_L2()
        assert abs(v.norm() - 1.0) < 1e-10
        assert abs(v[0] - 0.6) < 1e-10
        assert abs(v[1] - 0.8) < 1e-10
    
    def test_normalize_L2_zero(self):
        """Test L2 normalization with zero vector."""
        v = Vector([0, 0, 0])
        v.normalize_L2()
        assert list(v) == [0.0, 0.0, 0.0]


class TestNormsAndStats:
    """Test norms and statistical methods."""
    
    def test_normsq(self):
        """Test squared norm."""
        v = Vector([3, 4])
        assert v.normsq() == 25.0
    
    def test_norm(self):
        """Test L2 norm."""
        v = Vector([3, 4])
        assert v.norm() == 5.0
    
    def test_abs_norm(self):
        """Test L1 norm."""
        v = Vector([1, -2, 3, -4])
        assert v.abs_norm() == 10.0
    
    def test_max_abs(self):
        """Test maximum absolute value."""
        v = Vector([1, -5, 3, -2])
        assert v.max_abs() == 5.0
    
    def test_max_abs_empty(self):
        """Test maximum absolute value of empty vector."""
        v = Vector()
        assert v.max_abs() == -1.0
    
    def test_min_max(self):
        """Test min and max."""
        v = Vector([3, 1, 4, 1, 5, 9])
        assert v.min() == 1.0
        assert v.max() == 9.0
        assert v.imin() == 1  # First occurrence of min
        assert v.imax() == 5
    
    def test_sum_prod(self):
        """Test sum and product."""
        v = Vector([2, 3, 4])
        assert v.sum() == 9.0
        assert v.prod() == 24.0


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_all_finite(self):
        """Test all_finite check."""
        v1 = Vector([1, 2, 3])
        assert v1.all_finite()
        
        v2 = Vector([1, np.inf, 3])
        assert not v2.all_finite()
        
        v3 = Vector([1, np.nan, 3])
        assert not v3.all_finite()
    
    def test_set_to_zero(self):
        """Test setting all elements to zero."""
        v = Vector([1, 2, 3])
        v.set_to_zero()
        assert list(v) == [0.0, 0.0, 0.0]
    
    def test_zero_one(self):
        """Test zero() and one() methods."""
        v = Vector([1, 2, 3])
        v_zero = v.zero()
        v_one = v.one()
        
        assert len(v_zero) == 3
        assert list(v_zero) == [0.0, 0.0, 0.0]
        assert len(v_one) == 3
        assert list(v_one) == [1.0, 1.0, 1.0]
    
    def test_sort(self):
        """Test sorting."""
        v = Vector([3, 1, 4, 1, 5, 9])
        v.sort()
        assert list(v) == [1.0, 1.0, 3.0, 4.0, 5.0, 9.0]
    
    def test_concat(self):
        """Test concatenation."""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        v1.concat(v2)
        assert list(v1) == [1.0, 2.0, 3.0, 4.0]
    
    def test_concat_list(self):
        """Test concatenation with list."""
        v = Vector([1, 2])
        v.concat([3, 4])
        assert list(v) == [1.0, 2.0, 3.0, 4.0]
    
    def test_push_back(self):
        """Test push_back."""
        v = Vector([1, 2])
        v.push_back(3)
        assert list(v) == [1.0, 2.0, 3.0]
    
    def test_copy(self):
        """Test copy method."""
        v1 = Vector([1, 2, 3])
        v2 = v1.copy()
        assert v1 == v2
        assert v1 is not v2
        v2[0] = 999
        assert v1[0] == 1
    
    def test_swap(self):
        """Test swap method."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        v1.swap(v2)
        assert list(v1) == [4.0, 5.0, 6.0]
        assert list(v2) == [1.0, 2.0, 3.0]
    
    def test_to_numpy(self):
        """Test conversion to numpy."""
        v = Vector([1, 2, 3])
        arr = v.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
        # Check it's a copy
        arr[0] = 999
        assert v[0] == 1


class TestRandomization:
    """Test randomization methods."""
    
    def test_randomize(self):
        """Test uniform randomization."""
        v = Vector(100)
        rng = np.random.RandomState(42)
        v.randomize(rng)
        
        assert len(v) == 100
        assert all(0 <= x < 1 for x in v)
        assert 0.4 < v.sum() / 100 < 0.6  # Should average around 0.5
    
    def test_randomize_gaussian(self):
        """Test Gaussian randomization."""
        v = Vector(1000)
        rng = np.random.RandomState(42)
        v.randomize_gaussian(mean=5.0, sd=2.0, rng=rng)
        
        assert len(v) == 1000
        mean = v.sum() / 1000
        assert 4.8 < mean < 5.2  # Should be close to 5.0
        
        # Check standard deviation
        variance = sum((x - mean)**2 for x in v) / 1000
        sd = np.sqrt(variance)
        assert 1.8 < sd < 2.2  # Should be close to 2.0


class TestStringRepresentation:
    """Test string representations."""
    
    def test_str(self):
        """Test string representation."""
        v = Vector([1, 2, 3])
        s = str(v)
        assert "1" in s and "2" in s and "3" in s
    
    def test_repr(self):
        """Test detailed representation."""
        v = Vector([1, 2, 3])
        r = repr(v)
        assert "Vector" in r
        assert "1" in r and "2" in r and "3" in r


class TestEquality:
    """Test equality comparisons."""
    
    def test_equal_vectors(self):
        """Test equality of equal vectors."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([1, 2, 3])
        assert v1 == v2
    
    def test_unequal_vectors(self):
        """Test inequality of different vectors."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([1, 2, 4])
        assert v1 != v2
    
    def test_different_sizes(self):
        """Test vectors of different sizes."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([1, 2])
        assert v1 != v2
    
    def test_not_vector(self):
        """Test comparison with non-vector."""
        v = Vector([1, 2, 3])
        assert v != [1, 2, 3]
        assert v != "not a vector"