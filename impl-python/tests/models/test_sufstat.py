"""Tests for sufficient statistics classes."""

import pytest
import numpy as np
from boom.models.sufstat import *
from boom.linalg import Vector, Matrix


class TestGaussianSuf:
    """Test GaussianSuf class."""
    
    def test_construction(self):
        """Test construction."""
        suf = GaussianSuf()
        assert suf.n() == 0
        assert suf.sum() == 0.0
        assert suf.sumsq() == 0.0
    
    def test_update_single_values(self):
        """Test updating with single values."""
        suf = GaussianSuf()
        
        suf.update(3.0)
        assert suf.n() == 1
        assert suf.sum() == 3.0
        assert suf.sumsq() == 9.0
        
        suf.update(2.0)
        assert suf.n() == 2
        assert suf.sum() == 5.0
        assert suf.sumsq() == 13.0  # 9 + 4
    
    def test_update_list_values(self):
        """Test updating with list of values."""
        suf = GaussianSuf()
        
        suf.update([1.0, 2.0, 3.0])
        assert suf.n() == 3
        assert suf.sum() == 6.0
        assert suf.sumsq() == 14.0  # 1 + 4 + 9
    
    def test_statistics(self):
        """Test computed statistics."""
        suf = GaussianSuf()
        suf.update([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Mean
        assert suf.mean() == 3.0
        
        # Sample variance
        sample_var = suf.variance(sample=True)
        expected_sample_var = np.var([1, 2, 3, 4, 5], ddof=1)
        assert abs(sample_var - expected_sample_var) < 1e-10
        
        # Population variance
        pop_var = suf.variance(sample=False)
        expected_pop_var = np.var([1, 2, 3, 4, 5], ddof=0)
        assert abs(pop_var - expected_pop_var) < 1e-10
    
    def test_empty_statistics(self):
        """Test statistics with no data."""
        suf = GaussianSuf()
        
        assert suf.mean() == 0.0
        assert suf.variance(sample=True) == 0.0
        assert suf.variance(sample=False) == 0.0
    
    def test_single_point_statistics(self):
        """Test statistics with single data point."""
        suf = GaussianSuf()
        suf.update(5.0)
        
        assert suf.mean() == 5.0
        assert suf.variance(sample=True) == 0.0  # No variance with n=1
        assert suf.variance(sample=False) == 0.0
    
    def test_combine(self):
        """Test combining sufficient statistics."""
        suf1 = GaussianSuf()
        suf1.update([1.0, 2.0])
        
        suf2 = GaussianSuf()
        suf2.update([3.0, 4.0])
        
        combined = suf1.combine(suf2)
        
        assert combined.n() == 4
        assert combined.sum() == 10.0
        assert combined.sumsq() == 30.0  # 1 + 4 + 9 + 16
        assert combined.mean() == 2.5
    
    def test_combine_type_error(self):
        """Test combine with wrong type."""
        suf = GaussianSuf()
        
        with pytest.raises(TypeError):
            suf.combine("not a sufstat")
    
    def test_vectorization(self):
        """Test vectorization."""
        suf = GaussianSuf()
        suf.update([1.0, 2.0, 3.0])
        
        # Vectorize
        vec = suf.vectorize()
        assert len(vec) == 3
        assert vec[0] == 3.0  # n
        assert vec[1] == 6.0  # sum
        assert vec[2] == 14.0  # sumsq
        
        # Unvectorize
        new_suf = GaussianSuf()
        new_suf.unvectorize(Vector([2.0, 5.0, 13.0]))
        
        assert new_suf.n() == 2
        assert new_suf.sum() == 5.0
        assert new_suf.sumsq() == 13.0
    
    def test_unvectorize_error(self):
        """Test unvectorize with wrong size."""
        suf = GaussianSuf()
        
        with pytest.raises(ValueError):
            suf.unvectorize(Vector([1.0, 2.0]))  # Too short
    
    def test_clear(self):
        """Test clearing."""
        suf = GaussianSuf()
        suf.update([1.0, 2.0, 3.0])
        
        suf.clear()
        assert suf.n() == 0
        assert suf.sum() == 0.0
        assert suf.sumsq() == 0.0
    
    def test_clone(self):
        """Test cloning."""
        suf = GaussianSuf()
        suf.update([1.0, 2.0, 3.0])
        
        cloned = suf.clone()
        assert cloned.n() == suf.n()
        assert cloned.sum() == suf.sum()
        assert cloned.sumsq() == suf.sumsq()
        assert cloned is not suf
    
    def test_str_representation(self):
        """Test string representation."""
        suf = GaussianSuf()
        suf.update([1.0, 2.0, 3.0])
        
        s = str(suf)
        assert "GaussianSuf" in s
        assert "n=3" in s
        assert "mean=2.000" in s


class TestMultivariateGaussianSuf:
    """Test MultivariateGaussianSuf class."""
    
    def test_construction(self):
        """Test construction."""
        # With dimension
        suf = MultivariateGaussianSuf(3)
        assert suf.dim() == 3
        assert suf.n() == 0
        
        # Without dimension (inferred)
        suf = MultivariateGaussianSuf()
        assert suf.dim() == 0
        assert suf.n() == 0
    
    def test_update_infer_dimension(self):
        """Test updating with dimension inference."""
        suf = MultivariateGaussianSuf()
        
        # First update should set dimension
        suf.update(Vector([1, 2, 3]))
        assert suf.dim() == 3
        assert suf.n() == 1
        
        # Subsequent updates should match dimension
        suf.update([4, 5, 6])
        assert suf.n() == 2
    
    def test_update_dimension_mismatch(self):
        """Test updating with mismatched dimension."""
        suf = MultivariateGaussianSuf(3)
        
        with pytest.raises(ValueError):
            suf.update([1, 2])  # Wrong dimension
    
    def test_update_multiple_formats(self):
        """Test updating with different data formats."""
        suf = MultivariateGaussianSuf(2)
        
        # Vector
        suf.update(Vector([1, 2]))
        
        # List
        suf.update([3, 4])
        
        # NumPy array
        suf.update(np.array([5, 6]))
        
        assert suf.n() == 3
        sum_vec = suf.sum()
        assert list(sum_vec) == [9.0, 12.0]
    
    def test_statistics(self):
        """Test computed statistics."""
        suf = MultivariateGaussianSuf(2)
        
        # Add some data
        suf.update([1, 2])
        suf.update([3, 4])
        suf.update([5, 6])
        
        # Mean
        mean = suf.mean()
        assert list(mean) == [3.0, 4.0]
        
        # Sum
        sum_vec = suf.sum()
        assert list(sum_vec) == [9.0, 12.0]
        
        # Sum of squares matrix
        sumsq = suf.sumsq()
        expected_sumsq = np.array([[35, 44], [44, 56]])  # 1²+3²+5², 1*2+3*4+5*6, etc.
        np.testing.assert_array_equal(sumsq.to_numpy(), expected_sumsq)
    
    def test_covariance(self):
        """Test covariance computation."""
        suf = MultivariateGaussianSuf(2)
        
        # Add some data points
        data = [[1, 2], [3, 4], [5, 6]]
        for point in data:
            suf.update(point)
        
        # Sample covariance
        sample_cov = suf.covariance(sample=True)
        expected_cov = np.cov(np.array(data).T, ddof=1)
        np.testing.assert_array_almost_equal(sample_cov.to_numpy(), expected_cov)
        
        # Population covariance
        pop_cov = suf.covariance(sample=False)
        expected_pop_cov = np.cov(np.array(data).T, ddof=0)
        np.testing.assert_array_almost_equal(pop_cov.to_numpy(), expected_pop_cov)
    
    def test_empty_statistics(self):
        """Test statistics with no data."""
        suf = MultivariateGaussianSuf(2)
        
        assert suf.mean() is None
        assert suf.sum() is None
        assert suf.sumsq() is None
        assert suf.covariance() is None
    
    def test_combine(self):
        """Test combining sufficient statistics."""
        suf1 = MultivariateGaussianSuf(2)
        suf1.update([1, 2])
        suf1.update([3, 4])
        
        suf2 = MultivariateGaussianSuf(2)
        suf2.update([5, 6])
        
        combined = suf1.combine(suf2)
        
        assert combined.n() == 3
        assert combined.dim() == 2
        assert list(combined.sum()) == [9.0, 12.0]
    
    def test_combine_dimension_mismatch(self):
        """Test combine with mismatched dimensions."""
        suf1 = MultivariateGaussianSuf(2)
        suf2 = MultivariateGaussianSuf(3)
        
        with pytest.raises(ValueError):
            suf1.combine(suf2)
    
    def test_vectorization(self):
        """Test vectorization."""
        suf = MultivariateGaussianSuf(2)
        suf.update([1, 2])
        suf.update([3, 4])
        
        # Vectorize (minimal)
        vec = suf.vectorize(minimal=True)
        # Should contain: n, sum[0], sum[1], sumsq[0,0], sumsq[0,1], sumsq[1,1]
        assert len(vec) == 6  # n + 2 (sum) + 3 (upper triangle of 2x2)
        assert vec[0] == 2.0  # n
        assert vec[1] == 4.0  # sum[0]
        assert vec[2] == 6.0  # sum[1]
        
        # Unvectorize
        new_suf = MultivariateGaussianSuf(2)
        new_suf.unvectorize(vec, minimal=True)
        
        assert new_suf.n() == 2
        assert list(new_suf.sum()) == [4.0, 6.0]
    
    def test_clone(self):
        """Test cloning."""
        suf = MultivariateGaussianSuf(2)
        suf.update([1, 2])
        
        cloned = suf.clone()
        assert cloned.n() == suf.n()
        assert cloned.dim() == suf.dim()
        assert list(cloned.sum()) == list(suf.sum())
        assert cloned is not suf


class TestBinomialSuf:
    """Test BinomialSuf class."""
    
    def test_construction(self):
        """Test construction."""
        suf = BinomialSuf()
        assert suf.n() == 0
        assert suf.successes() == 0
        assert suf.failures() == 0
    
    def test_update_tuple(self):
        """Test updating with tuple."""
        suf = BinomialSuf()
        
        suf.update((10, 7))  # 10 trials, 7 successes
        assert suf.n() == 10
        assert suf.successes() == 7
        assert suf.failures() == 3
        
        suf.update((5, 2))  # Add more trials
        assert suf.n() == 15
        assert suf.successes() == 9
        assert suf.failures() == 6
    
    def test_update_dict(self):
        """Test updating with dictionary."""
        suf = BinomialSuf()
        
        suf.update({'trials': 8, 'successes': 3})
        assert suf.n() == 8
        assert suf.successes() == 3
        assert suf.failures() == 5
    
    def test_success_rate(self):
        """Test success rate computation."""
        suf = BinomialSuf()
        
        # No trials
        assert suf.success_rate() == 0.0
        
        # With trials
        suf.update((10, 3))
        assert suf.success_rate() == 0.3
        
        suf.update((10, 7))  # Total: 20 trials, 10 successes
        assert suf.success_rate() == 0.5
    
    def test_combine(self):
        """Test combining sufficient statistics."""
        suf1 = BinomialSuf()
        suf1.update((10, 4))
        
        suf2 = BinomialSuf()
        suf2.update((5, 3))
        
        combined = suf1.combine(suf2)
        
        assert combined.n() == 15
        assert combined.successes() == 7
        assert combined.failures() == 8
        assert abs(combined.success_rate() - 7/15) < 1e-10
    
    def test_vectorization(self):
        """Test vectorization."""
        suf = BinomialSuf()
        suf.update((10, 6))
        
        # Vectorize
        vec = suf.vectorize()
        assert len(vec) == 2
        assert vec[0] == 10.0  # n
        assert vec[1] == 6.0   # successes
        
        # Unvectorize
        new_suf = BinomialSuf()
        new_suf.unvectorize(Vector([8.0, 3.0]))
        
        assert new_suf.n() == 8
        assert new_suf.successes() == 3
    
    def test_unvectorize_error(self):
        """Test unvectorize with wrong size."""
        suf = BinomialSuf()
        
        with pytest.raises(ValueError):
            suf.unvectorize(Vector([1.0]))  # Too short
    
    def test_clear(self):
        """Test clearing."""
        suf = BinomialSuf()
        suf.update((10, 5))
        
        suf.clear()
        assert suf.n() == 0
        assert suf.successes() == 0
    
    def test_clone(self):
        """Test cloning."""
        suf = BinomialSuf()
        suf.update((10, 3))
        
        cloned = suf.clone()
        assert cloned.n() == suf.n()
        assert cloned.successes() == suf.successes()
        assert cloned is not suf
    
    def test_str_representation(self):
        """Test string representation."""
        suf = BinomialSuf()
        suf.update((10, 3))
        
        s = str(suf)
        assert "BinomialSuf" in s
        assert "n=10" in s
        assert "successes=3" in s
        assert "rate=0.300" in s