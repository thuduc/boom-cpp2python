"""Comprehensive tests for custom distributions."""

import pytest
import numpy as np
from boom.distributions import RNG
from boom.distributions.custom import *


class TestTriangularDistribution:
    """Test triangular distribution implementation."""
    
    def test_construction(self):
        """Test triangular distribution construction."""
        dist = TriangularDistribution(0, 2, 5)
        assert dist.left == 0
        assert dist.mode == 2
        assert dist.right == 5
    
    def test_construction_invalid(self):
        """Test invalid construction parameters."""
        with pytest.raises(ValueError):
            TriangularDistribution(3, 2, 5)  # mode < left
        
        with pytest.raises(ValueError):
            TriangularDistribution(0, 6, 5)  # mode > right
    
    def test_pdf(self):
        """Test probability density function."""
        dist = TriangularDistribution(0, 2, 4)
        
        # At the mode, should have maximum density
        mode_density = dist.pdf(2)
        assert mode_density > 0
        
        # Outside bounds should be zero
        assert dist.pdf(-1) == 0.0
        assert dist.pdf(5) == 0.0
        
        # At boundaries should be zero
        assert dist.pdf(0) == 0.0
        assert dist.pdf(4) == 0.0
        
        # Log scale
        log_density = dist.pdf(1, log=True)
        assert abs(log_density - np.log(dist.pdf(1))) < 1e-10
        
        # Log of zero should be -inf
        assert dist.pdf(-1, log=True) == -np.inf
    
    def test_cdf(self):
        """Test cumulative distribution function."""
        dist = TriangularDistribution(0, 2, 4)
        
        # At boundaries
        assert dist.cdf(-1) == 0.0
        assert dist.cdf(0) == 0.0
        assert dist.cdf(4) == 1.0
        assert dist.cdf(5) == 1.0
        
        # Should be monotonic increasing
        x_vals = np.linspace(0, 4, 100)
        cdf_vals = [dist.cdf(x) for x in x_vals]
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
    
    def test_quantile(self):
        """Test quantile function."""
        dist = TriangularDistribution(0, 2, 4)
        
        # At boundaries
        assert dist.quantile(0) == 0
        assert dist.quantile(1) == 4
        
        # Should be inverse of CDF
        x = 1.5
        p = dist.cdf(x)
        x_recovered = dist.quantile(p)
        assert abs(x - x_recovered) < 1e-10
    
    def test_quantile_invalid(self):
        """Test quantile with invalid probabilities."""
        dist = TriangularDistribution(0, 1, 2)
        
        with pytest.raises(ValueError):
            dist.quantile(-0.1)
        
        with pytest.raises(ValueError):
            dist.quantile(1.1)
    
    def test_rvs(self):
        """Test random sample generation."""
        rng = RNG(42)
        dist = TriangularDistribution(1, 3, 6)
        
        samples = [dist.rvs(rng) for _ in range(1000)]
        
        # All samples should be in bounds
        assert all(1 <= s <= 6 for s in samples)
        
        # Should have some variety
        assert len(set(samples)) > 50
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        left, mode, right = 0, 1, 3
        x = 1.5
        
        # PDF
        dist = TriangularDistribution(left, mode, right)
        expected_pdf = dist.pdf(x)
        assert abs(dtriangle(x, left, mode, right) - expected_pdf) < 1e-10
        
        # CDF
        expected_cdf = dist.cdf(x)
        assert abs(ptriangle(x, left, mode, right) - expected_cdf) < 1e-10
        
        # Quantile
        p = 0.3
        expected_q = dist.quantile(p)
        assert abs(qtriangle(p, left, mode, right) - expected_q) < 1e-10
        
        # Random
        rng = RNG(42)
        sample1 = rtriangle(left, mode, right, rng)
        
        rng = RNG(42)  # Reset
        sample2 = dist.rvs(rng)
        
        assert abs(sample1 - sample2) < 1e-10


class TestTruncatedNormal:
    """Test truncated normal distribution."""
    
    def test_construction(self):
        """Test construction."""
        dist = TruncatedNormal(0, 1, -2, 2)
        assert dist.mu == 0
        assert dist.sigma == 1
        assert dist.lower == -2
        assert dist.upper == 2
    
    def test_construction_invalid(self):
        """Test invalid construction parameters."""
        with pytest.raises(ValueError):
            TruncatedNormal(0, -1, -2, 2)  # negative sigma
        
        with pytest.raises(ValueError):
            TruncatedNormal(0, 1, 2, -2)  # lower > upper
    
    def test_pdf(self):
        """Test PDF."""
        dist = TruncatedNormal(0, 1, -1, 1)
        
        # Should be positive in the truncated region
        assert dist.pdf(0) > 0
        assert dist.pdf(0.5) > 0
        
        # Should be zero outside truncated region
        assert dist.pdf(-2) == 0
        assert dist.pdf(2) == 0
        
        # Log scale
        x = 0.5
        log_pdf = dist.pdf(x, log=True)
        assert abs(log_pdf - np.log(dist.pdf(x))) < 1e-10
    
    def test_cdf(self):
        """Test CDF."""
        dist = TruncatedNormal(0, 1, -1, 1)
        
        # At boundaries
        assert dist.cdf(-2) == 0
        assert dist.cdf(-1) == 0
        assert dist.cdf(1) == 1
        assert dist.cdf(2) == 1
        
        # Should be monotonic
        x_vals = np.linspace(-1, 1, 100)
        cdf_vals = [dist.cdf(x) for x in x_vals]
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
    
    def test_quantile(self):
        """Test quantile function."""
        dist = TruncatedNormal(0, 1, -1, 1)
        
        # At boundaries
        assert dist.quantile(0) == -1
        assert dist.quantile(1) == 1
        
        # Should be inverse of CDF
        x = 0.5
        p = dist.cdf(x)
        x_recovered = dist.quantile(p)
        assert abs(x - x_recovered) < 1e-10
    
    def test_rvs(self):
        """Test random sampling."""
        rng = RNG(42)
        dist = TruncatedNormal(0, 1, -2, 2)
        
        samples = [dist.rvs(rng) for _ in range(1000)]
        
        # All should be in truncated region
        assert all(-2 <= s <= 2 for s in samples)
        
        # Should have reasonable mean (close to original mean if not heavily truncated)
        mean = np.mean(samples)
        assert -0.5 < mean < 0.5


class TestInverseGamma:
    """Test inverse gamma distribution."""
    
    def test_construction(self):
        """Test construction."""
        dist = InverseGamma(2, 3)
        assert dist.alpha == 2
        assert dist.beta == 3
    
    def test_construction_invalid(self):
        """Test invalid construction."""
        with pytest.raises(ValueError):
            InverseGamma(-1, 1)  # negative alpha
        
        with pytest.raises(ValueError):
            InverseGamma(1, -1)  # negative beta
    
    def test_pdf(self):
        """Test PDF."""
        dist = InverseGamma(2, 1)
        
        # Should be positive for positive x
        assert dist.pdf(1) > 0
        assert dist.pdf(0.5) > 0
        
        # Should be zero for non-positive x
        assert dist.pdf(0) == 0
        assert dist.pdf(-1) == 0
        
        # Log scale
        x = 1.5
        log_pdf = dist.pdf(x, log=True)
        assert abs(log_pdf - np.log(dist.pdf(x))) < 1e-10
    
    def test_cdf(self):
        """Test CDF."""
        dist = InverseGamma(2, 1)
        
        # Should be zero at x=0
        assert dist.cdf(0) == 0
        
        # Should be monotonic increasing
        x_vals = np.linspace(0.1, 5, 100)
        cdf_vals = [dist.cdf(x) for x in x_vals]
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))
    
    def test_quantile(self):
        """Test quantile function."""
        dist = InverseGamma(2, 1)
        
        # At boundaries
        assert dist.quantile(0) == 0
        assert dist.quantile(1) == np.inf
        
        # Should be inverse of CDF
        x = 1.5
        p = dist.cdf(x)
        x_recovered = dist.quantile(p)
        assert abs(x - x_recovered) < 1e-10
    
    def test_rvs(self):
        """Test random sampling."""
        rng = RNG(42)
        dist = InverseGamma(3, 2)
        
        samples = [dist.rvs(rng) for _ in range(1000)]
        
        # All should be positive
        assert all(s > 0 for s in samples)
        
        # Check approximate mean
        expected_mean = dist.mean()
        actual_mean = np.mean(samples)
        assert 0.7 * expected_mean < actual_mean < 1.3 * expected_mean
    
    def test_mean_var(self):
        """Test mean and variance calculations."""
        # Mean exists for alpha > 1
        dist = InverseGamma(2, 3)
        mean = dist.mean()
        assert mean == 3.0  # beta / (alpha - 1) = 3 / 1 = 3
        
        # Variance exists for alpha > 2, so test with alpha = 3
        dist_with_var = InverseGamma(3, 2)
        var = dist_with_var.var()
        expected_var = (2**2) / ((3-1)**2 * (3-2))  # 4 / (4 * 1) = 1
        assert var == expected_var
        
        # Mean should be infinite for alpha <= 1
        dist_no_mean = InverseGamma(0.5, 1)
        assert dist_no_mean.mean() == np.inf
        
        # Variance should be infinite for alpha <= 2
        dist_no_var = InverseGamma(1.5, 1)
        assert dist_no_var.var() == np.inf


class TestDirichlet:
    """Test Dirichlet distribution."""
    
    def test_construction(self):
        """Test construction."""
        alpha = [1, 2, 3]
        dist = Dirichlet(alpha)
        
        np.testing.assert_array_equal(dist.alpha, np.array([1, 2, 3], dtype=float))
        assert dist.k == 3
        assert dist.alpha_sum == 6
    
    def test_construction_invalid(self):
        """Test invalid construction."""
        with pytest.raises(ValueError):
            Dirichlet([1, -1, 2])  # negative alpha
        
        with pytest.raises(ValueError):
            Dirichlet([0, 1, 2])  # zero alpha
    
    def test_pdf(self):
        """Test PDF."""
        dist = Dirichlet([2, 3, 4])
        
        # Valid simplex point
        x = [0.2, 0.3, 0.5]
        pdf_val = dist.pdf(x)
        assert pdf_val > 0
        
        # Invalid simplex point (doesn't sum to 1)
        x_invalid = [0.2, 0.3, 0.6]
        assert dist.pdf(x_invalid) == 0
        
        # Negative component
        x_negative = [0.2, -0.1, 0.9]
        assert dist.pdf(x_negative) == 0
        
        # Log scale
        x = [0.1, 0.4, 0.5]
        log_pdf = dist.pdf(x, log=True)
        assert abs(log_pdf - np.log(dist.pdf(x))) < 1e-10
    
    def test_pdf_wrong_dimension(self):
        """Test PDF with wrong dimension."""
        dist = Dirichlet([1, 2, 3])
        
        with pytest.raises(ValueError):
            dist.pdf([0.5, 0.5])  # Wrong dimension
    
    def test_rvs(self):
        """Test random sampling."""
        rng = RNG(42)
        dist = Dirichlet([2, 3, 4])
        
        samples = [dist.rvs(rng) for _ in range(100)]
        
        # Each sample should be on the simplex
        for sample in samples:
            assert len(sample) == 3
            assert all(s >= 0 for s in sample)
            assert abs(np.sum(sample) - 1.0) < 1e-10
    
    def test_mean(self):
        """Test mean calculation."""
        alpha = [2, 3, 1]
        dist = Dirichlet(alpha)
        
        expected_mean = np.array(alpha) / np.sum(alpha)
        actual_mean = dist.mean()
        
        np.testing.assert_array_almost_equal(actual_mean, expected_mean)
    
    def test_var(self):
        """Test variance calculation."""
        alpha = [2, 3, 1]
        dist = Dirichlet(alpha)
        
        mean = dist.mean()
        alpha_sum = np.sum(alpha)
        expected_var = mean * (1 - mean) / (alpha_sum + 1)
        actual_var = dist.var()
        
        np.testing.assert_array_almost_equal(actual_var, expected_var)


class TestCustomDistributionIntegration:
    """Test integration between custom distributions."""
    
    def test_sampling_consistency(self):
        """Test that sampling produces reasonable results."""
        rng = RNG(42)
        
        # Sample from triangular
        tri_dist = TriangularDistribution(0, 1, 3)
        tri_samples = [tri_dist.rvs(rng) for _ in range(100)]
        
        # All should be in bounds
        assert all(0 <= s <= 3 for s in tri_samples)
        
        # Sample from truncated normal
        tn_dist = TruncatedNormal(0, 1, -2, 2)
        tn_samples = [tn_dist.rvs(rng) for _ in range(100)]
        
        # All should be in bounds
        assert all(-2 <= s <= 2 for s in tn_samples)
    
    def test_pdf_integration(self):
        """Test that PDFs integrate to approximately 1."""
        # For triangular distribution
        dist = TriangularDistribution(0, 1, 2)
        
        # Numerical integration using trapezoidal rule
        x_vals = np.linspace(0, 2, 1000)
        y_vals = [dist.pdf(x) for x in x_vals]
        integral = np.trapz(y_vals, x_vals)
        
        assert abs(integral - 1.0) < 0.01  # Should be close to 1
    
    def test_moment_consistency(self):
        """Test moment consistency where calculable."""
        rng = RNG(42)
        
        # Inverse gamma with finite mean
        dist = InverseGamma(3, 2)
        samples = [dist.rvs(rng) for _ in range(5000)]
        
        theoretical_mean = dist.mean()
        sample_mean = np.mean(samples)
        
        # Should be close (within 10% due to sampling variability)
        assert abs(sample_mean - theoretical_mean) / theoretical_mean < 0.1