"""Tests for slice sampler."""

import pytest
import numpy as np
from boom.samplers.slice_sampler import SliceSampler, MultivariateSliceSampler, AdaptiveSliceSampler
from boom.linalg import Vector
from boom.distributions import RNG


class TestSliceSampler:
    """Test SliceSampler class."""
    
    def test_standard_normal_sampling(self):
        """Test sampling from standard normal distribution."""
        # Log density of standard normal
        def log_target(x):
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
        
        sampler = SliceSampler(log_target, RNG(seed=42))
        
        # Sample
        samples = sampler.sample(0.0, 1000, width=1.0)
        
        assert len(samples) == 1000
        
        # Check approximate properties
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        
        # Should be approximately N(0, 1)
        assert abs(sample_mean) < 0.1
        assert 0.8 < sample_var < 1.2
    
    def test_gamma_sampling(self):
        """Test sampling from Gamma distribution."""
        # Gamma(3, 2) distribution
        def log_target(x):
            if x <= 0:
                return -np.inf
            alpha, beta = 3.0, 2.0
            return (alpha - 1) * np.log(x) - beta * x
        
        sampler = SliceSampler(log_target, RNG(seed=123))
        
        # Sample
        samples = sampler.sample(1.0, 1000, width=2.0)
        
        assert len(samples) == 1000
        
        # All samples should be positive
        assert all(x > 0 for x in samples)
        
        # Check approximate mean (should be alpha/beta = 1.5)
        sample_mean = np.mean(samples)
        assert 1.2 < sample_mean < 1.8
    
    def test_beta_sampling(self):
        """Test sampling from Beta distribution."""
        # Beta(2, 3) distribution
        def log_target(x):
            if x <= 0 or x >= 1:
                return -np.inf
            alpha, beta = 2.0, 3.0
            return (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x)
        
        sampler = SliceSampler(log_target, RNG(seed=456))
        
        # Sample
        samples = sampler.sample(0.5, 500, width=0.5)
        
        assert len(samples) == 500
        
        # All samples should be in (0, 1)
        assert all(0 < x < 1 for x in samples)
        
        # Check approximate mean (should be alpha/(alpha+beta) = 0.4)
        sample_mean = np.mean(samples)
        assert 0.3 < sample_mean < 0.5
    
    def test_different_widths(self):
        """Test with different initial widths."""
        def log_target(x):
            return -0.5 * x**2
        
        sampler = SliceSampler(log_target, RNG(seed=789))
        
        # Test with small width
        samples_small = sampler.sample(0.0, 100, width=0.1)
        
        # Test with large width
        samples_large = sampler.sample(0.0, 100, width=10.0)
        
        # Both should work
        assert len(samples_small) == 100
        assert len(samples_large) == 100
        
        # Both should sample from approximately the same distribution
        assert abs(np.mean(samples_small)) < 0.5
        assert abs(np.mean(samples_large)) < 0.5
    
    def test_error_handling(self):
        """Test error handling with problematic target."""
        def problematic_target(x):
            if abs(x) > 3:
                raise ValueError("Out of bounds")
            return -0.5 * x**2
        
        sampler = SliceSampler(problematic_target, RNG(seed=101))
        
        # Should still work
        samples = sampler.sample(0.0, 100, width=1.0)
        
        assert len(samples) == 100
        # All samples should be in valid range
        assert all(abs(x) <= 3 for x in samples)


class TestMultivariateSliceSampler:
    """Test MultivariateSliceSampler class."""
    
    def test_bivariate_normal_sampling(self):
        """Test sampling from bivariate normal."""
        # Bivariate standard normal
        def log_target(x):
            if len(x) != 2:
                return -np.inf
            return -0.5 * (x[0]**2 + x[1]**2) - np.log(2 * np.pi)
        
        sampler = MultivariateSliceSampler(log_target, RNG(seed=42))
        
        # Sample
        initial_state = Vector([0.0, 0.0])
        samples = sampler.sample(initial_state, 500)
        
        assert len(samples) == 500
        
        # Extract coordinates
        x_vals = [s[0] for s in samples]
        y_vals = [s[1] for s in samples]
        
        # Check approximate properties
        assert abs(np.mean(x_vals)) < 0.2
        assert abs(np.mean(y_vals)) < 0.2
        assert 0.7 < np.var(x_vals) < 1.3
        assert 0.7 < np.var(y_vals) < 1.3
    
    def test_different_widths_per_coordinate(self):
        """Test with different widths for each coordinate."""
        # Independent normal with different variances
        def log_target(x):
            if len(x) != 2:
                return -np.inf
            return -0.5 * (x[0]**2 / 1.0 + x[1]**2 / 4.0)
        
        sampler = MultivariateSliceSampler(log_target, RNG(seed=123))
        
        # Use different widths
        initial_state = Vector([0.0, 0.0])
        samples = sampler.sample(initial_state, 300, widths=[1.0, 2.0])
        
        assert len(samples) == 300
        
        # Second coordinate should have larger variance
        x_vals = [s[0] for s in samples]
        y_vals = [s[1] for s in samples]
        
        assert np.var(y_vals) > np.var(x_vals)
    
    def test_scalar_width(self):
        """Test with scalar width applied to all coordinates."""
        def log_target(x):
            return -0.5 * sum(xi**2 for xi in x)
        
        sampler = MultivariateSliceSampler(log_target, RNG(seed=456))
        
        initial_state = Vector([0.0, 0.0, 0.0])
        samples = sampler.sample(initial_state, 200, widths=1.5)
        
        assert len(samples) == 200
        assert all(len(s) == 3 for s in samples)


class TestAdaptiveSliceSampler:
    """Test AdaptiveSliceSampler class."""
    
    def test_basic_adaptation(self):
        """Test basic adaptive functionality."""
        def log_target(x):
            return -0.5 * sum(xi**2 for xi in x)
        
        sampler = AdaptiveSliceSampler(log_target, RNG(seed=42))
        
        initial_state = Vector([0.0, 0.0])
        samples, final_widths = sampler.sample(initial_state, 250, 
                                              initial_widths=1.0)
        
        assert len(samples) == 250
        assert len(final_widths) == 2
        
        # Final widths should be positive
        assert all(w > 0 for w in final_widths)
    
    def test_adaptation_with_different_scales(self):
        """Test adaptation with different coordinate scales."""
        # Target with very different scales
        def log_target(x):
            if len(x) != 2:
                return -np.inf
            return -0.5 * (x[0]**2 / 0.01 + x[1]**2 / 100.0)
        
        sampler = AdaptiveSliceSampler(log_target, RNG(seed=789), 
                                     adaptation_window=50)
        
        initial_state = Vector([0.0, 0.0])
        samples, final_widths = sampler.sample(initial_state, 200,
                                              initial_widths=[0.1, 10.0])
        
        assert len(samples) == 200
        assert len(final_widths) == 2
        
        # Check that samples have reasonable ranges
        x_vals = [s[0] for s in samples]
        y_vals = [s[1] for s in samples]
        
        # First coordinate should have smaller range
        assert np.std(x_vals) < np.std(y_vals)
    
    def test_get_step_sizes(self):
        """Test getting current step sizes."""
        def log_target(x):
            return -0.5 * x[0]**2
        
        sampler = AdaptiveSliceSampler(log_target, RNG(seed=123))
        
        # Before any sampling
        initial_sizes = sampler.get_step_sizes()
        assert len(initial_sizes) == 1  # Default
        
        # After sampling
        initial_state = Vector([0.0])
        samples, final_widths = sampler.sample(initial_state, 50)
        
        current_sizes = sampler.get_step_sizes()
        assert len(current_sizes) == 1
        assert current_sizes[0] > 0