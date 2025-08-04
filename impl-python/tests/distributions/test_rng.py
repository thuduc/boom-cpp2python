"""Comprehensive tests for RNG class."""

import pytest
import numpy as np
from boom.distributions import RNG, GlobalRng, seed_rng


class TestRNGConstruction:
    """Test RNG construction and initialization."""
    
    def test_default_construction(self):
        """Test creating RNG with default seed."""
        rng = RNG()
        assert isinstance(rng, RNG)
        
        # Should generate different values
        values = [rng() for _ in range(10)]
        assert len(set(values)) > 5  # At least some variety
    
    def test_seeded_construction(self):
        """Test creating RNG with specific seed."""
        rng1 = RNG(42)
        rng2 = RNG(42)
        
        # Same seed should produce same sequence
        values1 = [rng1() for _ in range(10)]
        values2 = [rng2() for _ in range(10)]
        assert values1 == values2
    
    def test_different_seeds(self):
        """Test different seeds produce different sequences."""
        rng1 = RNG(42)
        rng2 = RNG(43)
        
        values1 = [rng1() for _ in range(10)]
        values2 = [rng2() for _ in range(10)]
        assert values1 != values2


class TestRNGSeeding:
    """Test RNG seeding methods."""
    
    def test_seed_method(self):
        """Test seed() method."""
        rng = RNG()
        seed_used = rng.seed(42)
        assert seed_used == 42
        
        # Should be reproducible
        rng2 = RNG(42)
        values1 = [rng() for _ in range(5)]
        values2 = [rng2() for _ in range(5)]
        assert values1 == values2
    
    def test_seed_none(self):
        """Test seeding with None generates random seed."""
        rng = RNG()
        seed1 = rng.seed(None)
        seed2 = rng.seed(None)
        
        # Should generate different seeds
        assert isinstance(seed1, int)
        assert isinstance(seed2, int)
        assert seed1 != seed2
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        rng = RNG()
        
        rng.seed(12345)
        sequence1 = [rng() for _ in range(20)]
        
        rng.seed(12345)
        sequence2 = [rng() for _ in range(20)]
        
        assert sequence1 == sequence2


class TestRNGUniform:
    """Test uniform random number generation."""
    
    def test_default_uniform(self):
        """Test default uniform generation [0, 1)."""
        rng = RNG(42)
        values = [rng() for _ in range(1000)]
        
        # All values should be in [0, 1)
        assert all(0.0 <= v < 1.0 for v in values)
        
        # Should have reasonable distribution
        mean = np.mean(values)
        assert 0.4 < mean < 0.6
    
    def test_uniform_range(self):
        """Test uniform generation with custom range."""
        rng = RNG(42)
        values = [rng.uniform(-10, 10) for _ in range(1000)]
        
        # All values should be in [-10, 10)
        assert all(-10.0 <= v < 10.0 for v in values)
        
        # Should have reasonable distribution
        mean = np.mean(values)
        assert -2 < mean < 2


class TestRNGDistributions:
    """Test various distribution methods."""
    
    def test_normal(self):
        """Test normal distribution."""
        rng = RNG(42)
        values = [rng.normal(0, 1) for _ in range(1000)]
        
        mean = np.mean(values)
        std = np.std(values)
        
        # Should be approximately standard normal
        assert -0.2 < mean < 0.2
        assert 0.8 < std < 1.2
    
    def test_normal_params(self):
        """Test normal with custom parameters."""
        rng = RNG(42)
        mu, sigma = 5.0, 2.0
        values = [rng.normal(mu, sigma) for _ in range(1000)]
        
        mean = np.mean(values)
        std = np.std(values)
        
        assert 4.5 < mean < 5.5
        assert 1.5 < std < 2.5
    
    def test_gamma(self):
        """Test gamma distribution."""
        rng = RNG(42)
        shape, scale = 2.0, 1.5
        values = [rng.gamma(shape, scale) for _ in range(1000)]
        
        # All values should be positive
        assert all(v > 0 for v in values)
        
        # Check approximate mean
        expected_mean = shape * scale
        actual_mean = np.mean(values)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean
    
    def test_beta(self):
        """Test beta distribution."""
        rng = RNG(42)
        alpha, beta = 2.0, 3.0
        values = [rng.beta(alpha, beta) for _ in range(1000)]
        
        # All values should be in [0, 1]
        assert all(0 <= v <= 1 for v in values)
        
        # Check approximate mean
        expected_mean = alpha / (alpha + beta)
        actual_mean = np.mean(values)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean
    
    def test_binomial(self):
        """Test binomial distribution."""
        rng = RNG(42)
        n, p = 10, 0.3
        values = [rng.binomial(n, p) for _ in range(1000)]
        
        # All values should be integers in [0, n]
        assert all(isinstance(v, int) and 0 <= v <= n for v in values)
        
        # Check approximate mean
        expected_mean = n * p
        actual_mean = np.mean(values)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean
    
    def test_poisson(self):
        """Test Poisson distribution."""
        rng = RNG(42)
        lam = 3.5
        values = [rng.poisson(lam) for _ in range(1000)]
        
        # All values should be non-negative integers
        assert all(isinstance(v, int) and v >= 0 for v in values)
        
        # Check approximate mean
        actual_mean = np.mean(values)
        assert 0.8 * lam < actual_mean < 1.2 * lam
    
    def test_exponential(self):
        """Test exponential distribution."""
        rng = RNG(42)
        scale = 2.0
        values = [rng.exponential(scale) for _ in range(1000)]
        
        # All values should be positive
        assert all(v > 0 for v in values)
        
        # Check approximate mean
        actual_mean = np.mean(values)
        assert 0.8 * scale < actual_mean < 1.2 * scale
    
    def test_chi_square(self):
        """Test chi-square distribution."""
        rng = RNG(42)
        df = 5.0
        values = [rng.chi_square(df) for _ in range(1000)]
        
        # All values should be positive
        assert all(v > 0 for v in values)
        
        # Check approximate mean (should equal df)
        actual_mean = np.mean(values)
        assert 0.8 * df < actual_mean < 1.2 * df
    
    def test_student_t(self):
        """Test Student's t distribution."""
        rng = RNG(42)
        df = 10.0
        values = [rng.student_t(df) for _ in range(1000)]
        
        # Should have approximately zero mean for large df
        actual_mean = np.mean(values)
        assert -0.3 < actual_mean < 0.3


class TestRNGUtilities:
    """Test utility methods."""
    
    def test_choice(self):
        """Test choice method."""
        rng = RNG(42)
        
        # Choice from range
        values = [rng.choice(5) for _ in range(100)]
        assert all(0 <= v < 5 for v in values)
        
        # Choice from array
        options = ['a', 'b', 'c']
        choices = [rng.choice(options) for _ in range(100)]
        assert all(c in options for c in choices)
    
    def test_choice_probabilities(self):
        """Test choice with custom probabilities."""
        rng = RNG(42)
        options = [0, 1, 2]
        probs = [0.1, 0.8, 0.1]  # Heavily favor option 1
        
        choices = [rng.choice(options, p=probs) for _ in range(1000)]
        
        # Should heavily favor 1
        count_1 = sum(1 for c in choices if c == 1)
        assert count_1 > 600  # Should be around 800
    
    def test_multinomial(self):
        """Test multinomial sampling."""
        rng = RNG(42)
        n = 100
        pvals = [0.2, 0.3, 0.5]
        
        result = rng.multinomial(n, pvals)
        
        # Should sum to n
        assert np.sum(result) == n
        
        # Should have correct length
        assert len(result) == len(pvals)
        
        # All counts should be non-negative
        assert all(count >= 0 for count in result)
    
    def test_multivariate_normal(self):
        """Test multivariate normal sampling."""
        rng = RNG(42)
        mean = [1, 2, 3]
        cov = [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]
        
        samples = [rng.multivariate_normal(mean, cov) for _ in range(100)]
        
        # Should have correct dimensions
        assert all(len(sample) == 3 for sample in samples)
        
        # Check approximate means
        sample_means = np.mean(samples, axis=0)
        np.testing.assert_allclose(sample_means, mean, rtol=0.3)


class TestRNGState:
    """Test state management."""
    
    def test_get_set_state(self):
        """Test getting and setting state."""
        rng = RNG(42)
        
        # Generate some values
        values1 = [rng() for _ in range(5)]
        
        # Save state
        state = rng.get_state()
        
        # Generate more values
        values2 = [rng() for _ in range(5)]
        
        # Restore state and generate same values
        rng.set_state(state)
        values3 = [rng() for _ in range(5)]
        
        assert values2 == values3
        assert values1 != values2
    
    def test_generator_property(self):
        """Test generator property access."""
        rng = RNG(42)
        generator = rng.generator
        
        assert hasattr(generator, 'random')
        assert hasattr(generator, 'normal')
        assert hasattr(generator, 'uniform')


class TestGlobalRng:
    """Test GlobalRng singleton."""
    
    def test_singleton_behavior(self):
        """Test that GlobalRng behaves as singleton."""
        rng1 = GlobalRng.get_rng()
        rng2 = GlobalRng.get_rng()
        
        # Should be the same instance
        assert rng1 is rng2
    
    def test_global_seeding(self):
        """Test global RNG seeding."""
        seed_used = GlobalRng.seed(42)
        assert seed_used == 42
        
        # Should affect the global instance
        rng = GlobalRng.get_rng()
        values1 = [rng() for _ in range(5)]
        
        # Re-seed and check reproducibility
        GlobalRng.seed(42)
        values2 = [rng() for _ in range(5)]
        
        assert values1 == values2
    
    def test_set_rng(self):
        """Test setting new global RNG."""
        original_rng = GlobalRng.get_rng()
        new_rng = RNG(99)
        
        GlobalRng.set_rng(new_rng)
        assert GlobalRng.get_rng() is new_rng
        
        # Restore original
        GlobalRng.set_rng(original_rng)


class TestSeedRng:
    """Test seed_rng utility function."""
    
    def test_seed_rng_default(self):
        """Test seed_rng with default (global) RNG."""
        seed_used = seed_rng()
        assert isinstance(seed_used, int)
        assert 0 <= seed_used < 2**32
    
    def test_seed_rng_custom(self):
        """Test seed_rng with custom RNG."""
        rng = RNG()
        seed_used = seed_rng(rng)
        
        assert isinstance(seed_used, int)
        assert 0 <= seed_used < 2**32
    
    def test_seed_rng_randomness(self):
        """Test that seed_rng generates different seeds."""
        seeds = [seed_rng() for _ in range(10)]
        
        # Should all be different (very high probability)
        assert len(set(seeds)) == len(seeds)