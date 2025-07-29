"""Tests for random number generation."""
import pytest
import numpy as np
from boom.distributions import rng, seed_rng, GlobalRng
from boom.linalg import SpdMatrix


class TestGlobalRng:
    """Test suite for GlobalRng class."""
    
    def test_seeding(self):
        """Test random seed setting."""
        # Create two RNG instances with same seed
        rng1 = GlobalRng(42)
        rng2 = GlobalRng(42)
        
        # Should produce same sequence
        assert rng1.runif() == rng2.runif()
        assert rng1.rnorm() == rng2.rnorm()
        
        # Test reseed
        rng1.seed(123)
        rng2.seed(123)
        assert rng1.runif() == rng2.runif()
        
        # Test module-level seed function
        seed_rng(99)
        val1 = rng.runif()
        seed_rng(99)
        val2 = rng.runif()
        assert val1 == val2
    
    def test_uniform(self):
        """Test uniform distribution."""
        seed_rng(42)
        
        # Single value
        u = rng.runif()
        assert 0 <= u <= 1
        
        u2 = rng.runif(10, 20)
        assert 10 <= u2 <= 20
        
        # Vector
        vec = rng.runif_vec(1000)
        assert len(vec) == 1000
        assert np.all((vec >= 0) & (vec <= 1))
        assert 0.4 < np.mean(vec) < 0.6  # Should be near 0.5
        
        vec2 = rng.runif_vec(100, -1, 1)
        assert np.all((vec2 >= -1) & (vec2 <= 1))
    
    def test_normal(self):
        """Test normal distribution."""
        seed_rng(42)
        
        # Single value
        n = rng.rnorm()
        assert isinstance(n, float)
        
        n2 = rng.rnorm(10, 2)
        assert isinstance(n2, float)
        
        # Mean/variance parameterization
        n3 = rng.rnorm_mt(0, 4)  # variance = 4, sd = 2
        
        # Vector
        vec = rng.rnorm_vec(10000)
        assert len(vec) == 10000
        assert -0.1 < np.mean(vec) < 0.1  # Should be near 0
        assert 0.9 < np.std(vec) < 1.1   # Should be near 1
        
        vec2 = rng.rnorm_vec(1000, 5, 2)
        assert 4.8 < np.mean(vec2) < 5.2
        assert 1.8 < np.std(vec2) < 2.2
    
    def test_exponential(self):
        """Test exponential distribution."""
        seed_rng(42)
        
        # Single value
        e = rng.rexp(2.0)
        assert e > 0
        
        # Vector
        vec = rng.rexp_vec(10000, 2.0)
        assert np.all(vec > 0)
        # Mean should be 1/lambda = 0.5
        assert 0.45 < np.mean(vec) < 0.55
    
    def test_gamma(self):
        """Test gamma distribution."""
        seed_rng(42)
        
        # Single value
        g = rng.rgamma(2.0, 3.0)
        assert g > 0
        
        # Vector
        vec = rng.rgamma_vec(10000, 2.0, 3.0)
        assert np.all(vec > 0)
        # Mean = shape * scale = 6
        assert 5.8 < np.mean(vec) < 6.2
    
    def test_beta(self):
        """Test beta distribution."""
        seed_rng(42)
        
        # Single value
        b = rng.rbeta(2.0, 3.0)
        assert 0 <= b <= 1
        
        # Vector
        vec = rng.rbeta_vec(10000, 2.0, 3.0)
        assert np.all((vec >= 0) & (vec <= 1))
        # Mean = a/(a+b) = 2/5 = 0.4
        assert 0.38 < np.mean(vec) < 0.42
    
    def test_chisq(self):
        """Test chi-square distribution."""
        seed_rng(42)
        
        # Single value
        c = rng.rchisq(5)
        assert c > 0
        
        # Vector
        vec = rng.rchisq_vec(10000, 5)
        assert np.all(vec > 0)
        # Mean = df = 5
        assert 4.8 < np.mean(vec) < 5.2
    
    def test_binomial(self):
        """Test binomial distribution."""
        seed_rng(42)
        
        # Single value
        b = rng.rbinom(10, 0.3)
        assert 0 <= b <= 10
        assert isinstance(b, int)
        
        # Vector
        vec = rng.rbinom_vec(10000, 10, 0.3)
        assert np.all((vec >= 0) & (vec <= 10))
        # Mean = n*p = 3
        assert 2.9 < np.mean(vec) < 3.1
    
    def test_poisson(self):
        """Test Poisson distribution."""
        seed_rng(42)
        
        # Single value
        p = rng.rpois(3.5)
        assert p >= 0
        assert isinstance(p, int)
        
        # Vector
        vec = rng.rpois_vec(10000, 3.5)
        assert np.all(vec >= 0)
        # Mean = lambda = 3.5
        assert 3.4 < np.mean(vec) < 3.6
    
    def test_multinomial(self):
        """Test multinomial distribution."""
        seed_rng(42)
        
        probs = np.array([0.2, 0.3, 0.5])
        counts = rng.rmulti(100, probs)
        
        assert len(counts) == 3
        assert np.sum(counts) == 100
        assert np.all(counts >= 0)
        
        # Test many samples
        total_counts = np.zeros(3)
        for _ in range(1000):
            total_counts += rng.rmulti(100, probs)
        
        # Should be approximately proportional to probs
        proportions = total_counts / np.sum(total_counts)
        assert np.allclose(proportions, probs, atol=0.02)
    
    def test_student_t(self):
        """Test Student t distribution."""
        seed_rng(42)
        
        # Single value
        t = rng.rt(10)
        assert isinstance(t, float)
        
        # Vector
        vec = rng.rt_vec(10000, 10)
        # Should have heavier tails than normal
        # but with df=10, should be close to normal
        assert -0.1 < np.mean(vec) < 0.1
    
    def test_f_distribution(self):
        """Test F distribution."""
        seed_rng(42)
        
        f = rng.rf(5, 10)
        assert f > 0
    
    def test_multivariate_normal(self):
        """Test multivariate normal distribution."""
        seed_rng(42)
        
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        # Single sample
        x = rng.rmvn(mean, cov)
        assert len(x) == 2
        
        # Many samples
        samples = np.array([rng.rmvn(mean, cov) for _ in range(10000)])
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)
        
        assert np.allclose(sample_mean, mean, atol=0.05)
        assert np.allclose(sample_cov, cov, atol=0.1)
        
        # Using Cholesky factor
        L = np.linalg.cholesky(cov)
        x2 = rng.rmvn_L(mean, L)
        assert len(x2) == 2
    
    def test_dirichlet(self):
        """Test Dirichlet distribution."""
        seed_rng(42)
        
        alpha = np.array([2.0, 3.0, 5.0])
        x = rng.rdirichlet(alpha)
        
        assert len(x) == 3
        assert np.allclose(np.sum(x), 1.0)
        assert np.all(x >= 0)
        
        # Many samples
        samples = np.array([rng.rdirichlet(alpha) for _ in range(10000)])
        mean = np.mean(samples, axis=0)
        expected_mean = alpha / np.sum(alpha)
        assert np.allclose(mean, expected_mean, atol=0.02)
    
    def test_wishart(self):
        """Test Wishart distribution."""
        seed_rng(42)
        
        df = 5
        scale = SpdMatrix([[2, 1], [1, 2]])
        
        W = rng.rwish(df, scale)
        assert W.shape == (2, 2)
        assert np.allclose(W, W.T)  # Symmetric
        
        # Should be positive definite
        eigvals = np.linalg.eigvalsh(W)
        assert np.all(eigvals > 0)
        
        # Many samples
        samples = [rng.rwish(df, scale) for _ in range(1000)]
        mean_W = np.mean(samples, axis=0)
        expected_mean = df * scale
        assert np.allclose(mean_W, expected_mean, rtol=0.1)
    
    def test_utility_functions(self):
        """Test utility functions."""
        seed_rng(42)
        
        # Random sample
        population = list(range(10))
        sample = rng.random_sample(population, 5, replace=False)
        assert len(sample) == 5
        assert len(set(sample)) == 5  # All unique
        assert all(x in population for x in sample)
        
        # With replacement
        sample2 = rng.random_sample(population, 20, replace=True)
        assert len(sample2) == 20
        
        # Shuffle
        arr = list(range(10))
        original = arr.copy()
        rng.shuffle(arr)
        assert sorted(arr) == sorted(original)
        assert arr != original  # Should be shuffled (very unlikely to be same)
        
        # Permutation
        perm = rng.permutation(10)
        assert len(perm) == 10
        assert sorted(perm) == list(range(10))
        
        # Permutation of array
        arr2 = np.array([1, 2, 3, 4, 5])
        perm2 = rng.permutation(arr2)
        assert sorted(perm2) == sorted(arr2)
    
    def test_deterministic_sequences(self):
        """Test that sequences are deterministic with fixed seed."""
        values1 = []
        seed_rng(12345)
        for _ in range(10):
            values1.append(rng.runif())
        
        values2 = []
        seed_rng(12345)
        for _ in range(10):
            values2.append(rng.runif())
        
        assert values1 == values2