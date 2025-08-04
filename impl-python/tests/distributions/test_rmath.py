"""Comprehensive tests for Rmath functions."""

import pytest
import numpy as np
import scipy.stats as stats
from boom.distributions import RNG
from boom.distributions.rmath import *


class TestNormalDistribution:
    """Test normal distribution functions."""
    
    def test_dnorm(self):
        """Test normal density function."""
        # Standard normal
        assert abs(dnorm(0.0) - stats.norm.pdf(0.0)) < 1e-10
        assert abs(dnorm(1.0) - stats.norm.pdf(1.0)) < 1e-10
        
        # Custom parameters
        mu, sig = 2.0, 1.5
        x = 2.5
        expected = stats.norm.pdf(x, loc=mu, scale=sig)
        assert abs(dnorm(x, mu, sig) - expected) < 1e-10
        
        # Log scale
        expected_log = stats.norm.logpdf(x, loc=mu, scale=sig)
        assert abs(dnorm(x, mu, sig, log=True) - expected_log) < 1e-10
    
    def test_pnorm(self):
        """Test normal CDF function."""
        # Standard normal
        assert abs(pnorm(0.0) - 0.5) < 1e-10
        assert abs(pnorm(1.96) - 0.975) < 1e-3
        
        # Custom parameters
        mu, sig = 1.0, 2.0
        x = 3.0
        expected = stats.norm.cdf(x, loc=mu, scale=sig)
        assert abs(pnorm(x, mu, sig) - expected) < 1e-10
        
        # Upper tail
        expected_upper = stats.norm.sf(x, loc=mu, scale=sig)
        assert abs(pnorm(x, mu, sig, lower_tail=False) - expected_upper) < 1e-10
    
    def test_qnorm(self):
        """Test normal quantile function."""
        # Standard normal
        assert abs(qnorm(0.5) - 0.0) < 1e-10
        assert abs(qnorm(0.975) - 1.96) < 1e-2
        
        # Custom parameters
        mu, sig = 1.0, 2.0
        p = 0.25
        expected = stats.norm.ppf(p, loc=mu, scale=sig)
        assert abs(qnorm(p, mu, sig) - expected) < 1e-10
    
    def test_rnorm(self):
        """Test normal random generation."""
        rng = RNG(42)
        
        # Generate samples
        samples = [rnorm(0, 1, rng) for _ in range(1000)]
        
        # Check basic properties
        mean = np.mean(samples)
        std = np.std(samples)
        assert -0.2 < mean < 0.2
        assert 0.8 < std < 1.2


class TestUniformDistribution:
    """Test uniform distribution functions."""
    
    def test_dunif(self):
        """Test uniform density function."""
        # Standard uniform [0, 1]
        assert dunif(0.5) == 1.0
        assert dunif(-0.1) == 0.0
        assert dunif(1.1) == 0.0
        
        # Custom range [2, 5]
        lo, hi = 2.0, 5.0
        assert dunif(3.5, lo, hi) == 1.0 / (hi - lo)
        assert dunif(1.0, lo, hi) == 0.0
    
    def test_punif(self):
        """Test uniform CDF function."""
        # Standard uniform
        assert punif(0.0) == 0.0
        assert punif(0.5) == 0.5
        assert punif(1.0) == 1.0
        
        # Custom range
        lo, hi = 1.0, 4.0
        x = 2.5
        expected = (x - lo) / (hi - lo)
        assert abs(punif(x, lo, hi) - expected) < 1e-10
    
    def test_qunif(self):
        """Test uniform quantile function."""
        # Standard uniform
        assert qunif(0.0) == 0.0
        assert qunif(0.5) == 0.5
        assert qunif(1.0) == 1.0
        
        # Custom range
        lo, hi = 2.0, 8.0
        p = 0.25
        expected = lo + p * (hi - lo)
        assert abs(qunif(p, lo, hi) - expected) < 1e-10
    
    def test_runif(self):
        """Test uniform random generation."""
        rng = RNG(42)
        
        # Generate samples in [0, 1]
        samples = [runif(0, 1, rng) for _ in range(1000)]
        assert all(0 <= s < 1 for s in samples)
        
        # Check approximate mean
        mean = np.mean(samples)
        assert 0.4 < mean < 0.6


class TestGammaDistribution:
    """Test gamma distribution functions."""
    
    def test_dgamma(self):
        """Test gamma density function."""
        shape, scale = 2.0, 1.5
        x = 3.0
        
        expected = stats.gamma.pdf(x, a=shape, scale=scale)
        assert abs(dgamma(x, shape, scale) - expected) < 1e-10
        
        # Log scale
        expected_log = stats.gamma.logpdf(x, a=shape, scale=scale)
        assert abs(dgamma(x, shape, scale, log=True) - expected_log) < 1e-10
    
    def test_pgamma(self):
        """Test gamma CDF function."""
        shape, scale = 2.0, 1.5
        x = 3.0
        
        expected = stats.gamma.cdf(x, a=shape, scale=scale)
        assert abs(pgamma(x, shape, scale) - expected) < 1e-10
        
        # Upper tail
        expected_upper = stats.gamma.sf(x, a=shape, scale=scale)
        assert abs(pgamma(x, shape, scale, lower_tail=False) - expected_upper) < 1e-10
    
    def test_rgamma(self):
        """Test gamma random generation."""
        rng = RNG(42)
        shape, scale = 2.0, 1.0
        
        samples = [rgamma(shape, scale, rng) for _ in range(1000)]
        
        # All should be positive
        assert all(s > 0 for s in samples)
        
        # Check approximate mean
        expected_mean = shape * scale
        actual_mean = np.mean(samples)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean


class TestBetaDistribution:
    """Test beta distribution functions."""
    
    def test_dbeta(self):
        """Test beta density function."""
        alpha, beta = 2.0, 3.0
        x = 0.4
        
        expected = stats.beta.pdf(x, a=alpha, b=beta)
        assert abs(dbeta(x, alpha, beta) - expected) < 1e-10
    
    def test_pbeta(self):
        """Test beta CDF function."""
        alpha, beta = 2.0, 3.0
        x = 0.4
        
        expected = stats.beta.cdf(x, a=alpha, b=beta)
        assert abs(pbeta(x, alpha, beta) - expected) < 1e-10
    
    def test_rbeta(self):
        """Test beta random generation."""
        rng = RNG(42)
        alpha, beta = 2.0, 3.0
        
        samples = [rbeta(alpha, beta, rng) for _ in range(1000)]
        
        # All should be in [0, 1]
        assert all(0 <= s <= 1 for s in samples)
        
        # Check approximate mean
        expected_mean = alpha / (alpha + beta)
        actual_mean = np.mean(samples)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean


class TestChiSquareDistribution:
    """Test chi-square distribution functions."""
    
    def test_dchisq(self):
        """Test chi-square density function."""
        df = 5.0
        x = 3.0
        
        expected = stats.chi2.pdf(x, df=df)
        assert abs(dchisq(x, df) - expected) < 1e-10
    
    def test_pchisq(self):
        """Test chi-square CDF function."""
        df = 5.0
        x = 3.0
        
        expected = stats.chi2.cdf(x, df=df)
        assert abs(pchisq(x, df) - expected) < 1e-10
    
    def test_rchisq(self):
        """Test chi-square random generation."""
        rng = RNG(42)
        df = 5.0
        
        samples = [rchisq(df, rng) for _ in range(1000)]
        
        # All should be positive
        assert all(s > 0 for s in samples)
        
        # Check approximate mean (should equal df)
        actual_mean = np.mean(samples)
        assert 0.8 * df < actual_mean < 1.2 * df


class TestStudentTDistribution:
    """Test Student's t distribution functions."""
    
    def test_dt(self):
        """Test t density function."""
        df = 10.0
        x = 1.5
        
        expected = stats.t.pdf(x, df=df)
        assert abs(dt(x, df) - expected) < 1e-10
    
    def test_pt(self):
        """Test t CDF function."""
        df = 10.0
        x = 1.5
        
        expected = stats.t.cdf(x, df=df)
        assert abs(pt(x, df) - expected) < 1e-10
    
    def test_rt(self):
        """Test t random generation."""
        rng = RNG(42)
        df = 10.0
        
        samples = [rt(df, rng) for _ in range(1000)]
        
        # Should have approximately zero mean for large df
        actual_mean = np.mean(samples)
        assert -0.3 < actual_mean < 0.3


class TestFDistribution:
    """Test F distribution functions."""
    
    def test_df_dist(self):
        """Test F density function."""
        dfn, dfd = 5.0, 10.0
        x = 2.0
        
        expected = stats.f.pdf(x, dfn=dfn, dfd=dfd)
        assert abs(df_dist(x, dfn, dfd) - expected) < 1e-10
    
    def test_pf(self):
        """Test F CDF function."""
        dfn, dfd = 5.0, 10.0
        x = 2.0
        
        expected = stats.f.cdf(x, dfn=dfn, dfd=dfd)
        assert abs(pf(x, dfn, dfd) - expected) < 1e-10
    
    def test_rf(self):
        """Test F random generation."""
        rng = RNG(42)
        dfn, dfd = 5.0, 10.0
        
        samples = [rf(dfn, dfd, rng) for _ in range(100)]
        
        # All should be positive
        assert all(s > 0 for s in samples)


class TestBinomialDistribution:
    """Test binomial distribution functions."""
    
    def test_dbinom(self):
        """Test binomial PMF function."""
        n, p = 10, 0.3
        x = 3
        
        expected = stats.binom.pmf(x, n=n, p=p)
        assert abs(dbinom(x, n, p) - expected) < 1e-10
    
    def test_pbinom(self):
        """Test binomial CDF function."""
        n, p = 10, 0.3
        x = 3
        
        expected = stats.binom.cdf(x, n=n, p=p)
        assert abs(pbinom(x, n, p) - expected) < 1e-10
    
    def test_rbinom(self):
        """Test binomial random generation."""
        rng = RNG(42)
        n, p = 10, 0.3
        
        samples = [rbinom(n, p, rng) for _ in range(1000)]
        
        # All should be integers in [0, n]
        assert all(isinstance(s, int) and 0 <= s <= n for s in samples)
        
        # Check approximate mean
        expected_mean = n * p
        actual_mean = np.mean(samples)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean


class TestPoissonDistribution:
    """Test Poisson distribution functions."""
    
    def test_dpois(self):
        """Test Poisson PMF function."""
        lam = 3.5
        x = 4
        
        expected = stats.poisson.pmf(x, mu=lam)
        assert abs(dpois(x, lam) - expected) < 1e-10
    
    def test_ppois(self):
        """Test Poisson CDF function."""
        lam = 3.5
        x = 4
        
        expected = stats.poisson.cdf(x, mu=lam)
        assert abs(ppois(x, lam) - expected) < 1e-10
    
    def test_rpois(self):
        """Test Poisson random generation."""
        rng = RNG(42)
        lam = 3.5
        
        samples = [rpois(lam, rng) for _ in range(1000)]
        
        # All should be non-negative integers
        assert all(isinstance(s, int) and s >= 0 for s in samples)
        
        # Check approximate mean
        actual_mean = np.mean(samples)
        assert 0.8 * lam < actual_mean < 1.2 * lam


class TestExponentialDistribution:
    """Test exponential distribution functions."""
    
    def test_dexp(self):
        """Test exponential density function."""
        rate = 2.0
        x = 1.0
        
        expected = stats.expon.pdf(x, scale=1.0/rate)
        assert abs(dexp(x, rate) - expected) < 1e-10
    
    def test_pexp(self):
        """Test exponential CDF function."""
        rate = 2.0
        x = 1.0
        
        expected = stats.expon.cdf(x, scale=1.0/rate)
        assert abs(pexp(x, rate) - expected) < 1e-10
    
    def test_rexp(self):
        """Test exponential random generation."""
        rng = RNG(42)
        rate = 2.0
        
        samples = [rexp(rate, rng) for _ in range(1000)]
        
        # All should be positive
        assert all(s > 0 for s in samples)
        
        # Check approximate mean
        expected_mean = 1.0 / rate
        actual_mean = np.mean(samples)
        assert 0.8 * expected_mean < actual_mean < 1.2 * expected_mean


class TestMultinomial:
    """Test multinomial distribution."""
    
    def test_rmultinom(self):
        """Test multinomial random generation."""
        rng = RNG(42)
        n = 100
        probs = [0.2, 0.3, 0.5]
        
        result = rmultinom(n, probs, rng)
        
        # Should sum to n
        assert np.sum(result) == n
        
        # Should have correct length
        assert len(result) == len(probs)
        
        # All counts should be non-negative
        assert all(count >= 0 for count in result)


class TestSpecialFunctions:
    """Test special mathematical functions."""
    
    def test_gamma_func(self):
        """Test gamma function."""
        # Gamma(1) = 1
        assert abs(gamma_func(1.0) - 1.0) < 1e-10
        
        # Gamma(2) = 1
        assert abs(gamma_func(2.0) - 1.0) < 1e-10
        
        # Gamma(3) = 2
        assert abs(gamma_func(3.0) - 2.0) < 1e-10
        
        # Gamma(0.5) = sqrt(pi)
        assert abs(gamma_func(0.5) - np.sqrt(np.pi)) < 1e-10
    
    def test_lgamma_func(self):
        """Test log gamma function."""
        x = 5.5
        expected = np.log(gamma_func(x))
        assert abs(lgamma_func(x) - expected) < 1e-10
    
    def test_digamma_func(self):
        """Test digamma function."""
        # Test at a few known values
        x = 1.0
        result = digamma_func(x)
        assert isinstance(result, float)
    
    def test_trigamma_func(self):
        """Test trigamma function."""
        x = 2.0
        result = trigamma_func(x)
        assert isinstance(result, float)
        assert result > 0  # Should be positive for x > 0
    
    def test_beta_func(self):
        """Test beta function."""
        a, b = 2.0, 3.0
        expected = gamma_func(a) * gamma_func(b) / gamma_func(a + b)
        assert abs(beta_func(a, b) - expected) < 1e-10
    
    def test_lbeta_func(self):
        """Test log beta function."""
        a, b = 2.0, 3.0
        expected = np.log(beta_func(a, b))
        assert abs(lbeta_func(a, b) - expected) < 1e-10
    
    def test_choose_func(self):
        """Test binomial coefficient function."""
        # C(5, 2) = 10
        assert abs(choose_func(5, 2) - 10.0) < 1e-10
        
        # C(10, 0) = 1
        assert abs(choose_func(10, 0) - 1.0) < 1e-10
        
        # C(n, k) = 0 for k > n
        assert choose_func(5, 10) == 0.0
        
        # C(n, k) = 0 for k < 0
        assert choose_func(5, -1) == 0.0
    
    def test_lchoose_func(self):
        """Test log binomial coefficient function."""
        n, k = 10, 3
        expected = np.log(choose_func(n, k))
        assert abs(lchoose_func(n, k) - expected) < 1e-10
        
        # Should return -inf for invalid combinations
        assert lchoose_func(5, 10) == -np.inf


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_sign(self):
        """Test sign function."""
        assert sign(5.0) == 1.0
        assert sign(-3.0) == -1.0
        assert sign(0.0) == 0.0
    
    def test_ftrunc(self):
        """Test truncation function."""
        assert ftrunc(3.7) == 3.0
        assert ftrunc(-2.9) == -2.0
        assert ftrunc(5.0) == 5.0


class TestConsistency:
    """Test consistency between distribution functions."""
    
    def test_normal_consistency(self):
        """Test p/q function consistency for normal distribution."""
        mu, sig = 1.0, 2.0
        x = 2.5
        
        # p(q(p)) = p
        p = pnorm(x, mu, sig)
        x_recovered = qnorm(p, mu, sig)
        assert abs(x - x_recovered) < 1e-10
    
    def test_gamma_consistency(self):
        """Test p/q function consistency for gamma distribution."""
        shape, scale = 2.0, 1.5
        x = 3.0
        
        # p(q(p)) = p
        p = pgamma(x, shape, scale)
        x_recovered = qgamma(p, shape, scale)
        assert abs(x - x_recovered) < 1e-10
    
    def test_beta_consistency(self):
        """Test p/q function consistency for beta distribution."""
        alpha, beta = 2.0, 3.0
        x = 0.6
        
        # p(q(p)) = p
        p = pbeta(x, alpha, beta)
        x_recovered = qbeta(p, alpha, beta)
        assert abs(x - x_recovered) < 1e-10