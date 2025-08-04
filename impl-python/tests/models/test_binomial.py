"""Tests for BinomialModel."""

import pytest
import numpy as np
from boom.models.binomial import BinomialModel
from boom.models.data import BinomialData
from boom.linalg import Vector


class TestBinomialModel:
    """Test BinomialModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = BinomialModel(p=0.3, n=10)
        assert model.p() == 0.3
        assert model.n() == 10
        
        # Test parameter validation
        with pytest.raises(ValueError):
            BinomialModel(p=-0.1)  # Invalid p
        
        with pytest.raises(ValueError):
            BinomialModel(p=1.5)   # Invalid p
        
        with pytest.raises(ValueError):
            BinomialModel(n=-1)    # Invalid n
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = BinomialModel()
        
        # Test setter validation
        with pytest.raises(ValueError):
            model.set_p(-0.1)
        
        with pytest.raises(ValueError):
            model.set_p(1.1)
        
        # Test valid setting
        model.set_p(0.7)
        assert model.p() == 0.7
        
        model.set_n(20)
        assert model.n() == 20
    
    def test_conjugate_prior(self):
        """Test conjugate Beta prior."""
        model = BinomialModel()
        
        # Set conjugate prior
        model.set_conjugate_prior(alpha=2.0, beta=3.0)
        
        # Test log prior
        model.set_p(0.4)
        log_prior = model.log_prior()
        
        # Should be Beta(2, 3) log pdf at 0.4
        from boom.distributions.rmath import dbeta
        expected = dbeta(0.4, 2.0, 3.0, log=True)
        assert abs(log_prior - expected) < 1e-10
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = BinomialModel()
        
        # Add data as tuples
        model.add_data((10, 7))  # 10 trials, 7 successes
        model.add_data((5, 2))   # 5 trials, 2 successes
        
        # Check sufficient statistics
        suf = model.suf()
        assert suf.n() == 15      # Total trials
        assert suf.successes() == 9  # Total successes
        assert suf.failures() == 6   # Total failures
        
        # Add data as dict
        model.add_data({'trials': 3, 'successes': 1})
        assert suf.n() == 18
        assert suf.successes() == 10
        
        # Add data as BinomialData
        data_obj = BinomialData(4, 3)
        model.add_data(data_obj)
        assert suf.n() == 22
        assert suf.successes() == 13
        
        # Test clearing data
        model.clear_data()
        assert suf.n() == 0
        assert suf.successes() == 0
    
    def test_log_likelihood(self):
        """Test log likelihood computation."""
        model = BinomialModel(p=0.6)
        
        # Add some data
        model.add_data([(10, 6), (5, 3), (8, 5)])  # Total: 23 trials, 14 successes
        
        # Test log likelihood using sufficient statistics
        log_lik = model.log_likelihood()
        
        # Manual calculation: 14*log(0.6) + 9*log(0.4)
        expected = 14 * np.log(0.6) + 9 * np.log(0.4)
        assert abs(log_lik - expected) < 1e-10
        
        # Test log likelihood with provided data
        test_data = [BinomialData(5, 3), BinomialData(3, 1)]
        log_lik_test = model.log_likelihood(test_data)
        
        # Manual calculation
        from boom.distributions.rmath import dbinom
        expected_test = (dbinom(3, 5, 0.6, log=True) + 
                        dbinom(1, 3, 0.6, log=True))
        assert abs(log_lik_test - expected_test) < 1e-10
    
    def test_simulation(self):
        """Test data simulation."""
        model = BinomialModel(p=0.4)
        
        simulated = model.simulate_data(n=100, trials_per_obs=10)
        
        assert len(simulated) == 100
        
        # Check that all simulated data are BinomialData with correct trials
        for data_point in simulated:
            assert isinstance(data_point, BinomialData)
            assert data_point.trials() == 10
            assert 0 <= data_point.successes() <= 10
        
        # Check approximate correctness (probabilistic test)
        success_rates = [d.success_rate() for d in simulated]
        mean_rate = np.mean(success_rates)
        
        # Should be approximately 0.4 (allow some variance)
        assert 0.3 < mean_rate < 0.5
    
    def test_parameter_vectorization(self):
        """Test parameter vectorization."""
        model = BinomialModel(p=0.3)
        
        # Test vectorization
        theta = model.vectorize_params()
        assert len(theta) == 1
        
        # Should be logit(0.3)
        expected_logit = np.log(0.3 / 0.7)
        assert abs(theta[0] - expected_logit) < 1e-10
        
        # Test unvectorization
        new_model = BinomialModel()
        new_model.unvectorize_params(theta)
        assert abs(new_model.p() - 0.3) < 1e-10
        
        # Test boundary cases
        model_low = BinomialModel(p=1e-10)
        theta_low = model_low.vectorize_params()
        assert theta_low[0] < -5  # Very negative logit
        
        model_high = BinomialModel(p=1-1e-10)
        theta_high = model_high.vectorize_params()
        assert theta_high[0] > 5   # Very positive logit
    
    def test_mle(self):
        """Test maximum likelihood estimation."""
        model = BinomialModel()
        
        # Add data with known MLE
        model.add_data([(100, 30), (50, 15), (25, 7)])  # Total: 175 trials, 52 successes
        
        model.mle()
        
        # MLE should be success rate
        expected_p = 52 / 175
        assert abs(model.p() - expected_p) < 1e-10
    
    def test_conjugate_inference(self):
        """Test conjugate Bayesian inference."""
        model = BinomialModel()
        model.set_conjugate_prior(alpha=3.0, beta=2.0)  # Beta(3, 2) prior
        
        # Add data
        model.add_data([(10, 7), (5, 2)])  # 15 trials, 9 successes
        
        # Posterior should be Beta(3+9, 2+6) = Beta(12, 8)
        
        # Test posterior mode
        p_mode = model.posterior_mode()
        expected_mode = 11 / 18  # (12-1)/(12+8-2) for Beta(12,8)
        assert abs(p_mode - expected_mode) < 1e-10
        
        # Test posterior mean
        p_mean = model.posterior_mean()
        expected_mean = 12 / 20  # 12/(12+8) for Beta(12,8)
        assert abs(p_mean - expected_mean) < 1e-10
        
        # Test posterior variance
        p_var = model.posterior_variance()
        expected_var = (12 * 8) / (20 * 20 * 21)  # Formula for Beta variance
        assert abs(p_var - expected_var) < 1e-10
    
    def test_posterior_sampling(self):
        """Test posterior sampling."""
        model = BinomialModel()
        model.set_conjugate_prior(alpha=2.0, beta=3.0)
        
        # Add some data
        model.add_data([(10, 4)])
        
        # Sample from posterior
        samples = model.sample_posterior(n=1000)
        
        assert len(samples) == 1000
        
        # All samples should be in [0, 1]
        for sample in samples:
            assert 0 <= sample <= 1
        
        # Check approximate posterior mean
        sample_mean = np.mean(samples)
        expected_mean = 6 / 16  # Beta(6, 9) mean
        assert abs(sample_mean - expected_mean) < 0.05  # Allow some Monte Carlo error
    
    def test_clone(self):
        """Test model cloning."""
        model = BinomialModel(p=0.7, n=15)
        model.set_conjugate_prior(alpha=4.0, beta=1.0)
        model.add_data([(10, 8), (5, 3)])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.p() == model.p()
        assert cloned.n() == model.n()
        
        # Check prior
        assert abs(cloned.log_prior() - model.log_prior()) < 1e-10
        
        # Check data
        assert cloned.suf().n() == model.suf().n()
        assert cloned.suf().successes() == model.suf().successes()
        
        # Ensure independence
        cloned.set_p(0.5)
        assert model.p() == 0.7  # Original unchanged
    
    def test_string_representation(self):
        """Test string representation."""
        model = BinomialModel(p=0.25, n=8)
        model.add_data([(10, 3)])
        
        s = str(model)
        assert "BinomialModel" in s
        assert "p=0.250" in s
        assert "n=8" in s
        assert "data_points=1" in s