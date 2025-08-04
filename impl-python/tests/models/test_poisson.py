"""Tests for PoissonModel."""

import pytest
import numpy as np
from boom.models.poisson import PoissonModel
from boom.models.data import DoubleData
from boom.linalg import Vector


class TestPoissonModel:
    """Test PoissonModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = PoissonModel(lam=2.5)
        assert model.lam() == 2.5
        assert model.lambda_() == 2.5  # Alias
        
        # Test parameter validation
        with pytest.raises(ValueError):
            PoissonModel(lam=-1.0)  # Invalid lambda
        
        with pytest.raises(ValueError):
            PoissonModel(lam=0.0)   # Invalid lambda
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = PoissonModel()
        
        # Test setter validation
        with pytest.raises(ValueError):
            model.set_lam(-0.5)
        
        with pytest.raises(ValueError):
            model.set_lam(0.0)
        
        # Test valid setting
        model.set_lam(3.2)
        assert model.lam() == 3.2
        assert model.lambda_() == 3.2
        
        # Test alias setter
        model.set_lambda(1.8)
        assert model.lam() == 1.8
    
    def test_conjugate_prior(self):
        """Test conjugate Gamma prior."""
        model = PoissonModel()
        
        # Set conjugate prior
        model.set_conjugate_prior(alpha=2.5, beta=1.5)
        
        # Test log prior
        model.set_lam(2.0)
        log_prior = model.log_prior()
        
        # Should be Gamma(2.5, 1.5) log pdf at 2.0
        from boom.distributions.rmath import dgamma
        expected = dgamma(2.0, 2.5, 1.5, log=True)
        assert abs(log_prior - expected) < 1e-10
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = PoissonModel()
        
        # Add data as integers/floats
        model.add_data([3, 1, 4, 2, 0])
        
        # Check sufficient statistics
        suf = model.suf()
        assert suf.n() == 5
        assert suf.sum() == 10.0  # 3+1+4+2+0
        assert abs(suf.mean() - 2.0) < 1e-10
        
        # Add more data
        model.add_data(DoubleData(5.0))
        assert suf.n() == 6
        assert suf.sum() == 15.0
        
        # Test invalid data
        with pytest.raises(ValueError):
            model.add_data(-1)  # Negative observation
        
        # Test clearing data
        model.clear_data()
        assert suf.n() == 0
        assert suf.sum() == 0.0
    
    def test_log_likelihood(self):
        """Test log likelihood computation."""
        model = PoissonModel(lam=1.5)
        
        # Add some data
        model.add_data([2, 0, 3, 1, 1])  # Total: 5 observations, sum = 7
        
        # Test log likelihood using sufficient statistics
        log_lik = model.log_likelihood()
        
        # Manual calculation: sum_x * log(lambda) - n * lambda
        # (factorial terms omitted as they don't depend on lambda)
        expected = 7 * np.log(1.5) - 5 * 1.5
        assert abs(log_lik - expected) < 1e-10
        
        # Test log likelihood with provided data
        test_data = [DoubleData(2.0), DoubleData(1.0)]
        log_lik_test = model.log_likelihood(test_data)
        
        # Manual calculation
        from boom.distributions.rmath import dpois
        expected_test = (dpois(2, 1.5, log=True) + 
                        dpois(1, 1.5, log=True))
        assert abs(log_lik_test - expected_test) < 1e-10
    
    def test_simulation(self):
        """Test data simulation."""
        model = PoissonModel(lam=3.0)
        
        simulated = model.simulate_data(n=1000)
        
        assert len(simulated) == 1000
        
        # Check that all simulated data are DoubleData with non-negative values
        values = []
        for data_point in simulated:
            assert isinstance(data_point, DoubleData)
            val = data_point.value()
            assert val >= 0
            values.append(val)
        
        # Check approximate correctness (probabilistic test)
        mean_val = np.mean(values)
        
        # Should be approximately 3.0 (allow some variance)
        assert 2.5 < mean_val < 3.5
        
        # Variance should also be approximately 3.0 for Poisson
        var_val = np.var(values)
        assert 2.0 < var_val < 4.0
    
    def test_parameter_vectorization(self):
        """Test parameter vectorization."""
        model = PoissonModel(lam=2.5)
        
        # Test vectorization
        theta = model.vectorize_params()
        assert len(theta) == 1
        
        # Should be log(2.5)
        expected_log = np.log(2.5)
        assert abs(theta[0] - expected_log) < 1e-10
        
        # Test unvectorization
        new_model = PoissonModel()
        new_model.unvectorize_params(theta)
        assert abs(new_model.lam() - 2.5) < 1e-10
    
    def test_mle(self):
        """Test maximum likelihood estimation."""
        model = PoissonModel()
        
        # Add data with known MLE
        model.add_data([2, 3, 1, 4, 0, 2, 1, 3])  # Sum = 16, n = 8
        
        model.mle()
        
        # MLE should be sample mean
        expected_lambda = 16 / 8
        assert abs(model.lam() - expected_lambda) < 1e-10
    
    def test_conjugate_inference(self):
        """Test conjugate Bayesian inference."""
        model = PoissonModel()
        model.set_conjugate_prior(alpha=3.0, beta=2.0)  # Gamma(3, 2) prior
        
        # Add data
        model.add_data([1, 2, 0, 3, 1])  # n=5, sum=7
        
        # Posterior should be Gamma(3+7, 2+5) = Gamma(10, 7)
        
        # Test posterior mode
        lam_mode = model.posterior_mode()
        expected_mode = 9 / 7  # (10-1)/7 for Gamma(10,7)
        assert abs(lam_mode - expected_mode) < 1e-10
        
        # Test posterior mean
        lam_mean = model.posterior_mean()
        expected_mean = 10 / 7  # 10/7 for Gamma(10,7)
        assert abs(lam_mean - expected_mean) < 1e-10
        
        # Test posterior variance
        lam_var = model.posterior_variance()
        expected_var = 10 / (7 * 7)  # alpha/beta^2 for Gamma
        assert abs(lam_var - expected_var) < 1e-10
    
    def test_posterior_sampling(self):
        """Test posterior sampling."""
        model = PoissonModel()
        model.set_conjugate_prior(alpha=2.0, beta=1.0)
        
        # Add some data
        model.add_data([3, 1, 2])  # n=3, sum=6
        
        # Sample from posterior (should be Gamma(8, 4))
        samples = model.sample_posterior(n=1000)
        
        assert len(samples) == 1000
        
        # All samples should be positive
        for sample in samples:
            assert sample > 0
        
        # Check approximate posterior mean
        sample_mean = np.mean(samples)
        expected_mean = 8 / 4  # Gamma(8, 4) mean
        assert abs(sample_mean - expected_mean) < 0.1  # Allow some Monte Carlo error
    
    def test_clone(self):
        """Test model cloning."""
        model = PoissonModel(lam=4.5)
        model.set_conjugate_prior(alpha=2.0, beta=1.5)
        model.add_data([3, 1, 4, 2])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.lam() == model.lam()
        
        # Check prior
        assert abs(cloned.log_prior() - model.log_prior()) < 1e-10
        
        # Check data
        assert cloned.suf().n() == model.suf().n()
        assert cloned.suf().sum() == model.suf().sum()
        
        # Ensure independence
        cloned.set_lam(2.0)
        assert model.lam() == 4.5  # Original unchanged
    
    def test_string_representation(self):
        """Test string representation."""
        model = PoissonModel(lam=1.8)
        model.add_data([2, 1])
        
        s = str(model)
        assert "PoissonModel" in s
        assert "lambda=1.800" in s
        assert "data_points=2" in s