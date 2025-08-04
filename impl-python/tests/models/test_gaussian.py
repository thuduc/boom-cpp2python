"""Tests for Gaussian model."""

import pytest
import numpy as np
from boom.models import GaussianModel
from boom.models.data import DoubleData
from boom.linalg import Vector


class TestGaussianModelConstruction:
    """Test GaussianModel construction."""
    
    def test_default_construction(self):
        """Test default construction."""
        model = GaussianModel()
        
        assert model.mu() == 0.0
        assert model.sigma_sq() == 1.0
        assert model.sigma() == 1.0
        assert model.sample_size() == 0
    
    def test_construction_with_params(self):
        """Test construction with parameters."""
        model = GaussianModel(mu=2.5, sigma_sq=4.0)
        
        assert model.mu() == 2.5
        assert model.sigma_sq() == 4.0
        assert model.sigma() == 2.0
    
    def test_parameter_setting(self):
        """Test parameter setting."""
        model = GaussianModel()
        
        # Set mu
        model.set_mu(3.0)
        assert model.mu() == 3.0
        
        # Set sigma_sq
        model.set_sigma_sq(9.0)
        assert model.sigma_sq() == 9.0
        assert model.sigma() == 3.0
        
        # Set sigma
        model.set_sigma(2.0)
        assert model.sigma() == 2.0
        assert model.sigma_sq() == 4.0
        
        # Set both parameters
        model.set_params(1.0, 0.25)
        assert model.mu() == 1.0
        assert model.sigma_sq() == 0.25
        assert model.sigma() == 0.5
    
    def test_invalid_variance(self):
        """Test setting invalid variance."""
        model = GaussianModel()
        
        with pytest.raises(ValueError):
            model.set_sigma_sq(-1.0)
        
        with pytest.raises(ValueError):
            model.set_sigma(-1.0)


class TestGaussianModelDataManagement:
    """Test data management in GaussianModel."""
    
    def test_add_data_float(self):
        """Test adding float data."""
        model = GaussianModel()
        
        model.add_data(3.0)
        assert model.sample_size() == 1
        
        # Check sufficient statistics
        suf = model.suf()
        assert suf.n() == 1
        assert suf.sum() == 3.0
        assert suf.sumsq() == 9.0
    
    def test_add_data_double_data(self):
        """Test adding DoubleData objects."""
        model = GaussianModel()
        
        data = DoubleData(2.5)
        model.add_data(data)
        
        assert model.sample_size() == 1
        assert model.suf().sum() == 2.5
    
    def test_add_data_list(self):
        """Test adding list of data."""
        model = GaussianModel()
        
        model.add_data([1.0, 2.0, 3.0])
        
        assert model.sample_size() == 3
        suf = model.suf()
        assert suf.n() == 3
        assert suf.sum() == 6.0
        assert suf.sumsq() == 14.0
    
    def test_clear_data(self):
        """Test clearing data."""
        model = GaussianModel()
        model.add_data([1.0, 2.0, 3.0])
        
        model.clear_data()
        
        assert model.sample_size() == 0
        assert model.suf().n() == 0
        assert model.suf().sum() == 0.0


class TestGaussianModelLikelihood:
    """Test likelihood computation."""
    
    def test_log_likelihood_empty(self):
        """Test log likelihood with no data."""
        model = GaussianModel(mu=0.0, sigma_sq=1.0)
        
        assert model.log_likelihood() == 0.0
    
    def test_log_likelihood_single_point(self):
        """Test log likelihood with single data point."""
        model = GaussianModel(mu=0.0, sigma_sq=1.0)
        model.add_data(0.0)
        
        # For x=0, mu=0, sigma=1: log likelihood = -0.5*log(2*pi) ≈ -0.91894
        log_lik = model.log_likelihood()
        expected = -0.5 * np.log(2 * np.pi)
        assert abs(log_lik - expected) < 1e-10
    
    def test_log_likelihood_multiple_points(self):
        """Test log likelihood with multiple data points."""
        model = GaussianModel(mu=2.0, sigma_sq=4.0)
        data = [1.0, 2.0, 3.0]
        model.add_data(data)
        
        # Manual calculation
        n = len(data)
        sigma_sq = 4.0
        mu = 2.0
        
        # Sum of squared deviations
        sum_sq_dev = sum((x - mu)**2 for x in data)
        
        expected = (-0.5 * n * np.log(2 * np.pi) - 
                   0.5 * n * np.log(sigma_sq) - 
                   0.5 * sum_sq_dev / sigma_sq)
        
        log_lik = model.log_likelihood()
        assert abs(log_lik - expected) < 1e-10
    
    def test_log_likelihood_provided_data(self):
        """Test log likelihood with provided data."""
        model = GaussianModel(mu=1.0, sigma_sq=1.0)
        
        # Don't add data to model, provide it directly
        data = [DoubleData(1.0), DoubleData(2.0)]
        log_lik = model.log_likelihood(data)
        
        # Should compute likelihood for provided data
        assert log_lik < 0  # Some finite negative value
        
        # Model should still have no data
        assert model.sample_size() == 0


class TestGaussianModelMLE:
    """Test maximum likelihood estimation."""
    
    def test_mle_no_data(self):
        """Test MLE with no data."""
        model = GaussianModel(mu=5.0, sigma_sq=2.0)
        
        # Should not change parameters
        model.mle()
        assert model.mu() == 5.0
        assert model.sigma_sq() == 2.0
    
    def test_mle_with_data(self):
        """Test MLE with data."""
        model = GaussianModel()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        model.add_data(data)
        
        model.mle()
        
        # MLE estimates
        expected_mu = np.mean(data)
        expected_sigma_sq = np.var(data, ddof=0)  # Population variance
        
        assert abs(model.mu() - expected_mu) < 1e-10
        assert abs(model.sigma_sq() - expected_sigma_sq) < 1e-10


class TestGaussianModelPriors:
    """Test prior distributions."""
    
    def test_conjugate_prior_setting(self):
        """Test setting conjugate prior."""
        model = GaussianModel()
        
        model.set_conjugate_prior(mu0=1.0, kappa0=2.0, alpha0=3.0, beta0=4.0)
        
        # Check that prior parameters are stored
        assert model._mu0 == 1.0
        assert model._kappa0 == 2.0
        assert model._alpha0 == 3.0
        assert model._beta0 == 4.0
    
    def test_log_prior_non_informative(self):
        """Test log prior with non-informative prior."""
        model = GaussianModel()
        
        # Default prior should be non-informative
        assert model.log_prior() == 0.0
    
    def test_log_prior_informative(self):
        """Test log prior with informative prior."""
        model = GaussianModel(mu=1.0, sigma_sq=1.0)
        model.set_conjugate_prior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        
        log_prior = model.log_prior()
        
        # Should be finite (not zero with informative prior)
        assert np.isfinite(log_prior)
        assert log_prior != 0.0
    
    def test_posterior_mode_non_informative(self):
        """Test posterior mode with non-informative prior."""
        model = GaussianModel()
        data = [1.0, 2.0, 3.0]
        model.add_data(data)
        
        mu_mode, sigma_sq_mode = model.posterior_mode()
        
        # Should equal MLE with non-informative prior
        expected_mu = np.mean(data)
        expected_sigma_sq = np.var(data, ddof=0)
        
        assert abs(mu_mode - expected_mu) < 1e-10
        assert abs(sigma_sq_mode - expected_sigma_sq) < 1e-10
    
    def test_posterior_mean_informative(self):
        """Test posterior mean with informative prior."""
        model = GaussianModel()
        model.set_conjugate_prior(mu0=0.0, kappa0=1.0, alpha0=3.0, beta0=2.0)
        
        data = [1.0, 2.0, 3.0]
        model.add_data(data)
        
        mu_mean, sigma_sq_mean = model.posterior_mean()
        
        # Should be finite
        assert np.isfinite(mu_mean)
        assert np.isfinite(sigma_sq_mean)
        assert sigma_sq_mean > 0
    
    def test_posterior_sampling_requires_informative_prior(self):
        """Test that posterior sampling requires informative prior."""
        model = GaussianModel()
        
        with pytest.raises(NotImplementedError):
            model.sample_posterior(1)
    
    def test_posterior_sampling_informative(self):
        """Test posterior sampling with informative prior."""
        model = GaussianModel()
        model.set_conjugate_prior(mu0=0.0, kappa0=1.0, alpha0=3.0, beta0=2.0)
        
        data = [1.0, 2.0]
        model.add_data(data)
        
        samples = model.sample_posterior(5)
        
        assert len(samples) == 5
        for mu_sample, sigma_sq_sample in samples:
            assert np.isfinite(mu_sample)
            assert np.isfinite(sigma_sq_sample)
            assert sigma_sq_sample > 0


class TestGaussianModelVectorization:
    """Test parameter vectorization."""
    
    def test_vectorize_params(self):
        """Test parameter vectorization."""
        model = GaussianModel(mu=2.0, sigma_sq=4.0)
        
        theta = model.vectorize_params()
        
        assert len(theta) == 2
        assert theta[0] == 2.0  # mu
        assert theta[1] == np.log(4.0)  # log(sigma_sq)
    
    def test_unvectorize_params(self):
        """Test parameter unvectorization."""
        model = GaussianModel()
        
        theta = Vector([1.5, np.log(9.0)])
        model.unvectorize_params(theta)
        
        assert model.mu() == 1.5
        assert abs(model.sigma_sq() - 9.0) < 1e-10
        assert abs(model.sigma() - 3.0) < 1e-10
    
    def test_unvectorize_wrong_size(self):
        """Test unvectorize with wrong size."""
        model = GaussianModel()
        
        with pytest.raises(ValueError):
            model.unvectorize_params(Vector([1.0]))  # Too short


class TestGaussianModelSimulation:
    """Test data simulation."""
    
    def test_simulate_data_empty(self):
        """Test simulation with n=0."""
        model = GaussianModel()
        
        simulated = model.simulate_data(0)
        assert len(simulated) == 0
    
    def test_simulate_data_explicit_n(self):
        """Test simulation with explicit n."""
        model = GaussianModel(mu=2.0, sigma_sq=1.0)
        
        simulated = model.simulate_data(10)
        
        assert len(simulated) == 10
        for data_point in simulated:
            assert isinstance(data_point, DoubleData)
            assert np.isfinite(data_point.value())
    
    def test_simulate_data_default_n(self):
        """Test simulation with default n (current sample size)."""
        model = GaussianModel()
        model.add_data([1.0, 2.0, 3.0])
        
        simulated = model.simulate_data()  # Should use n=3
        
        assert len(simulated) == 3


class TestGaussianModelUtilities:
    """Test utility methods."""
    
    def test_information_criteria(self):
        """Test AIC and BIC computation."""
        model = GaussianModel(mu=0.0, sigma_sq=1.0)
        model.add_data([0.0, 1.0, -1.0])
        
        # Should compute finite values
        aic = model.AIC()
        bic = model.BIC()
        deviance = model.deviance()
        
        assert np.isfinite(aic)
        assert np.isfinite(bic) 
        assert np.isfinite(deviance)
        assert deviance > 0  # Should be positive
    
    def test_clone(self):
        """Test cloning."""
        model = GaussianModel(mu=2.0, sigma_sq=3.0)
        model.set_conjugate_prior(mu0=1.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        model.add_data([1.0, 2.0])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.mu() == model.mu()
        assert cloned.sigma_sq() == model.sigma_sq()
        
        # Check prior parameters
        assert cloned._mu0 == model._mu0
        assert cloned._kappa0 == model._kappa0
        assert cloned._alpha0 == model._alpha0
        assert cloned._beta0 == model._beta0
        
        # Check data
        assert cloned.sample_size() == model.sample_size()
        assert cloned.suf().n() == model.suf().n()
        
        # Should be independent objects
        assert cloned is not model
        
        # Modifications should be independent
        cloned.set_mu(99.0)
        assert model.mu() == 2.0
    
    def test_str_representation(self):
        """Test string representation."""
        model = GaussianModel(mu=1.5, sigma_sq=2.25)
        model.add_data([1.0, 2.0])
        
        s = str(model)
        assert "GaussianModel" in s
        assert "mu=1.500" in s
        assert "sigma=1.500" in s
        assert "n=2" in s