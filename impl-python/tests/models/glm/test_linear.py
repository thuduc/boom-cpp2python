"""Tests for LinearRegressionModel."""

import pytest
import numpy as np
from boom.models.glm.linear import LinearRegressionModel
from boom.models.glm.base import GlmData
from boom.linalg import Vector, SpdMatrix


class TestLinearRegressionModel:
    """Test LinearRegressionModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = LinearRegressionModel(xdim=3, sigma_sq=2.5)
        assert model.xdim() == 3
        assert model.sigma_sq() == 2.5
        assert model.sigma() == np.sqrt(2.5)
        assert len(model.beta().to_numpy()) == 3
        
        # Test parameter validation
        with pytest.raises(ValueError):
            LinearRegressionModel(xdim=2, sigma_sq=-1.0)  # Invalid sigma_sq
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = LinearRegressionModel(xdim=2)
        
        # Test beta
        beta = Vector([1.5, -0.8])
        model.set_beta(beta)
        assert np.allclose(model.beta().to_numpy(), [1.5, -0.8])
        
        # Test sigma
        model.set_sigma(2.0)
        assert model.sigma() == 2.0
        assert model.sigma_sq() == 4.0
        
        model.set_sigma_sq(9.0)
        assert model.sigma_sq() == 9.0
        assert model.sigma() == 3.0
        
        # Test parameter validation
        with pytest.raises(ValueError):
            model.set_sigma(-1.0)
        
        with pytest.raises(ValueError):
            model.set_sigma_sq(0.0)
        
        with pytest.raises(ValueError):
            model.set_beta([1.0, 2.0, 3.0])  # Wrong dimension
    
    def test_conjugate_prior(self):
        """Test conjugate prior setup."""
        model = LinearRegressionModel(xdim=2)
        
        # Set conjugate prior
        beta_mean = Vector([0.0, 1.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 2.0]])
        model.set_conjugate_prior(beta_mean, beta_precision, sigma_a=3.0, sigma_b=2.0)
        
        # Test log prior
        model.set_beta(Vector([0.5, 1.2]))
        model.set_sigma_sq(1.0)
        log_prior = model.log_prior()
        assert isinstance(log_prior, float)
        assert not np.isnan(log_prior)
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = LinearRegressionModel(xdim=2)
        
        # Add data as tuples
        model.add_data((3.5, Vector([1.0, 2.0])))
        model.add_data((1.2, Vector([0.5, -1.0])))
        
        # Add data as GlmData objects
        data_obj = GlmData(2.8, Vector([1.5, 0.5]))
        model.add_data(data_obj)
        
        # Add batch data
        batch_data = [
            (4.1, Vector([2.0, 1.0])),
            (0.9, Vector([-0.5, 1.5]))
        ]
        model.add_data(batch_data)
        
        # Check sufficient statistics
        suf = model.suf()
        assert suf.n() == 5
        assert suf.yty() > 0
        assert len(suf.xty()) == 2
        assert suf.xtx().nrow() == 2
        
        # Test clearing data
        model.clear_data()
        assert suf.n() == 0
    
    def test_predictions(self):
        """Test prediction methods."""
        model = LinearRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Test single prediction
        x = Vector([2.0, 1.0])
        pred = model.predict(x)
        expected = 1.0 * 2.0 + (-0.5) * 1.0  # 1.5
        assert abs(pred - expected) < 1e-10
        
        # Test batch predictions
        X = np.array([[2.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
        preds = model.predict_batch(X)
        expected_preds = [1.5, -1.0, 1.0]
        assert np.allclose(preds.to_numpy(), expected_preds)
    
    def test_log_likelihood(self):
        """Test log likelihood computation."""
        model = LinearRegressionModel(xdim=2, sigma_sq=1.0)
        model.set_beta(Vector([1.0, 0.5]))
        
        # Add some data
        model.add_data([
            (2.0, Vector([1.0, 2.0])),  # y=2, mu=1*1+0.5*2=2
            (1.5, Vector([0.0, 1.0]))   # y=1.5, mu=0.5
        ])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        
        # Manual calculation for verification
        from boom.distributions.rmath import dnorm
        expected_log_lik = (dnorm(2.0, 2.0, 1.0, log=True) + 
                           dnorm(1.5, 0.5, 1.0, log=True))
        assert abs(log_lik - expected_log_lik) < 1e-10
    
    def test_mle(self):
        """Test maximum likelihood estimation."""
        model = LinearRegressionModel(xdim=2)
        
        # Generate some data with known parameters
        true_beta = np.array([2.0, -1.0])
        true_sigma = 0.5
        np.random.seed(42)
        
        X = np.random.randn(50, 2)
        y = X @ true_beta + true_sigma * np.random.randn(50)
        
        # Add data to model
        for i in range(50):
            model.add_data((y[i], Vector(X[i])))
        
        # Fit model
        model.mle()
        
        # Check that estimates are reasonable
        beta_hat = model.beta().to_numpy()
        assert np.allclose(beta_hat, true_beta, atol=0.2)  # Allow some error
        assert abs(model.sigma() - true_sigma) < 0.2
    
    def test_r_squared(self):
        """Test R-squared computation."""
        model = LinearRegressionModel(xdim=1)
        model.set_beta(Vector([1.0]))
        
        # Perfect fit data
        model.add_data([
            (1.0, Vector([1.0])),
            (2.0, Vector([2.0])),
            (3.0, Vector([3.0]))
        ])
        
        r_sq = model.r_squared()
        assert abs(r_sq - 1.0) < 1e-10  # Should be perfect fit
    
    def test_conjugate_inference(self):
        """Test conjugate Bayesian inference."""
        model = LinearRegressionModel(xdim=2)
        
        # Set conjugate prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_conjugate_prior(beta_mean, beta_precision, sigma_a=2.0, sigma_b=1.0)
        
        # Add some data
        model.add_data([
            (1.0, Vector([1.0, 0.0])),
            (2.0, Vector([0.0, 1.0])),
            (1.5, Vector([1.0, 1.0]))
        ])
        
        # Test posterior mode
        beta_mode, sigma_sq_mode = model.posterior_mode()
        assert len(beta_mode) == 2
        assert sigma_sq_mode > 0
        
        # Test posterior mean
        beta_mean_post, sigma_sq_mean = model.posterior_mean()
        assert len(beta_mean_post) == 2
        assert sigma_sq_mean > 0
    
    def test_posterior_sampling(self):
        """Test posterior sampling."""
        model = LinearRegressionModel(xdim=2)
        
        # Set conjugate prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_conjugate_prior(beta_mean, beta_precision, sigma_a=2.0, sigma_b=1.0)
        
        # Add some data
        model.add_data([
            (1.0, Vector([1.0, 0.0])),
            (2.0, Vector([0.0, 1.0]))
        ])
        
        # Sample from posterior
        samples = model.sample_posterior(n=100)
        assert len(samples) == 100
        
        for beta_sample, sigma_sq_sample in samples:
            assert len(beta_sample) == 2
            assert sigma_sq_sample > 0
    
    def test_simulation(self):
        """Test data simulation."""
        model = LinearRegressionModel(xdim=2, sigma_sq=1.0)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Design matrix
        X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
        
        simulated = model.simulate_data(n=3, X=X)
        assert len(simulated) == 3
        
        for i, data_point in enumerate(simulated):
            assert isinstance(data_point, GlmData)
            assert np.allclose(data_point.x().to_numpy(), X[i])
            # y should be approximately beta^T * x, but with noise
            expected_mean = np.dot([1.0, -0.5], X[i])
            assert abs(data_point.y() - expected_mean) < 5.0  # Allow for noise
    
    def test_clone(self):
        """Test model cloning."""
        model = LinearRegressionModel(xdim=2, sigma_sq=2.0)
        model.set_beta(Vector([1.5, -0.8]))
        model.add_data((3.0, Vector([1.0, 2.0])))
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_conjugate_prior(beta_mean, beta_precision, sigma_a=2.0, sigma_b=1.0)
        
        cloned = model.clone()
        
        # Check parameters
        assert np.allclose(cloned.beta().to_numpy(), model.beta().to_numpy())
        assert cloned.sigma_sq() == model.sigma_sq()
        
        # Check data
        assert len(cloned._data) == len(model._data)
        
        # Check prior
        assert abs(cloned.log_prior() - model.log_prior()) < 1e-10
        
        # Ensure independence
        cloned.set_beta(Vector([0.0, 0.0]))
        assert not np.allclose(cloned.beta().to_numpy(), model.beta().to_numpy())
    
    def test_vectorization(self):
        """Test parameter vectorization."""
        model = LinearRegressionModel(xdim=3)
        beta = Vector([1.5, -0.8, 2.1])
        model.set_beta(beta)
        
        # Test vectorization
        theta = model.vectorize_params()
        assert np.allclose(theta.to_numpy(), beta.to_numpy())
        
        # Test unvectorization
        new_model = LinearRegressionModel(xdim=3)
        new_model.unvectorize_params(theta)
        assert np.allclose(new_model.beta().to_numpy(), beta.to_numpy())
    
    def test_string_representation(self):
        """Test string representation."""
        model = LinearRegressionModel(xdim=2, sigma_sq=1.5)
        model.set_beta(Vector([1.2, -0.5]))
        model.add_data((2.0, Vector([1.0, 1.0])))
        
        s = str(model)
        assert "LinearRegressionModel" in s
        assert "xdim=2" in s
        assert "data_points=1" in s