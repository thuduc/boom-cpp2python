"""Tests for PoissonRegressionModel."""

import pytest
import numpy as np
from boom.models.glm.poisson import PoissonRegressionModel, PoissonRegressionData
from boom.linalg import Vector, SpdMatrix


class TestPoissonRegressionData:
    """Test PoissonRegressionData class."""
    
    def test_construction(self):
        """Test data construction."""
        data = PoissonRegressionData(5, Vector([1.0, 2.0]), exposure=2.0)
        assert data.count() == 5
        assert data.exposure() == 2.0
        assert data.rate() == 2.5  # 5/2
        assert np.allclose(data.x().to_numpy(), [1.0, 2.0])
        
        # Test default exposure
        data_default = PoissonRegressionData(3, Vector([1.0]))
        assert data_default.exposure() == 1.0
        
        # Test parameter validation
        with pytest.raises(ValueError):
            PoissonRegressionData(-1, Vector([1.0]))  # Negative count
        
        with pytest.raises(ValueError):
            PoissonRegressionData(5, Vector([1.0]), exposure=0.0)  # Zero exposure
    
    def test_clone(self):
        """Test data cloning."""
        data = PoissonRegressionData(8, Vector([1.5, -0.5]), exposure=3.0)
        cloned = data.clone()
        
        assert cloned.count() == data.count()
        assert cloned.exposure() == data.exposure()
        assert np.allclose(cloned.x().to_numpy(), data.x().to_numpy())
        
        # Ensure independence
        cloned.set_x(Vector([0.0, 0.0]))
        assert not np.allclose(cloned.x().to_numpy(), data.x().to_numpy())


class TestPoissonRegressionModel:
    """Test PoissonRegressionModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = PoissonRegressionModel(xdim=3)
        assert model.xdim() == 3
        assert len(model.beta().to_numpy()) == 3
        assert np.allclose(model.beta().to_numpy(), [0.0, 0.0, 0.0])
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = PoissonRegressionModel(xdim=2)
        
        # Test beta
        beta = Vector([1.5, -0.8])
        model.set_beta(beta)
        assert np.allclose(model.beta().to_numpy(), [1.5, -0.8])
        
        # Test parameter validation
        with pytest.raises(ValueError):
            model.set_beta([1.0, 2.0, 3.0])  # Wrong dimension
    
    def test_prior_setup(self):
        """Test prior setup."""
        model = PoissonRegressionModel(xdim=2)
        
        # Set prior
        beta_mean = Vector([0.0, 1.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 2.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Test log prior
        model.set_beta(Vector([0.5, 1.2]))
        log_prior = model.log_prior()
        assert isinstance(log_prior, float)
        assert not np.isnan(log_prior)
    
    def test_mean_function(self):
        """Test exponential mean function."""
        model = PoissonRegressionModel(xdim=1)
        
        # Test various inputs
        assert abs(model.mean_function(0.0) - 1.0) < 1e-10  # exp(0) = 1
        assert model.mean_function(1000.0) == np.exp(500)  # Should handle overflow
        assert model.mean_function(-1000.0) == np.exp(-500)  # Should handle underflow
        
        # Test reasonable range
        linear_pred = 2.0
        expected = np.exp(2.0)
        assert abs(model.mean_function(linear_pred) - expected) < 1e-10
    
    def test_variance_function(self):
        """Test Poisson variance function."""
        model = PoissonRegressionModel(xdim=1)
        
        # Variance = mean for Poisson
        assert abs(model.variance_function(5.0) - 5.0) < 1e-10
        assert abs(model.variance_function(0.1) - 0.1) < 1e-10
        assert abs(model.variance_function(10.5) - 10.5) < 1e-10
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = PoissonRegressionModel(xdim=2)
        
        # Add data as tuples (count, x)
        model.add_data((5, Vector([1.0, 2.0])))
        model.add_data((3, Vector([0.5, -1.0])))
        
        # Add data as tuples (count, x, exposure)
        model.add_data((8, Vector([1.5, 0.5]), 2.0))
        
        # Add data as PoissonRegressionData objects
        data_obj = PoissonRegressionData(12, Vector([2.0, 1.0]), exposure=3.0)
        model.add_data(data_obj)
        
        # Add batch data
        batch_data = [
            (4, Vector([2.0, 1.0])),
            (1, Vector([-0.5, 1.5]), 0.5)
        ]
        model.add_data(batch_data)
        
        assert len(model._data) == 6
        
        # Test clearing data
        model.clear_data()
        assert len(model._data) == 0
    
    def test_predictions(self):
        """Test prediction methods."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Test rate prediction
        x = Vector([2.0, 1.0])
        rate = model.predict_rate(x)
        linear_pred = 1.0 * 2.0 + (-0.5) * 1.0  # 1.5
        expected_rate = np.exp(1.5)
        assert abs(rate - expected_rate) < 1e-10
        
        # Test count prediction
        exposure = 2.0
        expected_count = model.predict_count(x, exposure)
        assert abs(expected_count - expected_rate * exposure) < 1e-10
        
        # Test batch predictions
        X = np.array([[2.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
        rates = model.predict_batch(X)
        assert len(rates) == 3
        for r in rates.to_numpy():
            assert r > 0  # Rates should be positive
    
    def test_log_likelihood(self):
        """Test log likelihood computation."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, 0.5]))
        
        # Add some data
        model.add_data([
            (8, Vector([1.0, 2.0]), 2.0),  # High count with positive predictors
            (1, Vector([-1.0, 0.0]), 1.0)  # Low count with negative predictors
        ])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        
        # Should be reasonable (negative but not too negative)
        assert log_lik < 0  # Log likelihood should be negative
        assert log_lik > -100  # But not extremely negative
    
    def test_mle(self):
        """Test maximum likelihood estimation."""
        model = PoissonRegressionModel(xdim=2)
        
        # Generate some data with known parameters
        true_beta = np.array([0.5, -0.3])
        np.random.seed(42)
        
        X = np.random.randn(100, 2)
        linear_preds = X @ true_beta
        rates = np.exp(linear_preds)
        
        # Generate Poisson data
        for i in range(100):
            count = np.random.poisson(rates[i])
            model.add_data((count, Vector(X[i])))
        
        # Fit model
        model.mle()
        
        # Check that estimates are reasonable
        beta_hat = model.beta().to_numpy()
        assert np.allclose(beta_hat, true_beta, atol=0.2)  # Allow some error
    
    def test_map_estimate(self):
        """Test MAP estimation with prior."""
        model = PoissonRegressionModel(xdim=2)
        
        # Set a reasonable prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add some data
        model.add_data([
            (8, Vector([2.0, 1.0])),
            (2, Vector([-1.0, 0.0])),
            (5, Vector([0.0, 1.0]))
        ])
        
        # Fit model
        model.map_estimate()
        
        # Check that we get reasonable estimates
        beta_hat = model.beta().to_numpy()
        assert len(beta_hat) == 2
        assert not np.any(np.isnan(beta_hat))
    
    def test_laplace_approximation(self):
        """Test Laplace approximation to posterior."""
        model = PoissonRegressionModel(xdim=2)
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add data
        model.add_data([
            (7, Vector([1.0, 0.0])),
            (3, Vector([0.0, 1.0])),
            (5, Vector([1.0, 1.0]))
        ])
        
        # Compute Laplace approximation
        posterior_mean, posterior_cov = model.laplace_approximation()
        
        assert len(posterior_mean) == 2
        assert posterior_cov.nrow() == 2
        assert posterior_cov.ncol() == 2
        
        # Covariance should be positive definite
        cov_array = posterior_cov.to_numpy()
        eigenvals = np.linalg.eigvals(cov_array)
        assert np.all(eigenvals > 0)
    
    def test_posterior_sampling(self):
        """Test posterior sampling via Laplace approximation."""
        model = PoissonRegressionModel(xdim=2)
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add data
        model.add_data([
            (6, Vector([1.0, 0.0])),
            (4, Vector([0.0, 1.0]))
        ])
        
        # Sample from posterior
        samples = model.sample_posterior_laplace(n=50)
        assert len(samples) == 50
        
        for sample in samples:
            assert len(sample) == 2
            assert not np.any(np.isnan(sample.to_numpy()))
    
    def test_model_diagnostics(self):
        """Test model diagnostic methods."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([0.5, -0.3]))
        
        # Add some data
        model.add_data([
            (5, Vector([1.0, 2.0])),
            (2, Vector([-1.0, 1.0])),
            (8, Vector([2.0, 0.0]))
        ])
        
        # Test deviance
        deviance = model.deviance()
        assert isinstance(deviance, float)
        assert deviance >= 0
        
        # Test AIC
        aic = model.aic()
        assert isinstance(aic, float)
        
        # Test BIC
        bic = model.bic()
        assert isinstance(bic, float)
        
        # For small datasets, BIC can be smaller than AIC
        # Just check that both are reasonable values
        assert aic > 0  # AIC should be positive for this data
        assert bic > 0  # BIC should be positive for this data
    
    def test_simulation(self):
        """Test data simulation."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Design matrix
        X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
        exposures = [1.0, 2.0, 0.5]
        
        simulated = model.simulate_data(n=3, X=X, exposures=exposures)
        assert len(simulated) == 3
        
        for i, data_point in enumerate(simulated):
            assert isinstance(data_point, PoissonRegressionData)
            assert np.allclose(data_point.x().to_numpy(), X[i])
            assert data_point.exposure() == exposures[i]
            assert data_point.count() >= 0  # Counts should be non-negative
    
    def test_clone(self):
        """Test model cloning."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([1.5, -0.8]))
        model.add_data((7, Vector([1.0, 2.0])))
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        cloned = model.clone()
        
        # Check parameters
        assert np.allclose(cloned.beta().to_numpy(), model.beta().to_numpy())
        
        # Check data
        assert len(cloned._data) == len(model._data)
        
        # Check prior
        assert abs(cloned.log_prior() - model.log_prior()) < 1e-10
        
        # Ensure independence
        cloned.set_beta(Vector([0.0, 0.0]))
        assert not np.allclose(cloned.beta().to_numpy(), model.beta().to_numpy())
    
    def test_vectorization(self):
        """Test parameter vectorization."""
        model = PoissonRegressionModel(xdim=3)
        beta = Vector([1.5, -0.8, 2.1])
        model.set_beta(beta)
        
        # Test vectorization
        theta = model.vectorize_params()
        assert np.allclose(theta.to_numpy(), beta.to_numpy())
        
        # Test unvectorization
        new_model = PoissonRegressionModel(xdim=3)
        new_model.unvectorize_params(theta)
        assert np.allclose(new_model.beta().to_numpy(), beta.to_numpy())
    
    def test_string_representation(self):
        """Test string representation."""
        model = PoissonRegressionModel(xdim=2)
        model.set_beta(Vector([1.2, -0.5]))
        model.add_data((5, Vector([1.0, 1.0])))
        
        s = str(model)
        assert "PoissonRegressionModel" in s
        assert "xdim=2" in s
        assert "data_points=1" in s