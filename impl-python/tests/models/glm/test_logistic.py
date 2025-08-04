"""Tests for LogisticRegressionModel."""

import pytest
import numpy as np
from boom.models.glm.logistic import LogisticRegressionModel, LogisticRegressionData
from boom.linalg import Vector, SpdMatrix


class TestLogisticRegressionData:
    """Test LogisticRegressionData class."""
    
    def test_construction(self):
        """Test data construction."""
        data = LogisticRegressionData(7, 10, Vector([1.0, 2.0]))
        assert data.successes() == 7
        assert data.trials() == 10
        assert data.failures() == 3
        assert data.proportion() == 0.7
        assert np.allclose(data.x().to_numpy(), [1.0, 2.0])
        
        # Test parameter validation
        with pytest.raises(ValueError):
            LogisticRegressionData(-1, 10, Vector([1.0]))  # Negative successes
        
        with pytest.raises(ValueError):
            LogisticRegressionData(15, 10, Vector([1.0]))  # Successes > trials
    
    def test_clone(self):
        """Test data cloning."""
        data = LogisticRegressionData(5, 8, Vector([1.5, -0.5]))
        cloned = data.clone()
        
        assert cloned.successes() == data.successes()
        assert cloned.trials() == data.trials()
        assert np.allclose(cloned.x().to_numpy(), data.x().to_numpy())
        
        # Ensure independence
        cloned.set_x(Vector([0.0, 0.0]))
        assert not np.allclose(cloned.x().to_numpy(), data.x().to_numpy())


class TestLogisticRegressionModel:
    """Test LogisticRegressionModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = LogisticRegressionModel(xdim=3)
        assert model.xdim() == 3
        assert len(model.beta().to_numpy()) == 3
        assert np.allclose(model.beta().to_numpy(), [0.0, 0.0, 0.0])
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = LogisticRegressionModel(xdim=2)
        
        # Test beta
        beta = Vector([1.5, -0.8])
        model.set_beta(beta)
        assert np.allclose(model.beta().to_numpy(), [1.5, -0.8])
        
        # Test parameter validation
        with pytest.raises(ValueError):
            model.set_beta([1.0, 2.0, 3.0])  # Wrong dimension
    
    def test_prior_setup(self):
        """Test prior setup."""
        model = LogisticRegressionModel(xdim=2)
        
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
        """Test logistic mean function."""
        model = LogisticRegressionModel(xdim=1)
        
        # Test various inputs
        assert abs(model.mean_function(0.0) - 0.5) < 1e-10  # logit(0) = 0.5
        assert model.mean_function(1000.0) == 1.0  # Should handle overflow
        assert model.mean_function(-1000.0) == 0.0  # Should handle underflow
        
        # Test reasonable range
        linear_pred = 1.0
        expected = np.exp(1.0) / (1.0 + np.exp(1.0))
        assert abs(model.mean_function(linear_pred) - expected) < 1e-10
    
    def test_variance_function(self):
        """Test binomial variance function."""
        model = LogisticRegressionModel(xdim=1)
        
        # Variance = p * (1 - p)
        assert abs(model.variance_function(0.5) - 0.25) < 1e-10
        assert abs(model.variance_function(0.0) - 0.0) < 1e-10
        assert abs(model.variance_function(1.0) - 0.0) < 1e-10
        assert abs(model.variance_function(0.3) - 0.21) < 1e-10
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = LogisticRegressionModel(xdim=2)
        
        # Add data as tuples
        model.add_data((7, 10, Vector([1.0, 2.0])))
        model.add_data((3, 5, Vector([0.5, -1.0])))
        
        # Add data as LogisticRegressionData objects
        data_obj = LogisticRegressionData(2, 8, Vector([1.5, 0.5]))
        model.add_data(data_obj)
        
        # Add batch data
        batch_data = [
            (4, 6, Vector([2.0, 1.0])),
            (1, 4, Vector([-0.5, 1.5]))
        ]
        model.add_data(batch_data)
        
        assert len(model._data) == 5
        
        # Test clearing data
        model.clear_data()
        assert len(model._data) == 0
    
    def test_predictions(self):
        """Test prediction methods."""
        model = LogisticRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Test probability prediction
        x = Vector([2.0, 1.0])
        prob = model.predict_probability(x)
        linear_pred = 1.0 * 2.0 + (-0.5) * 1.0  # 1.5
        expected_prob = np.exp(1.5) / (1.0 + np.exp(1.5))
        assert abs(prob - expected_prob) < 1e-10
        
        # Test class prediction
        class_pred = model.predict_class(x, threshold=0.5)
        assert class_pred == (1 if prob >= 0.5 else 0)
        
        # Test batch predictions
        X = np.array([[2.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
        probs = model.predict_batch(X)
        assert len(probs) == 3
        for p in probs.to_numpy():
            assert 0 <= p <= 1
    
    def test_log_likelihood(self):
        """Test log likelihood computation."""
        model = LogisticRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, 0.5]))
        
        # Add some data
        model.add_data([
            (7, 10, Vector([1.0, 2.0])),  # High success rate with positive predictors
            (2, 8, Vector([-1.0, 0.0]))   # Low success rate with negative predictors
        ])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        
        # Should be reasonable (negative but not too negative)
        assert log_lik < 0  # Log likelihood should be negative
        assert log_lik > -100  # But not extremely negative
    
    def test_mle(self):
        """Test maximum likelihood estimation."""
        model = LogisticRegressionModel(xdim=2)
        
        # Generate some data with known parameters
        true_beta = np.array([1.0, -0.5])
        np.random.seed(42)
        
        X = np.random.randn(100, 2)
        linear_preds = X @ true_beta
        probs = 1 / (1 + np.exp(-linear_preds))
        
        # Generate binomial data
        for i in range(100):
            trials = 10
            successes = np.random.binomial(trials, probs[i])
            model.add_data((successes, trials, Vector(X[i])))
        
        # Fit model
        model.mle()
        
        # Check that estimates are reasonable
        beta_hat = model.beta().to_numpy()
        assert np.allclose(beta_hat, true_beta, atol=0.3)  # Allow some error
    
    def test_map_estimate(self):
        """Test MAP estimation with prior."""
        model = LogisticRegressionModel(xdim=2)
        
        # Set a reasonable prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add some data
        model.add_data([
            (8, 10, Vector([2.0, 1.0])),
            (2, 10, Vector([-1.0, 0.0])),
            (5, 10, Vector([0.0, 1.0]))
        ])
        
        # Fit model
        model.map_estimate()
        
        # Check that we get reasonable estimates
        beta_hat = model.beta().to_numpy()
        assert len(beta_hat) == 2
        assert not np.any(np.isnan(beta_hat))
    
    def test_laplace_approximation(self):
        """Test Laplace approximation to posterior."""
        model = LogisticRegressionModel(xdim=2)
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add data
        model.add_data([
            (7, 10, Vector([1.0, 0.0])),
            (3, 10, Vector([0.0, 1.0])),
            (5, 10, Vector([1.0, 1.0]))
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
        model = LogisticRegressionModel(xdim=2)
        
        # Set prior
        beta_mean = Vector([0.0, 0.0])
        beta_precision = SpdMatrix([[1.0, 0.0], [0.0, 1.0]])
        model.set_prior(beta_mean, beta_precision)
        
        # Add data
        model.add_data([
            (6, 10, Vector([1.0, 0.0])),
            (4, 10, Vector([0.0, 1.0]))
        ])
        
        # Sample from posterior
        samples = model.sample_posterior_laplace(n=50)
        assert len(samples) == 50
        
        for sample in samples:
            assert len(sample) == 2
            assert not np.any(np.isnan(sample.to_numpy()))
    
    def test_simulation(self):
        """Test data simulation."""
        model = LogisticRegressionModel(xdim=2)
        model.set_beta(Vector([1.0, -0.5]))
        
        # Design matrix
        X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
        trials = [10, 5, 8]
        
        simulated = model.simulate_data(n=3, X=X, trials_per_obs=trials)
        assert len(simulated) == 3
        
        for i, data_point in enumerate(simulated):
            assert isinstance(data_point, LogisticRegressionData)
            assert np.allclose(data_point.x().to_numpy(), X[i])
            assert data_point.trials() == trials[i]
            assert 0 <= data_point.successes() <= trials[i]
    
    def test_clone(self):
        """Test model cloning."""
        model = LogisticRegressionModel(xdim=2)
        model.set_beta(Vector([1.5, -0.8]))
        model.add_data((7, 10, Vector([1.0, 2.0])))
        
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
        model = LogisticRegressionModel(xdim=3)
        beta = Vector([1.5, -0.8, 2.1])
        model.set_beta(beta)
        
        # Test vectorization
        theta = model.vectorize_params()
        assert np.allclose(theta.to_numpy(), beta.to_numpy())
        
        # Test unvectorization
        new_model = LogisticRegressionModel(xdim=3)
        new_model.unvectorize_params(theta)
        assert np.allclose(new_model.beta().to_numpy(), beta.to_numpy())
    
    def test_string_representation(self):
        """Test string representation."""
        model = LogisticRegressionModel(xdim=2)
        model.set_beta(Vector([1.2, -0.5]))
        model.add_data((5, 8, Vector([1.0, 1.0])))
        
        s = str(model)
        assert "LogisticRegressionModel" in s
        assert "xdim=2" in s
        assert "data_points=1" in s