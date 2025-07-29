"""Tests for GLM models."""
import pytest
import numpy as np
from boom.models.glm import RegressionModel, LogisticRegressionModel, PoissonRegressionModel
from boom.models.data import RegressionData
from boom.linalg import Vector, Matrix
from boom.distributions import rng


class TestRegressionModel:
    """Test linear regression model."""
    
    def test_basic_construction(self):
        """Test basic model construction."""
        # Empty model
        model = RegressionModel()
        assert model.xdim is None
        
        # With beta
        beta = Vector([1.0, 2.0, -0.5])
        model = RegressionModel(beta)
        assert model.xdim == 3
        np.testing.assert_array_almost_equal(model.beta, beta)
        assert model.sigma == 1.0
        
        # With dimension
        model = RegressionModel(xdim=4, sigma=2.0)
        assert model.xdim == 4
        assert model.sigma == 2.0
        np.testing.assert_array_almost_equal(model.beta, Vector.zero(4))
    
    def test_parameter_properties(self):
        """Test parameter getters and setters."""
        model = RegressionModel()
        
        # Set beta
        beta = Vector([1.0, -2.0])
        model.beta = beta
        assert model.xdim == 2
        np.testing.assert_array_almost_equal(model.beta, beta)
        
        # Set sigma
        model.sigma = 3.0
        assert model.sigma == 3.0
        assert model.sigsq == 9.0
        
        # Set sigsq
        model.sigsq = 16.0
        assert model.sigsq == 16.0
        assert model.sigma == 4.0
    
    def test_data_management(self):
        """Test data addition and management."""
        model = RegressionModel()
        
        # Add data points
        data1 = (2.5, [1.0, 0.5])
        data2 = (3.0, [1.0, 1.0])
        
        model.add_data(data1)
        assert model.xdim == 2
        assert len(model._data) == 1
        
        model.add_data(data2)
        assert len(model._data) == 2
        
        # Clear data
        model.clear_data()
        assert len(model._data) == 0
        
        # Set data
        model.set_data([data1, data2])
        assert len(model._data) == 2
    
    def test_prediction(self):
        """Test prediction methods."""
        beta = Vector([2.0, -1.0])
        model = RegressionModel(beta)
        
        # Single prediction
        x = Vector([1.0, 2.0])
        pred = model.predict(x)
        expected = 2.0 * 1.0 + (-1.0) * 2.0  # 0.0
        assert abs(pred - expected) < 1e-10
        
        # Batch prediction
        X = Matrix([[1.0, 2.0], [1.0, 0.0], [1.0, 1.0]])
        preds = model.predict_batch(X)
        expected_preds = Vector([0.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(preds, expected_preds)
    
    def test_ols_estimation(self):
        """Test OLS estimation."""
        rng.set_seed(123)
        
        # Generate synthetic data
        n = 100
        true_beta = Vector([1.0, 2.0, -0.5])
        true_sigma = 0.5
        
        X = Matrix(np.random.randn(n, 3))
        y_true = X @ true_beta
        y = Vector([y_true[i] + rng.rnorm(0, true_sigma) for i in range(n)])
        
        # Fit model
        model = RegressionModel()
        for i in range(n):
            model.add_data((y[i], X.row(i)))
        
        model.ols()
        
        # Check coefficients are close to true values
        np.testing.assert_array_almost_equal(model.beta, true_beta, decimal=1)
        assert abs(model.sigma - true_sigma) < 0.2
    
    def test_simulation(self):
        """Test data simulation."""
        rng.set_seed(456)
        
        beta = Vector([1.0, -0.5])
        sigma = 2.0
        model = RegressionModel(beta, sigma)
        
        # Generate design matrix
        X = Matrix([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        
        # Simulate data
        sim_data = model.simulate(X)
        assert len(sim_data) == 3
        
        for i, data in enumerate(sim_data):
            assert isinstance(data, RegressionData)
            np.testing.assert_array_almost_equal(data.x, X.row(i))
    
    def test_r_squared(self):
        """Test R-squared calculation."""
        # Perfect fit should give R-squared = 1
        model = RegressionModel(Vector([1.0, 2.0]))
        X = Matrix([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y_true = X @ model.beta
        
        for i in range(len(y_true)):
            model.add_data((y_true[i], X.row(i)))
        
        assert abs(model.r_squared() - 1.0) < 1e-10


class TestLogisticRegressionModel:
    """Test logistic regression model."""
    
    def test_basic_construction(self):
        """Test basic model construction."""
        # Empty model
        model = LogisticRegressionModel()
        assert model.xdim is None
        
        # With beta
        beta = Vector([0.5, -1.0])
        model = LogisticRegressionModel(beta)
        assert model.xdim == 2
        np.testing.assert_array_almost_equal(model.beta, beta)
        
        # With dimension
        model = LogisticRegressionModel(xdim=3)
        assert model.xdim == 3
        np.testing.assert_array_almost_equal(model.beta, Vector.zero(3))
    
    def test_data_validation(self):
        """Test data validation for binary responses."""
        model = LogisticRegressionModel()
        
        # Valid data
        model.add_data((1, [1.0, 0.5]))
        model.add_data((0, [1.0, 1.0]))
        
        # Invalid response
        with pytest.raises(ValueError, match="Response must be 0 or 1"):
            model.add_data((2, [1.0, 0.5]))
    
    def test_probability_prediction(self):
        """Test probability prediction."""
        beta = Vector([0.0, 1.0])  # Intercept=0, slope=1
        model = LogisticRegressionModel(beta)
        
        # At x=0, should get p=0.5
        prob = model.predict_prob([1.0, 0.0])
        assert abs(prob - 0.5) < 1e-10
        
        # At x=ln(3), should get p=0.75
        x_val = np.log(3)
        prob = model.predict_prob([1.0, x_val])
        expected = 3.0 / (1.0 + 3.0)  # 0.75
        assert abs(prob - expected) < 1e-10
    
    def test_binary_prediction(self):
        """Test binary prediction."""
        beta = Vector([0.0, 2.0])
        model = LogisticRegressionModel(beta)
        
        # Should predict 1 for positive values
        pred = model.predict([1.0, 1.0])  # prob > 0.5
        assert pred == 1
        
        # Should predict 0 for negative values
        pred = model.predict([1.0, -1.0])  # prob < 0.5
        assert pred == 0
    
    def test_mle_estimation(self):
        """Test MLE estimation."""
        rng.set_seed(789)
        
        # Generate synthetic data
        n = 500
        true_beta = Vector([0.5, 1.0])
        
        X = Matrix(np.column_stack([np.ones(n), np.random.randn(n)]))
        linear_preds = X @ true_beta
        probs = 1 / (1 + np.exp(-linear_preds))
        y = Vector([1 if rng.runif() < probs[i] else 0 for i in range(n)])
        
        # Fit model
        model = LogisticRegressionModel()
        for i in range(n):
            model.add_data((y[i], X.row(i)))
        
        model.mle()
        
        # Check coefficients are reasonable
        np.testing.assert_array_almost_equal(model.beta, true_beta, decimal=0)
    
    def test_pseudo_r_squared(self):
        """Test pseudo R-squared calculation."""
        # Create separable data
        model = LogisticRegressionModel(Vector([0.0, 5.0]))
        
        # Add perfectly separable data
        for i in range(10):
            model.add_data((0, [1.0, -2.0]))  # All negative -> y=0
            model.add_data((1, [1.0, 2.0]))   # All positive -> y=1
        
        pseudo_r2 = model.pseudo_r_squared()
        assert pseudo_r2 > 0.5  # Should have reasonable fit
    
    def test_simulation(self):
        """Test data simulation."""
        rng.set_seed(321)
        
        beta = Vector([0.0, 1.0])
        model = LogisticRegressionModel(beta)
        
        X = Matrix([[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]])
        sim_data = model.simulate(X)
        
        assert len(sim_data) == 3
        for i, data in enumerate(sim_data):
            assert isinstance(data, RegressionData)
            assert data.y in [0, 1]
            np.testing.assert_array_almost_equal(data.x, X.row(i))


class TestPoissonRegressionModel:
    """Test Poisson regression model."""
    
    def test_basic_construction(self):
        """Test basic model construction."""
        # Empty model
        model = PoissonRegressionModel()
        assert model.xdim is None
        
        # With beta
        beta = Vector([1.0, -0.5])
        model = PoissonRegressionModel(beta)
        assert model.xdim == 2
        np.testing.assert_array_almost_equal(model.beta, beta)
        
        # With dimension
        model = PoissonRegressionModel(xdim=3)
        assert model.xdim == 3
        np.testing.assert_array_almost_equal(model.beta, Vector.zero(3))
    
    def test_data_validation(self):
        """Test data validation for count responses."""
        model = PoissonRegressionModel()
        
        # Valid data
        model.add_data((0, [1.0, 0.5]))
        model.add_data((5, [1.0, 1.0]))
        
        # Invalid response (negative)
        with pytest.raises(ValueError, match="Response must be non-negative integer"):
            model.add_data((-1, [1.0, 0.5]))
    
    def test_rate_prediction(self):
        """Test rate prediction."""
        beta = Vector([0.0, 1.0])  # log(rate) = 0 + 1*x
        model = PoissonRegressionModel(beta)
        
        # At x=0, rate should be exp(0) = 1
        rate = model.predict_rate([1.0, 0.0])
        assert abs(rate - 1.0) < 1e-10
        
        # At x=ln(5), rate should be 5
        x_val = np.log(5)
        rate = model.predict_rate([1.0, x_val])
        assert abs(rate - 5.0) < 1e-10
    
    def test_mle_estimation(self):
        """Test MLE estimation."""
        rng.set_seed(654)
        
        # Generate synthetic data
        n = 200
        true_beta = Vector([1.0, 0.5])
        
        X = Matrix(np.column_stack([np.ones(n), np.random.randn(n)]))
        rates = np.exp(X @ true_beta)
        y = Vector([rng.rpois(rates[i]) for i in range(n)])
        
        # Fit model
        model = PoissonRegressionModel()
        for i in range(n):
            model.add_data((y[i], X.row(i)))
        
        model.mle()
        
        # Check coefficients are reasonable
        np.testing.assert_array_almost_equal(model.beta, true_beta, decimal=0)
    
    def test_residuals(self):
        """Test residual calculations."""
        model = PoissonRegressionModel(Vector([1.0, 0.0]))
        
        # Add some data
        model.add_data((3, [1.0, 0.0]))  # rate = exp(1) ≈ 2.72
        model.add_data((2, [1.0, 0.0]))
        model.add_data((4, [1.0, 0.0]))
        
        # Check residuals
        pearson_resid = model.pearson_residuals()
        dev_resid = model.deviance_residuals()
        
        assert len(pearson_resid) == 3
        assert len(dev_resid) == 3
    
    def test_simulation(self):
        """Test data simulation."""
        rng.set_seed(987)
        
        beta = Vector([0.0, 1.0])
        model = PoissonRegressionModel(beta)
        
        X = Matrix([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        sim_data = model.simulate(X)
        
        assert len(sim_data) == 3
        for i, data in enumerate(sim_data):
            assert isinstance(data, RegressionData)
            assert isinstance(data.y, (int, np.integer))
            assert data.y >= 0
            np.testing.assert_array_almost_equal(data.x, X.row(i))


if __name__ == "__main__":
    pytest.main([__file__])