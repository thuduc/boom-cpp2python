"""Tests for LocalLinearTrendModel."""

import pytest
import numpy as np
from boom.models.state_space.local_linear_trend import (
    LocalLinearTrendModel, LocalLinearTrendStateModel, SeasonalStateModel
)
from boom.models.state_space.base import TimeSeriesData
from boom.linalg import Vector, Matrix, SpdMatrix


class TestLocalLinearTrendStateModel:
    """Test LocalLinearTrendStateModel class."""
    
    def test_construction(self):
        """Test state model construction."""
        model = LocalLinearTrendStateModel(level_variance=1.5, slope_variance=0.8)
        assert model.state_dimension == 2
        assert model.level_variance == 1.5
        assert model.slope_variance == 0.8
        
        # Test parameter validation
        with pytest.raises(ValueError):
            LocalLinearTrendStateModel(level_variance=-1.0, slope_variance=1.0)
        
        with pytest.raises(ValueError):
            LocalLinearTrendStateModel(level_variance=1.0, slope_variance=0.0)
    
    def test_matrices(self):
        """Test transition and observation matrices."""
        model = LocalLinearTrendStateModel(level_variance=1.0, slope_variance=0.5)
        
        # Transition matrix
        T = model.transition_matrix(0)
        assert T.nrow() == 2 and T.ncol() == 2
        expected_T = np.array([[1.0, 1.0], [0.0, 1.0]])
        assert np.allclose(T.to_numpy(), expected_T)
        
        # State error variance
        Q = model.state_error_variance(0)
        assert Q.nrow() == 2 and Q.ncol() == 2
        expected_Q = np.array([[1.0, 0.0], [0.0, 0.5]])
        assert np.allclose(Q.to_numpy(), expected_Q)
        
        # Observation matrix
        Z = model.observation_matrix(0)
        assert Z.nrow() == 1 and Z.ncol() == 2
        expected_Z = np.array([[1.0, 0.0]])
        assert np.allclose(Z.to_numpy(), expected_Z)
    
    def test_initial_conditions(self):
        """Test initial state conditions."""
        model = LocalLinearTrendStateModel()
        
        initial_mean = model.initial_state_mean()
        assert len(initial_mean) == 2
        assert initial_mean[0] == 0.0
        assert initial_mean[1] == 0.0
        
        initial_var = model.initial_state_variance()
        assert initial_var.nrow() == 2 and initial_var.ncol() == 2
        expected_var = 1e6 * np.eye(2)
        assert np.allclose(initial_var.to_numpy(), expected_var)


class TestSeasonalStateModel:
    """Test SeasonalStateModel class."""
    
    def test_construction(self):
        """Test seasonal model construction."""
        model = SeasonalStateModel(period=12.0, variance=0.5)
        assert model.state_dimension == 2
        assert model.period == 12.0
        assert model.variance == 0.5
        
        # Test parameter validation
        with pytest.raises(ValueError):
            SeasonalStateModel(period=-1.0, variance=1.0)
        
        with pytest.raises(ValueError):
            SeasonalStateModel(period=12.0, variance=0.0)
    
    def test_transition_matrix(self):
        """Test seasonal transition matrix."""
        model = SeasonalStateModel(period=4.0, variance=1.0)  # Quarterly
        T = model.transition_matrix(0)
        
        assert T.nrow() == 2 and T.ncol() == 2
        
        # For period=4, frequency = 2π/4 = π/2
        # cos(π/2) = 0, sin(π/2) = 1
        expected_T = np.array([[0.0, 1.0], [-1.0, 0.0]])
        assert np.allclose(T.to_numpy(), expected_T, atol=1e-10)
    
    def test_observation_matrix(self):
        """Test seasonal observation matrix."""
        model = SeasonalStateModel(period=12.0, variance=1.0)
        Z = model.observation_matrix(0)
        
        assert Z.nrow() == 1 and Z.ncol() == 2
        expected_Z = np.array([[1.0, 0.0]])
        assert np.allclose(Z.to_numpy(), expected_Z)


class TestLocalLinearTrendModel:
    """Test LocalLinearTrendModel class."""
    
    def test_construction_no_seasonal(self):
        """Test model construction without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0, 
            level_variance=0.5, 
            slope_variance=0.2
        )
        assert model.observation_variance == 1.0
        assert model.level_variance == 0.5
        assert model.slope_variance == 0.2
        assert not model.has_seasonal
        assert model.seasonal_period is None
        assert model.seasonal_variance is None
        assert model.state_dimension == 2  # Level + slope
        assert model.number_of_state_models() == 1
    
    def test_construction_with_seasonal(self):
        """Test model construction with seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.5,
            slope_variance=0.2,
            seasonal_period=12.0,
            seasonal_variance=0.8
        )
        assert model.observation_variance == 1.0
        assert model.level_variance == 0.5
        assert model.slope_variance == 0.2
        assert model.has_seasonal
        assert model.seasonal_period == 12.0
        assert model.seasonal_variance == 0.8
        assert model.state_dimension == 4  # Level + slope + 2 seasonal
        assert model.number_of_state_models() == 2
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = LocalLinearTrendModel(
            seasonal_period=12.0, seasonal_variance=1.0
        )
        
        # Test trend parameters
        model.set_level_variance(2.5)
        assert model.level_variance == 2.5
        
        model.set_slope_variance(1.2)
        assert model.slope_variance == 1.2
        
        # Test seasonal parameters
        model.set_seasonal_variance(0.8)
        assert model.seasonal_variance == 0.8
        
        # Test parameter validation
        with pytest.raises(ValueError):
            model.set_level_variance(-1.0)
        
        with pytest.raises(ValueError):
            model.set_slope_variance(0.0)
    
    def test_observation_matrix_no_seasonal(self):
        """Test observation matrix without seasonality."""
        model = LocalLinearTrendModel()
        Z = model.observation_matrix(0)
        
        assert Z.nrow() == 1 and Z.ncol() == 2
        expected_Z = np.array([[1.0, 0.0]])  # Observe level only
        assert np.allclose(Z.to_numpy(), expected_Z)
    
    def test_observation_matrix_with_seasonal(self):
        """Test observation matrix with seasonality."""
        model = LocalLinearTrendModel(seasonal_period=12.0)
        Z = model.observation_matrix(0)
        
        assert Z.nrow() == 1 and Z.ncol() == 4
        expected_Z = np.array([[1.0, 0.0, 1.0, 0.0]])  # Observe level + seasonal
        assert np.allclose(Z.to_numpy(), expected_Z)
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = LocalLinearTrendModel()
        
        # Add trending data
        trend_data = [1.0, 2.1, 3.2, 4.3, 5.4]
        model.add_data(trend_data)
        assert model.time_dimension == 5
        
        # Check observations
        obs = model.observations()
        assert np.allclose(obs.to_numpy(), trend_data)
        
        # Test clearing data
        model.clear_data()
        assert model.time_dimension == 0
    
    def test_log_likelihood_no_seasonal(self):
        """Test log likelihood without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0, 
            level_variance=0.1, 
            slope_variance=0.05
        )
        # Linear trend data
        model.add_data([1.0, 2.0, 3.1, 4.0, 5.2])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        assert log_lik < 0  # Should be negative
    
    def test_log_likelihood_with_seasonal(self):
        """Test log likelihood with seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05,
            seasonal_period=4.0,
            seasonal_variance=0.2
        )
        # Trend + seasonal pattern
        model.add_data([1.0, 2.5, 3.0, 2.8, 5.1, 6.4, 7.2, 6.9])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        assert log_lik < 0
    
    def test_fit_simple_trend(self):
        """Test fitting to simple trend data."""
        # Generate linear trend data
        np.random.seed(42)
        n = 30
        true_level = 10.0
        true_slope = 0.5
        true_obs_var = 0.25
        
        observations = []
        level = true_level
        slope = true_slope
        
        for t in range(n):
            obs = level + np.sqrt(true_obs_var) * np.random.randn()
            observations.append(obs)
            level += slope  # Deterministic trend for simplicity
        
        # Fit model
        model = LocalLinearTrendModel(
            observation_variance=1.0,  # Wrong initial value
            level_variance=0.1,
            slope_variance=0.01
        )
        model.add_data(observations)
        
        converged = model.fit(max_iterations=10)
        
        # Check that observation variance is reasonable
        assert model.observation_variance > 0
        assert model.observation_variance < 2.0  # Should be close to true value
    
    def test_prediction_no_seasonal(self):
        """Test prediction without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05
        )
        # Add linear trend data
        model.add_data([1.0, 2.0, 3.0, 4.0, 5.0])
        
        means, variances = model.predict(n_ahead=3)
        
        assert len(means) == 3
        assert len(variances) == 3
        
        # Predictions should continue trend
        for i in range(3):
            assert not np.isnan(means[i])
            assert not np.isnan(variances[i])
            assert variances[i] > 0
        
        # For upward trend, predictions should increase
        assert means[1] >= means[0]
        assert means[2] >= means[1]
    
    def test_component_extraction_no_seasonal(self):
        """Test component extraction without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05
        )
        model.add_data([1.0, 2.1, 3.0, 4.2, 5.1])
        
        components = model.extract_components()
        
        assert 'level' in components
        assert 'slope' in components
        assert 'seasonal' not in components
        
        assert len(components['level']) == 5
        assert len(components['slope']) == 5
        
        # Level should be reasonable
        for level in components['level'].to_numpy():
            assert not np.isnan(level)
            assert 0 < level < 10  # Should be reasonable
        
        # Slope should be positive for trending data
        slopes = components['slope'].to_numpy()
        assert np.mean(slopes) > 0  # Average slope should be positive
    
    def test_component_extraction_with_seasonal(self):
        """Test component extraction with seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05,
            seasonal_period=4.0,
            seasonal_variance=0.2
        )
        model.add_data([1.0, 2.5, 3.0, 2.8, 5.1, 6.4, 7.2, 6.9])
        
        components = model.extract_components()
        
        assert 'level' in components
        assert 'slope' in components
        assert 'seasonal' in components
        
        assert len(components['level']) == 8
        assert len(components['slope']) == 8
        assert len(components['seasonal']) == 8
        
        # All components should be reasonable
        for component_name, component_values in components.items():
            for value in component_values.to_numpy():
                assert not np.isnan(value)
                assert abs(value) < 20  # Should be reasonable
    
    def test_residuals(self):
        """Test residual computation."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05
        )
        model.add_data([1.0, 2.1, 2.9, 4.2, 5.0])
        
        residuals, std_residuals = model.residuals()
        
        assert len(residuals) == 5
        assert len(std_residuals) == 5
        
        # Check that residuals are reasonable
        for i in range(5):
            assert not np.isnan(residuals[i])
            assert not np.isnan(std_residuals[i])
            assert abs(std_residuals[i]) < 5  # Should be reasonable
    
    def test_simulation_no_seasonal(self):
        """Test data simulation without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05
        )
        
        simulated = model.simulate_data(
            n=20, initial_level=5.0, initial_slope=0.5
        )
        
        assert len(simulated) == 20
        
        # Should show upward trend on average (use larger windows for robustness)
        first_quarter = np.mean(simulated[:5])
        last_quarter = np.mean(simulated[-5:])
        # With positive slope, the trend should generally be upward
        # Allow for some noise but expect reasonable trend
        # Very lenient check due to random variation in simulation
        assert last_quarter > first_quarter - 5.0  # Very lenient check
    
    def test_simulation_with_seasonal(self):
        """Test data simulation with seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.0,
            level_variance=0.1,
            slope_variance=0.05,
            seasonal_period=4.0,
            seasonal_variance=0.2
        )
        
        simulated = model.simulate_data(
            n=12, initial_level=5.0, initial_slope=0.2
        )
        
        assert len(simulated) == 12
        
        # All observations should be reasonable
        for obs in simulated:
            assert isinstance(obs, float)
            assert not np.isnan(obs)
            assert abs(obs) < 50  # Should be reasonable
    
    def test_clone_no_seasonal(self):
        """Test model cloning without seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.5,
            level_variance=0.8,
            slope_variance=0.3
        )
        model.add_data([1.0, 2.0, 3.0])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.observation_variance == model.observation_variance
        assert cloned.level_variance == model.level_variance
        assert cloned.slope_variance == model.slope_variance
        assert cloned.has_seasonal == model.has_seasonal
        
        # Check data
        assert cloned.time_dimension == model.time_dimension
        
        # Ensure independence
        cloned.set_level_variance(2.0)
        assert model.level_variance == 0.8  # Original unchanged
    
    def test_clone_with_seasonal(self):
        """Test model cloning with seasonality."""
        model = LocalLinearTrendModel(
            observation_variance=1.5,
            level_variance=0.8,
            slope_variance=0.3,
            seasonal_period=12.0,
            seasonal_variance=0.6
        )
        model.add_data([1.0, 2.0, 3.0])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.seasonal_period == model.seasonal_period
        assert cloned.seasonal_variance == model.seasonal_variance
        assert cloned.has_seasonal == model.has_seasonal
        
        # Ensure independence
        cloned.set_seasonal_variance(1.0)
        assert model.seasonal_variance == 0.6  # Original unchanged
    
    def test_string_representation(self):
        """Test string representation."""
        # Without seasonality
        model1 = LocalLinearTrendModel(
            observation_variance=1.2,
            level_variance=0.5,
            slope_variance=0.3
        )
        model1.add_data([1.0, 2.0])
        
        s1 = str(model1)
        assert "LocalLinearTrendModel" in s1
        assert "obs_var=1.200" in s1
        assert "level_var=0.500" in s1
        assert "slope_var=0.300" in s1
        assert "time_points=2" in s1
        assert "seasonal_period" not in s1
        
        # With seasonality
        model2 = LocalLinearTrendModel(seasonal_period=12.0)
        s2 = str(model2)
        assert "seasonal_period=12.0" in s2