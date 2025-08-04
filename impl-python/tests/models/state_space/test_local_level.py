"""Tests for LocalLevelModel."""

import pytest
import numpy as np
from boom.models.state_space.local_level import LocalLevelModel, LocalLevelStateModel
from boom.models.state_space.base import TimeSeriesData
from boom.linalg import Vector, Matrix, SpdMatrix


class TestLocalLevelStateModel:
    """Test LocalLevelStateModel class."""
    
    def test_construction(self):
        """Test state model construction."""
        model = LocalLevelStateModel(level_variance=2.5)
        assert model.state_dimension == 1
        assert model.level_variance == 2.5
        
        # Test parameter validation
        with pytest.raises(ValueError):
            LocalLevelStateModel(level_variance=-1.0)
    
    def test_matrices(self):
        """Test transition and observation matrices."""
        model = LocalLevelStateModel(level_variance=1.5)
        
        # Transition matrix should be identity
        T = model.transition_matrix(0)
        assert T.nrow() == 1 and T.ncol() == 1
        assert T[0, 0] == 1.0
        
        # State error variance
        Q = model.state_error_variance(0)
        assert Q.nrow() == 1 and Q.ncol() == 1
        assert Q[0, 0] == 1.5
        
        # Observation matrix
        Z = model.observation_matrix(0)
        assert Z.nrow() == 1 and Z.ncol() == 1
        assert Z[0, 0] == 1.0
    
    def test_initial_conditions(self):
        """Test initial state conditions."""
        model = LocalLevelStateModel()
        
        initial_mean = model.initial_state_mean()
        assert len(initial_mean) == 1
        assert initial_mean[0] == 0.0
        
        initial_var = model.initial_state_variance()
        assert initial_var.nrow() == 1 and initial_var.ncol() == 1
        assert initial_var[0, 0] == 1e6


class TestLocalLevelModel:
    """Test LocalLevelModel class."""
    
    def test_construction(self):
        """Test model construction."""
        model = LocalLevelModel(observation_variance=1.5, level_variance=0.8)
        assert model.observation_variance == 1.5
        assert model.level_variance == 0.8
        assert model.state_dimension == 1
        assert model.time_dimension == 0
        assert model.number_of_state_models() == 1
        
        # Test parameter validation
        with pytest.raises(ValueError):
            LocalLevelModel(observation_variance=-1.0)
    
    def test_parameter_access(self):
        """Test parameter getters and setters."""
        model = LocalLevelModel()
        
        # Test observation variance
        model.set_observation_variance(2.5)
        assert model.observation_variance == 2.5
        
        # Test level variance
        model.set_level_variance(1.2)
        assert model.level_variance == 1.2
        
        # Test parameter validation
        with pytest.raises(ValueError):
            model.set_observation_variance(0.0)
        
        with pytest.raises(ValueError):
            model.set_level_variance(-1.0)
    
    def test_data_management(self):
        """Test adding and managing data."""
        model = LocalLevelModel()
        
        # Add data as floats
        model.add_data([1.0, 2.5, -0.8, 3.2])
        assert model.time_dimension == 4
        
        # Add data as TimeSeriesData
        ts_data = TimeSeriesData(1.8, timestamp=4)
        model.add_data(ts_data)
        assert model.time_dimension == 5
        
        # Check data retrieval
        data_0 = model.get_data(0)
        assert data_0.y() == 1.0
        assert data_0.timestamp() == 0
        
        # Test observations vector
        obs = model.observations()
        expected = [1.0, 2.5, -0.8, 3.2, 1.8]
        assert np.allclose(obs.to_numpy(), expected)
        
        # Test clearing data
        model.clear_data()
        assert model.time_dimension == 0
    
    def test_missing_data(self):
        """Test handling of missing observations."""
        model = LocalLevelModel()
        
        # Add mix of observed and missing data
        model.add_data(TimeSeriesData(1.0, 0, True))
        model.add_data(TimeSeriesData(np.nan, 1, False))
        model.add_data(TimeSeriesData(2.0, 2, True))
        
        assert model.time_dimension == 3
        
        # Check observed mask
        observed_mask = model.observed_mask()
        assert observed_mask == [True, False, True]
        
        # Check observations (including NaN)
        obs = model.observations()
        assert obs[0] == 1.0
        assert np.isnan(obs[1])
        assert obs[2] == 2.0
    
    def test_log_likelihood_empty(self):
        """Test log likelihood with no data."""
        model = LocalLevelModel()
        assert model.log_likelihood() == 0.0
    
    def test_log_likelihood_with_data(self):
        """Test log likelihood computation."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        model.add_data([1.0, 1.1, 0.9, 1.2, 0.8])
        
        log_lik = model.log_likelihood()
        assert isinstance(log_lik, float)
        assert not np.isnan(log_lik)
        assert log_lik < 0  # Should be negative
    
    def test_fit_empty_data(self):
        """Test fitting with no data."""
        model = LocalLevelModel()
        converged = model.fit()
        assert converged is True  # Should succeed trivially
    
    def test_fit_with_data(self):
        """Test parameter estimation."""
        # Generate data from known model
        np.random.seed(42)
        true_obs_var = 1.0
        true_level_var = 0.25
        n = 50
        
        level = 0.0
        observations = []
        for t in range(n):
            obs = level + np.sqrt(true_obs_var) * np.random.randn()
            observations.append(obs)
            level += np.sqrt(true_level_var) * np.random.randn()
        
        # Fit model
        model = LocalLevelModel(observation_variance=2.0, level_variance=1.0)  # Wrong initial values
        model.add_data(observations)
        
        converged = model.fit(max_iterations=20)
        
        # Check that parameters are reasonable (allow some error)
        assert abs(model.observation_variance - true_obs_var) < 0.7
        assert abs(model.level_variance - true_level_var) < 0.7
    
    def test_prediction(self):
        """Test future prediction."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        # Add data with upward trend
        model.add_data([1.0, 1.1, 1.2, 1.3, 1.4])
        
        means, variances = model.predict(n_ahead=3)
        
        assert len(means) == 3
        assert len(variances) == 3
        
        # All predictions should be reasonable
        for i in range(3):
            assert not np.isnan(means[i])
            assert not np.isnan(variances[i])
            assert variances[i] > 0
        
        # Prediction variance should increase with horizon
        assert variances[1] >= variances[0]
        assert variances[2] >= variances[1]
    
    def test_extract_level(self):
        """Test level extraction."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        model.add_data([1.0, 1.2, 0.8, 1.1, 0.9])
        
        level_means, level_variances = model.extract_level()
        
        assert len(level_means) == 5
        assert len(level_variances) == 5
        
        # All estimates should be reasonable
        for i in range(5):
            assert not np.isnan(level_means[i])
            assert not np.isnan(level_variances[i])
            assert level_variances[i] > 0
            assert abs(level_means[i]) < 10  # Should be reasonable
    
    def test_residuals(self):
        """Test residual computation."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        model.add_data([1.0, 1.1, 0.9, 1.2, 0.8])
        
        residuals, std_residuals = model.residuals()
        
        assert len(residuals) == 5
        assert len(std_residuals) == 5
        
        # Check that residuals are reasonable
        for i in range(5):
            assert not np.isnan(residuals[i])
            assert not np.isnan(std_residuals[i])
            assert abs(std_residuals[i]) < 5  # Should be reasonable
    
    def test_simulation(self):
        """Test data simulation."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.25)
        
        simulated = model.simulate_data(n=10, initial_level=2.0)
        
        assert len(simulated) == 10
        
        # All observations should be reasonable
        for obs in simulated:
            assert isinstance(obs, float)
            assert not np.isnan(obs)
            assert abs(obs) < 20  # Should be reasonable given parameters
    
    def test_clone(self):
        """Test model cloning."""
        model = LocalLevelModel(observation_variance=1.5, level_variance=0.8)
        model.add_data([1.0, 2.0, 1.5])
        
        cloned = model.clone()
        
        # Check parameters
        assert cloned.observation_variance == model.observation_variance
        assert cloned.level_variance == model.level_variance
        
        # Check data
        assert cloned.time_dimension == model.time_dimension
        for i in range(model.time_dimension):
            assert cloned.get_data(i).y() == model.get_data(i).y()
        
        # Ensure independence
        cloned.set_observation_variance(2.0)
        assert model.observation_variance == 1.5  # Original unchanged
    
    def test_constant_level(self):
        """Test with constant level data."""
        model = LocalLevelModel(observation_variance=0.1, level_variance=0.0)  # No level variation
        constant_data = [5.0] * 10
        model.add_data(constant_data)
        
        # Should be able to compute likelihood
        log_lik = model.log_likelihood()
        assert not np.isnan(log_lik)
        
        # Level should be estimated close to constant value
        level_means, _ = model.extract_level()
        for level in level_means.to_numpy():
            assert abs(level - 5.0) < 1.0  # Should be close to true level
    
    def test_noisy_data(self):
        """Test with very noisy data."""
        model = LocalLevelModel(observation_variance=100.0, level_variance=0.1)
        # Very noisy observations
        noisy_data = [np.random.randn() * 10 for _ in range(5)]
        model.add_data(noisy_data)
        
        # Should still work
        log_lik = model.log_likelihood()
        assert not np.isnan(log_lik)
        
        # Predictions should have high variance
        means, variances = model.predict(n_ahead=1)
        assert variances[0] > 50.0  # Should reflect high noise
    
    def test_string_representation(self):
        """Test string representation."""
        model = LocalLevelModel(observation_variance=1.2, level_variance=0.5)
        model.add_data([1.0, 2.0])
        
        s = str(model)
        assert "LocalLevelModel" in s
        assert "obs_var=1.200" in s
        assert "level_var=0.500" in s
        assert "time_points=2" in s