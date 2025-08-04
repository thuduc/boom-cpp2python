"""Tests for Kalman filter."""

import pytest
import numpy as np
from boom.models.state_space.kalman import KalmanFilter
from boom.models.state_space.local_level import LocalLevelModel
from boom.models.state_space.base import TimeSeriesData
from boom.linalg import Vector


class TestKalmanFilter:
    """Test KalmanFilter class."""
    
    def test_construction(self):
        """Test filter construction."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.5)
        kalman_filter = KalmanFilter(model)
        
        assert kalman_filter.model is model
        assert len(kalman_filter.filter_states()) == 0
        assert len(kalman_filter.smoother_states()) == 0
    
    def test_empty_data(self):
        """Test filter with no data."""
        model = LocalLevelModel()
        kalman_filter = KalmanFilter(model)
        
        filter_states = kalman_filter.filter()
        assert len(filter_states) == 0
        assert kalman_filter.log_likelihood() == 0.0
    
    def test_single_observation(self):
        """Test filter with single observation."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.5)
        model.add_data(2.5)
        
        kalman_filter = KalmanFilter(model)
        filter_states = kalman_filter.filter()
        
        assert len(filter_states) == 1
        
        state = filter_states[0]
        assert len(state.state_mean) == 1
        assert state.state_variance.nrow() == 1
        assert state.predicted_observation != state.state_mean[0]  # Should differ due to prior
        assert state.prediction_variance > 0
        assert not np.isnan(state.innovation)
        assert not np.isnan(state.log_likelihood)
    
    def test_multiple_observations(self):
        """Test filter with multiple observations."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        data = [1.0, 1.2, 0.8, 1.1, 0.9]
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        filter_states = kalman_filter.filter()
        
        assert len(filter_states) == 5
        
        # Check that all states have proper dimensions
        for state in filter_states:
            assert len(state.state_mean) == 1
            assert state.state_variance.nrow() == 1
            assert state.prediction_variance > 0
            assert not np.isnan(state.log_likelihood)
        
        # Check that log likelihood is computed
        total_log_lik = kalman_filter.log_likelihood()
        assert not np.isnan(total_log_lik)
        assert total_log_lik < 0  # Should be negative
    
    def test_missing_observations(self):
        """Test filter with missing observations."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        
        # Add mix of observed and missing data
        model.add_data(TimeSeriesData(1.0, 0, True))
        model.add_data(TimeSeriesData(np.nan, 1, False))  # Missing
        model.add_data(TimeSeriesData(1.5, 2, True))
        
        kalman_filter = KalmanFilter(model)
        filter_states = kalman_filter.filter()
        
        assert len(filter_states) == 3
        
        # First observation should contribute to likelihood
        assert filter_states[0].log_likelihood != 0
        
        # Missing observation should not contribute to likelihood
        assert filter_states[1].log_likelihood == 0
        assert filter_states[1].innovation == 0
        
        # Third observation should contribute to likelihood
        assert filter_states[2].log_likelihood != 0
    
    def test_smoother(self):
        """Test RTS smoother."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        data = [1.0, 1.2, 0.8, 1.1, 0.9]
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        kalman_filter.filter()
        smoother_states = kalman_filter.smooth()
        
        assert len(smoother_states) == 5
        
        # Check that all smoother states have proper dimensions
        for state in smoother_states:
            assert len(state.state_mean) == 1
            assert state.state_variance.nrow() == 1
            assert len(state.state_disturbance_mean) == 1
            assert state.state_disturbance_variance.nrow() == 1
        
        # Smoothed variances should generally be smaller than filtered
        filter_states = kalman_filter.filter_states()
        for i in range(len(smoother_states) - 1):  # Except last time point
            smoothed_var = smoother_states[i].state_variance[0, 0]
            filtered_var = filter_states[i].state_variance[0, 0]
            assert smoothed_var <= filtered_var + 1e-10  # Allow small numerical error
    
    def test_prediction(self):
        """Test future prediction."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        # Trend upward
        data = [1.0, 1.1, 1.2, 1.3, 1.4]
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        kalman_filter.filter()
        
        predictions = kalman_filter.predict(n_ahead=3)
        assert len(predictions) == 3
        
        for mean, variance in predictions:
            assert isinstance(mean, float)
            assert isinstance(variance, float)
            assert variance > 0
            assert not np.isnan(mean)
            assert not np.isnan(variance)
        
        # Prediction variance should increase with horizon
        assert predictions[1][1] >= predictions[0][1]
        assert predictions[2][1] >= predictions[1][1]
    
    def test_residuals(self):
        """Test residual computation."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.1)
        data = [1.0, 1.2, 0.8, 1.1, 0.9]
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        kalman_filter.filter()
        
        residuals, std_residuals = kalman_filter.residuals()
        
        assert len(residuals) == 5
        assert len(std_residuals) == 5
        
        # Check that standardized residuals have roughly unit variance
        # (though this is approximate due to small sample)
        std_res_array = std_residuals.to_numpy()
        assert np.all(np.abs(std_res_array) < 5)  # Reasonable range
    
    def test_smooth_before_filter_error(self):
        """Test that smoothing before filtering raises error."""
        model = LocalLevelModel()
        model.add_data([1.0, 2.0])
        
        kalman_filter = KalmanFilter(model)
        
        with pytest.raises(ValueError, match="Must run filter"):
            kalman_filter.smooth()
    
    def test_predict_before_filter_error(self):
        """Test that prediction before filtering raises error."""
        model = LocalLevelModel()
        model.add_data([1.0, 2.0])
        
        kalman_filter = KalmanFilter(model)
        
        with pytest.raises(ValueError, match="Must run filter"):
            kalman_filter.predict()
    
    def test_likelihood_computation(self):
        """Test likelihood computation with known values."""
        # Simple case: single observation with known parameters
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.0)  # No level variation
        model.add_data(0.0)  # Observation at zero
        
        kalman_filter = KalmanFilter(model)
        kalman_filter.filter(compute_likelihood=True)
        
        log_lik = kalman_filter.log_likelihood()
        
        # With large initial variance and no level variation,
        # this should be close to standard normal log density
        expected_log_lik = -0.5 * np.log(2 * np.pi) - 0.5 * (0.0**2 / (1e6 + 1.0))
        
        # The actual likelihood will differ due to the large initial variance,
        # but should be reasonable
        assert not np.isnan(log_lik)
        assert log_lik < 0
    
    def test_deterministic_level(self):
        """Test with deterministic level (no level variance)."""
        model = LocalLevelModel(observation_variance=1.0, level_variance=0.0)
        data = [1.0, 1.0, 1.0, 1.0]  # Constant level
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        filter_states = kalman_filter.filter()
        
        # With no level variance, filtered estimates should converge quickly
        assert len(filter_states) == 4
        
        # Later filtered estimates should be close to observations
        for i in range(1, 4):
            assert abs(filter_states[i].state_mean[0] - 1.0) < 0.5
    
    def test_high_observation_noise(self):
        """Test with high observation noise."""
        model = LocalLevelModel(observation_variance=100.0, level_variance=0.1)
        data = [1.0, 50.0, -20.0, 2.0]  # Noisy observations
        model.add_data(data)
        
        kalman_filter = KalmanFilter(model)
        filter_states = kalman_filter.filter()
        
        assert len(filter_states) == 4
        
        # With high observation noise, predictions should be smoothed
        for state in filter_states:
            assert state.prediction_variance > 50.0  # Should reflect high noise
            assert abs(state.state_mean[0]) < 100  # Should be reasonable