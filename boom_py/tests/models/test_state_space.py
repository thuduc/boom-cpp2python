"""Tests for state space models."""
import numpy as np
from boom.models.state_space import (
    StateSpaceModel, LocalLevelModel, LocalLinearTrendModel, 
    SeasonalModel, KalmanFilter, KalmanSmoother
)
from boom.linalg import Vector, Matrix
from boom.distributions import rng


class TestLocalLevelModel:
    """Test local level model."""
    
    def test_construction(self):
        """Test model construction."""
        model = LocalLevelModel(sigma=0.5)
        assert model.sigma == 0.5
        assert model.variance == 0.25
        assert model.state_dimension() == 1
    
    def test_matrices(self):
        """Test system matrices."""
        model = LocalLevelModel(sigma=0.1)
        
        # Transition matrix should be [1]
        T = model.transition_matrix(0)
        np.testing.assert_array_equal(T, [[1.0]])
        
        # Observation matrix should be [1]
        Z = model.observation_matrix(0)
        np.testing.assert_array_equal(Z, [[1.0]])
        
        # State variance should be [0.01]
        Q = model.state_variance(0)
        np.testing.assert_array_almost_equal(Q, [[0.01]])
    
    def test_parameter_setting(self):
        """Test parameter setting."""
        model = LocalLevelModel()
        
        model.sigma = 2.0
        assert model.sigma == 2.0
        assert model.variance == 4.0
        
        model.variance = 9.0
        assert model.variance == 9.0
        assert model.sigma == 3.0


class TestLocalLinearTrendModel:
    """Test local linear trend model."""
    
    def test_construction(self):
        """Test model construction."""
        model = LocalLinearTrendModel(level_sigma=0.1, slope_sigma=0.05)
        assert model.level_sigma == 0.1
        assert model.slope_sigma == 0.05
        assert model.state_dimension() == 2
    
    def test_matrices(self):
        """Test system matrices."""
        model = LocalLinearTrendModel(level_sigma=0.1, slope_sigma=0.05)
        
        # Transition matrix should be [[1, 1], [0, 1]]
        T = model.transition_matrix(0)
        expected_T = [[1.0, 1.0], [0.0, 1.0]]
        np.testing.assert_array_equal(T, expected_T)
        
        # Observation matrix should be [1, 0]
        Z = model.observation_matrix(0)
        np.testing.assert_array_equal(Z, [[1.0, 0.0]])
        
        # State variance should be diagonal
        Q = model.state_variance(0)
        expected_Q = [[0.01, 0.0], [0.0, 0.0025]]
        np.testing.assert_array_almost_equal(Q, expected_Q)
    
    def test_contributions(self):
        """Test contribution methods."""
        model = LocalLinearTrendModel()
        state = Vector([5.0, 2.0])  # level=5, slope=2
        
        assert model.level_contribution(state) == 5.0
        assert model.slope_contribution(state) == 2.0
        assert model.contribution(state) == 5.0  # Only level is observed


class TestSeasonalModel:
    """Test seasonal model."""
    
    def test_construction(self):
        """Test model construction."""
        model = SeasonalModel(period=4, sigma=0.1)
        assert model.period == 4
        assert model.sigma == 0.1
        assert model.n_harmonics == 2
        assert model.state_dimension() == 1  # 2*(2-1) + 1 for even period
    
    def test_monthly_seasonal(self):
        """Test monthly seasonal component."""
        model = SeasonalModel(period=12, sigma=0.1)
        assert model.period == 12
        assert model.n_harmonics == 6
        assert model.state_dimension() == 10  # 2*(6-1) = 10
    
    def test_matrices(self):
        """Test system matrices for quarterly seasonal."""
        model = SeasonalModel(period=4, sigma=0.1)
        
        # Check dimensions
        T = model.transition_matrix(0)
        assert T.shape == (1, 1)
        
        Z = model.observation_matrix(0)
        assert Z.shape == (1, 1)
        
        Q = model.state_variance(0)
        assert Q.shape == (1, 1)
        np.testing.assert_almost_equal(Q[0, 0], 0.01)
    
    def test_seasonal_pattern(self):
        """Test seasonal pattern generation."""
        model = SeasonalModel(period=4, sigma=0.1)
        pattern = model.seasonal_pattern(n_periods=2)
        
        # Should have 8 time points for 2 periods of length 4
        assert len(pattern) == 8
        
        # Pattern should repeat after 4 periods
        np.testing.assert_array_almost_equal(pattern[:4], pattern[4:])


class TestStateSpaceModel:
    """Test composite state space model."""
    
    def test_empty_model(self):
        """Test model with no components."""
        model = StateSpaceModel(observation_variance=0.25)
        assert model.observation_variance == 0.25
        assert model.state_dimension() == 0
        assert len(model.components) == 0
    
    def test_single_component(self):
        """Test model with single component."""
        model = StateSpaceModel()
        level = LocalLevelModel(sigma=0.1)
        model.add_component(level)
        
        assert model.state_dimension() == 1
        assert len(model.components) == 1
        
        # Test combined matrices
        T = model.transition_matrix(0)
        assert T.shape == (1, 1)
        np.testing.assert_array_equal(T, [[1.0]])
    
    def test_multiple_components(self):
        """Test model with multiple components."""
        model = StateSpaceModel()
        
        # Add level and seasonal components
        level = LocalLevelModel(sigma=0.1)
        seasonal = SeasonalModel(period=4, sigma=0.05)
        
        model.add_component(level)
        model.add_component(seasonal)
        
        # Total state dimension should be sum of components
        expected_dim = level.state_dimension() + seasonal.state_dimension()
        assert model.state_dimension() == expected_dim
        
        # Test combined matrices have correct dimensions
        T = model.transition_matrix(0)
        assert T.shape == (expected_dim, expected_dim)
        
        Z = model.observation_matrix(0)
        assert Z.shape == (1, expected_dim)
    
    def test_observations(self):
        """Test observation management."""
        model = StateSpaceModel()
        
        # Add observations
        y_data = [1.0, 2.0, 1.5, 2.5]
        model.set_observations(y_data)
        
        observed = model.get_observations()
        np.testing.assert_array_equal(observed, y_data)
        
        # Add individual observation
        model.add_observation(3.0)
        observed = model.get_observations()
        expected = [1.0, 2.0, 1.5, 2.5, 3.0]
        np.testing.assert_array_equal(observed, expected)
    
    def test_simulation(self):
        """Test state and observation simulation."""
        rng.seed(123)
        
        model = StateSpaceModel(observation_variance=0.1)
        level = LocalLevelModel(sigma=0.2)
        model.add_component(level)
        
        # Simulate state sequence
        n_time = 10
        states = model.simulate_state_sequence(n_time, rng)
        assert states.shape == (n_time, 1)
        
        # Simulate observations
        obs = model.simulate_observations(n_time, rng)
        assert len(obs) == n_time


class TestKalmanFilter:
    """Test Kalman filter."""
    
    def test_no_state_model(self):
        """Test Kalman filter with no state (pure noise)."""
        model = StateSpaceModel(observation_variance=1.0)
        kf = KalmanFilter(model)
        
        y = Vector([1.0, -0.5, 2.0])
        result = kf.filter(y)
        
        # For pure noise model, prediction errors should equal observations
        np.testing.assert_array_almost_equal(result.prediction_errors, y)
        
        # Log likelihood should be calculable
        assert np.isfinite(result.log_likelihood)
    
    def test_local_level_filter(self):
        """Test Kalman filter with local level model."""
        model = StateSpaceModel(observation_variance=0.1)
        level = LocalLevelModel(sigma=0.2)
        model.add_component(level)
        
        kf = KalmanFilter(model)
        
        # Simple trending data
        y = Vector([1.0, 1.1, 1.2, 1.3, 1.4])
        result = kf.filter(y)
        
        # Check output dimensions
        assert result.filtered_states.shape == (5, 1)
        assert len(result.filtered_variances) == 5
        assert len(result.prediction_errors) == 5
        assert np.isfinite(result.log_likelihood)
        
        # Filtered states should track the trend
        filtered_levels = result.filtered_states[:, 0]
        assert filtered_levels[0] < filtered_levels[-1]  # Should be increasing
    
    def test_trend_model_filter(self):
        """Test Kalman filter with trend model."""
        model = StateSpaceModel(observation_variance=0.01)
        trend = LocalLinearTrendModel(level_sigma=0.1, slope_sigma=0.01)
        model.add_component(trend)
        
        kf = KalmanFilter(model)
        
        # Linear trend data
        y = Vector([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kf.filter(y)
        
        # Check dimensions
        assert result.filtered_states.shape == (5, 2)
        
        # Level should be close to observations
        filtered_levels = result.filtered_states[:, 0]
        np.testing.assert_allclose(filtered_levels, y, rtol=0.1)


class TestKalmanSmoother:
    """Test Kalman smoother."""
    
    def test_local_level_smoother(self):
        """Test Kalman smoother with local level model."""
        model = StateSpaceModel(observation_variance=0.1)
        level = LocalLevelModel(sigma=0.2)
        model.add_component(level)
        
        ks = KalmanSmoother(model)
        
        y = Vector([1.0, 1.5, 1.2, 1.8, 1.6])
        filter_result, smooth_result = ks.smooth(y)
        
        # Check dimensions
        assert smooth_result.smoothed_states.shape == (5, 1)
        assert len(smooth_result.smoothed_variances) == 5
        
        # Smoothed estimates should be different from filtered
        smoothed_levels = smooth_result.smoothed_states[:, 0]
        filtered_levels = filter_result.filtered_states[:, 0]
        
        # At least some differences (not exact due to smoothing)
        assert not np.allclose(smoothed_levels, filtered_levels, atol=1e-10)


class TestIntegratedModel:
    """Test complete state space model workflow."""
    
    def test_trend_seasonal_model(self):
        """Test model with trend and seasonal components."""
        rng.seed(456)
        
        # Create model with trend and seasonal components
        model = StateSpaceModel(observation_variance=0.1)
        
        trend = LocalLinearTrendModel(level_sigma=0.1, slope_sigma=0.01)
        seasonal = SeasonalModel(period=4, sigma=0.05)
        
        model.add_component(trend)
        model.add_component(seasonal)
        
        # Generate some synthetic seasonal data
        n_time = 20
        true_data = []
        for t in range(n_time):
            level = 10.0 + 0.1 * t  # Linear trend
            seasonal_effect = 2.0 * np.sin(2 * np.pi * t / 4)  # Quarterly pattern
            true_data.append(level + seasonal_effect)
        
        # Add noise
        y_obs = Vector([val + rng.rnorm(0, 0.3) for val in true_data])
        model.set_observations(y_obs)
        
        # Test likelihood computation
        log_lik = model.loglike()
        assert np.isfinite(log_lik)
        
        # Test filtering and smoothing
        kf = KalmanFilter(model)
        ks = KalmanSmoother(model)
        
        filter_result = kf.filter(y_obs)
        filter_result_2, smooth_result = ks.smooth(y_obs)
        
        # Check that results are reasonable
        assert filter_result.filtered_states.shape == (n_time, 3)  # 2 trend + 1 seasonal
        assert smooth_result.smoothed_states.shape == (n_time, 3)
        
        # Likelihood from filter should match model likelihood
        np.testing.assert_almost_equal(filter_result.log_likelihood, log_lik)


if __name__ == "__main__":
    # Run basic tests manually
    print("Testing LocalLevelModel...")
    test_ll = TestLocalLevelModel()
    test_ll.test_construction()
    test_ll.test_matrices()
    test_ll.test_parameter_setting()
    print("LocalLevelModel tests passed!")
    
    print("Testing LocalLinearTrendModel...")
    test_trend = TestLocalLinearTrendModel()
    test_trend.test_construction()
    test_trend.test_matrices()
    test_trend.test_contributions()
    print("LocalLinearTrendModel tests passed!")
    
    print("Testing SeasonalModel...")
    test_seasonal = TestSeasonalModel()
    test_seasonal.test_construction()
    test_seasonal.test_monthly_seasonal()
    test_seasonal.test_matrices()
    test_seasonal.test_seasonal_pattern()
    print("SeasonalModel tests passed!")
    
    print("Testing StateSpaceModel...")
    test_ss = TestStateSpaceModel()
    test_ss.test_empty_model()
    test_ss.test_single_component()
    test_ss.test_multiple_components()
    test_ss.test_observations()
    test_ss.test_simulation()
    print("StateSpaceModel tests passed!")
    
    print("Testing KalmanFilter...")
    test_kf = TestKalmanFilter()
    test_kf.test_no_state_model()
    test_kf.test_local_level_filter()
    test_kf.test_trend_model_filter()
    print("KalmanFilter tests passed!")
    
    print("Testing KalmanSmoother...")
    test_ks = TestKalmanSmoother()
    test_ks.test_local_level_smoother()
    print("KalmanSmoother tests passed!")
    
    print("Testing integrated model...")
    test_int = TestIntegratedModel()
    test_int.test_trend_seasonal_model()
    print("Integrated model tests passed!")
    
    print("All state space model tests passed successfully!")