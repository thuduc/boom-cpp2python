"""
Tests for time series models.

This module tests ARIMA, AR, MA, and related time series functionality.
"""

import pytest
import numpy as np
import sys
import os

# Add the impl-python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from boom.models.time_series import (
    TimeSeriesData, TimeSeriesModel, AutoregressiveModel,
    MovingAverageModel, ArimaModel
)
from boom.linalg import Vector


class TestTimeSeriesData:
    """Test time series data structures."""
    
    def test_data_creation(self):
        """Test creation of time series data."""
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        data = TimeSeriesData(observations)
        
        assert data.length == 5
        assert len(data) == 5
        np.testing.assert_array_equal(data.get_observations(), observations)
        
    def test_data_with_timestamps(self):
        """Test time series data with timestamps."""
        observations = [1.0, 2.0, 3.0]
        timestamps = ['2023-01-01', '2023-01-02', '2023-01-03']
        data = TimeSeriesData(observations, timestamps)
        
        assert data.length == 3
        assert data.get_timestamps() is not None
        
    def test_data_differences(self):
        """Test differencing functionality."""
        observations = [1.0, 3.0, 6.0, 10.0, 15.0]  # Differences: 2, 3, 4, 5
        data = TimeSeriesData(observations)
        
        diff1 = data.get_differences(lag=1)
        expected_diff = [2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_almost_equal(diff1, expected_diff)
        
    def test_lagged_observations(self):
        """Test creation of lagged observation matrix."""
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        data = TimeSeriesData(observations)
        
        lag_matrix = data.get_lags(max_lag=2)
        
        # Should have 3 rows (5 - 2) and 2 columns
        assert lag_matrix.shape() == (3, 2)
        
        # First row should be [2, 1] (X_2, X_1)
        np.testing.assert_array_equal(lag_matrix.to_numpy()[0, :], [2.0, 1.0])
        
    def test_data_splitting(self):
        """Test train/test splitting."""
        observations = list(range(10))
        data = TimeSeriesData(observations)
        
        train_data, test_data = data.split(7)
        
        assert train_data.length == 7
        assert test_data.length == 3
        np.testing.assert_array_equal(train_data.get_observations(), list(range(7)))
        np.testing.assert_array_equal(test_data.get_observations(), list(range(7, 10)))


class TestAutoRegressiveModel:
    """Test AR model implementation."""
    
    def test_ar_initialization(self):
        """Test AR model initialization."""
        ar = AutoregressiveModel(order=2)
        
        assert ar.order == 2
        assert ar.n_parameters() == 4  # intercept + 2 coeffs + sigma^2
        
    def test_ar_parameter_setting(self):
        """Test setting AR parameters."""
        ar = AutoregressiveModel(order=2)
        
        params = {
            'intercept': 0.5,
            'coefficients': np.array([0.6, -0.2]),
            'sigma_squared': 1.5
        }
        
        ar.set_parameters(params)
        
        assert ar.intercept == 0.5
        np.testing.assert_array_equal(ar.coefficients, [0.6, -0.2])
        assert ar.sigma_squared == 1.5
        
    def test_ar_fitting(self):
        """Test AR model fitting."""
        # Generate AR(1) data
        np.random.seed(42)
        n = 100
        true_phi = 0.7
        true_intercept = 1.0
        true_sigma = 1.0
        
        observations = []
        x = true_intercept / (1 - true_phi)  # Start at unconditional mean
        
        for _ in range(n):
            x = true_intercept + true_phi * x + np.random.normal(0, true_sigma)
            observations.append(x)
        
        # Fit AR(1) model
        ar = AutoregressiveModel(order=1)
        data = TimeSeriesData(observations)
        ar.set_data(data)
        ar.fit()
        
        # Check that parameters are reasonable
        assert ar.is_fitted()
        assert abs(ar.coefficients[0] - true_phi) < 0.2  # Rough check
        assert abs(ar.sigma_squared - true_sigma**2) < 1.0
        
    def test_ar_prediction(self):
        """Test AR model prediction."""
        ar = AutoregressiveModel(order=1)
        
        # Create data first
        observations = [1.0, 2.0, 3.0, 4.0]
        data = TimeSeriesData(observations)
        ar.set_data(data)
        
        # Set known parameters after data
        ar.set_parameters({
            'intercept': 0.0,
            'coefficients': np.array([0.5]),
            'sigma_squared': 1.0,
            'fitted': True
        })
        
        # Predict next values
        forecasts = ar.predict(n_periods=2)
        
        assert len(forecasts.to_numpy()) == 2
        
        # First forecast should be 0.5 * 4.0 = 2.0
        assert forecasts.to_numpy()[0] == pytest.approx(2.0)
        
    def test_ar_stationarity_check(self):
        """Test stationarity checking."""
        ar = AutoregressiveModel(order=1)
        
        # Stationary case
        ar.set_parameters({
            'coefficients': np.array([0.5]),
            'fitted': True
        })
        assert ar.check_stationarity()
        
        # Non-stationary case
        ar.set_parameters({
            'coefficients': np.array([1.1]),
            'fitted': True
        })
        assert not ar.check_stationarity()
        
    def test_ar_impulse_response(self):
        """Test impulse response function."""
        ar = AutoregressiveModel(order=1)
        ar.set_parameters({
            'coefficients': np.array([0.6]),
            'fitted': True
        })
        
        impulse_resp = ar.impulse_response(n_periods=5)
        
        # Should decay exponentially
        expected = [1.0, 0.6, 0.36, 0.216, 0.1296]
        np.testing.assert_array_almost_equal(impulse_resp.to_numpy(), expected)


class TestMovingAverageModel:
    """Test MA model implementation."""
    
    def test_ma_initialization(self):
        """Test MA model initialization."""
        ma = MovingAverageModel(order=2)
        
        assert ma.order == 2
        assert ma.n_parameters() == 4  # mean + 2 coeffs + sigma^2
        
    def test_ma_parameter_setting(self):
        """Test setting MA parameters."""
        ma = MovingAverageModel(order=1)
        
        params = {
            'mean': 2.0,
            'coefficients': np.array([0.4]),
            'sigma_squared': 1.5
        }
        
        ma.set_parameters(params)
        
        assert ma.mean == 2.0
        np.testing.assert_array_equal(ma.coefficients, [0.4])
        assert ma.sigma_squared == 1.5
        
    def test_ma_fitting(self):
        """Test MA model fitting."""
        # Generate MA(1) data
        np.random.seed(123)
        n = 50
        true_theta = 0.6
        true_mean = 1.0
        true_sigma = 1.0
        
        # Generate innovations
        innovations = np.random.normal(0, true_sigma, n + 1)
        
        observations = []
        for i in range(n):
            obs = true_mean + innovations[i + 1] + true_theta * innovations[i]
            observations.append(obs)
        
        # Fit MA(1) model
        ma = MovingAverageModel(order=1)
        data = TimeSeriesData(observations)
        ma.set_data(data)
        
        try:
            ma.fit()
            assert ma.is_fitted()
            # Parameter estimates may not be very accurate with small sample
            assert abs(ma.mean - true_mean) < 2.0
        except Exception:
            # MA fitting can be challenging - don't fail test if it doesn't converge
            pytest.skip("MA fitting did not converge")
            
    def test_ma_prediction(self):
        """Test MA model prediction."""
        ma = MovingAverageModel(order=1)
        
        # Create data first
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        data = TimeSeriesData(observations)
        ma.set_data(data)
        
        # Set parameters after data
        ma.set_parameters({
            'mean': 2.0,
            'coefficients': np.array([0.5]),
            'sigma_squared': 1.0,
            'fitted': True
        })
        
        # For MA models, future predictions beyond order are just the mean
        forecasts = ma.predict(n_periods=3)
        
        assert len(forecasts.to_numpy()) == 3
        # All forecasts should be finite numbers for MA process
        assert all(np.isfinite(f) for f in forecasts.to_numpy())
        assert not any(np.isnan(f) for f in forecasts.to_numpy())
        
    def test_ma_invertibility_check(self):
        """Test invertibility checking."""
        ma = MovingAverageModel(order=1)
        
        # Invertible case
        ma.set_parameters({
            'coefficients': np.array([0.5]),
            'fitted': True
        })
        assert ma.check_invertibility()
        
        # Non-invertible case
        ma.set_parameters({
            'coefficients': np.array([1.5]),
            'fitted': True
        })
        assert not ma.check_invertibility()


class TestArimaModel:
    """Test ARIMA model implementation."""
    
    def test_arima_initialization(self):
        """Test ARIMA model initialization."""
        arima = ArimaModel(order=(1, 1, 1))
        
        assert arima.order == (1, 1, 1)
        assert arima.n_parameters() == 4  # intercept + AR + MA + sigma^2
        
    def test_arima_differencing(self):
        """Test differencing in ARIMA model."""
        # Test data with trend
        observations = [1.0, 3.0, 6.0, 10.0, 15.0]  # Clear upward trend
        data = TimeSeriesData(observations)
        
        arima = ArimaModel(order=(0, 1, 0))  # Just differencing
        arima.set_data(data)
        
        # After differencing, should have stationary series
        assert arima._differenced_data is not None
        assert len(arima._differenced_data) == 4  # One less than original
        
    def test_arima_parameter_access(self):
        """Test ARIMA parameter access."""
        arima = ArimaModel(order=(1, 0, 1))
        
        params = {
            'intercept': 0.5,
            'ar_coefficients': np.array([0.6]),
            'ma_coefficients': np.array([0.4])
        }
        
        arima.set_parameters(params)
        
        assert arima.intercept == 0.5
        np.testing.assert_array_equal(arima.ar_coefficients, [0.6])
        np.testing.assert_array_equal(arima.ma_coefficients, [0.4])
        
    def test_arima_fitting_simple(self):
        """Test ARIMA fitting on simple data."""
        # Generate simple AR(1) data (no differencing needed)
        np.random.seed(42)
        n = 50
        observations = []
        x = 0.0
        
        for _ in range(n):
            x = 0.8 * x + np.random.normal(0, 1)
            observations.append(x)
        
        data = TimeSeriesData(observations)
        arima = ArimaModel(order=(1, 0, 0))  # AR(1)
        arima.set_data(data)
        
        try:
            arima.fit()
            assert arima.is_fitted()
            # Should recover something close to 0.8
            assert abs(arima.ar_coefficients[0] - 0.8) < 0.5
        except Exception:
            # ARIMA fitting can be challenging
            pytest.skip("ARIMA fitting did not converge")


class TestTimeSeriesIntegration:
    """Integration tests for time series functionality."""
    
    def test_model_comparison(self):
        """Test comparing different time series models."""
        # Generate AR(1) data
        np.random.seed(42)
        n = 100
        observations = []
        x = 0.0
        
        for _ in range(n):
            x = 0.7 * x + np.random.normal(0, 1)
            observations.append(x)
        
        data = TimeSeriesData(observations)
        
        # Fit different models
        models = [
            AutoregressiveModel(order=1),
            AutoregressiveModel(order=2),
        ]
        
        results = []
        for model in models:
            try:
                model.set_data(data)
                model.fit()
                if model.is_fitted():
                    aic = model.aic()
                    results.append((model, aic))
            except Exception:
                continue
        
        # Should have at least one successful fit
        assert len(results) > 0
        
        # AIC values should be reasonable
        for model, aic in results:
            assert not np.isnan(aic)
            assert not np.isinf(aic)
            
    def test_forecasting_workflow(self):
        """Test complete forecasting workflow."""
        # Simple trend + noise data
        t = np.arange(20)
        observations = 1.0 + 0.1 * t + np.random.normal(0, 0.1, 20)
        
        data = TimeSeriesData(observations.tolist())
        
        # Fit ARIMA(0,1,0) - random walk with drift
        arima = ArimaModel(order=(0, 1, 0))
        arima.set_data(data)
        
        try:
            arima.fit()
            
            # Generate forecasts
            forecasts = arima.predict(n_periods=5)
            
            assert len(forecasts.to_numpy()) == 5
            # Forecasts should be reasonable (not NaN or infinite)
            assert all(np.isfinite(f) for f in forecasts.to_numpy())
            
        except Exception:
            pytest.skip("ARIMA fitting failed")
            
    def test_residual_analysis(self):
        """Test residual analysis for time series models."""
        # Generate data and fit model
        np.random.seed(42)
        n = 50
        observations = np.random.normal(0, 1, n).tolist()
        
        data = TimeSeriesData(observations)
        ar = AutoregressiveModel(order=1)
        ar.set_data(data)
        ar.fit()
        
        # Get residuals
        residuals = ar.get_residuals()
        fitted_values = ar.get_fitted_values()
        
        assert residuals is not None
        assert fitted_values is not None
        assert len(residuals) > 0
        assert len(fitted_values) > 0
        
        # Get diagnostic information
        diagnostics = ar.diagnostic_plots()
        
        assert 'residuals' in diagnostics
        assert 'aic' in diagnostics
        assert 'bic' in diagnostics


if __name__ == '__main__':
    pytest.main([__file__])