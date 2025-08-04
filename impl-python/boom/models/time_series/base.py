"""
Base classes for time series models.

This module provides the foundational classes for all time series models
in the BOOM Python package.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple
from ...linalg import Vector, Matrix
from ..base import Model, Data


class TimeSeriesData(Data):
    """
    Data container for time series models.
    
    Stores time series observations with optional timestamps and metadata.
    """
    
    def __init__(self, observations: Union[List[float], np.ndarray, Vector],
                 timestamps: Optional[Union[List, np.ndarray]] = None):
        """
        Initialize time series data.
        
        Args:
            observations: Time series observations 
            timestamps: Optional timestamps for observations
        """
        if isinstance(observations, Vector):
            obs_array = observations.to_numpy()
        elif isinstance(observations, list):
            obs_array = np.array(observations)
        else:
            obs_array = np.asarray(observations)
        
        # Create data points
        data_points = []
        for i, value in enumerate(obs_array):
            timestamp = timestamps[i] if timestamps is not None else i
            data_points.append(TimeSeriesDataPoint(value, timestamp))
        
        super().__init__()
        self._data_points = data_points
        self._observations = obs_array
        self._timestamps = timestamps
    
    def clone(self) -> 'TimeSeriesData':
        """Create a deep copy of the time series data."""
        return TimeSeriesData(self._observations.copy(), 
                             self._timestamps.copy() if self._timestamps is not None else None)
    
    def n_observations(self) -> int:
        """Get number of observations."""
        return len(self._observations)
    
    def get_observations(self) -> np.ndarray:
        """Get observations as numpy array."""
        return self._observations.copy()
    
    def get_timestamps(self) -> Optional[np.ndarray]:
        """Get timestamps if available."""
        if self._timestamps is not None:
            return np.array(self._timestamps)
        return None
    
    def get_differences(self, lag: int = 1) -> np.ndarray:
        """
        Compute differenced time series.
        
        Args:
            lag: Lag for differencing
            
        Returns:
            Differenced series
        """
        if lag >= len(self._observations):
            raise ValueError(f"Lag {lag} too large for series of length {len(self._observations)}")
        
        return self._observations[lag:] - self._observations[:-lag]
    
    def get_lags(self, max_lag: int) -> Matrix:
        """
        Create lagged observations matrix.
        
        Args:
            max_lag: Maximum lag to include
            
        Returns:
            Matrix where column i contains observations lagged by i+1
        """
        n_obs = len(self._observations)
        if max_lag >= n_obs:
            raise ValueError(f"Max lag {max_lag} too large for series of length {n_obs}")
        
        n_valid = n_obs - max_lag
        lag_matrix = np.zeros((n_valid, max_lag))
        
        for i in range(max_lag):
            lag = i + 1
            # For lag=1, we want observations[max_lag-1:n_obs-1]
            # For lag=2, we want observations[max_lag-2:n_obs-2]
            for j in range(n_valid):
                lag_matrix[j, i] = self._observations[max_lag + j - lag]
        
        return Matrix(lag_matrix)
    
    def split(self, split_point: int) -> Tuple['TimeSeriesData', 'TimeSeriesData']:
        """
        Split time series into training and test sets.
        
        Args:
            split_point: Index where to split
            
        Returns:
            Tuple of (training_data, test_data)
        """
        if split_point <= 0 or split_point >= len(self._observations):
            raise ValueError(f"Split point {split_point} out of valid range")
        
        train_obs = self._observations[:split_point]
        test_obs = self._observations[split_point:]
        
        train_ts = None
        test_ts = None
        if self._timestamps is not None:
            train_ts = self._timestamps[:split_point]
            test_ts = self._timestamps[split_point:]
        
        return (TimeSeriesData(train_obs, train_ts),
                TimeSeriesData(test_obs, test_ts))
    
    @property
    def length(self) -> int:
        """Get length of time series."""
        return len(self._observations)
    
    def __len__(self) -> int:
        """Get length of time series."""
        return len(self._observations)


class TimeSeriesDataPoint(Data):
    """Single time series observation with timestamp."""
    
    def __init__(self, value: float, timestamp: Any = None):
        """
        Initialize time series data point.
        
        Args:
            value: Observation value
            timestamp: Optional timestamp
        """
        super().__init__()
        self._value = float(value)
        self._timestamp = timestamp
    
    def clone(self) -> 'TimeSeriesDataPoint':
        """Create a deep copy of the data point."""
        return TimeSeriesDataPoint(self._value, self._timestamp)
    
    def y(self) -> float:
        """Get observation value."""
        return self._value
    
    def timestamp(self) -> Any:
        """Get timestamp."""
        return self._timestamp


class TimeSeriesModel(Model, ABC):
    """
    Abstract base class for time series models.
    
    Provides common functionality for time series modeling including
    forecasting, residual analysis, and parameter estimation.
    """
    
    def __init__(self):
        """Initialize time series model."""
        super().__init__()
        self._data: Optional[TimeSeriesData] = None
        self._fitted = False
        self._residuals: Optional[np.ndarray] = None
        self._fitted_values: Optional[np.ndarray] = None
    
    def set_data(self, data: TimeSeriesData) -> None:
        """
        Set the time series data.
        
        Args:
            data: Time series data
        """
        self._data = data
        self._fitted = False
        self._residuals = None
        self._fitted_values = None
        self._notify_observers()
    
    def get_data(self) -> Optional[TimeSeriesData]:
        """Get the time series data."""
        return self._data
    
    @abstractmethod
    def fit(self) -> None:
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, n_periods: int = 1) -> Vector:
        """
        Generate forecasts.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Vector of forecasts
        """
        pass
    
    @abstractmethod
    def log_likelihood(self) -> float:
        """Compute log likelihood of current parameters."""
        pass
    
    def forecast(self, n_periods: int = 1, 
                 confidence_level: float = 0.95) -> Tuple[Vector, Vector, Vector]:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            n_periods: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (forecasts, lower_bounds, upper_bounds)
        """
        # Default implementation - subclasses should override for proper intervals
        forecasts = self.predict(n_periods)
        
        # Rough approximation using residual standard error
        if self._residuals is not None:
            residual_std = np.std(self._residuals)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # rough approximation
            margin = z_score * residual_std
            
            lower = Vector(forecasts.to_numpy() - margin)
            upper = Vector(forecasts.to_numpy() + margin)
        else:
            # No residuals available
            lower = forecasts
            upper = forecasts
        
        return forecasts, lower, upper
    
    def compute_residuals(self) -> np.ndarray:
        """Compute model residuals."""
        if not self._fitted or self._data is None:
            raise RuntimeError("Model must be fitted before computing residuals")
        
        if self._fitted_values is None:
            raise RuntimeError("Fitted values not available")
        
        observations = self._data.get_observations()
        if len(self._fitted_values) != len(observations):
            # Handle case where fitted values might be shorter (e.g., due to lags)
            n_fit = len(self._fitted_values)
            self._residuals = observations[-n_fit:] - self._fitted_values
        else:
            self._residuals = observations - self._fitted_values
        
        return self._residuals.copy()
    
    def get_residuals(self) -> Optional[np.ndarray]:
        """Get model residuals."""
        if self._residuals is None and self._fitted:
            self.compute_residuals()
        return self._residuals.copy() if self._residuals is not None else None
    
    def get_fitted_values(self) -> Optional[np.ndarray]:
        """Get fitted values."""
        return self._fitted_values.copy() if self._fitted_values is not None else None
    
    def aic(self) -> float:
        """Compute Akaike Information Criterion."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing AIC")
        
        log_lik = self.log_likelihood()
        n_params = self.n_parameters()
        
        return -2 * log_lik + 2 * n_params
    
    def bic(self) -> float:
        """Compute Bayesian Information Criterion."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing BIC")
        
        log_lik = self.log_likelihood()
        n_params = self.n_parameters()
        n_obs = len(self._data.get_observations()) if self._data else 0
        
        return -2 * log_lik + np.log(n_obs) * n_params
    
    @abstractmethod
    def n_parameters(self) -> int:
        """Get number of model parameters."""
        pass
    
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._fitted
    
    def diagnostic_plots(self) -> dict:
        """
        Generate diagnostic information for plotting.
        
        Returns:
            Dictionary with diagnostic data
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before generating diagnostics")
        
        residuals = self.get_residuals()
        fitted_values = self.get_fitted_values()
        
        diagnostics = {
            'residuals': residuals,
            'fitted_values': fitted_values,
            'aic': self.aic(),
            'bic': self.bic(),
            'log_likelihood': self.log_likelihood()
        }
        
        if residuals is not None:
            diagnostics.update({
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_min': np.min(residuals),
                'residual_max': np.max(residuals)
            })
        
        return diagnostics
    
    def simulate(self, n_periods: int, n_simulations: int = 1) -> Matrix:
        """
        Simulate future paths from the model.
        
        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            
        Returns:
            Matrix of simulations [n_periods x n_simulations]
        """
        # Default implementation - subclasses should override
        simulations = np.zeros((n_periods, n_simulations))
        
        for i in range(n_simulations):
            forecast = self.predict(n_periods)
            simulations[:, i] = forecast.to_numpy()
        
        return Matrix(simulations)
    
    def cross_validate(self, n_folds: int = 5) -> dict:
        """
        Perform time series cross-validation.
        
        Args:
            n_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        if self._data is None:
            raise RuntimeError("No data available for cross-validation")
        
        observations = self._data.get_observations()
        n_obs = len(observations)
        
        if n_folds >= n_obs:
            raise ValueError("Number of folds too large for data size")
        
        fold_size = n_obs // n_folds
        errors = []
        
        for i in range(n_folds):
            # Split data - use expanding window approach for time series
            train_end = (i + 1) * fold_size
            if train_end >= n_obs:
                break
            
            train_data = TimeSeriesData(observations[:train_end])
            
            # Create temporary model for this fold
            temp_model = self.__class__()
            temp_model.set_data(train_data)
            temp_model.fit()
            
            # Predict next observation
            if train_end < n_obs:
                forecast = temp_model.predict(1)
                actual = observations[train_end]
                error = abs(forecast.to_numpy()[0] - actual)
                errors.append(error)
        
        return {
            'mean_absolute_error': np.mean(errors) if errors else np.nan,
            'errors': errors
        }