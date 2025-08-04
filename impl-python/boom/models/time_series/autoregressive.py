"""
Autoregressive (AR) Model implementation.

This module provides autoregressive time series models of various orders.
"""

import numpy as np
import scipy.linalg
from typing import Optional, Union, List
from ...linalg import Vector, Matrix
from .base import TimeSeriesModel, TimeSeriesData, TimeSeriesDataPoint
from ..base import Data


class AutoregressiveModel(TimeSeriesModel):
    """
    Autoregressive model of order p: AR(p).
    
    The model is: X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t
    where ε_t ~ N(0, σ²)
    """
    
    def __init__(self, order: int):
        """
        Initialize AR(p) model.
        
        Args:
            order: Order of the autoregressive model (p)
        """
        super().__init__()
        if order < 1:
            raise ValueError("AR order must be at least 1")
        
        self._order = order
        self._intercept = 0.0
        self._coefficients = np.zeros(order)
        self._sigma_squared = 1.0
        
        # For efficient computation
        self._design_matrix: Optional[Matrix] = None
        self._response_vector: Optional[Vector] = None
    
    def fit(self) -> None:
        """Fit the AR model using ordinary least squares."""
        if self._data is None:
            raise RuntimeError("No data available for fitting")
        
        observations = self._data.get_observations()
        n_obs = len(observations)
        
        if n_obs <= self._order:
            raise ValueError(f"Need at least {self._order + 1} observations for AR({self._order}) model")
        
        # Create design matrix and response vector
        self._create_design_matrix()
        
        # Solve least squares: β = (X'X)⁻¹X'y
        X = self._design_matrix.to_numpy()
        y = self._response_vector.to_numpy()
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
            # Use scipy for numerical stability
            coeffs, residuals, rank, s = scipy.linalg.lstsq(X_with_intercept, y)
            
            self._intercept = coeffs[0]
            self._coefficients = coeffs[1:]
            
            # Compute residual variance
            if hasattr(residuals, '__len__') and len(residuals) > 0:
                self._sigma_squared = residuals[0] / (len(y) - len(coeffs))
            else:
                # Compute manually if residuals not returned
                fitted = X_with_intercept @ coeffs
                residuals_manual = y - fitted
                self._sigma_squared = np.sum(residuals_manual**2) / (len(y) - len(coeffs))
            
            # Store fitted values and residuals
            self._fitted_values = X_with_intercept @ coeffs
            self._residuals = y - self._fitted_values
            
        except scipy.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to fit AR model: {e}")
        
        self._fitted = True
        self._notify_observers()
    
    def _create_design_matrix(self) -> None:
        """Create the design matrix for AR model fitting."""
        observations = self._data.get_observations()
        n_obs = len(observations)
        n_valid = n_obs - self._order
        
        # Design matrix: each row has [X_{t-1}, X_{t-2}, ..., X_{t-p}]
        X = np.zeros((n_valid, self._order))
        y = np.zeros(n_valid)
        
        for i in range(n_valid):
            # Response is X_{t}
            y[i] = observations[i + self._order]
            
            # Predictors are X_{t-1}, X_{t-2}, ..., X_{t-p}
            for j in range(self._order):
                X[i, j] = observations[i + self._order - 1 - j]
        
        self._design_matrix = Matrix(X)
        self._response_vector = Vector(y)
    
    def predict(self, n_periods: int = 1) -> Vector:
        """
        Generate AR forecasts.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Vector of forecasts
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if self._data is None:
            raise RuntimeError("No data available")
        
        observations = self._data.get_observations()
        forecasts = np.zeros(n_periods)
        
        # Initialize with last p observations
        history = observations[-self._order:].copy()
        
        for i in range(n_periods):
            # Compute forecast: c + φ₁X_{t-1} + ... + φₚX_{t-p}
            forecast = self._intercept + np.sum(self._coefficients * history[::-1])
            forecasts[i] = forecast
            
            # Update history for next prediction
            history = np.roll(history, 1)
            history[0] = forecast
        
        return Vector(forecasts)
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of current parameters."""
        if not self._fitted or self._residuals is None:
            raise RuntimeError("Model must be fitted before computing log likelihood")
        
        n = len(self._residuals)
        log_lik = -0.5 * n * np.log(2 * np.pi * self._sigma_squared)
        log_lik -= 0.5 * np.sum(self._residuals**2) / self._sigma_squared
        
        return log_lik
    
    def n_parameters(self) -> int:
        """Get number of parameters (intercept + AR coefficients + variance)."""
        return self._order + 2  # intercept + p coefficients + sigma²
    
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted
    
    def aic(self) -> float:
        """Compute Akaike Information Criterion."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing AIC")
        
        log_lik = self.log_likelihood()
        k = self.n_parameters()
        return 2 * k - 2 * log_lik
    
    def forecast(self, n_periods: int = 1, 
                 confidence_level: float = 0.95) -> tuple:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            n_periods: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (forecasts, lower_bounds, upper_bounds)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        forecasts = self.predict(n_periods)
        
        # Compute forecast standard errors
        # For AR model, forecast variance increases with horizon
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        
        forecast_vars = np.zeros(n_periods)
        forecast_vars[0] = self._sigma_squared
        
        # Approximate forecast variance growth (exact formula is more complex)
        for i in range(1, n_periods):
            # Rough approximation - exact calculation requires recursive formula
            forecast_vars[i] = forecast_vars[i-1] * (1 + np.sum(self._coefficients**2))
        
        forecast_stds = np.sqrt(forecast_vars)
        margins = z_score * forecast_stds
        
        forecasts_array = forecasts.to_numpy()
        lower = Vector(forecasts_array - margins)
        upper = Vector(forecasts_array + margins)
        
        return forecasts, lower, upper
    
    def check_stationarity(self) -> bool:
        """
        Check if the AR model is stationary.
        
        Returns:
            True if model is stationary
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before checking stationarity")
        
        # Create characteristic polynomial: 1 - φ₁z - φ₂z² - ... - φₚzᵖ
        # For stationarity, all roots must be outside unit circle
        poly_coeffs = np.zeros(self._order + 1)
        poly_coeffs[0] = 1.0
        poly_coeffs[1:] = -self._coefficients
        
        roots = np.roots(poly_coeffs[::-1])  # numpy roots expects descending order
        
        # Check if all roots are outside unit circle
        return np.all(np.abs(roots) > 1.0)
    
    def impulse_response(self, n_periods: int = 20) -> Vector:
        """
        Compute impulse response function.
        
        Args:
            n_periods: Number of periods for impulse response
            
        Returns:
            Vector of impulse responses
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing impulse response")
        
        # Initialize impulse response
        psi = np.zeros(n_periods)
        psi[0] = 1.0  # Initial impulse
        
        # Compute recursive impulse responses
        for i in range(1, n_periods):
            for j in range(min(i, self._order)):
                psi[i] += self._coefficients[j] * psi[i - 1 - j]
        
        return Vector(psi)
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        return {
            'order': self._order,
            'intercept': self._intercept,
            'coefficients': self._coefficients.copy(),
            'sigma_squared': self._sigma_squared,
            'fitted': self._fitted
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set model parameters."""
        if 'intercept' in params:
            self._intercept = params['intercept']
        if 'coefficients' in params:
            coeffs = params['coefficients']
            if len(coeffs) != self._order:
                raise ValueError(f"Expected {self._order} coefficients, got {len(coeffs)}")
            self._coefficients = np.array(coeffs)
        if 'sigma_squared' in params:
            if params['sigma_squared'] <= 0:
                raise ValueError("sigma_squared must be positive")
            self._sigma_squared = params['sigma_squared']
        if 'fitted' in params:
            self._fitted = params['fitted']
        
        self._notify_observers()
    
    def simulate(self, n_periods: int, n_simulations: int = 1) -> Matrix:
        """
        Simulate future paths from the AR model.
        
        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            
        Returns:
            Matrix of simulations [n_periods x n_simulations]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before simulation")
        
        if self._data is None:
            raise RuntimeError("No data available")
        
        observations = self._data.get_observations()
        simulations = np.zeros((n_periods, n_simulations))
        
        for sim in range(n_simulations):
            # Initialize with last p observations
            history = observations[-self._order:].copy()
            
            for i in range(n_periods):
                # Generate innovation
                epsilon = np.random.normal(0, np.sqrt(self._sigma_squared))
                
                # Compute next value
                next_val = self._intercept + np.sum(self._coefficients * history[::-1]) + epsilon
                simulations[i, sim] = next_val
                
                # Update history
                history = np.roll(history, 1)
                history[0] = next_val
        
        return Matrix(simulations)
    
    def simulate_data(self, n: Optional[int] = None) -> List[Data]:
        """
        Simulate data from the AR model.
        
        Args:
            n: Number of observations to simulate
            
        Returns:
            List of TimeSeriesDataPoint objects
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before simulation")
        
        if n is None:
            n = self.sample_size() if hasattr(self, 'sample_size') else 100
        
        # Simulate time series
        data_points = []
        history = np.zeros(self._order)  # Start with zeros
        
        for t in range(n):
            # Generate next value
            if t < self._order:
                # For initial values, use intercept plus noise
                next_val = self._intercept + np.random.normal(0, np.sqrt(self._sigma_squared))
                history[t] = next_val
            else:
                # AR process
                ar_part = np.sum(self._coefficients * history[::-1])
                next_val = self._intercept + ar_part + np.random.normal(0, np.sqrt(self._sigma_squared))
                
                # Update history
                history = np.roll(history, 1)
                history[0] = next_val
            
            data_points.append(TimeSeriesDataPoint(next_val, t))
        
        return data_points
    
    @property
    def order(self) -> int:
        """Get AR order."""
        return self._order
    
    @property
    def intercept(self) -> float:
        """Get intercept parameter."""
        return self._intercept
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get AR coefficients."""
        return self._coefficients.copy()
    
    @property
    def sigma_squared(self) -> float:
        """Get residual variance."""
        return self._sigma_squared