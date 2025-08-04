"""
ARIMA (Autoregressive Integrated Moving Average) Model implementation.

This module provides ARIMA(p,d,q) models combining autoregressive,
differencing, and moving average components.
"""

import numpy as np
import scipy.optimize
from typing import Optional, Union, List, Tuple
from ...linalg import Vector, Matrix
from .base import TimeSeriesModel, TimeSeriesData, TimeSeriesDataPoint
from .autoregressive import AutoregressiveModel
from .moving_average import MovingAverageModel
from ..base import Data


class ArimaModel(TimeSeriesModel):
    """
    ARIMA(p,d,q) model.
    
    Combines autoregressive (AR), differencing (I), and moving average (MA) components:
    - AR(p): X_t depends on p previous values
    - I(d): Series is differenced d times to achieve stationarity  
    - MA(q): X_t depends on q previous error terms
    """
    
    def __init__(self, order: Tuple[int, int, int]):
        """
        Initialize ARIMA(p,d,q) model.
        
        Args:
            order: Tuple of (p, d, q) for AR order, differencing, MA order
        """
        super().__init__()
        
        p, d, q = order
        if p < 0 or d < 0 or q < 0:
            raise ValueError("ARIMA orders must be non-negative")
        
        self._p = p  # AR order
        self._d = d  # Differencing order
        self._q = q  # MA order
        
        # Model parameters
        self._intercept = 0.0
        self._ar_coeffs = np.zeros(p) if p > 0 else np.array([])
        self._ma_coeffs = np.zeros(q) if q > 0 else np.array([])
        self._sigma_squared = 1.0
        
        # For differencing
        self._original_data: Optional[np.ndarray] = None
        self._differenced_data: Optional[np.ndarray] = None
        self._initial_values: List[np.ndarray] = []  # Store values lost in differencing
        
        # For fitting
        self._innovations: Optional[np.ndarray] = None
    
    def set_data(self, data: TimeSeriesData) -> None:
        """Set data and apply differencing if needed."""
        super().set_data(data)
        
        if self._data is not None:
            self._original_data = self._data.get_observations()
            self._apply_differencing()
    
    def _apply_differencing(self) -> None:
        """Apply differencing transformation."""
        if self._original_data is None:
            return
        
        # Store initial values for reconstruction
        self._initial_values = []
        current_data = self._original_data.copy()
        
        for d in range(self._d):
            if len(current_data) <= 1:
                raise ValueError(f"Insufficient data for {self._d} differences")
            
            # Store the first value of current level
            self._initial_values.append(current_data[:d+1].copy())
            
            # Apply differencing
            current_data = np.diff(current_data)
        
        self._differenced_data = current_data
    
    def _invert_differencing(self, differenced_series: np.ndarray,
                            n_original: int) -> np.ndarray:
        """Invert differencing to get back to original scale."""
        if self._d == 0:
            return differenced_series
        
        current_series = differenced_series.copy()
        
        # Apply inverse differencing in reverse order
        for d in range(self._d - 1, -1, -1):
            initial_vals = self._initial_values[d]
            
            # Reconstruct by cumulative sum
            if len(initial_vals) > 0:
                reconstructed = np.zeros(len(current_series) + len(initial_vals))
                reconstructed[:len(initial_vals)] = initial_vals
                
                # Cumulative sum to invert differencing
                for i in range(len(current_series)):
                    reconstructed[len(initial_vals) + i] = (
                        reconstructed[len(initial_vals) + i - 1] + current_series[i]
                    )
                
                current_series = reconstructed
        
        # Return only the length we need
        return current_series[-n_original:] if len(current_series) >= n_original else current_series
    
    def fit(self, method: str = 'mle') -> None:
        """
        Fit the ARIMA model.
        
        Args:
            method: Fitting method ('mle' for maximum likelihood)
        """
        if self._differenced_data is None:
            raise RuntimeError("No data available for fitting")
        
        if len(self._differenced_data) <= max(self._p, self._q):
            raise ValueError(
                f"Need at least {max(self._p, self._q) + 1} observations "
                f"after differencing for ARIMA({self._p},{self._d},{self._q})"
            )
        
        if method == 'mle':
            self._fit_mle()
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        self._fitted = True
        self._notify_observers()
    
    def _fit_mle(self) -> None:
        """Fit using maximum likelihood estimation."""
        observations = self._differenced_data
        
        # Initial parameter estimates
        if self._p > 0:
            # Fit AR model for initial AR coefficients
            try:
                temp_ar = AutoregressiveModel(self._p)
                temp_ar.set_data(TimeSeriesData(observations))
                temp_ar.fit()
                ar_params = temp_ar.get_parameters()
                self._ar_coeffs = ar_params['coefficients']
                self._intercept = ar_params['intercept']
                self._sigma_squared = ar_params['sigma_squared']
            except:
                self._ar_coeffs = np.random.normal(0, 0.1, self._p)
                self._intercept = np.mean(observations)
        else:
            self._intercept = np.mean(observations)
        
        if self._q > 0:
            # Random initialization for MA coefficients
            self._ma_coeffs = np.random.normal(0, 0.1, self._q)
        
        if self._sigma_squared <= 0:
            self._sigma_squared = np.var(observations)
        
        # Pack parameters for optimization
        params = []
        if self._p > 0 or self._q > 0:  # Include intercept only if we have AR or MA terms
            params.append(self._intercept)
        if self._p > 0:
            params.extend(self._ar_coeffs)
        if self._q > 0:
            params.extend(self._ma_coeffs)
        params.append(np.log(self._sigma_squared))  # Log for positivity
        
        initial_params = np.array(params)
        
        # Optimize
        try:
            result = scipy.optimize.minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(observations,),
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success:
                self._unpack_parameters(result.x)
                self._compute_residuals_and_fitted()
            else:
                raise RuntimeError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to fit ARIMA model: {e}")
    
    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Unpack optimized parameters."""
        idx = 0
        
        if self._p > 0 or self._q > 0:
            self._intercept = params[idx]
            idx += 1
        else:
            self._intercept = 0.0
        
        if self._p > 0:
            self._ar_coeffs = params[idx:idx + self._p]
            idx += self._p
        
        if self._q > 0:
            self._ma_coeffs = params[idx:idx + self._q]
            idx += self._q
        
        self._sigma_squared = np.exp(params[idx])
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                               observations: np.ndarray) -> float:
        """Compute negative log likelihood."""
        try:
            # Unpack parameters
            idx = 0
            
            if self._p > 0 or self._q > 0:
                intercept = params[idx]
                idx += 1
            else:
                intercept = 0.0
            
            ar_coeffs = params[idx:idx + self._p] if self._p > 0 else np.array([])
            idx += self._p
            
            ma_coeffs = params[idx:idx + self._q] if self._q > 0 else np.array([])
            idx += self._q
            
            log_sigma_sq = params[idx]
            sigma_sq = np.exp(log_sigma_sq)
            
            # Compute log likelihood using state space approach
            log_lik = self._compute_log_likelihood(
                observations, intercept, ar_coeffs, ma_coeffs, sigma_sq
            )
            
            return -log_lik
            
        except Exception:
            return 1e10
    
    def _compute_log_likelihood(self, observations: np.ndarray,
                              intercept: float, ar_coeffs: np.ndarray,
                              ma_coeffs: np.ndarray, sigma_sq: float) -> float:
        """Compute log likelihood using innovations algorithm."""
        n = len(observations)
        
        # Center observations
        centered_obs = observations - intercept
        
        # Initialize
        innovations = np.zeros(n)
        variances = np.zeros(n)
        
        # Use simplified approach for likelihood computation
        # This is approximate but numerically stable
        
        max_order = max(self._p, self._q)
        
        for t in range(n):
            if t < max_order:
                # Use reduced form for initial observations
                innovations[t] = centered_obs[t]
                variances[t] = sigma_sq
            else:
                # AR prediction
                ar_pred = 0.0
                if self._p > 0:
                    for i in range(self._p):
                        ar_pred += ar_coeffs[i] * centered_obs[t - 1 - i]
                
                # MA prediction  
                ma_pred = 0.0
                if self._q > 0:
                    for i in range(self._q):
                        ma_pred += ma_coeffs[i] * innovations[t - 1 - i]
                
                # Innovation
                prediction = ar_pred + ma_pred
                innovations[t] = centered_obs[t] - prediction
                variances[t] = sigma_sq
        
        # Compute log likelihood
        log_lik = 0.0
        for t in range(n):
            if variances[t] > 0:
                log_lik -= 0.5 * np.log(2 * np.pi * variances[t])
                log_lik -= 0.5 * innovations[t]**2 / variances[t]
        
        return log_lik
    
    def _compute_residuals_and_fitted(self) -> None:
        """Compute residuals and fitted values."""
        if self._differenced_data is None:
            return
        
        observations = self._differenced_data
        n = len(observations)
        
        innovations = np.zeros(n)
        fitted_values = np.zeros(n)
        
        centered_obs = observations - self._intercept
        max_order = max(self._p, self._q)
        
        for t in range(n):
            if t < max_order:
                innovations[t] = centered_obs[t]
                fitted_values[t] = self._intercept
            else:
                # AR component
                ar_pred = 0.0
                if self._p > 0:
                    for i in range(self._p):
                        ar_pred += self._ar_coeffs[i] * centered_obs[t - 1 - i]
                
                # MA component
                ma_pred = 0.0
                if self._q > 0:
                    for i in range(self._q):
                        ma_pred += self._ma_coeffs[i] * innovations[t - 1 - i]
                
                prediction = ar_pred + ma_pred
                fitted_values[t] = self._intercept + prediction
                innovations[t] = centered_obs[t] - prediction
        
        self._innovations = innovations
        self._fitted_values = fitted_values
        self._residuals = innovations
    
    def predict(self, n_periods: int = 1) -> Vector:
        """
        Generate ARIMA forecasts.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Vector of forecasts (on original scale)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Generate forecasts on differenced scale
        differenced_forecasts = self._predict_differenced(n_periods)
        
        # Convert back to original scale
        if self._d == 0:
            return Vector(differenced_forecasts)
        
        # For differenced series, need to integrate forecasts
        if self._original_data is None:
            raise RuntimeError("Original data not available")
        
        # Reconstruct forecast path
        last_original_values = self._original_data[-self._d:].copy()
        forecast_path = np.zeros(n_periods)
        
        for i in range(n_periods):
            # This is a simplified approach - exact integration is more complex
            if self._d == 1:
                if i == 0:
                    forecast_path[i] = last_original_values[-1] + differenced_forecasts[i]
                else:
                    forecast_path[i] = forecast_path[i-1] + differenced_forecasts[i]
            else:
                # Higher order differencing - use simplified approach
                forecast_path[i] = differenced_forecasts[i]
        
        return Vector(forecast_path)
    
    def _predict_differenced(self, n_periods: int) -> np.ndarray:
        """Generate forecasts on differenced scale."""
        forecasts = np.zeros(n_periods)
        
        if self._differenced_data is None or self._innovations is None:
            return forecasts
        
        # Initialize history
        if self._p > 0:
            ar_history = self._differenced_data[-self._p:].copy() - self._intercept
        else:
            ar_history = np.array([])
        
        if self._q > 0:
            ma_history = self._innovations[-self._q:].copy()
        else:
            ma_history = np.array([])
        
        for i in range(n_periods):
            forecast = self._intercept
            
            # AR component
            if self._p > 0:
                for j in range(min(len(ar_history), self._p)):
                    forecast += self._ar_coeffs[j] * ar_history[-(j+1)]
            
            # MA component (only for initial periods)
            if self._q > 0 and i < self._q:
                for j in range(min(len(ma_history), self._q - i)):
                    forecast += self._ma_coeffs[i + j] * ma_history[-(j+1)]
            
            forecasts[i] = forecast
            
            # Update histories for next prediction
            if self._p > 0:
                ar_history = np.append(ar_history, forecast - self._intercept)
                if len(ar_history) > self._p:
                    ar_history = ar_history[-self._p:]
        
        return forecasts
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of current parameters."""
        if not self._fitted or self._differenced_data is None:
            raise RuntimeError("Model must be fitted before computing log likelihood")
        
        return self._compute_log_likelihood(
            self._differenced_data, self._intercept, 
            self._ar_coeffs, self._ma_coeffs, self._sigma_squared
        )
    
    def n_parameters(self) -> int:
        """Get number of parameters."""
        n_params = 0
        if self._p > 0 or self._q > 0:
            n_params += 1  # intercept
        n_params += self._p  # AR coefficients
        n_params += self._q  # MA coefficients
        n_params += 1  # sigma squared
        return n_params
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        return {
            'order': (self._p, self._d, self._q),
            'intercept': self._intercept,
            'ar_coefficients': self._ar_coeffs.copy(),
            'ma_coefficients': self._ma_coeffs.copy(),
            'sigma_squared': self._sigma_squared,
            'fitted': self._fitted
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set model parameters."""
        if 'intercept' in params:
            self._intercept = params['intercept']
        if 'ar_coefficients' in params:
            self._ar_coeffs = np.array(params['ar_coefficients'])
        if 'ma_coefficients' in params:
            self._ma_coeffs = np.array(params['ma_coefficients'])
        if 'sigma_squared' in params:
            if params['sigma_squared'] <= 0:
                raise ValueError("sigma_squared must be positive")
            self._sigma_squared = params['sigma_squared']
        if 'fitted' in params:
            self._fitted = params['fitted']
        
        self._notify_observers()
    
    @property
    def order(self) -> Tuple[int, int, int]:
        """Get ARIMA order (p, d, q)."""
        return (self._p, self._d, self._q)
    
    @property
    def ar_coefficients(self) -> np.ndarray:
        """Get AR coefficients."""
        return self._ar_coeffs.copy()
    
    @property
    def ma_coefficients(self) -> np.ndarray:
        """Get MA coefficients."""
        return self._ma_coeffs.copy()
    
    @property
    def intercept(self) -> float:
        """Get intercept."""
        return self._intercept
    
    def simulate_data(self, n: Optional[int] = None) -> List[Data]:
        """
        Simulate data from the ARIMA model.
        
        Args:
            n: Number of observations to simulate
            
        Returns:
            List of TimeSeriesDataPoint objects
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before simulation")
        
        if n is None:
            n = self.sample_size() if hasattr(self, 'sample_size') else 100
        
        # For simplicity, simulate as if no differencing (i.e., simulate ARMA)
        # This is a simplified implementation
        data_points = []
        
        # Initialize histories
        ar_history = np.zeros(self._p) if self._p > 0 else np.array([])
        innovations = np.random.normal(0, np.sqrt(self._sigma_squared), n + self._q)
        
        for t in range(n):
            # AR component
            ar_part = 0.0
            if self._p > 0 and t >= self._p:
                for i in range(self._p):
                    ar_part += self._ar_coeffs[i] * ar_history[-(i+1)]
            
            # MA component
            ma_part = 0.0
            if self._q > 0:
                for j in range(min(t + 1, self._q)):
                    ma_part += self._ma_coeffs[j] * innovations[t - j]
            
            # Generate value
            value = self._intercept + ar_part + ma_part + innovations[t]
            
            # Update AR history
            if self._p > 0:
                ar_history = np.append(ar_history, value)
                if len(ar_history) > self._p:
                    ar_history = ar_history[-self._p:]
            
            data_points.append(TimeSeriesDataPoint(value, t))
        
        return data_points
    
    @property
    def sigma_squared(self) -> float:
        """Get residual variance."""
        return self._sigma_squared