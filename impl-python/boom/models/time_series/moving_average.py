"""
Moving Average (MA) Model implementation.

This module provides moving average time series models of various orders.
"""

import numpy as np
import scipy.optimize
from typing import Optional, Union, List
from ...linalg import Vector, Matrix
from .base import TimeSeriesModel, TimeSeriesData, TimeSeriesDataPoint
from ..base import Data


class MovingAverageModel(TimeSeriesModel):
    """
    Moving Average model of order q: MA(q).
    
    The model is: X_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
    where ε_t ~ N(0, σ²)
    """
    
    def __init__(self, order: int):
        """
        Initialize MA(q) model.
        
        Args:
            order: Order of the moving average model (q)
        """
        super().__init__()
        if order < 1:
            raise ValueError("MA order must be at least 1")
        
        self._order = order
        self._mean = 0.0
        self._coefficients = np.zeros(order)  # θ₁, θ₂, ..., θ_q
        self._sigma_squared = 1.0
        
        # For innovation algorithm
        self._innovations: Optional[np.ndarray] = None
        self._innovation_variances: Optional[np.ndarray] = None
    
    def fit(self, method: str = 'mle') -> None:
        """
        Fit the MA model.
        
        Args:
            method: Fitting method ('mle' for maximum likelihood estimation)
        """
        if self._data is None:
            raise RuntimeError("No data available for fitting")
        
        observations = self._data.get_observations()
        n_obs = len(observations)
        
        if n_obs <= self._order:
            raise ValueError(f"Need at least {self._order + 1} observations for MA({self._order}) model")
        
        if method == 'mle':
            self._fit_mle(observations)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        self._fitted = True
        self._notify_observers()
    
    def _fit_mle(self, observations: np.ndarray) -> None:
        """Fit using maximum likelihood estimation."""
        # Initial parameter guess
        self._mean = np.mean(observations)
        self._coefficients = np.random.normal(0, 0.1, self._order)
        self._sigma_squared = np.var(observations)
        
        # Pack parameters for optimization
        initial_params = np.concatenate([
            [self._mean],
            self._coefficients,
            [np.log(self._sigma_squared)]  # Use log for positivity constraint
        ])
        
        # Optimize log likelihood
        try:
            result = scipy.optimize.minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(observations,),
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Unpack optimized parameters
                self._mean = result.x[0]
                self._coefficients = result.x[1:1+self._order]
                self._sigma_squared = np.exp(result.x[-1])
                
                # Compute fitted values and residuals using innovation algorithm
                self._compute_innovations(observations)
                
            else:
                raise RuntimeError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to fit MA model: {e}")
    
    def _negative_log_likelihood(self, params: np.ndarray, observations: np.ndarray) -> float:
        """Compute negative log likelihood for optimization."""
        try:
            # Unpack parameters
            mean = params[0]
            coeffs = params[1:1+self._order]
            log_sigma_sq = params[-1]
            sigma_sq = np.exp(log_sigma_sq)
            
            # Compute log likelihood using innovation algorithm
            centered_obs = observations - mean
            log_lik = self._innovation_log_likelihood(centered_obs, coeffs, sigma_sq)
            
            return -log_lik
            
        except Exception:
            return 1e10  # Return large value if computation fails
    
    def _innovation_log_likelihood(self, centered_obs: np.ndarray, 
                                  coeffs: np.ndarray, sigma_sq: float) -> float:
        """Compute log likelihood using innovation algorithm."""
        n = len(centered_obs)
        
        # Initialize innovations and variances
        innovations = np.zeros(n)
        variances = np.zeros(n)
        
        # Compute autocovariances for MA(q) process
        gamma = self._compute_autocovariances(coeffs, sigma_sq)
        
        # Innovation algorithm
        for t in range(n):
            if t == 0:
                variances[0] = gamma[0]
                innovations[0] = centered_obs[0]
            else:
                # Compute prediction based on previous innovations
                prediction = 0.0
                for j in range(min(t, self._order)):
                    theta_j = coeffs[j] if j < len(coeffs) else 0.0
                    prediction += theta_j * innovations[t-1-j]
                
                # Prediction error
                innovations[t] = centered_obs[t] - prediction
                
                # Update variance
                variance_sum = 0.0
                for j in range(min(t, self._order)):
                    theta_j = coeffs[j] if j < len(coeffs) else 0.0
                    variance_sum += theta_j**2 * variances[t-1-j]
                
                variances[t] = sigma_sq + variance_sum
        
        # Compute log likelihood
        log_lik = 0.0
        for t in range(n):
            if variances[t] > 0:
                log_lik -= 0.5 * np.log(2 * np.pi * variances[t])
                log_lik -= 0.5 * innovations[t]**2 / variances[t]
            else:
                return -1e10  # Invalid variance
        
        return log_lik
    
    def _compute_autocovariances(self, coeffs: np.ndarray, sigma_sq: float) -> np.ndarray:
        """Compute autocovariances for MA(q) process."""
        # For MA(q): γ(k) = σ²(θ_k + θ₁θ_{k+1} + ... + θ_{q-k}θ_q) for k ≤ q
        # where θ₀ = 1
        
        max_lag = max(self._order + 1, 10)  # Need enough lags for algorithm
        gamma = np.zeros(max_lag)
        
        # θ coefficients with θ₀ = 1
        theta = np.zeros(self._order + 1)
        theta[0] = 1.0
        theta[1:] = coeffs
        
        for k in range(max_lag):
            if k <= self._order:
                for j in range(self._order + 1 - k):
                    gamma[k] += theta[j] * theta[j + k]
                gamma[k] *= sigma_sq
        
        return gamma
    
    def _compute_innovations(self, observations: np.ndarray) -> None:
        """Compute innovations and fitted values."""
        n = len(observations)
        centered_obs = observations - self._mean
        
        self._innovations = np.zeros(n)
        self._innovation_variances = np.zeros(n)
        fitted_values = np.zeros(n)
        
        # Compute autocovariances
        gamma = self._compute_autocovariances(self._coefficients, self._sigma_squared)
        
        # Innovation algorithm
        for t in range(n):
            if t == 0:
                self._innovation_variances[0] = gamma[0]
                self._innovations[0] = centered_obs[0]
                fitted_values[0] = self._mean
            else:
                # Compute prediction
                prediction = 0.0
                for j in range(min(t, self._order)):
                    theta_j = self._coefficients[j]
                    prediction += theta_j * self._innovations[t-1-j]
                
                fitted_values[t] = self._mean + prediction
                
                # Innovation
                self._innovations[t] = centered_obs[t] - prediction
                
                # Update variance
                variance_sum = 0.0
                for j in range(min(t, self._order)):
                    theta_j = self._coefficients[j]
                    variance_sum += theta_j**2 * self._innovation_variances[t-1-j]
                
                self._innovation_variances[t] = self._sigma_squared + variance_sum
        
        self._fitted_values = fitted_values
        self._residuals = self._innovations.copy()
    
    def predict(self, n_periods: int = 1) -> Vector:
        """
        Generate MA forecasts.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Vector of forecasts
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if self._innovations is None:
            # Compute innovations if not already done
            observations = self._data.get_observations()
            self._compute_innovations(observations)
        
        forecasts = np.zeros(n_periods)
        
        # For MA model, forecasts beyond order q are just the mean
        for i in range(n_periods):
            forecast = self._mean
            
            # Add MA components for forecasts within order
            for j in range(min(i + 1, self._order)):
                if len(self._innovations) > j:
                    theta_j = self._coefficients[j]
                    innovation_idx = len(self._innovations) - 1 - j
                    if innovation_idx >= 0:
                        forecast += theta_j * self._innovations[innovation_idx]
            
            forecasts[i] = forecast
        
        return Vector(forecasts)
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of current parameters."""
        if not self._fitted or self._data is None:
            raise RuntimeError("Model must be fitted before computing log likelihood")
        
        observations = self._data.get_observations()
        centered_obs = observations - self._mean
        
        return self._innovation_log_likelihood(centered_obs, self._coefficients, self._sigma_squared)
    
    def n_parameters(self) -> int:
        """Get number of parameters (mean + MA coefficients + variance)."""
        return self._order + 2  # mean + q coefficients + sigma²
    
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
    
    def check_invertibility(self) -> bool:
        """
        Check if the MA model is invertible.
        
        Returns:
            True if model is invertible
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before checking invertibility")
        
        # Create characteristic polynomial: 1 + θ₁z + θ₂z² + ... + θ_qz^q
        # For invertibility, all roots must be outside unit circle
        poly_coeffs = np.zeros(self._order + 1)
        poly_coeffs[0] = 1.0
        poly_coeffs[1:] = self._coefficients
        
        roots = np.roots(poly_coeffs[::-1])  # numpy roots expects descending order
        
        # Check if all roots are outside unit circle
        return np.all(np.abs(roots) > 1.0)
    
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
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        
        forecast_vars = np.zeros(n_periods)
        
        # For MA(q), forecast variance depends on how many MA terms are still active
        for i in range(n_periods):
            var_sum = self._sigma_squared  # Base variance
            
            # Add variance from MA terms
            for j in range(min(i + 1, self._order)):
                if j < len(self._coefficients):
                    var_sum += self._coefficients[j]**2 * self._sigma_squared
            
            forecast_vars[i] = var_sum
        
        forecast_stds = np.sqrt(forecast_vars)
        margins = z_score * forecast_stds
        
        forecasts_array = forecasts.to_numpy()
        lower = Vector(forecasts_array - margins)
        upper = Vector(forecasts_array + margins)
        
        return forecasts, lower, upper
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        return {
            'order': self._order,
            'mean': self._mean,
            'coefficients': self._coefficients.copy(),
            'sigma_squared': self._sigma_squared,
            'fitted': self._fitted
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set model parameters."""
        if 'mean' in params:
            self._mean = params['mean']
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
        Simulate future paths from the MA model.
        
        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            
        Returns:
            Matrix of simulations [n_periods x n_simulations]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before simulation")
        
        simulations = np.zeros((n_periods, n_simulations))
        
        for sim in range(n_simulations):
            # Generate innovations
            innovations = np.random.normal(0, np.sqrt(self._sigma_squared), n_periods + self._order)
            
            for i in range(n_periods):
                # Compute MA value
                value = self._mean + innovations[i + self._order]
                
                # Add MA terms
                for j in range(self._order):
                    value += self._coefficients[j] * innovations[i + self._order - 1 - j]
                
                simulations[i, sim] = value
        
        return Matrix(simulations)
    
    def simulate_data(self, n: Optional[int] = None) -> List[Data]:
        """
        Simulate data from the MA model.
        
        Args:
            n: Number of observations to simulate
            
        Returns:
            List of TimeSeriesDataPoint objects
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before simulation")
        
        if n is None:
            n = self.sample_size() if hasattr(self, 'sample_size') else 100
        
        # Generate innovations
        innovations = np.random.normal(0, np.sqrt(self._sigma_squared), n + self._order)
        
        data_points = []
        for t in range(n):
            # Compute MA value
            value = self._mean + innovations[t + self._order]
            
            # Add MA terms
            for j in range(self._order):
                value += self._coefficients[j] * innovations[t + self._order - 1 - j]
            
            data_points.append(TimeSeriesDataPoint(value, t))
        
        return data_points
    
    @property
    def order(self) -> int:
        """Get MA order."""
        return self._order
    
    @property
    def mean(self) -> float:
        """Get mean parameter."""
        return self._mean
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get MA coefficients."""
        return self._coefficients.copy()
    
    @property
    def sigma_squared(self) -> float:
        """Get residual variance."""
        return self._sigma_squared