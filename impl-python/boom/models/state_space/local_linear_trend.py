"""Local linear trend model for time series analysis."""

import numpy as np
from typing import Optional

from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import RNG
from .base import StateModel, StateSpaceModel
from .kalman import KalmanFilter


class LocalLinearTrendStateModel(StateModel):
    """Local linear trend state model: level and slope both follow random walks.
    
    State equation:
    [level_{t+1}]   [1 1] [level_t]   [eta_level_t]
    [slope_{t+1}] = [0 1] [slope_t] + [eta_slope_t]
    
    where eta_level_t ~ N(0, sigma_level^2) and eta_slope_t ~ N(0, sigma_slope^2)
    
    This represents a local linear trend with stochastically evolving level and slope.
    """
    
    def __init__(self, level_variance: float = 1.0, slope_variance: float = 1.0):
        """Initialize local linear trend state model.
        
        Args:
            level_variance: Variance of level innovations
            slope_variance: Variance of slope innovations
        """
        super().__init__(state_dimension=2)
        self._level_variance = float(level_variance)
        self._slope_variance = float(slope_variance)
        
        if self._level_variance <= 0:
            raise ValueError("Level variance must be positive")
        if self._slope_variance <= 0:
            raise ValueError("Slope variance must be positive")
    
    @property
    def level_variance(self) -> float:
        """Get level variance."""
        return self._level_variance
    
    @property
    def slope_variance(self) -> float:
        """Get slope variance."""
        return self._slope_variance
    
    def set_level_variance(self, variance: float):
        """Set level variance."""
        if variance <= 0:
            raise ValueError("Level variance must be positive")
        self._level_variance = float(variance)
    
    def set_slope_variance(self, variance: float):
        """Set slope variance."""
        if variance <= 0:
            raise ValueError("Slope variance must be positive")
        self._slope_variance = float(variance)
    
    def transition_matrix(self, t: int) -> Matrix:
        """Get transition matrix for local linear trend."""
        return Matrix(np.array([
            [1.0, 1.0],  # level_{t+1} = level_t + slope_t + eta_level_t
            [0.0, 1.0]   # slope_{t+1} = slope_t + eta_slope_t
        ]))
    
    def state_error_variance(self, t: int) -> SpdMatrix:
        """Get state error variance (diagonal)."""
        return SpdMatrix(np.array([
            [self._level_variance, 0.0],
            [0.0, self._slope_variance]
        ]))
    
    def observation_matrix(self, t: int) -> Matrix:
        """Get observation matrix (observe level only)."""
        return Matrix(np.array([[1.0, 0.0]]))
    
    def initial_state_mean(self) -> Vector:
        """Get initial state mean (level=0, slope=0)."""
        return Vector([0.0, 0.0])
    
    def initial_state_variance(self) -> SpdMatrix:
        """Get initial state variance (diffuse)."""
        return SpdMatrix(1e6 * np.eye(2))


class SeasonalStateModel(StateModel):
    """Seasonal state model with trigonometric representation.
    
    Uses a sum of sine and cosine terms to model seasonal patterns.
    For a single frequency, the state is [s_t, s*_t] where:
    
    [s_{t+1}]   [cos(2π*freq) sin(2π*freq)] [s_t]     [eta_s_t]
    [s*_{t+1}] = [-sin(2π*freq) cos(2π*freq)] [s*_t] + [eta_s*_t]
    """
    
    def __init__(self, period: float, variance: float = 1.0):
        """Initialize seasonal state model.
        
        Args:
            period: Seasonal period (e.g., 12 for monthly data)
            variance: Variance of seasonal innovations
        """
        super().__init__(state_dimension=2)
        self._period = float(period)
        self._variance = float(variance)
        
        if self._period <= 0:
            raise ValueError("Period must be positive")
        if self._variance <= 0:
            raise ValueError("Variance must be positive")
        
        # Frequency for trigonometric representation
        self._frequency = 2 * np.pi / self._period
    
    @property
    def period(self) -> float:
        """Get seasonal period."""
        return self._period
    
    @property
    def variance(self) -> float:
        """Get seasonal variance."""
        return self._variance
    
    def set_variance(self, variance: float):
        """Set seasonal variance."""
        if variance <= 0:
            raise ValueError("Variance must be positive")
        self._variance = float(variance)
    
    def transition_matrix(self, t: int) -> Matrix:
        """Get transition matrix for seasonal component."""
        cos_freq = np.cos(self._frequency)
        sin_freq = np.sin(self._frequency)
        
        return Matrix(np.array([
            [cos_freq, sin_freq],
            [-sin_freq, cos_freq]
        ]))
    
    def state_error_variance(self, t: int) -> SpdMatrix:
        """Get state error variance."""
        return SpdMatrix(self._variance * np.eye(2))
    
    def observation_matrix(self, t: int) -> Matrix:
        """Get observation matrix (observe first component only)."""
        return Matrix(np.array([[1.0, 0.0]]))
    
    def initial_state_mean(self) -> Vector:
        """Get initial state mean."""
        return Vector([0.0, 0.0])
    
    def initial_state_variance(self) -> SpdMatrix:
        """Get initial state variance."""
        return SpdMatrix(1e6 * np.eye(2))


class LocalLinearTrendModel(StateSpaceModel):
    """Local linear trend model with optional seasonal component.
    
    Model:
    y_t = level_t + seasonal_t + epsilon_t,  epsilon_t ~ N(0, sigma_obs^2)
    
    Trend component:
    level_{t+1} = level_t + slope_t + eta_level_t
    slope_{t+1} = slope_t + eta_slope_t
    
    Optional seasonal component (if period > 0):
    Uses trigonometric representation for seasonal pattern.
    """
    
    def __init__(self, observation_variance: float = 1.0, 
                 level_variance: float = 1.0, slope_variance: float = 1.0,
                 seasonal_period: Optional[float] = None, seasonal_variance: float = 1.0):
        """Initialize local linear trend model.
        
        Args:
            observation_variance: Observation error variance
            level_variance: Level innovation variance
            slope_variance: Slope innovation variance
            seasonal_period: Seasonal period (None for no seasonality)
            seasonal_variance: Seasonal innovation variance
        """
        super().__init__(observation_variance)
        
        # Add trend component
        self._trend_model = LocalLinearTrendStateModel(level_variance, slope_variance)
        self.add_state_model(self._trend_model)
        
        # Add seasonal component if specified
        self._seasonal_model = None
        if seasonal_period is not None and seasonal_period > 0:
            self._seasonal_model = SeasonalStateModel(seasonal_period, seasonal_variance)
            self.add_state_model(self._seasonal_model)
    
    @property
    def level_variance(self) -> float:
        """Get level variance."""
        return self._trend_model.level_variance
    
    @property
    def slope_variance(self) -> float:
        """Get slope variance."""
        return self._trend_model.slope_variance
    
    @property
    def has_seasonal(self) -> bool:
        """Check if model has seasonal component."""
        return self._seasonal_model is not None
    
    @property
    def seasonal_period(self) -> Optional[float]:
        """Get seasonal period."""
        return self._seasonal_model.period if self._seasonal_model else None
    
    @property
    def seasonal_variance(self) -> Optional[float]:
        """Get seasonal variance."""
        return self._seasonal_model.variance if self._seasonal_model else None
    
    def set_level_variance(self, variance: float):
        """Set level variance."""
        self._trend_model.set_level_variance(variance)
        self._notify_observers()
    
    def set_slope_variance(self, variance: float):
        """Set slope variance."""
        self._trend_model.set_slope_variance(variance)
        self._notify_observers()
    
    def set_seasonal_variance(self, variance: float):
        """Set seasonal variance."""
        if self._seasonal_model:
            self._seasonal_model.set_variance(variance)
            self._notify_observers()
    
    def observation_matrix(self, t: int) -> Matrix:
        """Get observation matrix (sum of all components)."""
        if self.has_seasonal:
            # Observe trend (level) + seasonal
            return Matrix(np.array([[1.0, 0.0, 1.0, 0.0]]))
        else:
            # Observe trend (level) only
            return Matrix(np.array([[1.0, 0.0]]))
    
    def log_likelihood(self) -> float:
        """Compute log likelihood using Kalman filter."""
        if self.time_dimension == 0:
            return 0.0
        
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter(compute_likelihood=True)
        return kalman_filter.log_likelihood()
    
    def fit(self, max_iterations: int = 100, tolerance: float = 1e-6) -> bool:
        """Fit model parameters using maximum likelihood estimation.
        
        Args:
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            True if converged, False otherwise
        """
        if self.time_dimension == 0:
            return True
        
        # Initialize parameters if needed
        if self.observation_variance <= 0:
            self.set_observation_variance(np.var([d.y() for d in self._data if d.is_observed()]))
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Run Kalman filter and smoother
            kalman_filter = KalmanFilter(self)
            kalman_filter.filter(compute_likelihood=True)
            kalman_filter.smooth()
            
            current_log_likelihood = kalman_filter.log_likelihood()
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                return True
            
            prev_log_likelihood = current_log_likelihood
            
            # M-step: Update parameters (simplified implementation)
            self._m_step(kalman_filter)
        
        return False  # Did not converge
    
    def _m_step(self, kalman_filter: KalmanFilter):
        """M-step of EM algorithm: update parameters (simplified)."""
        filter_states = kalman_filter.filter_states()
        smoother_states = kalman_filter.smoother_states()
        
        if not smoother_states:
            return
        
        T = self.time_dimension
        
        # Update observation variance
        sse_obs = 0.0
        n_obs = 0
        
        for t in range(T):
            data_t = self.get_data(t)
            if data_t.is_observed():
                # Predicted observation from state
                Z_t = self.observation_matrix(t)
                predicted_obs = float((Z_t @ smoother_states[t].state_mean)[0])
                
                residual = data_t.y() - predicted_obs
                sse_obs += residual**2
                n_obs += 1
        
        if n_obs > 0:
            new_obs_variance = sse_obs / n_obs
            self.set_observation_variance(max(new_obs_variance, 1e-8))
    
    def predict(self, n_ahead: int = 1) -> tuple[Vector, Vector]:
        """Predict future observations.
        
        Args:
            n_ahead: Number of steps ahead to predict
            
        Returns:
            Tuple of (means, variances) for predicted observations
        """
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter()
        
        predictions = kalman_filter.predict(n_ahead)
        
        means = Vector([pred[0] for pred in predictions])
        variances = Vector([pred[1] for pred in predictions])
        
        return means, variances
    
    def extract_components(self) -> dict:
        """Extract smoothed estimates of trend and seasonal components.
        
        Returns:
            Dictionary with 'level', 'slope', and optionally 'seasonal' components
        """
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter()
        kalman_filter.smooth()
        
        smoother_states = kalman_filter.smoother_states()
        
        components = {}
        
        # Extract trend components
        components['level'] = Vector([state.state_mean[0] for state in smoother_states])
        components['slope'] = Vector([state.state_mean[1] for state in smoother_states])
        
        # Extract seasonal component if present
        if self.has_seasonal:
            components['seasonal'] = Vector([state.state_mean[2] for state in smoother_states])
        
        return components
    
    def residuals(self) -> tuple[Vector, Vector]:
        """Compute standardized residuals.
        
        Returns:
            Tuple of (residuals, standardized_residuals)
        """
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter()
        return kalman_filter.residuals()
    
    def simulate_data(self, n: int, initial_level: float = 0.0, initial_slope: float = 0.0,
                     rng: Optional[RNG] = None) -> list:
        """Simulate time series data from the local linear trend model.
        
        Args:
            n: Number of time points to simulate
            initial_level: Initial level value
            initial_slope: Initial slope value
            rng: Random number generator
            
        Returns:
            List of simulated observations
        """
        if rng is None:
            rng = RNG()
        
        observations = []
        level = initial_level
        slope = initial_slope
        
        # Initialize seasonal state if present
        seasonal_state = np.array([0.0, 0.0]) if self.has_seasonal else None
        
        for t in range(n):
            # Generate observation
            obs_mean = level
            if self.has_seasonal and seasonal_state is not None:
                obs_mean += seasonal_state[0]  # Add seasonal component
            
            obs_error = np.sqrt(self.observation_variance) * rng.normal()
            y_t = obs_mean + obs_error
            observations.append(y_t)
            
            # Evolve trend state
            level_error = np.sqrt(self.level_variance) * rng.normal()
            slope_error = np.sqrt(self.slope_variance) * rng.normal()
            
            # Update level and slope
            new_level = level + slope + level_error
            new_slope = slope + slope_error
            
            level = new_level
            slope = new_slope
            
            # Evolve seasonal state if present
            if self.has_seasonal and seasonal_state is not None:
                T_seasonal = self._seasonal_model.transition_matrix(t).to_numpy()
                seasonal_error = np.sqrt(self.seasonal_variance) * np.array([rng.normal(), rng.normal()])
                seasonal_state = T_seasonal @ seasonal_state + seasonal_error
        
        return observations
    
    def clone(self) -> 'LocalLinearTrendModel':
        """Create a copy of this model."""
        cloned = LocalLinearTrendModel(
            self.observation_variance, 
            self.level_variance, 
            self.slope_variance,
            self.seasonal_period,
            self.seasonal_variance
        )
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        seasonal_str = f", seasonal_period={self.seasonal_period}" if self.has_seasonal else ""
        return (f"LocalLinearTrendModel(obs_var={self.observation_variance:.3f}, "
                f"level_var={self.level_variance:.3f}, slope_var={self.slope_variance:.3f}"
                f"{seasonal_str}, time_points={self.time_dimension})")