"""Local level model for time series analysis."""

import numpy as np
from typing import Optional

from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import RNG
from .base import StateModel, StateSpaceModel
from .kalman import KalmanFilter


class LocalLevelStateModel(StateModel):
    """Local level state model: level follows a random walk.
    
    State equation: level_{t+1} = level_t + eta_t
    where eta_t ~ N(0, sigma_level^2)
    
    This represents a slowly evolving level component.
    """
    
    def __init__(self, level_variance: float = 1.0):
        """Initialize local level state model.
        
        Args:
            level_variance: Variance of level innovations
        """
        super().__init__(state_dimension=1)
        self._level_variance = float(level_variance)
        
        if self._level_variance < 0:
            raise ValueError("Level variance must be non-negative")
    
    @property
    def level_variance(self) -> float:
        """Get level variance."""
        return self._level_variance
    
    def set_level_variance(self, variance: float):
        """Set level variance."""
        if variance < 0:
            raise ValueError("Level variance must be non-negative")
        self._level_variance = float(variance)
    
    def transition_matrix(self, t: int) -> Matrix:
        """Get transition matrix (identity for random walk)."""
        return Matrix(np.array([[1.0]]))
    
    def state_error_variance(self, t: int) -> SpdMatrix:
        """Get state error variance."""
        return SpdMatrix(np.array([[self._level_variance]]))
    
    def observation_matrix(self, t: int) -> Matrix:
        """Get observation matrix (observe level directly)."""
        return Matrix(np.array([[1.0]]))
    
    def initial_state_mean(self) -> Vector:
        """Get initial level mean (default: 0)."""
        return Vector([0.0])
    
    def initial_state_variance(self) -> SpdMatrix:
        """Get initial level variance (default: diffuse)."""
        return SpdMatrix(np.array([[1e6]]))


class LocalLevelModel(StateSpaceModel):
    """Local level model for time series with slowly evolving mean.
    
    Model:
    y_t = level_t + epsilon_t,  epsilon_t ~ N(0, sigma_obs^2)
    level_{t+1} = level_t + eta_t,  eta_t ~ N(0, sigma_level^2)
    
    This is the simplest structural time series model, representing
    a time series as a stochastic level plus white noise.
    """
    
    def __init__(self, observation_variance: float = 1.0, level_variance: float = 1.0):
        """Initialize local level model.
        
        Args:
            observation_variance: Observation error variance (sigma_obs^2)
            level_variance: Level innovation variance (sigma_level^2)
        """
        super().__init__(observation_variance)
        
        # Add local level state model
        self._level_model = LocalLevelStateModel(level_variance)
        self.add_state_model(self._level_model)
    
    @property
    def level_variance(self) -> float:
        """Get level variance."""
        return self._level_model.level_variance
    
    def set_level_variance(self, variance: float):
        """Set level variance."""
        self._level_model.set_level_variance(variance)
        self._notify_observers()
    
    def log_likelihood(self) -> float:
        """Compute log likelihood using Kalman filter."""
        if self.time_dimension == 0:
            return 0.0
        
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter(compute_likelihood=True)
        return kalman_filter.log_likelihood()
    
    def fit(self, max_iterations: int = 100, tolerance: float = 1e-6) -> bool:
        """Fit model parameters using maximum likelihood estimation.
        
        Uses EM algorithm to estimate observation and level variances.
        
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
            
            # M-step: Update parameters
            self._m_step(kalman_filter)
        
        return False  # Did not converge
    
    def _m_step(self, kalman_filter: KalmanFilter):
        """M-step of EM algorithm: update parameters."""
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
                smoother_t = smoother_states[t]
                residual = data_t.y() - smoother_t.state_mean[0]
                sse_obs += residual**2 + smoother_t.state_variance[0, 0]
                n_obs += 1
        
        if n_obs > 0:
            new_obs_variance = sse_obs / n_obs
            self.set_observation_variance(max(new_obs_variance, 1e-8))
        
        # Update level variance
        sse_level = 0.0
        n_level = 0
        
        for t in range(1, T):
            smoother_t_minus_1 = smoother_states[t-1]
            smoother_t = smoother_states[t]
            
            # Level innovation: level_t - level_{t-1}
            level_diff = smoother_t.state_mean[0] - smoother_t_minus_1.state_mean[0]
            level_var = (smoother_t.state_variance[0, 0] + 
                        smoother_t_minus_1.state_variance[0, 0])
            
            sse_level += level_diff**2 + level_var
            n_level += 1
        
        if n_level > 0:
            new_level_variance = sse_level / n_level
            self.set_level_variance(max(new_level_variance, 1e-8))
    
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
    
    def extract_level(self) -> tuple[Vector, Vector]:
        """Extract smoothed level estimates.
        
        Returns:
            Tuple of (level_means, level_variances)
        """
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter()
        kalman_filter.smooth()
        
        smoother_states = kalman_filter.smoother_states()
        
        level_means = Vector([state.state_mean[0] for state in smoother_states])
        level_variances = Vector([state.state_variance[0, 0] for state in smoother_states])
        
        return level_means, level_variances
    
    def residuals(self) -> tuple[Vector, Vector]:
        """Compute standardized residuals.
        
        Returns:
            Tuple of (residuals, standardized_residuals)
        """
        kalman_filter = KalmanFilter(self)
        kalman_filter.filter()
        return kalman_filter.residuals()
    
    def simulate_data(self, n: int, initial_level: float = 0.0, 
                     rng: Optional[RNG] = None) -> list:
        """Simulate time series data from the local level model.
        
        Args:
            n: Number of time points to simulate
            initial_level: Initial level value
            rng: Random number generator
            
        Returns:
            List of simulated observations
        """
        if rng is None:
            rng = RNG()
        
        observations = []
        level = initial_level
        
        for t in range(n):
            # Generate observation
            obs_error = np.sqrt(self.observation_variance) * rng.normal()
            y_t = level + obs_error
            observations.append(y_t)
            
            # Evolve level (random walk)
            level_error = np.sqrt(self.level_variance) * rng.normal()
            level += level_error
        
        return observations
    
    def clone(self) -> 'LocalLevelModel':
        """Create a copy of this model."""
        cloned = LocalLevelModel(self.observation_variance, self.level_variance)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"LocalLevelModel(obs_var={self.observation_variance:.3f}, "
                f"level_var={self.level_variance:.3f}, time_points={self.time_dimension})")