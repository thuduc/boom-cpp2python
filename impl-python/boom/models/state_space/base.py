"""Base classes for state space models."""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from abc import ABC, abstractmethod

from boom.models.base import Model
from boom.models.data import Data
from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import RNG


class TimeSeriesData(Data):
    """Data point for time series observations."""
    
    def __init__(self, y: float, timestamp: Optional[int] = None, is_observed: bool = True):
        """Initialize time series data point.
        
        Args:
            y: Observed value
            timestamp: Time index (optional)
            is_observed: Whether the observation is observed or missing
        """
        super().__init__()
        self._y = float(y) if not np.isnan(y) else np.nan
        self._timestamp = timestamp
        self._is_observed = is_observed and not np.isnan(y)
    
    def y(self) -> float:
        """Get observed value."""
        return self._y
    
    def timestamp(self) -> Optional[int]:
        """Get timestamp."""
        return self._timestamp
    
    def is_observed(self) -> bool:
        """Check if observation is observed (not missing)."""
        return self._is_observed
    
    def set_y(self, y: float):
        """Set observed value."""
        self._y = float(y) if not np.isnan(y) else np.nan
        self._is_observed = not np.isnan(self._y)
        # Note: Observer pattern not implemented for simple data classes
    
    def set_timestamp(self, timestamp: Optional[int]):
        """Set timestamp."""
        self._timestamp = timestamp
        # Note: Observer pattern not implemented for simple data classes
    
    def clone(self) -> 'TimeSeriesData':
        """Create a copy of this data point."""
        return TimeSeriesData(self._y, self._timestamp, self._is_observed)
    
    def __repr__(self) -> str:
        return f"TimeSeriesData(y={self._y}, timestamp={self._timestamp}, observed={self._is_observed})"


class StateModel(ABC):
    """Base class for state models that describe state evolution."""
    
    def __init__(self, state_dimension: int):
        """Initialize state model.
        
        Args:
            state_dimension: Dimension of the state vector
        """
        self._state_dimension = state_dimension
    
    @property
    def state_dimension(self) -> int:
        """Get state dimension."""
        return self._state_dimension
    
    @abstractmethod
    def transition_matrix(self, t: int) -> Matrix:
        """Get transition matrix T_t at time t.
        
        Args:
            t: Time index
            
        Returns:
            Transition matrix T_t such that state_{t+1} = T_t * state_t + error_t
        """
        pass
    
    @abstractmethod
    def state_error_variance(self, t: int) -> SpdMatrix:
        """Get state error variance Q_t at time t.
        
        Args:
            t: Time index
            
        Returns:
            State error covariance matrix Q_t
        """
        pass
    
    @abstractmethod
    def observation_matrix(self, t: int) -> Matrix:
        """Get observation matrix Z_t at time t.
        
        Args:
            t: Time index
            
        Returns:
            Observation matrix Z_t such that y_t = Z_t * state_t + obs_error_t
        """
        pass
    
    def state_error_dimension(self) -> int:
        """Get dimension of state error term (default: same as state dimension)."""
        return self._state_dimension
    
    def simulate_state_error(self, t: int, rng: Optional[RNG] = None) -> Vector:
        """Simulate state error at time t.
        
        Args:
            t: Time index
            rng: Random number generator
            
        Returns:
            Simulated state error vector
        """
        if rng is None:
            rng = RNG()
        
        Q = self.state_error_variance(t)
        error_dim = self.state_error_dimension()
        
        # Generate standard normal random vector
        z = Vector([rng.randn() for _ in range(error_dim)])
        
        # Transform to have covariance Q
        if error_dim == 1:
            return Vector([z[0] * np.sqrt(Q[0, 0])])
        else:
            # Use Cholesky decomposition for multivariate case
            try:
                L = np.linalg.cholesky(Q.to_numpy())
                return Vector(L @ z.to_numpy())
            except np.linalg.LinAlgError:
                # If not positive definite, use eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(Q.to_numpy())
                eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
                sqrt_Q = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
                return Vector(sqrt_Q @ z.to_numpy())
    
    def initial_state_mean(self) -> Vector:
        """Get initial state mean (default: zero vector)."""
        return Vector(np.zeros(self._state_dimension))
    
    def initial_state_variance(self) -> SpdMatrix:
        """Get initial state variance (default: large diagonal matrix)."""
        return SpdMatrix(1e6 * np.eye(self._state_dimension))


class StateSpaceModel(Model, ABC):
    """Base class for state space models.
    
    A state space model has the form:
    y_t = Z_t * alpha_t + epsilon_t    (observation equation)
    alpha_{t+1} = T_t * alpha_t + eta_t  (state equation)
    
    where:
    - y_t is the observation at time t
    - alpha_t is the state vector at time t
    - Z_t is the observation matrix
    - T_t is the transition matrix
    - epsilon_t ~ N(0, H_t) is observation noise
    - eta_t ~ N(0, Q_t) is state noise
    """
    
    def __init__(self, observation_variance: float = 1.0):
        """Initialize state space model.
        
        Args:
            observation_variance: Observation error variance
        """
        super().__init__()
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive")
        self._observation_variance = float(observation_variance)
        self._data: List[TimeSeriesData] = []
        self._state_models: List[StateModel] = []
    
    @property
    def observation_variance(self) -> float:
        """Get observation variance."""
        return self._observation_variance
    
    def set_observation_variance(self, variance: float):
        """Set observation variance."""
        if variance <= 0:
            raise ValueError("Observation variance must be positive")
        self._observation_variance = float(variance)
        self._notify_observers()
    
    @property
    def state_dimension(self) -> int:
        """Get total state dimension (sum of all state models)."""
        return sum(model.state_dimension for model in self._state_models)
    
    @property
    def time_dimension(self) -> int:
        """Get number of time points."""
        return len(self._data)
    
    def add_state_model(self, state_model: StateModel):
        """Add a state model component."""
        self._state_models.append(state_model)
    
    def state_model(self, index: int) -> StateModel:
        """Get state model by index."""
        return self._state_models[index]
    
    def number_of_state_models(self) -> int:
        """Get number of state models."""
        return len(self._state_models)
    
    def add_data(self, data: Union[TimeSeriesData, float, List[Union[TimeSeriesData, float]]]):
        """Add time series data."""
        if isinstance(data, TimeSeriesData):
            self._add_single_data(data)
        elif isinstance(data, (int, float)):
            ts_data = TimeSeriesData(float(data))
            self._add_single_data(ts_data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, TimeSeriesData):
                    self._add_single_data(item)
                elif isinstance(item, (int, float)):
                    ts_data = TimeSeriesData(float(item))
                    self._add_single_data(ts_data)
                else:
                    raise ValueError(f"Invalid data item: {item}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    def _add_single_data(self, data: TimeSeriesData):
        """Add a single time series data point."""
        if data.timestamp() is None:
            data.set_timestamp(len(self._data))
        self._data.append(data)
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
    
    def get_data(self, t: int) -> TimeSeriesData:
        """Get data at time t."""
        return self._data[t]
    
    def observations(self) -> Vector:
        """Get all observations as a vector."""
        return Vector([data.y() for data in self._data])
    
    def observed_mask(self) -> List[bool]:
        """Get mask indicating which observations are observed."""
        return [data.is_observed() for data in self._data]
    
    def transition_matrix(self, t: int) -> Matrix:
        """Get combined transition matrix at time t."""
        if not self._state_models:
            return Matrix(np.eye(1))
        
        # For now, assume block diagonal structure
        blocks = [model.transition_matrix(t) for model in self._state_models]
        
        if len(blocks) == 1:
            return blocks[0]
        
        # Create block diagonal matrix
        total_dim = sum(block.nrow() for block in blocks)
        combined = np.zeros((total_dim, total_dim))
        
        row_offset = 0
        col_offset = 0
        for block in blocks:
            block_array = block.to_numpy()
            combined[row_offset:row_offset+block.nrow(), 
                    col_offset:col_offset+block.ncol()] = block_array
            row_offset += block.nrow()
            col_offset += block.ncol()
        
        return Matrix(combined)
    
    def state_error_variance(self, t: int) -> SpdMatrix:
        """Get combined state error variance at time t."""
        if not self._state_models:
            return SpdMatrix(np.eye(1))
        
        # For now, assume block diagonal structure
        blocks = [model.state_error_variance(t) for model in self._state_models]
        
        if len(blocks) == 1:
            return blocks[0]
        
        # Create block diagonal matrix
        total_dim = sum(block.nrow() for block in blocks)
        combined = np.zeros((total_dim, total_dim))
        
        row_offset = 0
        col_offset = 0
        for block in blocks:
            block_array = block.to_numpy()
            combined[row_offset:row_offset+block.nrow(), 
                    col_offset:col_offset+block.ncol()] = block_array
            row_offset += block.nrow()
            col_offset += block.ncol()
        
        return SpdMatrix(combined)
    
    def observation_matrix(self, t: int) -> Matrix:
        """Get combined observation matrix at time t.
        
        Default implementation assumes first state component is observed directly.
        """
        if not self._state_models:
            return Matrix(np.array([[1.0]]))
        
        # Default: observe first component of first state model
        total_dim = self.state_dimension
        Z = np.zeros((1, total_dim))
        Z[0, 0] = 1.0
        
        return Matrix(Z)
    
    def initial_state_mean(self) -> Vector:
        """Get initial state mean."""
        if not self._state_models:
            return Vector([0.0])
        
        means = [model.initial_state_mean() for model in self._state_models]
        combined = np.concatenate([mean.to_numpy() for mean in means])
        return Vector(combined)
    
    def initial_state_variance(self) -> SpdMatrix:
        """Get initial state variance."""
        if not self._state_models:
            return SpdMatrix(np.array([[1e6]]))
        
        # Block diagonal structure
        blocks = [model.initial_state_variance() for model in self._state_models]
        
        if len(blocks) == 1:
            return blocks[0]
        
        total_dim = sum(block.nrow() for block in blocks)
        combined = np.zeros((total_dim, total_dim))
        
        row_offset = 0
        col_offset = 0
        for block in blocks:
            block_array = block.to_numpy()
            combined[row_offset:row_offset+block.nrow(), 
                    col_offset:col_offset+block.ncol()] = block_array
            row_offset += block.nrow()
            col_offset += block.ncol()
        
        return SpdMatrix(combined)
    
    def simulate_data(self, n: int, rng: Optional[RNG] = None) -> List[TimeSeriesData]:
        """Simulate time series data from the model.
        
        Args:
            n: Number of time points to simulate
            rng: Random number generator
            
        Returns:
            List of simulated time series data
        """
        if rng is None:
            rng = RNG()
        
        if not self._state_models:
            raise ValueError("No state models added")
        
        simulated_data = []
        
        # Initialize state
        state = self.initial_state_mean()
        
        for t in range(n):
            # Observation equation: y_t = Z_t * alpha_t + epsilon_t
            Z = self.observation_matrix(t)
            mean_obs = float((Z @ state.to_numpy())[0])
            obs_error = np.sqrt(self._observation_variance) * rng.normal()
            y_t = mean_obs + obs_error
            
            simulated_data.append(TimeSeriesData(y_t, timestamp=t))
            
            # State equation: alpha_{t+1} = T_t * alpha_t + eta_t
            if t < n - 1:  # Don't evolve state after last observation
                T = self.transition_matrix(t)
                state_error = Vector([model.simulate_state_error(t, rng) for model in self._state_models])
                state_error_combined = np.concatenate([err.to_numpy() for err in state_error])
                
                new_state = T @ state.to_numpy() + state_error_combined
                state = Vector(new_state)
        
        return simulated_data
    
    @abstractmethod
    def log_likelihood(self) -> float:
        """Compute log likelihood of the model."""
        pass
    
    def clone(self) -> 'StateSpaceModel':
        """Create a copy of this model."""
        cloned = self.__class__(self._observation_variance)
        
        # Copy state models (need to implement clone for each state model)
        for state_model in self._state_models:
            # For now, just copy the reference (would need proper cloning)
            cloned.add_state_model(state_model)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(state_dim={self.state_dimension}, "
                f"obs_var={self._observation_variance:.3f}, time_points={self.time_dimension})")