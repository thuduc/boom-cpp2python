"""Base classes for state space models."""
import numpy as np
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from ..base import Model, VectorParameter, PositiveParameter
from ...linalg import Vector, Matrix, SpdMatrix


class StateComponent(ABC):
    """Base class for state space components."""
    
    def __init__(self, name: str = ""):
        """Initialize state component.
        
        Args:
            name: Name of the component
        """
        self.name = name
        self._state_dimension = None
    
    @abstractmethod
    def state_dimension(self) -> int:
        """Return the dimension of the state vector for this component."""
        pass
    
    @abstractmethod
    def transition_matrix(self, t: int) -> Matrix:
        """Return the state transition matrix at time t.
        
        Args:
            t: Time index
            
        Returns:
            Transition matrix T_t
        """
        pass
    
    @abstractmethod
    def state_variance(self, t: int) -> SpdMatrix:
        """Return the state innovation variance at time t.
        
        Args:
            t: Time index
            
        Returns:
            State variance Q_t
        """
        pass
    
    @abstractmethod
    def observation_matrix(self, t: int) -> Matrix:
        """Return the observation matrix at time t.
        
        Args:
            t: Time index
            
        Returns:
            Observation matrix Z_t
        """
        pass
    
    def initial_state_mean(self) -> Vector:
        """Return the initial state mean."""
        return Vector.zero(self.state_dimension())
    
    def initial_state_variance(self) -> SpdMatrix:
        """Return the initial state variance."""
        return SpdMatrix.identity(self.state_dimension())
    
    def simulate_initial_state(self, rng) -> Vector:
        """Simulate initial state."""
        mean = self.initial_state_mean()
        var = self.initial_state_variance()
        return Vector(rng.rmvn(mean, var))
    
    def simulate_state_innovation(self, t: int, rng) -> Vector:
        """Simulate state innovation at time t."""
        var = self.state_variance(t)
        return Vector(rng.rmvn(Vector.zero(self.state_dimension()), var))


class StateSpaceModel(Model):
    """Base class for state space models."""
    
    def __init__(self, observation_variance: float = 1.0):
        """Initialize state space model.
        
        Args:
            observation_variance: Observation noise variance
        """
        super().__init__()
        self.components: List[StateComponent] = []
        self._params['observation_variance'] = PositiveParameter(observation_variance, 'observation_variance')
        self._observations = []
        self._time_dimension = 0
    
    @property
    def observation_variance(self) -> float:
        """Get observation variance."""
        return self._params['observation_variance'].value
    
    @observation_variance.setter
    def observation_variance(self, value: float):
        """Set observation variance."""
        if value <= 0:
            raise ValueError("Observation variance must be positive")
        self._params['observation_variance'].value = value
    
    def add_component(self, component: StateComponent):
        """Add a state component to the model."""
        self.components.append(component)
    
    def state_dimension(self) -> int:
        """Total state dimension across all components."""
        return sum(comp.state_dimension() for comp in self.components)
    
    def add_observation(self, y: float, time_index: Optional[int] = None):
        """Add an observation.
        
        Args:
            y: Observed value
            time_index: Time index (if None, uses sequential indexing)
        """
        if time_index is None:
            time_index = len(self._observations)
        
        self._observations.append((time_index, y))
        self._time_dimension = max(self._time_dimension, time_index + 1)
    
    def set_observations(self, y: Union[List[float], np.ndarray]):
        """Set all observations.
        
        Args:
            y: Array of observations
        """
        self._observations = [(i, float(val)) for i, val in enumerate(y)]
        self._time_dimension = len(y)
    
    def get_observations(self) -> np.ndarray:
        """Get observations as array."""
        if not self._observations:
            return np.array([])
        
        # Sort by time index
        sorted_obs = sorted(self._observations)
        return np.array([obs[1] for obs in sorted_obs])
    
    def transition_matrix(self, t: int) -> Matrix:
        """Combined transition matrix at time t."""
        if not self.components:
            return Matrix.identity(0)
        
        # Block diagonal matrix of component transitions
        blocks = [comp.transition_matrix(t) for comp in self.components]
        return Matrix(self._block_diagonal(blocks))
    
    def state_variance(self, t: int) -> SpdMatrix:
        """Combined state variance at time t."""
        if not self.components:
            return SpdMatrix.identity(0)
        
        # Block diagonal matrix of component variances
        blocks = [comp.state_variance(t) for comp in self.components]
        return SpdMatrix(self._block_diagonal(blocks))
    
    def observation_matrix(self, t: int) -> Matrix:
        """Combined observation matrix at time t."""
        if not self.components:
            return Matrix.zero(1, 0)
        
        # Horizontal concatenation of component observation matrices
        matrices = [comp.observation_matrix(t) for comp in self.components]
        return Matrix(np.hstack(matrices))
    
    def initial_state_mean(self) -> Vector:
        """Combined initial state mean."""
        if not self.components:
            return Vector([])
        
        means = [comp.initial_state_mean() for comp in self.components]
        return Vector(np.concatenate(means))
    
    def initial_state_variance(self) -> SpdMatrix:
        """Combined initial state variance."""
        if not self.components:
            return SpdMatrix.identity(0)
        
        blocks = [comp.initial_state_variance() for comp in self.components]
        return SpdMatrix(self._block_diagonal(blocks))
    
    def simulate_state_sequence(self, n_time_points: int, rng) -> Matrix:
        """Simulate a sequence of states.
        
        Args:
            n_time_points: Number of time points
            rng: Random number generator
            
        Returns:
            Matrix of states (time x state_dim)
        """
        state_dim = self.state_dimension()
        if state_dim == 0:
            return Matrix.zero(n_time_points, 0)
        
        states = Matrix.zero(n_time_points, state_dim)
        
        # Initial state
        if n_time_points > 0:
            states[0, :] = self.simulate_initial_state(rng)
        
        # Subsequent states
        for t in range(1, n_time_points):
            T = self.transition_matrix(t)
            innovation = self.simulate_state_innovation(t, rng)
            states[t, :] = T @ states[t-1, :] + innovation
        
        return Matrix(states)
    
    def simulate_observations(self, n_time_points: int, rng) -> Vector:
        """Simulate observations from the model.
        
        Args:
            n_time_points: Number of time points
            rng: Random number generator
            
        Returns:
            Vector of simulated observations
        """
        states = self.simulate_state_sequence(n_time_points, rng)
        observations = Vector.zero(n_time_points)
        
        for t in range(n_time_points):
            Z = self.observation_matrix(t)
            mean_obs = float((Z @ states.row(t))[0]) if Z.ncol() > 0 else 0.0
            observations[t] = rng.rnorm(mean_obs, np.sqrt(self.observation_variance))
        
        return observations
    
    def simulate_initial_state(self, rng) -> Vector:
        """Simulate initial state from all components."""
        if not self.components:
            return Vector([])
        
        states = []
        for comp in self.components:
            states.append(comp.simulate_initial_state(rng))
        
        return Vector(np.concatenate(states))
    
    def simulate_state_innovation(self, t: int, rng) -> Vector:
        """Simulate state innovation from all components."""
        if not self.components:
            return Vector([])
        
        innovations = []
        for comp in self.components:
            innovations.append(comp.simulate_state_innovation(t, rng))
        
        return Vector(np.concatenate(innovations))
    
    def loglike(self) -> float:
        """Compute log likelihood using Kalman filter."""
        from .kalman import KalmanFilter
        
        if not self._observations:
            return 0.0
        
        y = self.get_observations()
        kf = KalmanFilter(self)
        return kf.log_likelihood(y)
    
    def _block_diagonal(self, blocks: List[np.ndarray]) -> np.ndarray:
        """Create block diagonal matrix from list of blocks."""
        if not blocks:
            return np.array([[]])
        
        # Handle scalar blocks
        blocks = [np.atleast_2d(block) for block in blocks]
        
        total_rows = sum(block.shape[0] for block in blocks)
        total_cols = sum(block.shape[1] for block in blocks)
        
        result = np.zeros((total_rows, total_cols))
        
        row_start = 0
        col_start = 0
        
        for block in blocks:
            rows, cols = block.shape
            result[row_start:row_start+rows, col_start:col_start+cols] = block
            row_start += rows
            col_start += cols
        
        return result
    
    def add_data(self, data):
        """Add data to the model (for Model interface compatibility)."""
        if isinstance(data, (list, tuple)) and len(data) == 2:
            time_idx, obs = data
            self.add_observation(obs, time_idx)
        else:
            # Assume it's just an observation
            self.add_observation(float(data))
    
    def clear_data(self):
        """Clear all observations."""
        self._observations = []
        self._time_dimension = 0
    
    def simulate(self, n_time_points: int, rng=None):
        """Simulate data from the model."""
        if rng is None:
            from ...distributions import rng as global_rng
            rng = global_rng
        
        return self.simulate_observations(n_time_points, rng)
    
    def clone(self) -> 'StateSpaceModel':
        """Create a copy of the model."""
        model = StateSpaceModel(self.observation_variance)
        
        # Copy components (assuming they are immutable or have clone methods)
        for comp in self.components:
            model.add_component(comp)
        
        # Copy observations
        for time_idx, obs in self._observations:
            model.add_observation(obs, time_idx)
        
        return model