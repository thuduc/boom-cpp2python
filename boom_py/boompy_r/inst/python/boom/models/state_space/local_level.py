"""Local level model (random walk) component."""
import numpy as np
from .base import StateComponent
from ..base import PositiveParameter
from ...linalg import Vector, Matrix, SpdMatrix


class LocalLevelModel(StateComponent):
    """Local level model: alpha_t = alpha_{t-1} + eta_t.
    
    This represents a random walk component where the level
    can change randomly over time.
    """
    
    def __init__(self, sigma: float = 1.0, name: str = "level"):
        """Initialize local level model.
        
        Args:
            sigma: Standard deviation of level innovations
            name: Component name
        """
        super().__init__(name)
        self._sigma_param = PositiveParameter(sigma, f"{name}_sigma")
        self._state_dimension = 1
    
    @property
    def sigma(self) -> float:
        """Get innovation standard deviation."""
        return self._sigma_param.value
    
    @sigma.setter
    def sigma(self, value: float):
        """Set innovation standard deviation."""
        if value <= 0:
            raise ValueError("Sigma must be positive")
        self._sigma_param.value = value
    
    @property
    def variance(self) -> float:
        """Get innovation variance."""
        return self.sigma ** 2
    
    @variance.setter
    def variance(self, value: float):
        """Set innovation variance."""
        if value <= 0:
            raise ValueError("Variance must be positive")
        self.sigma = np.sqrt(value)
    
    def state_dimension(self) -> int:
        """Return state dimension (1 for local level)."""
        return 1
    
    def transition_matrix(self, t: int) -> Matrix:
        """Return transition matrix T = [1]."""
        return Matrix([[1.0]])
    
    def state_variance(self, t: int) -> SpdMatrix:
        """Return state innovation variance Q = [sigma^2]."""
        return SpdMatrix([[self.variance]])
    
    def observation_matrix(self, t: int) -> Matrix:
        """Return observation matrix Z = [1]."""
        return Matrix([[1.0]])
    
    def initial_state_mean(self) -> Vector:
        """Return initial state mean (0)."""
        return Vector([0.0])
    
    def initial_state_variance(self) -> SpdMatrix:
        """Return initial state variance (large value for diffuse prior)."""
        return SpdMatrix([[1e6]])
    
    def contribution(self, state: Vector) -> float:
        """Get the contribution of this component to the observation.
        
        Args:
            state: Full state vector
            
        Returns:
            Contribution (just the level value)
        """
        return float(state[0])
    
    def clone(self) -> 'LocalLevelModel':
        """Create a copy of the component."""
        return LocalLevelModel(self.sigma, self.name)