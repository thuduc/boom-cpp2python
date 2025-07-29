"""Local linear trend model component."""
import numpy as np
from .base import StateComponent
from ..base import PositiveParameter
from ...linalg import Vector, Matrix, SpdMatrix


class LocalLinearTrendModel(StateComponent):
    """Local linear trend model with level and slope.
    
    State equation:
    [alpha_t]   [1 1] [alpha_{t-1}]   [eta_alpha_t]
    [beta_t ] = [0 1] [beta_{t-1} ] + [eta_beta_t ]
    
    where alpha_t is the level and beta_t is the slope.
    """
    
    def __init__(self, level_sigma: float = 1.0, slope_sigma: float = 1.0, 
                 name: str = "trend"):
        """Initialize local linear trend model.
        
        Args:
            level_sigma: Standard deviation of level innovations
            slope_sigma: Standard deviation of slope innovations  
            name: Component name
        """
        super().__init__(name)
        self._level_sigma_param = PositiveParameter(level_sigma, f"{name}_level_sigma")
        self._slope_sigma_param = PositiveParameter(slope_sigma, f"{name}_slope_sigma")
        self._state_dimension = 2
    
    @property
    def level_sigma(self) -> float:
        """Get level innovation standard deviation."""
        return self._level_sigma_param.value
    
    @level_sigma.setter
    def level_sigma(self, value: float):
        """Set level innovation standard deviation."""
        if value <= 0:
            raise ValueError("Level sigma must be positive")
        self._level_sigma_param.value = value
    
    @property
    def slope_sigma(self) -> float:
        """Get slope innovation standard deviation."""
        return self._slope_sigma_param.value
    
    @slope_sigma.setter
    def slope_sigma(self, value: float):
        """Set slope innovation standard deviation."""
        if value <= 0:
            raise ValueError("Slope sigma must be positive")
        self._slope_sigma_param.value = value
    
    @property
    def level_variance(self) -> float:
        """Get level innovation variance."""
        return self.level_sigma ** 2
    
    @level_variance.setter
    def level_variance(self, value: float):
        """Set level innovation variance."""
        if value <= 0:
            raise ValueError("Level variance must be positive")
        self.level_sigma = np.sqrt(value)
    
    @property
    def slope_variance(self) -> float:
        """Get slope innovation variance."""
        return self.slope_sigma ** 2
    
    @slope_variance.setter
    def slope_variance(self, value: float):
        """Set slope innovation variance."""
        if value <= 0:
            raise ValueError("Slope variance must be positive")
        self.slope_sigma = np.sqrt(value)
    
    def state_dimension(self) -> int:
        """Return state dimension (2 for level and slope)."""
        return 2
    
    def transition_matrix(self, t: int) -> Matrix:
        """Return transition matrix [[1, 1], [0, 1]]."""
        return Matrix([[1.0, 1.0],
                      [0.0, 1.0]])
    
    def state_variance(self, t: int) -> SpdMatrix:
        """Return state innovation variance matrix."""
        return SpdMatrix([[self.level_variance, 0.0],
                         [0.0, self.slope_variance]])
    
    def observation_matrix(self, t: int) -> Matrix:
        """Return observation matrix [1, 0] (only level is observed)."""
        return Matrix([[1.0, 0.0]])
    
    def initial_state_mean(self) -> Vector:
        """Return initial state mean [0, 0]."""
        return Vector([0.0, 0.0])
    
    def initial_state_variance(self) -> SpdMatrix:
        """Return initial state variance (diffuse prior)."""
        return SpdMatrix([[1e6, 0.0],
                         [0.0, 1e6]])
    
    def level_contribution(self, state: Vector) -> float:
        """Get the level contribution.
        
        Args:
            state: Full state vector
            
        Returns:
            Level value
        """
        return float(state[0])
    
    def slope_contribution(self, state: Vector) -> float:
        """Get the slope contribution.
        
        Args:
            state: Full state vector
            
        Returns:
            Slope value
        """
        return float(state[1])
    
    def contribution(self, state: Vector) -> float:
        """Get the total contribution (level only, since slope is not directly observed).
        
        Args:
            state: Full state vector
            
        Returns:
            Level contribution
        """
        return self.level_contribution(state)
    
    def clone(self) -> 'LocalLinearTrendModel':
        """Create a copy of the component."""
        return LocalLinearTrendModel(self.level_sigma, self.slope_sigma, self.name)