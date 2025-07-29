"""Seasonal component for state space models."""
import numpy as np
from .base import StateComponent
from ..base import PositiveParameter
from ...linalg import Vector, Matrix, SpdMatrix


class SeasonalModel(StateComponent):
    """Seasonal component with trigonometric representation.
    
    For period S, uses S/2 (or (S-1)/2 for odd S) harmonic components.
    Each harmonic has a sine and cosine component that evolve according to:
    
    [s_{j,t}  ]   [cos(2πj/S)  sin(2πj/S)] [s_{j,t-1}  ]   [w_{j,t}  ]
    [s*_{j,t} ] = [-sin(2πj/S) cos(2πj/S)] [s*_{j,t-1} ] + [w*_{j,t} ]
    """
    
    def __init__(self, period: int, sigma: float = 1.0, name: str = "seasonal"):
        """Initialize seasonal component.
        
        Args:
            period: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
            sigma: Standard deviation of seasonal innovations
            name: Component name
        """
        super().__init__(name)
        if period < 2:
            raise ValueError("Period must be at least 2")
        
        self.period = period
        self._sigma_param = PositiveParameter(sigma, f"{name}_sigma")
        
        # Number of harmonic frequencies (excluding the Nyquist frequency for even periods)
        self.n_harmonics = period // 2
        if period % 2 == 0:
            # For even periods, the Nyquist frequency component is handled separately
            self._state_dimension = 2 * (self.n_harmonics - 1) + 1
        else:
            self._state_dimension = 2 * self.n_harmonics
    
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
        """Return state dimension."""
        return self._state_dimension
    
    def transition_matrix(self, t: int) -> Matrix:
        """Return seasonal transition matrix."""
        dim = self.state_dimension()
        T = Matrix.zero(dim, dim)
        
        idx = 0
        for j in range(1, self.n_harmonics):
            # Frequency for this harmonic
            freq = 2 * np.pi * j / self.period
            cos_freq = np.cos(freq)
            sin_freq = np.sin(freq)
            
            # 2x2 rotation matrix for this harmonic
            T[idx, idx] = cos_freq
            T[idx, idx+1] = sin_freq
            T[idx+1, idx] = -sin_freq
            T[idx+1, idx+1] = cos_freq
            
            idx += 2
        
        # Handle Nyquist frequency for even periods
        if self.period % 2 == 0 and idx < dim:
            T[idx, idx] = -1.0  # Nyquist frequency alternates sign
        
        return T
    
    def state_variance(self, t: int) -> SpdMatrix:
        """Return state innovation variance matrix."""
        dim = self.state_dimension()
        return SpdMatrix(self.variance * np.eye(dim))
    
    def observation_matrix(self, t: int) -> Matrix:
        """Return observation matrix (sum of all seasonal components)."""
        dim = self.state_dimension()
        Z = Matrix.zero(1, dim)
        
        # First component of each harmonic contributes to observation
        idx = 0
        for j in range(1, self.n_harmonics):
            Z[0, idx] = 1.0  # First component of harmonic
            idx += 2
        
        # Handle Nyquist frequency for even periods
        if self.period % 2 == 0 and idx < dim:
            Z[0, idx] = 1.0
        
        return Z
    
    def initial_state_mean(self) -> Vector:
        """Return initial state mean (zeros)."""
        return Vector.zero(self.state_dimension())
    
    def initial_state_variance(self) -> SpdMatrix:
        """Return initial state variance (diffuse prior)."""
        dim = self.state_dimension()
        
        # For seasonal components, use a reasonably large but finite variance
        # to represent lack of knowledge about initial seasonal pattern
        initial_var = 100.0  # Much smaller than trend components
        return SpdMatrix(initial_var * np.eye(dim))
    
    def seasonal_pattern(self, n_periods: int = 1) -> Vector:
        """Generate the seasonal pattern over n_periods.
        
        Args:
            n_periods: Number of complete seasonal periods to generate
            
        Returns:
            Vector of seasonal values over time
        """
        n_time = n_periods * self.period
        pattern = Vector.zero(n_time)
        
        for t in range(n_time):
            # Sum contributions from all harmonics
            seasonal_value = 0.0
            
            for j in range(1, self.n_harmonics):
                freq = 2 * np.pi * j / self.period
                # Assume unit amplitude for each harmonic
                seasonal_value += np.cos(freq * t)
            
            # Handle Nyquist frequency for even periods
            if self.period % 2 == 0:
                seasonal_value += (-1) ** t
            
            pattern[t] = seasonal_value
        
        return pattern
    
    def contribution(self, state: Vector) -> float:
        """Get the seasonal contribution.
        
        Args:
            state: Full state vector for this component
            
        Returns:
            Seasonal contribution
        """
        # Sum the first component of each harmonic
        contribution = 0.0
        idx = 0
        
        for j in range(1, self.n_harmonics):
            contribution += state[idx]
            idx += 2
        
        # Handle Nyquist frequency for even periods
        if self.period % 2 == 0 and idx < len(state):
            contribution += state[idx]
        
        return contribution
    
    def clone(self) -> 'SeasonalModel':
        """Create a copy of the component."""
        return SeasonalModel(self.period, self.sigma, self.name)