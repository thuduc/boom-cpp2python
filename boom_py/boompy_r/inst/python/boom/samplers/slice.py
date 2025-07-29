"""Slice samplers for BOOM."""
import numpy as np
from typing import Callable, Optional, Tuple
from .base import Sampler
from ..distributions.rng import GlobalRng


class SliceSampler(Sampler):
    """Univariate slice sampler."""
    
    def __init__(self, target_log_density: Callable[[float], float],
                 lower_bound: float = -np.inf,
                 upper_bound: float = np.inf,
                 step_size: float = 1.0,
                 max_steps: int = 100,
                 rng: Optional[GlobalRng] = None):
        """Initialize slice sampler.
        
        Args:
            target_log_density: Log density function (univariate)
            lower_bound: Lower bound for support
            upper_bound: Upper bound for support
            step_size: Initial step size for finding slice
            max_steps: Maximum steps for expanding slice
            rng: Random number generator
        """
        super().__init__(target_log_density)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.max_steps = max_steps
        self.rng = rng or GlobalRng()
        
        # Current state (scalar)
        self.current_value = None
        self.current_log_density = None
    
    def set_initial_value(self, value: float):
        """Set initial value."""
        self.current_value = float(value)
        self.current_log_density = self.target_log_density(self.current_value)
    
    def draw(self) -> float:
        """Draw one sample using slice sampling."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Step 1: Draw slice level
        slice_level = (self.current_log_density + 
                      np.log(self.rng.runif()))
        
        # Step 2: Find slice interval
        left, right = self._find_slice_interval(slice_level)
        
        # Step 3: Sample from slice
        while True:
            # Draw candidate point
            candidate = self.rng.runif(left, right)
            
            # Check if candidate is in slice
            try:
                candidate_log_density = self.target_log_density(candidate)
            except:
                candidate_log_density = -np.inf
            
            if candidate_log_density >= slice_level:
                # Accept candidate
                self.current_value = candidate
                self.current_log_density = candidate_log_density
                self.n_accepted += 1
                break
            else:
                # Shrink interval
                if candidate < self.current_value:
                    left = candidate
                else:
                    right = candidate
        
        self.n_draws += 1
        return self.current_value
    
    def _find_slice_interval(self, slice_level: float) -> Tuple[float, float]:
        """Find interval containing the slice."""
        # Initial interval
        left = max(self.current_value - self.step_size * self.rng.runif(), 
                  self.lower_bound)
        right = min(self.current_value + self.step_size * self.rng.runif(),
                   self.upper_bound)
        
        # Expand left
        steps = 0
        while (steps < self.max_steps and 
               left > self.lower_bound):
            try:
                left_log_density = self.target_log_density(left)
            except:
                left_log_density = -np.inf
            
            if left_log_density < slice_level:
                break
            
            left = max(left - self.step_size, self.lower_bound)
            steps += 1
        
        # Expand right
        steps = 0
        while (steps < self.max_steps and 
               right < self.upper_bound):
            try:
                right_log_density = self.target_log_density(right)
            except:
                right_log_density = -np.inf
            
            if right_log_density < slice_level:
                break
            
            right = min(right + self.step_size, self.upper_bound)
            steps += 1
        
        return left, right


class DoubleSliceSampler(Sampler):
    """Slice sampler with doubling procedure."""
    
    def __init__(self, target_log_density: Callable[[float], float],
                 step_size: float = 1.0,
                 max_doublings: int = 10,
                 rng: Optional[GlobalRng] = None):
        """Initialize doubling slice sampler.
        
        Args:
            target_log_density: Log density function
            step_size: Initial step size
            max_doublings: Maximum number of doublings
            rng: Random number generator
        """
        super().__init__(target_log_density)
        self.step_size = step_size
        self.max_doublings = max_doublings
        self.rng = rng or GlobalRng()
        
        self.current_value = None
        self.current_log_density = None
    
    def set_initial_value(self, value: float):
        """Set initial value."""
        self.current_value = float(value)
        self.current_log_density = self.target_log_density(self.current_value)
    
    def draw(self) -> float:
        """Draw using doubling slice sampling."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Draw slice level
        slice_level = (self.current_log_density + 
                      np.log(self.rng.runif()))
        
        # Create initial interval
        u = self.rng.runif()
        left = self.current_value - self.step_size * u
        right = self.current_value + self.step_size * (1 - u)
        
        # Double until bounds are outside slice
        for _ in range(self.max_doublings):
            left_ok = True
            right_ok = True
            
            try:
                if self.target_log_density(left) >= slice_level:
                    left_ok = False
            except:
                pass
            
            try:
                if self.target_log_density(right) >= slice_level:
                    right_ok = False
            except:
                pass
            
            if left_ok and right_ok:
                break
            
            # Double the interval
            if self.rng.runif() < 0.5:
                left = left - (right - left)
            else:
                right = right + (right - left)
        
        # Sample from interval with shrinkage
        while True:
            candidate = self.rng.runif(left, right)
            
            try:
                candidate_log_density = self.target_log_density(candidate)
            except:
                candidate_log_density = -np.inf
            
            if candidate_log_density >= slice_level:
                # Accept
                self.current_value = candidate
                self.current_log_density = candidate_log_density
                self.n_accepted += 1
                break
            else:
                # Shrink
                if candidate < self.current_value:
                    left = candidate
                else:
                    right = candidate
        
        self.n_draws += 1
        return self.current_value


class MultivariateSliceSampler(Sampler):
    """Multivariate slice sampler using coordinate-wise updates."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float],
                 step_sizes: Optional[np.ndarray] = None,
                 rng: Optional[GlobalRng] = None):
        """Initialize multivariate slice sampler.
        
        Args:
            target_log_density: Log density function
            step_sizes: Step sizes for each coordinate
            rng: Random number generator
        """
        super().__init__(target_log_density)
        self.step_sizes = step_sizes
        self.rng = rng or GlobalRng()
        self.coordinate_samplers = []
    
    def set_initial_value(self, value: np.ndarray):
        """Set initial value and create coordinate samplers."""
        self.current_value = np.array(value)
        self.current_log_density = self.target_log_density(self.current_value)
        
        dim = len(self.current_value)
        if self.step_sizes is None:
            self.step_sizes = np.ones(dim)
        
        # Create slice sampler for each coordinate
        self.coordinate_samplers = []
        for i in range(dim):
            def coord_log_density(x_i, coord=i):
                x_temp = self.current_value.copy()
                x_temp[coord] = x_i
                return self.target_log_density(x_temp)
            
            sampler = SliceSampler(
                coord_log_density,
                step_size=self.step_sizes[i],
                rng=self.rng
            )
            self.coordinate_samplers.append(sampler)
    
    def draw(self) -> np.ndarray:
        """Draw by updating each coordinate."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Update each coordinate
        for i, coord_sampler in enumerate(self.coordinate_samplers):
            # Update the coordinate sampler's target function
            def coord_log_density(x_i):
                x_temp = self.current_value.copy()
                x_temp[i] = x_i
                return self.target_log_density(x_temp)
            
            coord_sampler.target_log_density = coord_log_density
            coord_sampler.set_initial_value(self.current_value[i])
            
            # Draw new value for this coordinate
            self.current_value[i] = coord_sampler.draw()
        
        # Update log density
        self.current_log_density = self.target_log_density(self.current_value)
        self.n_draws += 1
        self.n_accepted += 1  # Always accept in slice sampling
        
        return self.current_value.copy()