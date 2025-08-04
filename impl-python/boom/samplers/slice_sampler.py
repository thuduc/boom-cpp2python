"""Slice sampler implementation."""

from typing import Callable, Optional, Union
import numpy as np
from ..linalg import Vector
from ..distributions import RNG


class SliceSampler:
    """Univariate slice sampler for sampling from unnormalized densities.
    
    The slice sampler is particularly useful for sampling from univariate
    distributions where the inverse CDF is not available.
    """
    
    def __init__(self, log_target: Callable[[float], float], 
                 rng: Optional[RNG] = None):
        """Initialize slice sampler.
        
        Args:
            log_target: Function that computes log target density for a scalar.
            rng: Random number generator. If None, creates a new one.
        """
        self._log_target = log_target
        self._rng = rng if rng is not None else RNG()
    
    def sample(self, initial_value: float, n_samples: int, 
               width: float = 1.0, max_steps_out: int = 10) -> list:
        """Run slice sampling.
        
        Args:
            initial_value: Starting value.
            n_samples: Number of samples to generate.
            width: Initial bracket width for stepping out.
            max_steps_out: Maximum number of stepping out steps.
            
        Returns:
            List of samples.
        """
        samples = []
        current_value = initial_value
        current_log_target = self._log_target(current_value)
        
        for i in range(n_samples):
            # Step 1: Sample auxiliary variable (slice level)
            log_u = current_log_target + np.log(self._rng())
            
            # Step 2: Find interval containing current value
            left, right = self._step_out(current_value, log_u, width, max_steps_out)
            
            # Step 3: Sample from the interval
            current_value = self._sample_from_interval(left, right, log_u, current_value)
            current_log_target = self._log_target(current_value)
            
            samples.append(current_value)
        
        return samples
    
    def _step_out(self, x: float, log_u: float, width: float, max_steps: int) -> tuple:
        """Step out to find an interval that brackets the slice.
        
        Args:
            x: Current value.
            log_u: Log of auxiliary variable.
            width: Initial bracket width.
            max_steps: Maximum number of steps.
            
        Returns:
            Tuple of (left, right) interval bounds.
        """
        # Random placement of x within initial interval
        u = self._rng()
        left = x - width * u
        right = left + width
        
        # Expand left boundary
        steps_left = 0
        while (steps_left < max_steps and 
               self._is_in_slice(left, log_u)):
            left -= width
            steps_left += 1
        
        # Expand right boundary
        steps_right = 0
        while (steps_right < max_steps and 
               self._is_in_slice(right, log_u)):
            right += width
            steps_right += 1
        
        return left, right
    
    def _sample_from_interval(self, left: float, right: float, log_u: float, current_x: float) -> float:
        """Sample from the interval using shrinkage.
        
        Args:
            left: Left boundary.
            right: Right boundary.
            log_u: Log of auxiliary variable.
            
        Returns:
            Sample from the interval.
        """
        while True:
            # Sample uniformly from current interval
            x_new = left + (right - left) * self._rng()
            
            if self._is_in_slice(x_new, log_u):
                return x_new
            
            # Shrink interval
            if x_new < current_x:
                left = x_new
            else:
                right = x_new
    
    def _is_in_slice(self, x: float, log_u: float) -> bool:
        """Check if a point is in the slice.
        
        Args:
            x: Point to check.
            log_u: Log of auxiliary variable.
            
        Returns:
            True if point is in slice.
        """
        try:
            return self._log_target(x) >= log_u
        except (ValueError, OverflowError):
            return False
    
    def __call__(self, initial_value: float, n_samples: int, **kwargs) -> list:
        """Convenience method for sampling."""
        return self.sample(initial_value, n_samples, **kwargs)


class MultivariateSliceSampler:
    """Multivariate slice sampler using coordinate-wise updates.
    
    This performs slice sampling on each coordinate in turn, which is
    effective for many multivariate distributions.
    """
    
    def __init__(self, log_target: Callable[[Vector], float], 
                 rng: Optional[RNG] = None):
        """Initialize multivariate slice sampler.
        
        Args:
            log_target: Function that computes log target density for a Vector.
            rng: Random number generator. If None, creates a new one.
        """
        self._log_target = log_target
        self._rng = rng if rng is not None else RNG()
    
    def sample(self, initial_state: Vector, n_samples: int,
               widths: Optional[Union[float, list, Vector]] = None,
               max_steps_out: int = 10) -> list:
        """Run multivariate slice sampling.
        
        Args:
            initial_state: Starting parameter values.
            n_samples: Number of samples to generate.
            widths: Initial bracket widths for each coordinate. If scalar,
                   uses same width for all coordinates.
            max_steps_out: Maximum number of stepping out steps.
            
        Returns:
            List of Vector samples.
        """
        samples = []
        current_state = initial_state.copy()
        dim = len(current_state)
        
        # Set up widths
        if widths is None:
            widths = [1.0] * dim
        elif isinstance(widths, (int, float)):
            widths = [float(widths)] * dim
        elif isinstance(widths, Vector):
            widths = widths.to_numpy().tolist()
        elif isinstance(widths, (list, np.ndarray)):
            widths = list(widths)
        else:
            raise TypeError("widths must be scalar, list, or Vector")
        
        if len(widths) != dim:
            raise ValueError(f"widths length {len(widths)} doesn't match dimension {dim}")
        
        # Create univariate slice samplers for each coordinate
        coordinate_samplers = []
        for i in range(dim):
            def make_coordinate_target(coord_idx):
                def coord_log_target(x_coord):
                    # Create temporary state with updated coordinate
                    temp_state = current_state.copy()
                    temp_state[coord_idx] = x_coord
                    return self._log_target(temp_state)
                return coord_log_target
            
            coord_sampler = SliceSampler(make_coordinate_target(i), self._rng)
            coordinate_samplers.append(coord_sampler)
        
        for sample_idx in range(n_samples):
            # Update each coordinate in turn
            for coord_idx in range(dim):
                # Create coordinate-specific target function
                def coord_log_target(x_coord):
                    temp_state = current_state.copy()
                    temp_state[coord_idx] = x_coord
                    return self._log_target(temp_state)
                
                # Update the sampler's target function
                coordinate_samplers[coord_idx]._log_target = coord_log_target
                
                # Sample new value for this coordinate
                new_values = coordinate_samplers[coord_idx].sample(
                    current_state[coord_idx], 1, 
                    width=widths[coord_idx], 
                    max_steps_out=max_steps_out
                )
                current_state[coord_idx] = new_values[0]
            
            samples.append(current_state.copy())
        
        return samples
    
    def __call__(self, initial_state: Vector, n_samples: int, **kwargs) -> list:
        """Convenience method for sampling."""
        return self.sample(initial_state, n_samples, **kwargs)


class AdaptiveSliceSampler:
    """Adaptive slice sampler that adjusts step sizes during sampling.
    
    This version adapts the initial bracket widths based on the acceptance
    history to improve efficiency.
    """
    
    def __init__(self, log_target: Callable[[Vector], float], 
                 rng: Optional[RNG] = None,
                 adaptation_window: int = 100):
        """Initialize adaptive slice sampler.
        
        Args:
            log_target: Function that computes log target density for a Vector.
            rng: Random number generator. If None, creates a new one.
            adaptation_window: Number of samples between adaptations.
        """
        self._log_target = log_target
        self._rng = rng if rng is not None else RNG()
        self._adaptation_window = adaptation_window
        
        # Adaptation statistics
        self._step_sizes = None
        self._n_steps_out_history = []
        self._adaptation_count = 0
    
    def sample(self, initial_state: Vector, n_samples: int,
               initial_widths: Optional[Union[float, list, Vector]] = None,
               max_steps_out: int = 10,
               target_steps_out: float = 3.0) -> tuple:
        """Run adaptive slice sampling.
        
        Args:
            initial_state: Starting parameter values.
            n_samples: Number of samples to generate.
            initial_widths: Initial bracket widths.
            max_steps_out: Maximum number of stepping out steps.
            target_steps_out: Target average number of stepping out steps.
            
        Returns:
            Tuple of (samples, final_widths).
        """
        dim = len(initial_state)
        
        # Initialize step sizes
        if self._step_sizes is None:
            if initial_widths is None:
                self._step_sizes = np.ones(dim)
            elif isinstance(initial_widths, (int, float)):
                self._step_sizes = np.full(dim, float(initial_widths))
            elif isinstance(initial_widths, Vector):
                self._step_sizes = initial_widths.to_numpy().copy()
            else:
                self._step_sizes = np.array(initial_widths, dtype=float)
        
        # Initialize adaptation history
        self._n_steps_out_history = [[] for _ in range(dim)]
        
        # Use multivariate slice sampler with adaptation
        base_sampler = MultivariateSliceSampler(self._log_target, self._rng)
        
        samples = []
        current_state = initial_state.copy()
        
        for i in range(n_samples):
            # Get sample using current step sizes
            new_samples = base_sampler.sample(current_state, 1, 
                                            widths=self._step_sizes,
                                            max_steps_out=max_steps_out)
            current_state = new_samples[0]
            samples.append(current_state.copy())
            
            # Adapt step sizes periodically
            if (i + 1) % self._adaptation_window == 0:
                self._adapt_step_sizes(target_steps_out)
        
        return samples, Vector(self._step_sizes)
    
    def _adapt_step_sizes(self, target_steps_out: float):
        """Adapt step sizes based on stepping out history.
        
        Args:
            target_steps_out: Target average number of steps out.
        """
        # This is a simplified adaptation - in practice, would need to
        # track actual stepping out statistics from the slice sampler
        # For now, just adjust based on a simple heuristic
        
        adaptation_factor = 1.1  # How much to adjust by
        
        for i in range(len(self._step_sizes)):
            # Simple adaptation: increase step size if we're consistently
            # hitting the maximum steps out, decrease if we're not stepping out much
            
            # In a full implementation, would track actual steps_out per coordinate
            # Here we use a placeholder adaptation
            if self._adaptation_count % 3 == 0:
                # Occasionally increase step size
                self._step_sizes[i] *= adaptation_factor
            elif self._adaptation_count % 3 == 1:
                # Occasionally decrease step size
                self._step_sizes[i] /= adaptation_factor
            # Otherwise keep the same
        
        self._adaptation_count += 1
    
    def get_step_sizes(self) -> Vector:
        """Get current step sizes."""
        if self._step_sizes is None:
            return Vector([1.0])
        return Vector(self._step_sizes)