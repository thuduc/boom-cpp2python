"""Gibbs sampler implementation."""

from typing import Callable, Optional, List, Union, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from ..linalg import Vector
from ..distributions import RNG


class ConditionalSampler(ABC):
    """Abstract base class for conditional samplers used in Gibbs sampling."""
    
    @abstractmethod
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> Any:
        """Sample from the conditional distribution.
        
        Args:
            current_state: Dictionary containing current values of all variables.
            rng: Random number generator.
            
        Returns:
            Sample from the conditional distribution.
        """
        pass
    
    @abstractmethod
    def variable_name(self) -> str:
        """Return the name of the variable this sampler updates."""
        pass


class FunctionalConditionalSampler(ConditionalSampler):
    """Conditional sampler that uses a user-provided function."""
    
    def __init__(self, variable_name: str, 
                 conditional_sampler: Callable[[Dict[str, Any], RNG], Any]):
        """Initialize functional conditional sampler.
        
        Args:
            variable_name: Name of the variable to update.
            conditional_sampler: Function that samples from conditional distribution.
        """
        self._variable_name = variable_name
        self._conditional_sampler = conditional_sampler
    
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> Any:
        """Sample using the provided function."""
        return self._conditional_sampler(current_state, rng)
    
    def variable_name(self) -> str:
        """Return variable name."""
        return self._variable_name


class GaussianConditionalSampler(ConditionalSampler):
    """Conditional sampler for Gaussian distributions."""
    
    def __init__(self, variable_name: str,
                 mean_func: Callable[[Dict[str, Any]], float],
                 variance_func: Callable[[Dict[str, Any]], float]):
        """Initialize Gaussian conditional sampler.
        
        Args:
            variable_name: Name of the variable to update.
            mean_func: Function that computes conditional mean.
            variance_func: Function that computes conditional variance.
        """
        self._variable_name = variable_name
        self._mean_func = mean_func
        self._variance_func = variance_func
    
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> float:
        """Sample from conditional Gaussian."""
        mean = self._mean_func(current_state)
        variance = self._variance_func(current_state)
        
        if variance <= 0:
            raise ValueError(f"Conditional variance for {self._variable_name} must be positive")
        
        return rng.normal(mean, np.sqrt(variance))
    
    def variable_name(self) -> str:
        """Return variable name."""
        return self._variable_name


class BetaConditionalSampler(ConditionalSampler):
    """Conditional sampler for Beta distributions."""
    
    def __init__(self, variable_name: str,
                 alpha_func: Callable[[Dict[str, Any]], float],
                 beta_func: Callable[[Dict[str, Any]], float]):
        """Initialize Beta conditional sampler.
        
        Args:
            variable_name: Name of the variable to update.
            alpha_func: Function that computes conditional alpha parameter.
            beta_func: Function that computes conditional beta parameter.
        """
        self._variable_name = variable_name
        self._alpha_func = alpha_func
        self._beta_func = beta_func
    
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> float:
        """Sample from conditional Beta."""
        alpha = self._alpha_func(current_state)
        beta = self._beta_func(current_state)
        
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Conditional Beta parameters for {self._variable_name} must be positive")
        
        return rng.beta(alpha, beta)
    
    def variable_name(self) -> str:
        """Return variable name."""
        return self._variable_name


class GammaConditionalSampler(ConditionalSampler):
    """Conditional sampler for Gamma distributions."""
    
    def __init__(self, variable_name: str,
                 shape_func: Callable[[Dict[str, Any]], float],
                 rate_func: Callable[[Dict[str, Any]], float]):
        """Initialize Gamma conditional sampler.
        
        Args:
            variable_name: Name of the variable to update.
            shape_func: Function that computes conditional shape parameter.
            rate_func: Function that computes conditional rate parameter.
        """
        self._variable_name = variable_name
        self._shape_func = shape_func
        self._rate_func = rate_func
    
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> float:
        """Sample from conditional Gamma."""
        shape = self._shape_func(current_state)
        rate = self._rate_func(current_state)
        
        if shape <= 0 or rate <= 0:
            raise ValueError(f"Conditional Gamma parameters for {self._variable_name} must be positive")
        
        return rng.gamma(shape, 1.0 / rate)  # numpy uses scale parameterization
    
    def variable_name(self) -> str:
        """Return variable name."""
        return self._variable_name


class DirichletConditionalSampler(ConditionalSampler):
    """Conditional sampler for Dirichlet distributions."""
    
    def __init__(self, variable_name: str,
                 alpha_func: Callable[[Dict[str, Any]], Union[List[float], np.ndarray, Vector]]):
        """Initialize Dirichlet conditional sampler.
        
        Args:
            variable_name: Name of the variable to update.
            alpha_func: Function that computes conditional concentration parameters.
        """
        self._variable_name = variable_name
        self._alpha_func = alpha_func
    
    def sample(self, current_state: Dict[str, Any], rng: RNG) -> Vector:
        """Sample from conditional Dirichlet."""
        alpha = self._alpha_func(current_state)
        
        if isinstance(alpha, Vector):
            alpha_array = alpha.to_numpy()
        else:
            alpha_array = np.array(alpha, dtype=float)
        
        if np.any(alpha_array <= 0):
            raise ValueError(f"Conditional Dirichlet parameters for {self._variable_name} must be positive")
        
        # Sample from independent Gamma distributions and normalize
        gamma_samples = np.array([rng.gamma(a, 1.0) for a in alpha_array])
        return Vector(gamma_samples / np.sum(gamma_samples))
    
    def variable_name(self) -> str:
        """Return variable name."""
        return self._variable_name


class GibbsSampler:
    """Gibbs sampler for multivariate distributions."""
    
    def __init__(self, conditional_samplers: List[ConditionalSampler],
                 rng: Optional[RNG] = None):
        """Initialize Gibbs sampler.
        
        Args:
            conditional_samplers: List of conditional samplers, one for each variable.
            rng: Random number generator. If None, creates a new one.
        """
        self._conditional_samplers = conditional_samplers
        self._rng = rng if rng is not None else RNG()
        
        # Validate that variable names are unique
        variable_names = [sampler.variable_name() for sampler in conditional_samplers]
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("Variable names must be unique")
        
        self._variable_names = variable_names
    
    def sample(self, initial_state: Dict[str, Any], n_samples: int,
               burn_in: int = 0, thin: int = 1) -> List[Dict[str, Any]]:
        """Run Gibbs sampling.
        
        Args:
            initial_state: Dictionary with initial values for all variables.
            n_samples: Number of samples to return (after burn-in and thinning).
            burn_in: Number of burn-in samples to discard.
            thin: Thinning interval (keep every thin-th sample).
            
        Returns:
            List of sample dictionaries.
        """
        # Validate initial state
        for var_name in self._variable_names:
            if var_name not in initial_state:
                raise ValueError(f"Initial state missing variable: {var_name}")
        
        current_state = initial_state.copy()
        samples = []
        total_iterations = burn_in + n_samples * thin
        
        for iteration in range(total_iterations):
            # Update each variable in turn
            for sampler in self._conditional_samplers:
                var_name = sampler.variable_name()
                try:
                    new_value = sampler.sample(current_state, self._rng)
                    current_state[var_name] = new_value
                except (ValueError, OverflowError) as e:
                    # If sampling fails, keep current value
                    print(f"Warning: Sampling failed for {var_name}: {e}")
            
            # Store sample if past burn-in and on thinning schedule
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(current_state.copy())
        
        return samples
    
    def sample_chains(self, initial_states: List[Dict[str, Any]], 
                     n_samples: int, burn_in: int = 0, thin: int = 1) -> List[List[Dict[str, Any]]]:
        """Run multiple Gibbs sampling chains.
        
        Args:
            initial_states: List of initial states for each chain.
            n_samples: Number of samples per chain.
            burn_in: Number of burn-in samples per chain.
            thin: Thinning interval.
            
        Returns:
            List of sample lists, one per chain.
        """
        chains = []
        for initial_state in initial_states:
            chain_samples = self.sample(initial_state, n_samples, burn_in, thin)
            chains.append(chain_samples)
        
        return chains
    
    def variable_names(self) -> List[str]:
        """Get list of variable names."""
        return self._variable_names.copy()
    
    def n_variables(self) -> int:
        """Get number of variables."""
        return len(self._variable_names)


class AdaptiveGibbsSampler(GibbsSampler):
    """Adaptive Gibbs sampler that can adjust conditional samplers during sampling."""
    
    def __init__(self, conditional_samplers: List[ConditionalSampler],
                 rng: Optional[RNG] = None,
                 adaptation_window: int = 100):
        """Initialize adaptive Gibbs sampler.
        
        Args:
            conditional_samplers: List of conditional samplers.
            rng: Random number generator.
            adaptation_window: Number of samples between adaptations.
        """
        super().__init__(conditional_samplers, rng)
        self._adaptation_window = adaptation_window
        self._adaptation_count = 0
        
        # Store sample history for adaptation
        self._sample_history = {name: [] for name in self._variable_names}
    
    def sample(self, initial_state: Dict[str, Any], n_samples: int,
               burn_in: int = 0, thin: int = 1,
               adapt_during_burnin: bool = True) -> List[Dict[str, Any]]:
        """Run adaptive Gibbs sampling.
        
        Args:
            initial_state: Dictionary with initial values for all variables.
            n_samples: Number of samples to return.
            burn_in: Number of burn-in samples.
            thin: Thinning interval.
            adapt_during_burnin: Whether to adapt during burn-in period.
            
        Returns:
            List of sample dictionaries.
        """
        current_state = initial_state.copy()
        samples = []
        total_iterations = burn_in + n_samples * thin
        
        for iteration in range(total_iterations):
            # Update each variable
            for sampler in self._conditional_samplers:
                var_name = sampler.variable_name()
                try:
                    new_value = sampler.sample(current_state, self._rng)
                    current_state[var_name] = new_value
                    
                    # Store for adaptation
                    if adapt_during_burnin or iteration >= burn_in:
                        self._sample_history[var_name].append(new_value)
                        
                except (ValueError, OverflowError) as e:
                    print(f"Warning: Sampling failed for {var_name}: {e}")
            
            # Adapt samplers periodically
            if ((iteration + 1) % self._adaptation_window == 0 and 
                (adapt_during_burnin or iteration >= burn_in)):
                self._adapt_samplers()
            
            # Store sample
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(current_state.copy())
        
        return samples
    
    def _adapt_samplers(self):
        """Adapt conditional samplers based on sample history.
        
        This is a placeholder implementation. In practice, specific
        adaptation strategies would depend on the sampler types.
        """
        # Simple adaptation: clear history to prevent memory growth
        for var_name in self._variable_names:
            if len(self._sample_history[var_name]) > 1000:
                # Keep only recent samples
                self._sample_history[var_name] = self._sample_history[var_name][-500:]
        
        self._adaptation_count += 1
    
    def get_sample_history(self, variable_name: str) -> List[Any]:
        """Get sample history for a variable."""
        if variable_name not in self._sample_history:
            raise ValueError(f"Unknown variable: {variable_name}")
        return self._sample_history[variable_name].copy()
    
    def clear_sample_history(self):
        """Clear all sample history."""
        for var_name in self._variable_names:
            self._sample_history[var_name].clear()