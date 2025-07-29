"""Base classes for MCMC samplers."""
import numpy as np
from typing import Callable, Optional, Any
from abc import ABC, abstractmethod
from ..models.base import Model


class Sampler(ABC):
    """Base class for MCMC samplers."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float]):
        """Initialize sampler.
        
        Args:
            target_log_density: Function that computes log density
        """
        self.target_log_density = target_log_density
        self.current_value = None
        self.current_log_density = None
        self.n_draws = 0
        self.n_accepted = 0
    
    @abstractmethod
    def draw(self) -> np.ndarray:
        """Draw a sample from the target distribution."""
        pass
    
    def set_initial_value(self, value: np.ndarray):
        """Set initial value for the chain."""
        self.current_value = np.array(value)
        self.current_log_density = self.target_log_density(self.current_value)
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate."""
        if self.n_draws == 0:
            return 0.0
        return self.n_accepted / self.n_draws
    
    def reset_counters(self):
        """Reset draw and acceptance counters."""
        self.n_draws = 0
        self.n_accepted = 0


class PosteriorSampler(ABC):
    """Base class for model-specific posterior samplers."""
    
    def __init__(self, model: Model, name: str = ""):
        """Initialize posterior sampler.
        
        Args:
            model: The model to sample from
            name: Optional name for the sampler
        """
        self.model = model
        self.name = name
        self.n_draws = 0
    
    @abstractmethod
    def draw(self):
        """Update model parameters with a posterior draw."""
        pass
    
    def sample_posterior(self):
        """Sample from posterior (calls draw)."""
        self.draw()
        self.n_draws += 1
    
    def reset_counters(self):
        """Reset draw counter."""
        self.n_draws = 0


class AdaptiveSampler(Sampler):
    """Base class for adaptive samplers."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float],
                 adaptation_frequency: int = 100):
        """Initialize adaptive sampler.
        
        Args:
            target_log_density: Function that computes log density
            adaptation_frequency: How often to adapt (in draws)
        """
        super().__init__(target_log_density)
        self.adaptation_frequency = adaptation_frequency
        self.adaptation_phase = True
        self.last_adaptation = 0
    
    @abstractmethod
    def adapt(self):
        """Adapt sampler parameters based on recent performance."""
        pass
    
    def should_adapt(self) -> bool:
        """Check if adaptation should occur."""
        return (self.adaptation_phase and 
                self.n_draws - self.last_adaptation >= self.adaptation_frequency)
    
    def stop_adaptation(self):
        """Stop adaptation phase."""
        self.adaptation_phase = False
    
    def draw(self) -> np.ndarray:
        """Draw with adaptation."""
        sample = self._draw_impl()
        
        if self.should_adapt():
            self.adapt()
            self.last_adaptation = self.n_draws
        
        return sample
    
    @abstractmethod
    def _draw_impl(self) -> np.ndarray:
        """Implementation of the actual draw."""
        pass


class CompositeSampler(PosteriorSampler):
    """Composite sampler that runs multiple samplers in sequence."""
    
    def __init__(self, samplers: list):
        """Initialize composite sampler.
        
        Args:
            samplers: List of PosteriorSampler objects
        """
        if not samplers:
            raise ValueError("Must provide at least one sampler")
        
        # Use first sampler's model
        super().__init__(samplers[0].model, "CompositeSampler")
        self.samplers = samplers
        
        # Verify all samplers use the same model
        for sampler in samplers:
            if sampler.model is not self.model:
                raise ValueError("All samplers must use the same model")
    
    def draw(self):
        """Draw from each sampler in sequence."""
        for sampler in self.samplers:
            sampler.draw()
    
    def reset_counters(self):
        """Reset counters for all samplers."""
        super().reset_counters()
        for sampler in self.samplers:
            sampler.reset_counters()
    
    @property
    def acceptance_rates(self) -> dict:
        """Get acceptance rates for all samplers."""
        rates = {}
        for sampler in self.samplers:
            name = sampler.name or f"sampler_{id(sampler)}"
            if hasattr(sampler, 'acceptance_rate'):
                rates[name] = sampler.acceptance_rate
        return rates


class MoveAccounting:
    """Class to track sampler performance."""
    
    def __init__(self, name: str = ""):
        """Initialize move accounting.
        
        Args:
            name: Name of the move
        """
        self.name = name
        self.n_attempts = 0
        self.n_accepted = 0
        self.log_sum_proposal_prob = 0.0
    
    def record_attempt(self, accepted: bool, log_proposal_prob: float = 0.0):
        """Record a sampling attempt.
        
        Args:
            accepted: Whether the proposal was accepted
            log_proposal_prob: Log probability of the proposal
        """
        self.n_attempts += 1
        if accepted:
            self.n_accepted += 1
        self.log_sum_proposal_prob += log_proposal_prob
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate."""
        if self.n_attempts == 0:
            return 0.0
        return self.n_accepted / self.n_attempts
    
    @property
    def average_log_proposal_prob(self) -> float:
        """Get average log proposal probability."""
        if self.n_attempts == 0:
            return 0.0
        return self.log_sum_proposal_prob / self.n_attempts
    
    def reset(self):
        """Reset counters."""
        self.n_attempts = 0
        self.n_accepted = 0
        self.log_sum_proposal_prob = 0.0
    
    def __repr__(self):
        return (f"MoveAccounting(name='{self.name}', "
                f"acceptance_rate={self.acceptance_rate:.3f}, "
                f"n_attempts={self.n_attempts})")