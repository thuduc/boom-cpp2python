"""Base classes for mixture models."""
import numpy as np
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from ..base import Model, VectorParameter, PositiveParameter
from ...linalg import Vector, Matrix


class MixtureComponent(ABC):
    """Base class for mixture components."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize mixture component.
        
        Args:
            weight: Component weight (will be normalized)
        """
        self._weight = weight
    
    @property
    def weight(self) -> float:
        """Get component weight."""
        return self._weight
    
    @weight.setter
    def weight(self, value: float):
        """Set component weight."""
        if value < 0:
            raise ValueError("Weight must be non-negative")
        self._weight = value
    
    @abstractmethod
    def logpdf(self, x) -> float:
        """Log probability density function."""
        pass
    
    @abstractmethod
    def sample(self, rng) -> float:
        """Sample from the component."""
        pass
    
    @abstractmethod
    def fit(self, data: Vector, weights: Vector):
        """Fit component parameters to weighted data."""
        pass
    
    def pdf(self, x) -> float:
        """Probability density function."""
        return np.exp(self.logpdf(x))


class MixtureModel(Model):
    """Base class for mixture models."""
    
    def __init__(self, n_components: int):
        """Initialize mixture model.
        
        Args:
            n_components: Number of mixture components
        """
        super().__init__()
        self.n_components = n_components
        self.components: List[MixtureComponent] = []
        self._mixing_weights = Vector.ones(n_components) / n_components
        self._data = []
        
        # Parameters for mixing weights (Dirichlet prior)
        self._params['mixing_weights'] = VectorParameter(self._mixing_weights, 'mixing_weights')
    
    @property
    def mixing_weights(self) -> Vector:
        """Get mixing weights."""
        return self._params['mixing_weights'].value
    
    @mixing_weights.setter
    def mixing_weights(self, weights: Union[Vector, np.ndarray, List[float]]):
        """Set mixing weights."""
        weights = Vector(weights)
        if len(weights) != self.n_components:
            raise ValueError(f"Expected {self.n_components} weights, got {len(weights)}")
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
        if abs(float(weights.sum()) - 1.0) > 1e-10:
            # Normalize if not already normalized
            weights = weights / float(weights.sum())
        self._params['mixing_weights'].value = weights
        self._mixing_weights = weights
    
    def add_component(self, component: MixtureComponent):
        """Add a mixture component."""
        if len(self.components) >= self.n_components:
            raise ValueError(f"Cannot add more than {self.n_components} components")
        self.components.append(component)
    
    def add_data(self, x: float):
        """Add a data point."""
        self._data.append(float(x))
    
    def set_data(self, data: Union[List[float], Vector, np.ndarray]):
        """Set all data."""
        self._data = [float(x) for x in data]
    
    def clear_data(self):
        """Clear all data."""
        self._data = []
    
    def get_data(self) -> Vector:
        """Get data as Vector."""
        return Vector(self._data)
    
    def logpdf(self, x: float) -> float:
        """Log probability density of mixture."""
        if not self.components:
            raise ValueError("No components added to mixture")
        
        # Log-sum-exp trick for numerical stability
        log_densities = []
        for i, component in enumerate(self.components):
            log_weight = np.log(self.mixing_weights[i]) if self.mixing_weights[i] > 0 else -np.inf
            log_densities.append(log_weight + component.logpdf(x))
        
        max_log_density = max(log_densities)
        log_sum = max_log_density + np.log(sum(np.exp(ld - max_log_density) for ld in log_densities))
        return log_sum
    
    def pdf(self, x: float) -> float:
        """Probability density of mixture."""
        return np.exp(self.logpdf(x))
    
    def loglike(self) -> float:
        """Log likelihood of data."""
        if not self._data:
            return 0.0
        return sum(self.logpdf(x) for x in self._data)
    
    def component_posteriors(self, x: float) -> Vector:
        """Posterior probabilities of components given x."""
        if not self.components:
            raise ValueError("No components added to mixture")
        
        log_posteriors = []
        for i, component in enumerate(self.components):
            log_weight = np.log(self.mixing_weights[i]) if self.mixing_weights[i] > 0 else -np.inf
            log_posteriors.append(log_weight + component.logpdf(x))
        
        # Normalize using log-sum-exp
        max_log_post = max(log_posteriors)
        exp_log_posts = [np.exp(lp - max_log_post) for lp in log_posteriors]
        sum_exp = sum(exp_log_posts)
        
        return Vector([elp / sum_exp for elp in exp_log_posts])
    
    def sample_component(self, rng) -> int:
        """Sample a component index according to mixing weights."""
        return rng.rmulti(1, self.mixing_weights).argmax()
    
    def sample(self, rng) -> float:
        """Sample from the mixture."""
        if not self.components:
            raise ValueError("No components added to mixture")
        
        # Sample component
        component_idx = self.sample_component(rng)
        
        # Sample from that component
        return self.components[component_idx].sample(rng)
    
    def simulate(self, n: int, rng=None) -> Vector:
        """Simulate n samples from the mixture."""
        if rng is None:
            from ...distributions import rng as global_rng
            rng = global_rng
        
        samples = []
        for _ in range(n):
            samples.append(self.sample(rng))
        
        return Vector(samples)
    
    def classify(self, x: float) -> int:
        """Classify x to most likely component."""
        posteriors = self.component_posteriors(x)
        return int(np.argmax(posteriors))
    
    def clone(self) -> 'MixtureModel':
        """Create a copy of the model."""
        # This is abstract - subclasses should implement
        raise NotImplementedError("Subclasses must implement clone()")
    
    @abstractmethod
    def fit(self, max_iter: int = 100, tol: float = 1e-6):
        """Fit the mixture model to data."""
        pass