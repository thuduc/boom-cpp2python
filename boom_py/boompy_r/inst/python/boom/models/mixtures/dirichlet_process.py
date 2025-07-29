"""Dirichlet Process mixture models."""
import numpy as np
from typing import Optional, Union, List
from .base import MixtureModel, MixtureComponent
from .finite_mixture import GaussianComponent
from ..base import PositiveParameter
from ...linalg import Vector, Matrix


class DirichletProcessMixtureModel(MixtureModel):
    """Dirichlet Process mixture model with stick-breaking construction.
    
    This implements a truncated Dirichlet Process mixture where we
    assume a maximum number of components but allow the model to
    determine how many are actually used.
    """
    
    def __init__(self, max_components: int = 20, alpha: float = 1.0, 
                 base_component_type: str = "gaussian"):
        """Initialize Dirichlet Process mixture model.
        
        Args:
            max_components: Maximum number of components (truncation level)
            alpha: Concentration parameter
            base_component_type: Type of base components
        """
        # Use max_components as the truncation level
        super().__init__(max_components)
        self.max_components = max_components
        self._alpha_param = PositiveParameter(alpha, 'alpha')
        self.base_component_type = base_component_type
        
        # Stick-breaking weights (before normalization)
        self._stick_lengths = Vector.ones(max_components)
        self._active_components = 0
        
        self._initialize_components()
        self._update_mixing_weights()
    
    @property
    def alpha(self) -> float:
        """Get concentration parameter."""
        return self._alpha_param.value
    
    @alpha.setter
    def alpha(self, value: float):
        """Set concentration parameter."""
        if value <= 0:
            raise ValueError("Alpha must be positive")
        self._alpha_param.value = value
    
    def _initialize_components(self):
        """Initialize all potential components."""
        self.components = []
        
        if self.base_component_type == "gaussian":
            for i in range(self.max_components):
                # Initialize with different means
                mean = np.random.normal(0, 1)  # Random initialization
                sigma = 1.0
                component = GaussianComponent(mean=mean, sigma=sigma)
                self.components.append(component)
        else:
            raise ValueError(f"Unknown component type: {self.base_component_type}")
    
    def _sample_stick_lengths(self, rng):
        """Sample stick lengths using stick-breaking process."""
        # Beta(1, alpha) for stick-breaking
        for i in range(self.max_components - 1):
            self._stick_lengths[i] = rng.rbeta(1.0, self.alpha)
        
        # Last stick gets the remainder
        self._stick_lengths[-1] = 1.0
    
    def _update_mixing_weights(self):
        """Update mixing weights from stick lengths."""
        # Convert stick lengths to mixing weights using stick-breaking
        weights = Vector.zero(self.max_components)
        remaining_stick = 1.0
        
        for i in range(self.max_components):
            if i == self.max_components - 1:
                weights[i] = remaining_stick
            else:
                weights[i] = self._stick_lengths[i] * remaining_stick
                remaining_stick *= (1.0 - self._stick_lengths[i])
        
        self.mixing_weights = weights
    
    def add_data(self, x: float):
        """Add a data point."""
        super().add_data(x)
        # Update the number of active components based on data
        self._update_active_components()
    
    def _update_active_components(self):
        """Update number of active components based on current assignments."""
        if not self._data:
            self._active_components = 0
            return
        
        # Simple heuristic: use components with weight > threshold
        threshold = 1.0 / (10 * self.max_components)
        self._active_components = sum(1 for w in self.mixing_weights if w > threshold)
        self._active_components = max(1, min(self._active_components, self.max_components))
    
    def effective_n_components(self) -> int:
        """Get the effective number of components (with significant weight)."""
        threshold = 1.0 / (10 * self.max_components)
        return sum(1 for w in self.mixing_weights if w > threshold)
    
    def sample_component_assignment(self, x: float, rng) -> int:
        """Sample component assignment for data point x using Gibbs sampling."""
        # Compute probabilities for each component
        log_probs = []
        
        for i in range(self.max_components):
            if self.mixing_weights[i] > 1e-10:
                log_weight = np.log(self.mixing_weights[i])
                log_likelihood = self.components[i].logpdf(x)
                log_probs.append(log_weight + log_likelihood)
            else:
                log_probs.append(-np.inf)
        
        # Convert to probabilities using log-sum-exp
        max_log_prob = max(log_probs)
        probs = [np.exp(lp - max_log_prob) for lp in log_probs]
        prob_sum = sum(probs)
        
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]
        else:
            # Uniform fallback
            probs = [1.0 / self.max_components] * self.max_components
        
        # Sample component
        return rng.rmulti(1, Vector(probs)).argmax()
    
    def gibbs_step(self, rng):
        """Perform one Gibbs sampling step."""
        if not self._data:
            return
        
        # Sample component assignments
        assignments = []
        for x in self._data:
            assignment = self.sample_component_assignment(x, rng)
            assignments.append(assignment)
        
        # Update component parameters based on assignments
        data_vec = Vector(self._data)
        
        for k in range(self.max_components):
            # Find data points assigned to component k
            assigned_indices = [i for i, a in enumerate(assignments) if a == k]
            
            if assigned_indices:
                assigned_data = Vector([data_vec[i] for i in assigned_indices])
                weights = Vector.ones(len(assigned_data))
                self.components[k].fit(assigned_data, weights)
        
        # Update stick lengths and mixing weights
        self._sample_stick_lengths(rng)
        self._update_mixing_weights()
    
    def fit(self, n_iter: int = 1000, burn_in: int = 500):
        """Fit the model using Gibbs sampling.
        
        Args:
            n_iter: Number of Gibbs iterations
            burn_in: Number of burn-in iterations
        """
        if not self._data:
            raise ValueError("No data to fit")
        
        from ...distributions import rng
        
        # Initialize components from data
        self._initialize_from_data()
        
        # Gibbs sampling
        for iteration in range(n_iter):
            self.gibbs_step(rng)
            
            # Track effective number of components after burn-in
            if iteration >= burn_in:
                self._update_active_components()
    
    def _initialize_from_data(self):
        """Initialize components from data."""
        if not self._data:
            return
        
        data_vec = Vector(self._data)
        data_mean = np.mean(data_vec)
        data_std = np.std(data_vec)
        
        # Initialize components around the data
        for i, component in enumerate(self.components):
            if isinstance(component, GaussianComponent):
                # Spread means around data mean
                offset = (i - self.max_components // 2) * data_std / self.max_components
                component.mean = data_mean + offset
                component.sigma = data_std
    
    def get_significant_components(self) -> List[int]:
        """Get indices of components with significant weight."""
        threshold = 1.0 / (10 * self.max_components)
        return [i for i, w in enumerate(self.mixing_weights) if w > threshold]
    
    def clone(self) -> 'DirichletProcessMixtureModel':
        """Create a copy of the model."""
        model = DirichletProcessMixtureModel(
            self.max_components, 
            self.alpha, 
            self.base_component_type
        )
        
        # Copy components
        model.components = [comp.clone() for comp in self.components] 
        
        # Copy stick lengths and weights
        model._stick_lengths = self._stick_lengths.copy()
        model.mixing_weights = self.mixing_weights.copy()
        model._active_components = self._active_components
        
        # Copy data
        model._data = self._data.copy()
        
        return model