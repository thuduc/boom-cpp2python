"""
Dirichlet Process Mixture Model implementation.

This module provides a Dirichlet Process mixture model for non-parametric
Bayesian clustering with an unknown number of components.
"""

import numpy as np
import scipy.stats
from typing import Any, Optional, Union, List, Tuple, Dict
from ...linalg import Vector, Matrix
from .base import MixtureModel, MixtureData


class DirichletProcessMixture(MixtureModel):
    """
    Dirichlet Process mixture model with Gaussian components.
    
    Uses the Chinese Restaurant Process representation for inference.
    The number of active components can grow dynamically.
    """
    
    def __init__(self, concentration: float, max_components: int = 50,
                 base_mean: Optional[Vector] = None,
                 base_precision: float = 1.0,
                 base_scale: float = 1.0,
                 base_dof: float = 3.0):
        """
        Initialize Dirichlet Process mixture model.
        
        Args:
            concentration: DP concentration parameter (α)
            max_components: Maximum number of components to track
            base_mean: Prior mean for component means
            base_precision: Prior precision for component means
            base_scale: Prior scale for component precisions
            base_dof: Prior degrees of freedom for component precisions
        """
        super().__init__(max_components)
        
        self._concentration = concentration
        self._max_components = max_components
        self._active_components = 0
        
        # Base distribution parameters (Normal-Inverse-Gamma)
        self._base_mean = base_mean
        self._base_precision = base_precision
        self._base_scale = base_scale
        self._base_dof = base_dof
        
        # Component parameters
        self._means = []
        self._precisions = []  # Precision (inverse variance) for each component
        
        # Component assignments for each observation
        self._assignments = []
        
        # Component counts (for CRP)
        self._component_counts = np.zeros(max_components, dtype=int)
        
        # Dimension (will be set when data is added)
        self._dimension = None
    
    def set_data(self, data: MixtureData) -> None:
        """Set the data and initialize assignments."""
        super().set_data(data)
        
        if self.n_observations() > 0:
            # Infer dimension from first observation
            first_obs = self._data[0].y()
            if isinstance(first_obs, Vector):
                self._dimension = len(first_obs.to_numpy())
            else:
                self._dimension = len(np.asarray(first_obs))
            
            # Set default base mean if not provided
            if self._base_mean is None:
                self._base_mean = Vector(np.zeros(self._dimension))
            
            # Initialize assignments randomly
            self._initialize_assignments()
    
    def _initialize_assignments(self) -> None:
        """Initialize component assignments using Chinese Restaurant Process."""
        self._assignments = []
        self._component_counts.fill(0)
        self._active_components = 0
        
        for i in range(self.n_observations()):
            # Compute probabilities for existing components and new component
            probs = []
            
            # Existing components
            for k in range(self._active_components):
                prob = self._component_counts[k] / (i + self._concentration)
                probs.append(prob)
            
            # New component
            prob_new = self._concentration / (i + self._concentration)
            probs.append(prob_new)
            
            # Sample assignment
            probs = np.array(probs)
            probs /= np.sum(probs)
            assignment = np.random.choice(len(probs), p=probs)
            
            if assignment == self._active_components:
                # Create new component
                self._create_new_component()
                self._active_components += 1
            
            self._assignments.append(assignment)
            self._component_counts[assignment] += 1
        
        # Update component parameters
        self._update_component_parameters()
    
    def _create_new_component(self) -> None:
        """Create a new component with parameters drawn from base distribution."""
        if self._active_components >= self._max_components:
            raise RuntimeError("Maximum number of components reached")
        
        # Sample mean from base distribution
        base_mean = self._base_mean.to_numpy()
        cov = np.eye(self._dimension) / self._base_precision
        mean = np.random.multivariate_normal(base_mean, cov)
        
        # Sample precision from base distribution (Gamma)
        precision = np.random.gamma(self._base_dof / 2, 2 / self._base_scale)
        
        self._means.append(Vector(mean))
        self._precisions.append(precision)
    
    def _update_component_parameters(self) -> None:
        """Update component parameters given current assignments."""
        for k in range(self._active_components):
            # Get observations assigned to component k
            assigned_obs = []
            for i, assignment in enumerate(self._assignments):
                if assignment == k:
                    obs = self._data[i].y()
                    if isinstance(obs, Vector):
                        assigned_obs.append(obs.to_numpy())
                    else:
                        assigned_obs.append(np.asarray(obs))
            
            if not assigned_obs:
                continue
            
            assigned_obs = np.array(assigned_obs)
            n_k = len(assigned_obs)
            
            # Update mean (posterior of Normal-Normal)
            sample_mean = np.mean(assigned_obs, axis=0)
            
            posterior_precision = self._base_precision + n_k * self._precisions[k]
            posterior_mean = (self._base_precision * self._base_mean.to_numpy() +
                            n_k * self._precisions[k] * sample_mean) / posterior_precision
            
            self._means[k] = Vector(posterior_mean)
            
            # Update precision (posterior of Gamma)
            sample_var = np.var(assigned_obs, axis=0, ddof=0)
            mean_sample_var = np.mean(sample_var)
            
            posterior_dof = self._base_dof + n_k
            posterior_scale = self._base_scale + 0.5 * n_k * mean_sample_var
            
            # For simplicity, use scalar precision
            self._precisions[k] = posterior_dof / (2 * posterior_scale)
    
    def component_log_density(self, component: int, observation: Any) -> float:
        """
        Compute log density of observation under specified component.
        
        Args:
            component: Component index
            observation: Observation
            
        Returns:
            Log probability density
        """
        if component >= self._active_components:
            raise ValueError(f"Component {component} not active")
        
        if isinstance(observation, Vector):
            y = observation.to_numpy()
        else:
            y = np.asarray(observation)
        
        mean = self._means[component].to_numpy()
        precision = self._precisions[component]
        
        # Multivariate normal with spherical covariance
        variance = 1.0 / precision
        log_density = scipy.stats.multivariate_normal.logpdf(
            y, mean=mean, cov=variance * np.eye(self._dimension)
        )
        
        return log_density
    
    def gibbs_step(self) -> None:
        """Perform one Gibbs sampling step."""
        # Update assignments
        for i in range(self.n_observations()):
            # Remove observation from current assignment
            old_assignment = self._assignments[i]
            self._component_counts[old_assignment] -= 1
            
            # If component becomes empty, consider removing it
            if self._component_counts[old_assignment] == 0:
                # For simplicity, keep empty components for now
                pass
            
            # Compute assignment probabilities
            probs = []
            
            # Existing components
            for k in range(self._active_components):
                count_prob = self._component_counts[k] / (self.n_observations() - 1 + self._concentration)
                data_prob = np.exp(self.component_log_density(k, self._data[i].y()))
                probs.append(count_prob * data_prob)
            
            # New component
            new_count_prob = self._concentration / (self.n_observations() - 1 + self._concentration)
            # Use base distribution predictive probability
            new_data_prob = self._base_predictive_density(self._data[i].y())
            probs.append(new_count_prob * new_data_prob)
            
            # Sample new assignment
            probs = np.array(probs)
            if np.sum(probs) == 0:
                probs = np.ones(len(probs))
            probs /= np.sum(probs)
            
            new_assignment = np.random.choice(len(probs), p=probs)
            
            if new_assignment == self._active_components:
                # Create new component
                if self._active_components < self._max_components:
                    self._create_new_component()
                    self._active_components += 1
                else:
                    # Fallback to existing component
                    new_assignment = np.random.choice(self._active_components)
            
            self._assignments[i] = new_assignment
            self._component_counts[new_assignment] += 1
        
        # Update component parameters
        self._update_component_parameters()
        self._notify_observers()
    
    def _base_predictive_density(self, observation: Any) -> float:
        """Compute predictive density under base distribution."""
        if isinstance(observation, Vector):
            y = observation.to_numpy()
        else:
            y = np.asarray(observation)
        
        # Marginal likelihood under Normal-Inverse-Gamma base
        # This is a multivariate t-distribution
        base_mean = self._base_mean.to_numpy()
        
        # Parameters of resulting t-distribution
        dof = self._base_dof
        scale = self._base_scale * (1 + 1/self._base_precision)
        
        # Simplified: use multivariate normal approximation
        variance = scale / (dof / 2)
        cov = variance * np.eye(self._dimension)
        
        density = scipy.stats.multivariate_normal.pdf(y, mean=base_mean, cov=cov)
        return density
    
    def predict_component(self, observation: Any) -> int:
        """Predict most likely component for new observation."""
        if self._active_components == 0:
            return 0
        
        probs = []
        for k in range(self._active_components):
            count_prob = self._component_counts[k] / self.n_observations()
            data_prob = np.exp(self.component_log_density(k, observation))
            probs.append(count_prob * data_prob)
        
        return int(np.argmax(probs))
    
    def get_active_components(self) -> int:
        """Get number of active components."""
        return self._active_components
    
    def get_component_assignments(self) -> List[int]:
        """Get component assignments for all observations."""
        return self._assignments.copy()
    
    def get_component_counts(self) -> np.ndarray:
        """Get counts for each component."""
        return self._component_counts[:self._active_components].copy()
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of current configuration."""
        if self._active_components == 0:
            return -np.inf
        
        log_lik = 0.0
        for i in range(self.n_observations()):
            assignment = self._assignments[i]
            log_lik += self.component_log_density(assignment, self._data[i].y())
        
        return log_lik
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        return {
            'concentration': self._concentration,
            'active_components': self._active_components,
            'means': [mean.to_numpy() for mean in self._means[:self._active_components]],
            'precisions': self._precisions[:self._active_components],
            'assignments': self._assignments.copy(),
            'component_counts': self._component_counts[:self._active_components].copy()
        }
    
    def set_concentration(self, concentration: float) -> None:
        """Set concentration parameter."""
        if concentration <= 0:
            raise ValueError("Concentration must be positive")
        self._concentration = concentration
        self._notify_observers()
    
    @property
    def concentration(self) -> float:
        """Get concentration parameter."""
        return self._concentration
    
    @property
    def active_components(self) -> int:
        """Get number of active components."""
        return self._active_components
    
    def m_step(self, posterior_probs: Matrix) -> None:
        """
        M-step not applicable for DP mixture - use Gibbs sampling instead.
        """
        raise NotImplementedError("Use gibbs_step() for DP mixture inference")


class DirichletProcessMixtureData(MixtureData):
    """Data container for Dirichlet Process mixture models."""
    
    def __init__(self, observations: Union[List[Vector], np.ndarray]):
        """
        Initialize with observations.
        
        Args:
            observations: List of Vector observations or 2D array
        """
        if isinstance(observations, np.ndarray):
            obs_list = [Vector(row) for row in observations]
        else:
            obs_list = observations
        
        super().__init__(obs_list)
    
    @property
    def dimension(self) -> int:
        """Get observation dimension."""
        if not self._observations:
            return 0
        return len(self._observations[0].y().to_numpy())