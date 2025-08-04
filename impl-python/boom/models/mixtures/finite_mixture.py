"""
Finite Mixture Model implementation.

This module provides a general finite mixture model framework that can
work with various component distributions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple, Dict, Callable
from ...linalg import Vector, Matrix
from .base import MixtureModel, MixtureData


class ComponentDistribution(ABC):
    """Abstract base class for mixture component distributions."""
    
    @abstractmethod
    def log_density(self, observation: Any) -> float:
        """Compute log density of observation."""
        pass
    
    @abstractmethod
    def update_parameters(self, observations: List[Any], weights: np.ndarray) -> None:
        """Update parameters given weighted observations."""
        pass
    
    @abstractmethod
    def sample(self) -> Any:
        """Sample from this component."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """Get component parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """Set component parameters."""
        pass


class GaussianComponent(ComponentDistribution):
    """Gaussian component distribution for mixture models."""
    
    def __init__(self, dimension: int, mean: Optional[np.ndarray] = None,
                 covariance: Optional[np.ndarray] = None):
        """
        Initialize Gaussian component.
        
        Args:
            dimension: Dimension of observations
            mean: Initial mean vector
            covariance: Initial covariance matrix
        """
        self._dimension = dimension
        self._mean = mean if mean is not None else np.zeros(dimension)
        self._covariance = covariance if covariance is not None else np.eye(dimension)
        self._inv_covariance = np.linalg.inv(self._covariance)
        self._log_det_covariance = np.linalg.slogdet(self._covariance)[1]
    
    def log_density(self, observation: Any) -> float:
        """Compute log density of observation."""
        if isinstance(observation, Vector):
            y = observation.to_numpy()
        else:
            y = np.asarray(observation)
        
        diff = y - self._mean
        mahalanobis = np.dot(diff, np.dot(self._inv_covariance, diff))
        
        log_density = -0.5 * (mahalanobis + self._log_det_covariance + 
                              self._dimension * np.log(2 * np.pi))
        return log_density
    
    def update_parameters(self, observations: List[Any], weights: np.ndarray) -> None:
        """Update parameters given weighted observations."""
        if len(observations) == 0:
            return
        
        # Convert observations to numpy arrays
        obs_array = []
        for obs in observations:
            if isinstance(obs, Vector):
                obs_array.append(obs.to_numpy())
            else:
                obs_array.append(np.asarray(obs))
        obs_array = np.array(obs_array)
        
        # Weighted mean
        total_weight = np.sum(weights)
        if total_weight < 1e-10:
            return
        
        weighted_mean = np.sum(obs_array * weights.reshape(-1, 1), axis=0) / total_weight
        self._mean = weighted_mean
        
        # Weighted covariance
        diff = obs_array - self._mean
        weighted_cov = np.zeros((self._dimension, self._dimension))
        for i in range(len(observations)):
            weighted_cov += weights[i] * np.outer(diff[i], diff[i])
        weighted_cov /= total_weight
        
        # Add regularization
        min_eigenval = np.min(np.linalg.eigvals(weighted_cov))
        if min_eigenval < 1e-6:
            weighted_cov += (1e-6 - min_eigenval) * np.eye(self._dimension)
        
        self._covariance = weighted_cov
        self._inv_covariance = np.linalg.inv(self._covariance)
        self._log_det_covariance = np.linalg.slogdet(self._covariance)[1]
    
    def sample(self) -> Vector:
        """Sample from this component."""
        sample = np.random.multivariate_normal(self._mean, self._covariance)
        return Vector(sample)
    
    def get_parameters(self) -> dict:
        """Get component parameters."""
        return {
            'mean': self._mean.copy(),
            'covariance': self._covariance.copy()
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set component parameters."""
        if 'mean' in params:
            self._mean = params['mean'].copy()
        if 'covariance' in params:
            self._covariance = params['covariance'].copy()
            self._inv_covariance = np.linalg.inv(self._covariance)
            self._log_det_covariance = np.linalg.slogdet(self._covariance)[1]


class CategoricalComponent(ComponentDistribution):
    """Categorical component distribution for mixture models."""
    
    def __init__(self, n_categories: int, probabilities: Optional[np.ndarray] = None):
        """
        Initialize categorical component.
        
        Args:
            n_categories: Number of categories
            probabilities: Initial category probabilities
        """
        self._n_categories = n_categories
        if probabilities is not None:
            self._probabilities = probabilities.copy()
        else:
            self._probabilities = np.ones(n_categories) / n_categories
    
    def log_density(self, observation: Any) -> float:
        """Compute log density of observation."""
        if isinstance(observation, (int, np.integer)):
            category = int(observation)
        else:
            category = int(observation)
        
        if category < 0 or category >= self._n_categories:
            return -np.inf
        
        return np.log(max(self._probabilities[category], 1e-10))
    
    def update_parameters(self, observations: List[Any], weights: np.ndarray) -> None:
        """Update parameters given weighted observations."""
        if len(observations) == 0:
            return
        
        # Count weighted occurrences
        weighted_counts = np.zeros(self._n_categories)
        for i, obs in enumerate(observations):
            if isinstance(obs, (int, np.integer)):
                category = int(obs)
            else:
                category = int(obs)
            
            if 0 <= category < self._n_categories:
                weighted_counts[category] += weights[i]
        
        # Update probabilities
        total_weight = np.sum(weighted_counts)
        if total_weight > 1e-10:
            self._probabilities = weighted_counts / total_weight
        
        # Add small regularization
        self._probabilities += 1e-10
        self._probabilities /= np.sum(self._probabilities)
    
    def sample(self) -> int:
        """Sample from this component."""
        return np.random.choice(self._n_categories, p=self._probabilities)
    
    def get_parameters(self) -> dict:
        """Get component parameters."""
        return {'probabilities': self._probabilities.copy()}
    
    def set_parameters(self, params: dict) -> None:
        """Set component parameters."""
        if 'probabilities' in params:
            self._probabilities = params['probabilities'].copy()


class FiniteMixture(MixtureModel):
    """
    General finite mixture model with configurable component distributions.
    """
    
    def __init__(self, components: List[ComponentDistribution]):
        """
        Initialize finite mixture model.
        
        Args:
            components: List of component distributions
        """
        super().__init__(len(components))
        self._components = components.copy()
    
    def component_log_density(self, component: int, observation: Any) -> float:
        """
        Compute log density of observation under specified component.
        
        Args:
            component: Component index
            observation: Observation
            
        Returns:
            Log probability density
        """
        if component >= len(self._components):
            raise ValueError(f"Component {component} does not exist")
        
        return self._components[component].log_density(observation)
    
    def m_step(self, posterior_probs: Matrix) -> None:
        """
        Perform M-step of EM algorithm.
        
        Args:
            posterior_probs: Posterior probabilities [n_obs x n_components]
        """
        n_obs = self.n_observations()
        
        # Update mixing weights
        for k in range(self._n_components):
            self._mixing_weights[k] = np.sum(posterior_probs[:, k]) / n_obs
        
        # Update component parameters
        observations = [self._data[i].y() for i in range(n_obs)]
        
        for k in range(self._n_components):
            weights = posterior_probs[:, k]
            self._components[k].update_parameters(observations, weights)
        
        self._notify_observers()
    
    def predict_component(self, observation: Any) -> int:
        """Predict most likely component for new observation."""
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(max(self._mixing_weights[k], 1e-10))
            log_density = self.component_log_density(k, observation)
            log_probs.append(log_weight + log_density)
        
        return int(np.argmax(log_probs))
    
    def predict_proba(self, observation: Any) -> Vector:
        """Predict component probabilities for new observation."""
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(max(self._mixing_weights[k], 1e-10))
            log_density = self.component_log_density(k, observation)
            log_probs.append(log_weight + log_density)
        
        # Convert to probabilities using log-sum-exp
        log_probs = np.array(log_probs)
        max_log_prob = np.max(log_probs)
        exp_probs = np.exp(log_probs - max_log_prob)
        probs = exp_probs / np.sum(exp_probs)
        
        return Vector(probs)
    
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Generate samples from the mixture model."""
        samples = []
        
        for _ in range(n_samples):
            # Sample component
            component = np.random.choice(self._n_components, p=self._mixing_weights)
            
            # Sample from component
            sample = self._components[component].sample()
            samples.append(sample)
        
        return samples
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        component_params = []
        for comp in self._components:
            component_params.append(comp.get_parameters())
        
        return {
            'mixing_weights': self._mixing_weights.copy(),
            'components': component_params
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set all model parameters."""
        if 'mixing_weights' in params:
            self._mixing_weights = params['mixing_weights'].copy()
        
        if 'components' in params:
            component_params = params['components']
            for i, comp_params in enumerate(component_params):
                if i < len(self._components):
                    self._components[i].set_parameters(comp_params)
        
        self._notify_observers()
    
    def add_component(self, component: ComponentDistribution) -> None:
        """Add a new component to the mixture."""
        self._components.append(component)
        self._n_components += 1
        
        # Update mixing weights
        old_weights = self._mixing_weights.copy()
        self._mixing_weights = np.zeros(self._n_components)
        
        # Redistribute weights
        if len(old_weights) > 0:
            weight_factor = 0.9  # Reserve some weight for new component
            self._mixing_weights[:-1] = old_weights * weight_factor
            self._mixing_weights[-1] = 1.0 - np.sum(self._mixing_weights[:-1])
        else:
            self._mixing_weights[-1] = 1.0
        
        self._notify_observers()
    
    def remove_component(self, component_idx: int) -> None:
        """Remove a component from the mixture."""
        if component_idx >= self._n_components:
            raise ValueError(f"Component {component_idx} does not exist")
        
        # Remove component
        del self._components[component_idx]
        self._n_components -= 1
        
        # Update mixing weights
        old_weights = self._mixing_weights.copy()
        self._mixing_weights = np.zeros(self._n_components)
        
        # Redistribute weights (excluding removed component)
        remaining_weight = 1.0 - old_weights[component_idx]
        idx = 0
        for i in range(len(old_weights)):
            if i != component_idx:
                self._mixing_weights[idx] = old_weights[i] / remaining_weight
                idx += 1
        
        self._notify_observers()
    
    @property
    def components(self) -> List[ComponentDistribution]:
        """Get component distributions."""
        return self._components.copy()


def create_gaussian_mixture(n_components: int, dimension: int) -> FiniteMixture:
    """
    Create a finite mixture with Gaussian components.
    
    Args:
        n_components: Number of components
        dimension: Dimension of observations
        
    Returns:
        FiniteMixture with Gaussian components
    """
    components = []
    for _ in range(n_components):
        component = GaussianComponent(dimension)
        components.append(component)
    
    return FiniteMixture(components)


def create_categorical_mixture(n_components: int, n_categories: int) -> FiniteMixture:
    """
    Create a finite mixture with categorical components.
    
    Args:
        n_components: Number of components
        n_categories: Number of categories for each component
        
    Returns:
        FiniteMixture with categorical components
    """
    components = []
    for _ in range(n_components):
        component = CategoricalComponent(n_categories)
        components.append(component)
    
    return FiniteMixture(components)