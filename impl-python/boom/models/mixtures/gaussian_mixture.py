"""
Gaussian Mixture Model implementation.

This module provides a mixture model with Gaussian component distributions.
"""

import numpy as np
import scipy.stats
from typing import Any, Optional, Union, List, Tuple
from ...linalg import Vector, Matrix
from .base import MixtureModel, MixtureData


class GaussianMixture(MixtureModel):
    """
    Gaussian mixture model with multivariate Gaussian components.
    
    Each component is a multivariate Gaussian distribution with its own
    mean vector and covariance matrix.
    """
    
    def __init__(self, n_components: int, dimension: int):
        """
        Initialize Gaussian mixture model.
        
        Args:
            n_components: Number of mixture components
            dimension: Dimension of observations
        """
        super().__init__(n_components)
        self._dimension = dimension
        
        # Component parameters
        self._means = np.zeros((n_components, dimension))
        self._covariances = np.array([np.eye(dimension) for _ in range(n_components)])
        self._inv_covariances = np.array([np.eye(dimension) for _ in range(n_components)])
        self._log_det_covariances = np.zeros(n_components)
        
        # Initialize with random parameters
        self._initialize_parameters()
        self._update_derived_quantities()
    
    def _initialize_parameters(self) -> None:
        """Initialize parameters with reasonable defaults."""
        # Random means
        self._means = np.random.randn(self._n_components, self._dimension)
        
        # Identity covariances (will be updated during EM)
        for k in range(self._n_components):
            self._covariances[k] = np.eye(self._dimension)
    
    def _update_derived_quantities(self) -> None:
        """Update inverse covariances and log determinants."""
        for k in range(self._n_components):
            try:
                self._inv_covariances[k] = np.linalg.inv(self._covariances[k])
                sign, log_det = np.linalg.slogdet(self._covariances[k])
                if sign <= 0:
                    raise np.linalg.LinAlgError("Non-positive definite covariance")
                self._log_det_covariances[k] = log_det
            except np.linalg.LinAlgError:
                # Regularize covariance matrix
                reg_cov = self._covariances[k] + 1e-6 * np.eye(self._dimension)
                self._covariances[k] = reg_cov
                self._inv_covariances[k] = np.linalg.inv(reg_cov)
                sign, log_det = np.linalg.slogdet(reg_cov)
                self._log_det_covariances[k] = log_det
    
    def component_log_density(self, component: int, observation: Any) -> float:
        """
        Compute log density of observation under specified component.
        
        Args:
            component: Component index
            observation: Observation (Vector or array-like)
            
        Returns:
            Log probability density
        """
        if isinstance(observation, Vector):
            y = observation.to_numpy()
        else:
            y = np.asarray(observation)
        
        if len(y) != self._dimension:
            raise ValueError(f"Observation dimension {len(y)} != model dimension {self._dimension}")
        
        mean = self._means[component]
        inv_cov = self._inv_covariances[component]
        log_det = self._log_det_covariances[component]
        
        # Compute log density: -0.5 * [(y-μ)ᵀ Σ⁻¹ (y-μ) + log|Σ| + d*log(2π)]
        diff = y - mean
        mahalanobis = np.dot(diff, np.dot(inv_cov, diff))
        
        log_density = -0.5 * (mahalanobis + log_det + self._dimension * np.log(2 * np.pi))
        return log_density
    
    def m_step(self, posterior_probs: Matrix) -> None:
        """
        Perform M-step of EM algorithm.
        
        Args:
            posterior_probs: Posterior probabilities [n_obs x n_components]
        """
        n_obs = self.n_observations()
        
        for k in range(self._n_components):
            # Effective sample size for component k
            n_k = np.sum(posterior_probs[:, k])
            
            if n_k < 1e-10:
                # Component has no support - use prior
                continue
            
            # Update mixing weight
            self._mixing_weights[k] = n_k / n_obs
            
            # Update mean
            weighted_sum = np.zeros(self._dimension)
            for i in range(n_obs):
                if isinstance(self._data[i].y(), Vector):
                    y = self._data[i].y().to_numpy()
                else:
                    y = np.asarray(self._data[i].y())
                weighted_sum += posterior_probs[i, k] * y
            
            self._means[k] = weighted_sum / n_k
            
            # Update covariance
            weighted_cov = np.zeros((self._dimension, self._dimension))
            for i in range(n_obs):
                if isinstance(self._data[i].y(), Vector):
                    y = self._data[i].y().to_numpy()
                else:
                    y = np.asarray(self._data[i].y())
                diff = y - self._means[k]
                weighted_cov += posterior_probs[i, k] * np.outer(diff, diff)
            
            self._covariances[k] = weighted_cov / n_k
            
            # Add regularization to ensure positive definiteness
            min_eigenval = np.min(np.linalg.eigvals(self._covariances[k]))
            if min_eigenval < 1e-6:
                self._covariances[k] += (1e-6 - min_eigenval) * np.eye(self._dimension)
        
        # Update derived quantities
        self._update_derived_quantities()
        self._notify_observers()
    
    def predict_component(self, observation: Any) -> int:
        """
        Predict most likely component for new observation.
        
        Args:
            observation: New observation
            
        Returns:
            Most likely component index
        """
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(self._mixing_weights[k])
            log_density = self.component_log_density(k, observation)
            log_probs.append(log_weight + log_density)
        
        return int(np.argmax(log_probs))
    
    def predict_proba(self, observation: Any) -> Vector:
        """
        Predict component probabilities for new observation.
        
        Args:
            observation: New observation
            
        Returns:
            Vector of component probabilities
        """
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(self._mixing_weights[k])
            log_density = self.component_log_density(k, observation)
            log_probs.append(log_weight + log_density)
        
        # Convert to probabilities using log-sum-exp
        log_probs = np.array(log_probs)
        max_log_prob = np.max(log_probs)
        exp_probs = np.exp(log_probs - max_log_prob)
        probs = exp_probs / np.sum(exp_probs)
        
        return Vector(probs)
    
    def sample(self, n_samples: int = 1) -> List[Vector]:
        """
        Generate samples from the mixture model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of sampled vectors
        """
        samples = []
        
        for _ in range(n_samples):
            # Sample component
            component = np.random.choice(self._n_components, p=self._mixing_weights)
            
            # Sample from component
            mean = self._means[component]
            cov = self._covariances[component]
            sample = np.random.multivariate_normal(mean, cov)
            samples.append(Vector(sample))
        
        return samples
    
    def get_parameters(self) -> dict:
        """Get all model parameters."""
        return {
            'mixing_weights': self._mixing_weights.copy(),
            'means': self._means.copy(),
            'covariances': self._covariances.copy(),
            'dimension': self._dimension
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set all model parameters."""
        if 'mixing_weights' in params:
            self._mixing_weights = params['mixing_weights'].copy()
        if 'means' in params:
            self._means = params['means'].copy()
        if 'covariances' in params:
            self._covariances = params['covariances'].copy()
            self._update_derived_quantities()
        if 'dimension' in params:
            self._dimension = params['dimension']
        
        self._notify_observers()
    
    def vectorize_params(self) -> Vector:
        """
        Convert parameters to vector form.
        
        Returns:
            Parameter vector
        """
        # Pack: mixing_weights (K-1), means (K*d), covariances (K*d*(d+1)/2)
        params_list = []
        
        # Mixing weights (omit last one due to sum-to-1 constraint)
        params_list.extend(self._mixing_weights[:-1])
        
        # Means
        params_list.extend(self._means.flatten())
        
        # Covariances (upper triangular parts)
        for k in range(self._n_components):
            cov = self._covariances[k]
            # Extract upper triangular part including diagonal
            for i in range(self._dimension):
                for j in range(i, self._dimension):
                    params_list.append(cov[i, j])
        
        return Vector(np.array(params_list))
    
    def unvectorize_params(self, params: Vector) -> None:
        """
        Set parameters from vector form.
        
        Args:
            params: Parameter vector
        """
        params_array = params.to_numpy()
        idx = 0
        
        # Mixing weights
        self._mixing_weights[:-1] = params_array[idx:idx + self._n_components - 1]
        self._mixing_weights[-1] = 1.0 - np.sum(self._mixing_weights[:-1])
        idx += self._n_components - 1
        
        # Means
        n_mean_params = self._n_components * self._dimension
        means_flat = params_array[idx:idx + n_mean_params]
        self._means = means_flat.reshape((self._n_components, self._dimension))
        idx += n_mean_params
        
        # Covariances
        for k in range(self._n_components):
            cov = np.zeros((self._dimension, self._dimension))
            for i in range(self._dimension):
                for j in range(i, self._dimension):
                    cov[i, j] = params_array[idx]
                    if i != j:
                        cov[j, i] = params_array[idx]  # Symmetric
                    idx += 1
            self._covariances[k] = cov
        
        self._update_derived_quantities()
        self._notify_observers()
    
    @property
    def dimension(self) -> int:
        """Get observation dimension."""
        return self._dimension
    
    @property
    def means(self) -> np.ndarray:
        """Get component means."""
        return self._means.copy()
    
    @property
    def covariances(self) -> np.ndarray:
        """Get component covariances."""
        return self._covariances.copy()


class GaussianMixtureData(MixtureData):
    """Data container for Gaussian mixture models."""
    
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
        
        # Validate dimensions
        if obs_list:
            dim = len(obs_list[0].to_numpy())
            for i, obs in enumerate(obs_list):
                if len(obs.to_numpy()) != dim:
                    raise ValueError(f"Observation {i} has dimension {len(obs.to_numpy())}, expected {dim}")
    
    @property
    def dimension(self) -> int:
        """Get observation dimension."""
        if not self._observations:
            return 0
        return len(self._observations[0].y().to_numpy())
    
    def get_data_matrix(self) -> np.ndarray:
        """
        Get observations as a 2D numpy array.
        
        Returns:
            Data matrix [n_observations x dimension]
        """
        if not self._observations:
            return np.array([])
        
        dim = self.dimension
        n_obs = len(self._observations)
        data_matrix = np.zeros((n_obs, dim))
        
        for i, obs in enumerate(self._observations):
            data_matrix[i] = obs.y().to_numpy()
        
        return data_matrix