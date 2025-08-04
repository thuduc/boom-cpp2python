"""Base classes for mixture models."""

import numpy as np
from typing import List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

from boom.models.base import Model
from boom.models.data import Data
from boom.linalg import Vector, Matrix
from boom.distributions import RNG


class MixtureData(Data):
    """Data point for mixture models."""
    
    def __init__(self, y: Union[float, np.ndarray, Vector], component: Optional[int] = None):
        """Initialize mixture data point.
        
        Args:
            y: Observation (scalar or vector)
            component: Component assignment (if known)
        """
        super().__init__()
        if isinstance(y, (list, np.ndarray)):
            self._y = Vector(y) if len(np.array(y).shape) == 1 else y
        elif isinstance(y, Vector):
            self._y = y.copy()
        else:
            self._y = float(y)
        
        self._component = component
    
    def y(self) -> Union[float, Vector]:
        """Get observation."""
        if isinstance(self._y, Vector):
            return self._y.copy()
        return self._y
    
    def component(self) -> Optional[int]:
        """Get component assignment."""
        return self._component
    
    def set_y(self, y: Union[float, np.ndarray, Vector]):
        """Set observation."""
        if isinstance(y, (list, np.ndarray)):
            self._y = Vector(y) if len(np.array(y).shape) == 1 else y
        elif isinstance(y, Vector):
            self._y = y.copy()
        else:
            self._y = float(y)
        self._notify_observers("y")
    
    def set_component(self, component: Optional[int]):
        """Set component assignment."""
        self._component = component
        self._notify_observers("component")
    
    def is_scalar(self) -> bool:
        """Check if observation is scalar."""
        return isinstance(self._y, (int, float))
    
    def dimension(self) -> int:
        """Get dimension of observation."""
        if self.is_scalar():
            return 1
        return len(self._y)
    
    def clone(self) -> 'MixtureData':
        """Create a copy of this data point."""
        return MixtureData(self._y, self._component)
    
    def __repr__(self) -> str:
        return f"MixtureData(y={self._y}, component={self._component})"


class MixtureModel(Model, ABC):
    """Base class for mixture models.
    
    A mixture model has the form:
    y_i ~ Σ_k π_k * f_k(y_i | θ_k)
    
    where:
    - π_k are mixing weights (sum to 1)
    - f_k are component densities
    - θ_k are component parameters
    """
    
    def __init__(self, n_components: int):
        """Initialize mixture model.
        
        Args:
            n_components: Number of mixture components
        """
        super().__init__()
        self._n_components = n_components
        self._mixing_weights = Vector(np.ones(n_components) / n_components)
        self._data: List[MixtureData] = []
    
    @property
    def n_components(self) -> int:
        """Get number of components."""
        return self._n_components
    
    def mixing_weights(self) -> Vector:
        """Get mixing weights."""
        return self._mixing_weights.copy()
    
    def set_mixing_weights(self, weights: Union[List[float], np.ndarray, Vector]):
        """Set mixing weights."""
        if isinstance(weights, (list, np.ndarray)):
            weights_vec = Vector(weights)
        elif isinstance(weights, Vector):
            weights_vec = weights.copy()
        else:
            raise ValueError(f"weights must be list, ndarray, or Vector, got {type(weights)}")
        
        if len(weights_vec) != self._n_components:
            raise ValueError(f"weights length {len(weights_vec)} doesn't match n_components {self._n_components}")
        
        if not np.allclose(np.sum(weights_vec.to_numpy()), 1.0):
            raise ValueError("Mixing weights must sum to 1")
        
        if np.any(weights_vec.to_numpy() < 0):
            raise ValueError("Mixing weights must be non-negative")
        
        self._mixing_weights = weights_vec
        self._notify_observers("mixing_weights")
    
    def add_data(self, data: Union[MixtureData, float, Vector, List]):
        """Add data to the model."""
        if isinstance(data, MixtureData):
            self._add_single_data(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, MixtureData):
                    self._add_single_data(item)
                else:
                    mix_data = MixtureData(item)
                    self._add_single_data(mix_data)
        else:
            mix_data = MixtureData(data)
            self._add_single_data(mix_data)
    
    def _add_single_data(self, data: MixtureData):
        """Add a single data point."""
        self._data.append(data)
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
    
    def get_data(self, i: int) -> MixtureData:
        """Get data point i."""
        return self._data[i]
    
    def n_observations(self) -> int:
        """Get number of observations."""
        return len(self._data)
    
    @abstractmethod
    def component_log_density(self, k: int, y: Union[float, Vector]) -> float:
        """Compute log density of observation y under component k.
        
        Args:
            k: Component index
            y: Observation
            
        Returns:
            Log density f_k(y | θ_k)
        """
        pass
    
    @abstractmethod
    def sample_component(self, k: int, rng: Optional[RNG] = None) -> Union[float, Vector]:
        """Sample from component k.
        
        Args:
            k: Component index
            rng: Random number generator
            
        Returns:
            Sample from component k
        """
        pass
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of the data."""
        if self.n_observations() == 0:
            return 0.0
        
        log_lik = 0.0
        
        for i in range(self.n_observations()):
            y = self._data[i].y()
            
            # Compute log P(y_i) = log Σ_k π_k * f_k(y_i)
            log_probs = []
            for k in range(self._n_components):
                log_weight = np.log(self._mixing_weights[k])
                log_density = self.component_log_density(k, y)
                log_probs.append(log_weight + log_density)
            
            log_lik += self._log_sum_exp(np.array(log_probs))
        
        return log_lik
    
    def posterior_probs(self) -> Matrix:
        """Compute posterior probabilities P(Z_i = k | y_i).
        
        Returns:
            Matrix where gamma[i, k] = P(Z_i = k | y_i)
        """
        n = self.n_observations()
        gamma = Matrix((n, self._n_components))
        
        for i in range(n):
            y = self._data[i].y()
            
            # Compute log P(Z_i = k | y_i) ∝ π_k * f_k(y_i)
            log_probs = []
            for k in range(self._n_components):
                log_weight = np.log(self._mixing_weights[k])
                log_density = self.component_log_density(k, y)
                log_probs.append(log_weight + log_density)
            
            # Normalize
            log_norm = self._log_sum_exp(np.array(log_probs))
            
            for k in range(self._n_components):
                gamma[i, k] = np.exp(log_probs[k] - log_norm)
        
        return gamma
    
    def predict_component_probs(self, y: Union[float, Vector]) -> Vector:
        """Predict component probabilities for new observation.
        
        Args:
            y: New observation
            
        Returns:
            Vector of component probabilities
        """
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(self._mixing_weights[k])
            log_density = self.component_log_density(k, y)
            log_probs.append(log_weight + log_density)
        
        # Normalize
        log_norm = self._log_sum_exp(np.array(log_probs))
        probs = np.exp(np.array(log_probs) - log_norm)
        
        return Vector(probs)
    
    def predict_density(self, y: Union[float, Vector]) -> float:
        """Predict density for new observation.
        
        Args:
            y: New observation
            
        Returns:
            Predicted density
        """
        log_probs = []
        for k in range(self._n_components):
            log_weight = np.log(self._mixing_weights[k])
            log_density = self.component_log_density(k, y)
            log_probs.append(log_weight + log_density)
        
        return np.exp(self._log_sum_exp(np.array(log_probs)))
    
    def classify(self, y: Union[float, Vector]) -> int:
        """Classify observation to most likely component.
        
        Args:
            y: Observation to classify
            
        Returns:
            Most likely component index
        """
        probs = self.predict_component_probs(y)
        return int(np.argmax(probs.to_numpy()))
    
    def sample(self, n: int = 1, rng: Optional[RNG] = None) -> List[Tuple[Union[float, Vector], int]]:
        """Sample from the mixture model.
        
        Args:
            n: Number of samples
            rng: Random number generator
            
        Returns:
            List of (observation, component) pairs
        """
        if rng is None:
            rng = RNG()
        
        samples = []
        weights = self._mixing_weights.to_numpy()
        
        for _ in range(n):
            # Sample component
            k = rng.choice(self._n_components, p=weights)
            
            # Sample from component
            y = self.sample_component(k, rng)
            
            samples.append((y, k))
        
        return samples
    
    def fit_em(self, max_iterations: int = 100, tolerance: float = 1e-6,
               verbose: bool = False) -> Tuple[List[float], bool]:
        """Fit model parameters using Expectation-Maximization algorithm.
        
        Args:
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance for log-likelihood
            verbose: Whether to print progress
            
        Returns:
            Tuple of (log_likelihood_history, converged)
        """
        if self.n_observations() == 0:
            raise ValueError("No data to fit")
        
        log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: compute posterior probabilities
            gamma = self.posterior_probs()
            
            # M-step: update parameters
            self._update_mixing_weights(gamma)
            self._update_component_params(gamma)
            
            # Compute log-likelihood
            current_log_likelihood = self.log_likelihood()
            log_likelihood_history.append(current_log_likelihood)
            
            if verbose:
                print(f"EM Iteration {iteration + 1}: Log-likelihood = {current_log_likelihood:.6f}")
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                if verbose:
                    print(f"EM converged after {iteration + 1} iterations")
                return log_likelihood_history, True
            
            prev_log_likelihood = current_log_likelihood
        
        if verbose:
            print(f"EM did not converge after {max_iterations} iterations")
        
        return log_likelihood_history, False
    
    def _update_mixing_weights(self, gamma: Matrix):
        """Update mixing weights."""
        n = self.n_observations()
        new_weights = np.zeros(self._n_components)
        
        for k in range(self._n_components):
            new_weights[k] = np.sum([gamma[i, k] for i in range(n)]) / n
        
        self.set_mixing_weights(new_weights)
    
    @abstractmethod
    def _update_component_params(self, gamma: Matrix):
        """Update component parameters (to be implemented by subclasses)."""
        pass
    
    def _log_sum_exp(self, log_values: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        if len(log_values) == 0:
            return -np.inf
        
        max_val = np.max(log_values)
        if max_val == -np.inf:
            return -np.inf
        
        return max_val + np.log(np.sum(np.exp(log_values - max_val)))
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Vectorize parameters for optimization."""
        params = []
        
        # Mixing weights (use log-ratio to ensure simplex constraint)
        if minimal and self._n_components > 1:
            weights = self._mixing_weights.to_numpy()
            log_ratios = np.log(weights[:-1] / weights[-1])
            params.extend(log_ratios)
        else:
            params.extend(self._mixing_weights.to_numpy())
        
        return Vector(params)
    
    def unvectorize_params(self, theta: Vector):
        """Set parameters from vector."""
        theta_array = theta.to_numpy()
        idx = 0
        
        # Mixing weights
        if self._n_components > 1:
            log_ratios = theta_array[idx:idx + self._n_components - 1]
            idx += self._n_components - 1
            
            # Convert from log-ratio to probabilities
            ratios = np.exp(log_ratios)
            weights = np.zeros(self._n_components)
            weights[:-1] = ratios
            weights[-1] = 1.0
            weights /= np.sum(weights)
            
            self.set_mixing_weights(weights)
    
    def clone(self) -> 'MixtureModel':
        """Create a copy of this model."""
        cloned = self.__class__(self._n_components)
        cloned.set_mixing_weights(self._mixing_weights)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(n_components={self._n_components}, "
                f"observations={self.n_observations()})")