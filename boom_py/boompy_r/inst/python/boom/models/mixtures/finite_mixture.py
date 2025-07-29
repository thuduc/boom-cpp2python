"""Finite mixture models with EM algorithm."""
import numpy as np
from typing import List, Optional, Union
from .base import MixtureModel, MixtureComponent
from ..base import PositiveParameter
from ...linalg import Vector, Matrix
from ...distributions.continuous import Normal


class GaussianComponent(MixtureComponent):
    """Gaussian mixture component."""
    
    def __init__(self, mean: float = 0.0, sigma: float = 1.0, weight: float = 1.0):
        """Initialize Gaussian component.
        
        Args:
            mean: Component mean
            sigma: Component standard deviation
            weight: Component weight
        """
        super().__init__(weight)
        self.mean = mean
        self._sigma_param = PositiveParameter(sigma, 'sigma')
    
    @property
    def sigma(self) -> float:
        """Get standard deviation."""
        return self._sigma_param.value
    
    @sigma.setter
    def sigma(self, value: float):
        """Set standard deviation."""
        if value <= 0:
            raise ValueError("Sigma must be positive")
        self._sigma_param.value = value
    
    @property
    def variance(self) -> float:
        """Get variance."""
        return self.sigma ** 2
    
    def logpdf(self, x: float) -> float:
        """Log probability density function."""
        return Normal(self.mean, self.sigma).logpdf(x)
    
    def pdf(self, x: float) -> float:
        """Probability density function."""
        return Normal(self.mean, self.sigma).pdf(x)
    
    def sample(self, rng) -> float:
        """Sample from the component."""
        return rng.rnorm(self.mean, self.sigma)
    
    def fit(self, data: Vector, weights: Vector):
        """Fit component to weighted data using MLE."""
        if len(data) != len(weights):
            raise ValueError("Data and weights must have same length")
        
        if len(weights) == 0:
            return
        
        # Weighted mean
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            self.mean = np.sum(weights * data) / weight_sum
            
            # Weighted variance
            weighted_sq_diff = weights * (data - self.mean) ** 2
            variance = np.sum(weighted_sq_diff) / weight_sum
            self.sigma = np.sqrt(max(variance, 1e-6))  # Prevent zero variance
    
    def clone(self) -> 'GaussianComponent':
        """Create a copy of the component."""
        return GaussianComponent(self.mean, self.sigma, self.weight)


class FiniteMixtureModel(MixtureModel):
    """Finite mixture model with EM algorithm."""
    
    def __init__(self, n_components: int, component_type: str = "gaussian"):
        """Initialize finite mixture model.
        
        Args:
            n_components: Number of components
            component_type: Type of components ("gaussian" supported)
        """
        super().__init__(n_components)
        self.component_type = component_type
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with default parameters."""
        self.components = []
        
        if self.component_type == "gaussian":
            for i in range(self.n_components):
                # Initialize with different means
                mean = i - (self.n_components - 1) / 2.0
                component = GaussianComponent(mean=mean, sigma=1.0)
                self.components.append(component)
        else:
            raise ValueError(f"Unknown component type: {self.component_type}")
    
    def initialize_from_data(self, method: str = "kmeans"):
        """Initialize components from data.
        
        Args:
            method: Initialization method ("kmeans", "random")
        """
        if not self._data:
            raise ValueError("No data available for initialization")
        
        data = Vector(self._data)
        n = len(data)
        
        if method == "random":
            # Random initialization
            from ...distributions import rng
            # Simple random sampling without replacement
            n_sample = min(self.n_components, n)
            indices = []
            available = list(range(n))
            for _ in range(n_sample):
                idx = int(rng.runif() * len(available))
                indices.append(available.pop(idx))
            
            for i, component in enumerate(self.components):
                if isinstance(component, GaussianComponent):
                    component.mean = data[indices[i]]
                    component.sigma = np.std(np.array(data))
        
        elif method == "kmeans":
            # Simple k-means initialization
            # Start with equally spaced quantiles
            sorted_data = np.sort(np.array(data))
            quantiles = np.linspace(0, 1, self.n_components + 1)[1:-1]
            means = [np.quantile(sorted_data, q) for q in quantiles]
            
            # Add endpoints
            if self.n_components == 1:
                means = [np.mean(np.array(data))]
            else:
                means = [sorted_data[0]] + means + [sorted_data[-1]]
                means = means[:self.n_components]
            
            for i, component in enumerate(self.components):
                if isinstance(component, GaussianComponent):
                    component.mean = means[i]
                    component.sigma = np.std(np.array(data)) / np.sqrt(self.n_components)
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def expectation_step(self) -> Matrix:
        """E-step: compute component responsibilities."""
        if not self._data:
            return Matrix.zero(0, self.n_components)
        
        n = len(self._data)
        responsibilities = Matrix.zero(n, self.n_components)
        
        for i, x in enumerate(self._data):
            posteriors = self.component_posteriors(x)
            responsibilities[i, :] = posteriors
        
        return responsibilities
    
    def maximization_step(self, responsibilities: Matrix):
        """M-step: update parameters."""
        if not self._data:
            return
        
        data = Vector(self._data)
        n = len(data)
        
        # Update mixing weights
        n_k = Vector([float(responsibilities[:, k].sum()) for k in range(self.n_components)])
        self.mixing_weights = n_k / n
        
        # Update component parameters
        for k, component in enumerate(self.components):
            weights = Vector(responsibilities[:, k])
            component.fit(data, weights)
    
    def fit(self, max_iter: int = 100, tol: float = 1e-6, init_method: str = "kmeans"):
        """Fit mixture model using EM algorithm.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            init_method: Initialization method
        """
        if not self._data:
            raise ValueError("No data to fit")
        
        # Initialize components
        self.initialize_from_data(init_method)
        
        prev_loglike = -np.inf
        
        for iteration in range(max_iter):
            # E-step
            responsibilities = self.expectation_step()
            
            # M-step
            self.maximization_step(responsibilities)
            
            # Check convergence
            current_loglike = self.loglike()
            
            if abs(current_loglike - prev_loglike) < tol:
                break
            
            prev_loglike = current_loglike
    
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        n = len(self._data)
        if n == 0:
            return 0.0
        
        # Number of free parameters
        # For Gaussian: each component has mean + variance, plus mixing weights
        n_params = self.n_components * 2 + (self.n_components - 1)
        
        return -2 * self.loglike() + n_params * np.log(n)
    
    def aic(self) -> float:
        """Akaike Information Criterion."""
        # Number of free parameters
        n_params = self.n_components * 2 + (self.n_components - 1)
        
        return -2 * self.loglike() + 2 * n_params
    
    def clone(self) -> 'FiniteMixtureModel':
        """Create a copy of the model."""
        model = FiniteMixtureModel(self.n_components, self.component_type)
        
        # Copy components
        model.components = [comp.clone() for comp in self.components]
        
        # Copy mixing weights
        model.mixing_weights = self.mixing_weights.copy()
        
        # Copy data
        model._data = self._data.copy()
        
        return model