"""Base classes for BOOM models."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import numpy as np
from ..linalg import Vector, Matrix


class Data(ABC):
    """Abstract base class for all data types in BOOM.
    
    This mimics the C++ BOOM::Data class, providing a common interface
    for all data objects that can be used in models.
    """
    
    def __init__(self):
        """Initialize data object."""
        self._missing = False
    
    def is_missing(self) -> bool:
        """Check if data is missing."""
        return self._missing
    
    def set_missing(self, missing: bool = True):
        """Set missing status."""
        self._missing = missing
    
    @abstractmethod
    def clone(self) -> 'Data':
        """Create a deep copy of the data object."""
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(missing={self._missing})"


class Model(ABC):
    """Abstract base class for all BOOM models.
    
    This provides the basic interface that all BOOM models must satisfy,
    including methods for computing log likelihood, sampling, and parameter
    management.
    """
    
    def __init__(self):
        """Initialize model."""
        self._data: List[Data] = []
        self._parameters: Dict[str, Any] = {}
        self._observers: List[Any] = []
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Data):
        """Add a data point to the model."""
        self._data.append(data)
        self._notify_observers()
    
    def add_data_batch(self, data_list: List[Data]):
        """Add multiple data points efficiently."""
        self._data.extend(data_list)
        self._notify_observers()
    
    def clear_data(self):
        """Remove all data from the model."""
        self._data.clear()
        self._notify_observers()
    
    def data(self) -> List[Data]:
        """Get all data points."""
        return self._data.copy()
    
    def sample_size(self) -> int:
        """Get number of data points."""
        return len(self._data)
    
    # ============================================================================
    # Parameter Management
    # ============================================================================
    
    def set_parameter(self, name: str, value: Any):
        """Set a parameter value."""
        old_value = self._parameters.get(name)
        self._parameters[name] = value
        if old_value != value:
            self._notify_observers()
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self._parameters.get(name, default)
    
    def parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self._parameters.keys())
    
    def parameters(self) -> Dict[str, Any]:
        """Get all parameters as dictionary."""
        return self._parameters.copy()
    
    # ============================================================================
    # Abstract Methods - must be implemented by subclasses
    # ============================================================================
    
    @abstractmethod
    def log_likelihood(self, data: Optional[List[Data]] = None) -> float:
        """Compute log likelihood.
        
        Args:
            data: Data to compute likelihood for. If None, uses model's data.
            
        Returns:
            Log likelihood value.
        """
        pass
    
    @abstractmethod
    def simulate_data(self, n: Optional[int] = None) -> List[Data]:
        """Simulate data from the model.
        
        Args:
            n: Number of data points to simulate. If None, uses current sample size.
            
        Returns:
            List of simulated data points.
        """
        pass
    
    # ============================================================================
    # Optional Methods - can be overridden by subclasses
    # ============================================================================
    
    def log_prior(self) -> float:
        """Compute log prior probability.
        
        Default implementation returns 0 (uniform prior).
        """
        return 0.0
    
    def log_posterior(self, data: Optional[List[Data]] = None) -> float:
        """Compute log posterior probability.
        
        Default implementation is log_likelihood + log_prior.
        """
        return self.log_likelihood(data) + self.log_prior()
    
    def mle(self):
        """Compute maximum likelihood estimate.
        
        Default implementation does nothing. Subclasses should override
        to provide specific MLE computation.
        """
        pass
    
    def gradient(self, data: Optional[List[Data]] = None) -> Vector:
        """Compute gradient of log likelihood.
        
        Default implementation raises NotImplementedError.
        Subclasses should override if gradient is available.
        """
        raise NotImplementedError("Gradient not implemented for this model")
    
    def hessian(self, data: Optional[List[Data]] = None) -> Matrix:
        """Compute Hessian of log likelihood.
        
        Default implementation raises NotImplementedError.
        Subclasses should override if Hessian is available.
        """
        raise NotImplementedError("Hessian not implemented for this model")
    
    # ============================================================================
    # Parameter Vectorization (for optimization/MCMC)
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form.
        
        Args:
            minimal: If True, use minimal representation.
            
        Returns:
            Parameter vector.
        """
        # Default implementation - subclasses should override
        return Vector([])
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form.
        
        Args:
            theta: Parameter vector.
            minimal: If True, interpret as minimal representation.
        """
        # Default implementation - subclasses should override
        pass
    
    # ============================================================================
    # Observer Pattern for Parameter Changes
    # ============================================================================
    
    def add_observer(self, observer):
        """Add an observer that gets notified of parameter changes."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer):
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self):
        """Notify all observers of parameter changes."""
        for observer in self._observers:
            if hasattr(observer, 'update'):
                observer.update(self)
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'Model':
        """Create a deep copy of the model.
        
        Default implementation raises NotImplementedError.
        Subclasses should override.
        """
        raise NotImplementedError("Clone not implemented for this model")
    
    def __str__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"sample_size={self.sample_size()}, "
                f"parameters={list(self._parameters.keys())})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)


class PriorMixin:
    """Mixin class for models with explicit prior distributions.
    
    This provides a framework for models that have explicit prior
    distributions on their parameters.
    """
    
    def __init__(self):
        """Initialize prior mixin."""
        self._priors: Dict[str, Any] = {}
    
    def set_prior(self, param_name: str, prior):
        """Set prior distribution for a parameter.
        
        Args:
            param_name: Name of the parameter.
            prior: Prior distribution object.
        """
        self._priors[param_name] = prior
    
    def get_prior(self, param_name: str):
        """Get prior distribution for a parameter.
        
        Args:
            param_name: Name of the parameter.
            
        Returns:
            Prior distribution object or None if not set.
        """
        return self._priors.get(param_name)
    
    def has_prior(self, param_name: str) -> bool:
        """Check if parameter has a prior."""
        return param_name in self._priors
    
    def log_prior(self) -> float:
        """Compute log prior probability.
        
        Sums log prior probabilities across all parameters.
        """
        log_prior_sum = 0.0
        
        for param_name, prior in self._priors.items():
            if hasattr(self, 'get_parameter'):
                param_value = self.get_parameter(param_name)
                if param_value is not None and hasattr(prior, 'logpdf'):
                    log_prior_sum += prior.logpdf(param_value)
        
        return log_prior_sum


class LoglikeModel(Model):
    """Base class for models where log likelihood is the primary interface.
    
    This provides some default implementations based on log likelihood
    computation.
    """
    
    def likelihood(self, data: Optional[List[Data]] = None) -> float:
        """Compute likelihood (non-log).
        
        Args:
            data: Data to compute likelihood for.
            
        Returns:
            Likelihood value.
        """
        return np.exp(self.log_likelihood(data))
    
    def AIC(self, data: Optional[List[Data]] = None) -> float:
        """Compute Akaike Information Criterion.
        
        Args:
            data: Data to compute AIC for.
            
        Returns:
            AIC value.
        """
        k = len(self.vectorize_params())  # Number of parameters
        return 2 * k - 2 * self.log_likelihood(data)
    
    def BIC(self, data: Optional[List[Data]] = None) -> float:
        """Compute Bayesian Information Criterion.
        
        Args:
            data: Data to compute BIC for.
            
        Returns:
            BIC value.
        """
        if data is None:
            data = self._data
        
        k = len(self.vectorize_params())  # Number of parameters
        n = len(data)  # Sample size
        
        return np.log(n) * k - 2 * self.log_likelihood(data)
    
    def deviance(self, data: Optional[List[Data]] = None) -> float:
        """Compute deviance.
        
        Args:
            data: Data to compute deviance for.
            
        Returns:
            Deviance value.
        """
        return -2 * self.log_likelihood(data)


class ConjugateModel(LoglikeModel):
    """Base class for models with conjugate priors.
    
    This provides a framework for models where the posterior
    distribution has a closed form.
    """
    
    @abstractmethod
    def posterior_mode(self):
        """Compute posterior mode.
        
        For conjugate models, this often has a closed form.
        """
        pass
    
    @abstractmethod
    def posterior_mean(self):
        """Compute posterior mean.
        
        For conjugate models, this often has a closed form.
        """
        pass
    
    def sample_posterior(self, n: int = 1) -> List:
        """Sample from posterior distribution.
        
        Default implementation raises NotImplementedError.
        Subclasses should override for specific sampling.
        """
        raise NotImplementedError("Posterior sampling not implemented")