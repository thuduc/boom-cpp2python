"""Base classes for BOOM models."""
import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from ..linalg import Vector, Matrix


class Data:
    """Base class for data objects."""
    pass


class Parameter(ABC):
    """Base class for model parameters."""
    
    def __init__(self, value: Any, name: str = ""):
        """Initialize parameter.
        
        Args:
            value: Initial parameter value
            name: Optional parameter name
        """
        self._value = value
        self.name = name
        self._fixed = False
    
    @property
    def value(self):
        """Get parameter value."""
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Set parameter value."""
        if self._fixed:
            raise ValueError(f"Cannot modify fixed parameter {self.name}")
        self._validate(new_value)
        self._value = new_value
    
    @abstractmethod
    def _validate(self, value):
        """Validate parameter value."""
        pass
    
    def fix(self):
        """Fix parameter (prevent updates)."""
        self._fixed = True
    
    def unfix(self):
        """Unfix parameter (allow updates)."""
        self._fixed = False
    
    @property
    def is_fixed(self):
        """Check if parameter is fixed."""
        return self._fixed
    
    @abstractmethod
    def size(self) -> int:
        """Number of elements in parameter."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value}, name='{self.name}')"


class ScalarParameter(Parameter):
    """Scalar parameter."""
    
    def _validate(self, value):
        if not np.isscalar(value):
            raise ValueError("Value must be scalar")
    
    def size(self):
        return 1


class PositiveParameter(ScalarParameter):
    """Positive scalar parameter."""
    
    def _validate(self, value):
        super()._validate(value)
        if value <= 0:
            raise ValueError("Value must be positive")


class ProbabilityParameter(ScalarParameter):
    """Probability parameter (in [0, 1])."""
    
    def _validate(self, value):
        super()._validate(value)
        if not 0 <= value <= 1:
            raise ValueError("Value must be in [0, 1]")


class VectorParameter(Parameter):
    """Vector parameter."""
    
    def _validate(self, value):
        self._value = Vector(value)
    
    def size(self):
        return len(self._value)


class MatrixParameter(Parameter):
    """Matrix parameter."""
    
    def _validate(self, value):
        self._value = Matrix(value)
    
    def size(self):
        return self._value.size


class SufStat:
    """Base class for sufficient statistics."""
    
    @abstractmethod
    def update(self, data):
        """Update sufficient statistics with new data."""
        pass
    
    @abstractmethod
    def combine(self, other: 'SufStat'):
        """Combine with another sufficient statistic."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear sufficient statistics."""
        pass
    
    @property
    @abstractmethod
    def sample_size(self) -> int:
        """Number of observations."""
        pass


class Model(ABC):
    """Base class for statistical models."""
    
    def __init__(self):
        """Initialize model."""
        self._data = []
        self._params = {}
        self._samplers = []
    
    @abstractmethod
    def clear_data(self):
        """Clear all data from the model."""
        pass
    
    @abstractmethod
    def add_data(self, data):
        """Add data to the model."""
        pass
    
    @property
    def sample_size(self) -> int:
        """Number of data points."""
        return len(self._data)
    
    @abstractmethod
    def loglike(self, data=None) -> float:
        """Log likelihood of the data."""
        pass
    
    def likelihood(self, data=None) -> float:
        """Likelihood of the data."""
        return np.exp(self.loglike(data))
    
    @abstractmethod
    def simulate(self, n: int = 1, rng=None) -> Union[Data, List[Data]]:
        """Simulate data from the model."""
        pass
    
    def parameter_vector(self) -> Vector:
        """Get all parameters as a vector."""
        values = []
        for param in self._params.values():
            if isinstance(param.value, (int, float)):
                values.append(param.value)
            else:
                values.extend(np.ravel(param.value))
        return Vector(values)
    
    def vectorize_params(self, include_fixed: bool = True) -> Vector:
        """Get parameter vector."""
        values = []
        for param in self._params.values():
            if not include_fixed and param.is_fixed:
                continue
            if isinstance(param.value, (int, float)):
                values.append(param.value)
            else:
                values.extend(np.ravel(param.value))
        return Vector(values)
    
    def unvectorize_params(self, v: Vector, include_fixed: bool = True):
        """Set parameters from vector."""
        idx = 0
        for param in self._params.values():
            if not include_fixed and param.is_fixed:
                continue
            size = param.size()
            if size == 1:
                param.value = float(v[idx])
            else:
                param_shape = param.value.shape
                param.value = v[idx:idx+size].reshape(param_shape)
            idx += size
    
    def set_data(self, data):
        """Set data (replaces existing data)."""
        self.clear_data()
        if hasattr(data, '__iter__'):
            for d in data:
                self.add_data(d)
        else:
            self.add_data(data)
    
    def register_sampler(self, sampler):
        """Register a posterior sampler."""
        self._samplers.append(sampler)
    
    def sample_posterior(self):
        """Sample from posterior using registered samplers."""
        for sampler in self._samplers:
            sampler.draw()
    
    @abstractmethod
    def clone(self) -> 'Model':
        """Create a copy of the model."""
        pass


class MixtureComponent(Model):
    """Base class for mixture components."""
    
    @abstractmethod
    def add_mixture_data(self, data, weight: float):
        """Add weighted data for EM algorithm."""
        pass
    
    @abstractmethod
    def clear_mixture_data(self):
        """Clear mixture data."""
        pass


class PriorModel(Model):
    """Base class for models that can serve as priors."""
    
    @abstractmethod
    def logp(self, theta) -> float:
        """Log prior density."""
        pass
    
    def log_prior(self, theta) -> float:
        """Alias for logp."""
        return self.logp(theta)