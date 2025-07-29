"""Gaussian (Normal) models for BOOM."""
import numpy as np
from typing import Union, List, Optional
from .base import Model, PriorModel, Parameter
from .data import DoubleData
from .sufstat import GaussianSuf
from ..distributions.rng import GlobalRng
from ..distributions.continuous import Normal


class GaussianModel(Model, PriorModel):
    """Univariate Gaussian model."""
    
    def __init__(self, mean: float = 0.0, sd: float = 1.0):
        """Initialize Gaussian model.
        
        Args:
            mean: Mean parameter
            sd: Standard deviation (must be positive)
        """
        super().__init__()
        self._params['mean'] = Parameter(mean, 'mean')
        self._params['sd'] = Parameter(sd, 'sd')
        self._params['variance'] = Parameter(sd**2, 'variance')
        self._suf = GaussianSuf()
        self._dist = Normal(mean, sd)
    
    @property
    def mean(self) -> float:
        """Get mean parameter."""
        return self._params['mean'].value
    
    @mean.setter
    def mean(self, value: float):
        """Set mean parameter."""
        self._params['mean'].value = value
        self._dist.mean_param = value
    
    @property
    def sd(self) -> float:
        """Get standard deviation."""
        return self._params['sd'].value
    
    @sd.setter
    def sd(self, value: float):
        """Set standard deviation."""
        if value <= 0:
            raise ValueError("Standard deviation must be positive")
        self._params['sd'].value = value
        self._params['variance'].value = value ** 2
        self._dist.sd = value
        self._dist.variance_param = value ** 2
    
    @property
    def variance(self) -> float:
        """Get variance."""
        return self._params['variance'].value
    
    @variance.setter
    def variance(self, value: float):
        """Set variance."""
        if value <= 0:
            raise ValueError("Variance must be positive")
        self._params['variance'].value = value
        sd = np.sqrt(value)
        self._params['sd'].value = sd
        self._dist.sd = sd
        self._dist.variance_param = value
    
    @property
    def precision(self) -> float:
        """Get precision (1/variance)."""
        return 1.0 / self.variance
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._suf.clear()
    
    def add_data(self, data: Union[float, DoubleData]):
        """Add single observation."""
        if not isinstance(data, DoubleData):
            data = DoubleData(data)
        self._data.append(data)
        self._suf.update(data)
    
    def set_data(self, data: Union[List[float], np.ndarray]):
        """Set data (replaces existing)."""
        self.clear_data()
        for x in data:
            self.add_data(x)
    
    def suf(self) -> GaussianSuf:
        """Get sufficient statistics."""
        return self._suf
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            n = self._suf.n
            if n == 0:
                return 0.0
            
            centered_ss = self._suf.centered_sumsq(self.mean)
            return -0.5 * n * np.log(2 * np.pi * self.variance) - \
                   0.5 * centered_ss / self.variance
        else:
            # Single observation
            if isinstance(data, DoubleData):
                x = data.value
            else:
                x = float(data)
            return self._dist.logpdf(x)
    
    def pdf(self, x: float) -> float:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def logpdf(self, x: float) -> float:
        """Log probability density function."""
        return self._dist.logpdf(x)
    
    def simulate(self, n: int = 1, rng: Optional[GlobalRng] = None) -> Union[DoubleData, List[DoubleData]]:
        """Simulate data from the model."""
        if rng is None:
            from ..distributions import rng as global_rng
            rng = global_rng
        
        values = rng.rnorm_vec(n, self.mean, self.sd)
        data = [DoubleData(x) for x in values]
        
        if n == 1:
            return data[0]
        return data
    
    def mle(self):
        """Maximum likelihood estimation."""
        if self._suf.n > 0:
            self.mean = self._suf.mean()
            if self._suf.n > 1:
                self.variance = self._suf.sample_variance()
    
    def clone(self) -> 'GaussianModel':
        """Create a copy of the model."""
        model = GaussianModel(self.mean, self.sd)
        model.set_data([d.value for d in self._data])
        return model
    
    # PriorModel interface
    def logp(self, theta: Union[float, np.ndarray]) -> float:
        """Log prior density (for use as prior)."""
        return self.logpdf(theta)


class ZeroMeanGaussianModel(GaussianModel):
    """Gaussian model with fixed zero mean."""
    
    def __init__(self, sd: float = 1.0):
        """Initialize zero-mean Gaussian model.
        
        Args:
            sd: Standard deviation (must be positive)
        """
        super().__init__(mean=0.0, sd=sd)
        self._params['mean'].fix()  # Fix mean at zero
    
    @property
    def mean(self) -> float:
        """Get mean (always 0)."""
        return 0.0
    
    @mean.setter
    def mean(self, value: float):
        """Cannot set mean for zero-mean model."""
        if value != 0:
            raise ValueError("Cannot change mean of zero-mean Gaussian model")
    
    def mle(self):
        """Maximum likelihood estimation (only variance)."""
        if self._suf.n > 0:
            # For zero mean, variance = sum(x^2) / n
            self.variance = self._suf.sumsq / self._suf.n
    
    def clone(self) -> 'ZeroMeanGaussianModel':
        """Create a copy of the model."""
        model = ZeroMeanGaussianModel(self.sd)
        model.set_data([d.value for d in self._data])
        return model


class GaussianModelGivenSigma(GaussianModel):
    """Gaussian model with fixed variance."""
    
    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        """Initialize Gaussian model with fixed sigma.
        
        Args:
            mean: Mean parameter
            sigma: Fixed standard deviation (must be positive)
        """
        super().__init__(mean=mean, sd=sigma)
        self._params['sd'].fix()
        self._params['variance'].fix()
    
    @property
    def sd(self) -> float:
        """Get fixed standard deviation."""
        return self._params['sd'].value
    
    @sd.setter
    def sd(self, value: float):
        """Cannot change fixed standard deviation."""
        raise ValueError("Cannot change sigma in GaussianModelGivenSigma")
    
    @property
    def variance(self) -> float:
        """Get fixed variance."""
        return self._params['variance'].value
    
    @variance.setter
    def variance(self, value: float):
        """Cannot change fixed variance."""
        raise ValueError("Cannot change variance in GaussianModelGivenSigma")
    
    def mle(self):
        """Maximum likelihood estimation (only mean)."""
        if self._suf.n > 0:
            self.mean = self._suf.mean()
    
    def clone(self) -> 'GaussianModelGivenSigma':
        """Create a copy of the model."""
        model = GaussianModelGivenSigma(self.mean, self.sd)
        model.set_data([d.value for d in self._data])
        return model