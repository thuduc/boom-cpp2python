"""Sufficient statistics for BOOM models."""
import numpy as np
from typing import Union, Optional
from ..linalg import Vector, Matrix, SpdMatrix
from .base import SufStat
from .data import DoubleData, VectorData, RegressionData


class GaussianSuf(SufStat):
    """Sufficient statistics for Gaussian distribution."""
    
    def __init__(self):
        """Initialize empty sufficient statistics."""
        self.clear()
    
    def update(self, data: Union[float, DoubleData]):
        """Update with a single observation."""
        if isinstance(data, DoubleData):
            x = data.value
        else:
            x = float(data)
        
        self.n += 1
        self.sum += x
        self.sumsq += x * x
    
    def update_raw(self, x: float):
        """Update with raw value."""
        self.n += 1
        self.sum += x
        self.sumsq += x * x
    
    def update_many(self, data: np.ndarray):
        """Update with multiple observations."""
        data = np.asarray(data)
        self.n += len(data)
        self.sum += np.sum(data)
        self.sumsq += np.sum(data * data)
    
    def combine(self, other: 'GaussianSuf'):
        """Combine with another sufficient statistic."""
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def mean(self) -> float:
        """Sample mean."""
        if self.n == 0:
            return 0.0
        return self.sum / self.n
    
    def sample_variance(self) -> float:
        """Sample variance (dividing by n)."""
        if self.n == 0:
            return 0.0
        mean = self.mean()
        return self.sumsq / self.n - mean * mean
    
    def variance(self) -> float:
        """Unbiased sample variance (dividing by n-1)."""
        if self.n <= 1:
            return 0.0
        return self.n * self.sample_variance() / (self.n - 1)
    
    def sd(self) -> float:
        """Sample standard deviation."""
        return np.sqrt(self.variance())
    
    def centered_sumsq(self, mu: float) -> float:
        """Sum of squared deviations from mu."""
        return self.sumsq - 2 * mu * self.sum + self.n * mu * mu


class MvnSuf(SufStat):
    """Sufficient statistics for multivariate normal distribution."""
    
    def __init__(self, dim: Optional[int] = None):
        """Initialize sufficient statistics.
        
        Args:
            dim: Dimension (inferred from first observation if None)
        """
        self.dim = dim
        self.clear()
    
    def update(self, data: Union[Vector, VectorData, np.ndarray]):
        """Update with a single observation."""
        if isinstance(data, VectorData):
            x = data.value
        else:
            x = Vector(data)
        
        if self.dim is None:
            self.dim = len(x)
            self.sum = Vector.zero(self.dim)
            self.sumsq = Matrix.zero(self.dim, self.dim)
        elif len(x) != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {len(x)}")
        
        self.n += 1
        self.sum += x
        self.sumsq += x.outer(x)
    
    def update_raw(self, x: Union[Vector, np.ndarray]):
        """Update with raw vector."""
        self.update(x)
    
    def combine(self, other: 'MvnSuf'):
        """Combine with another sufficient statistic."""
        if self.dim is None:
            self.dim = other.dim
            self.sum = other.sum.copy() if other.sum is not None else None
            self.sumsq = other.sumsq.copy() if other.sumsq is not None else None
            self.n = other.n
        else:
            if other.dim != self.dim:
                raise ValueError("Dimensions must match")
            self.n += other.n
            self.sum += other.sum
            self.sumsq += other.sumsq
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        if self.dim is not None:
            self.sum = Vector.zero(self.dim)
            self.sumsq = Matrix.zero(self.dim, self.dim)
        else:
            self.sum = None
            self.sumsq = None
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def mean(self) -> Vector:
        """Sample mean vector."""
        if self.n == 0:
            return Vector.zero(self.dim)
        return self.sum / self.n
    
    def sample_covariance(self) -> SpdMatrix:
        """Sample covariance matrix (dividing by n)."""
        if self.n == 0:
            return SpdMatrix.identity(self.dim)
        mean = self.mean()
        centered_sumsq = self.sumsq - self.n * mean.outer(mean)
        return SpdMatrix(centered_sumsq / self.n)
    
    def covariance(self) -> SpdMatrix:
        """Unbiased sample covariance (dividing by n-1)."""
        if self.n <= 1:
            return SpdMatrix.identity(self.dim)
        return SpdMatrix(self.sample_covariance() * self.n / (self.n - 1))
    
    def centered_sumsq(self, mu: Vector) -> Matrix:
        """Sum of squared deviations from mu."""
        return self.sumsq - 2 * mu.outer(self.sum) + self.n * mu.outer(mu)


class RegressionSuf(SufStat):
    """Sufficient statistics for regression models."""
    
    def __init__(self, xdim: Optional[int] = None):
        """Initialize regression sufficient statistics.
        
        Args:
            xdim: Dimension of predictors (inferred if None)
        """
        self.xdim = xdim
        self.clear()
    
    def update(self, data: Union[RegressionData, tuple]):
        """Update with regression data."""
        if isinstance(data, RegressionData):
            y = data.y
            x = data.x
        else:
            y, x = data
            x = Vector(x)
        
        if self.xdim is None:
            self.xdim = len(x)
            self.xtx = Matrix.zero(self.xdim, self.xdim)
            self.xty = Vector.zero(self.xdim)
        elif len(x) != self.xdim:
            raise ValueError(f"Expected x dimension {self.xdim}, got {len(x)}")
        
        self.n += 1
        self.yty += y * y
        self.xtx += x.outer(x)
        self.xty += x * y
    
    def combine(self, other: 'RegressionSuf'):
        """Combine with another sufficient statistic."""
        if self.xdim is None:
            self.xdim = other.xdim
            self.xtx = other.xtx.copy() if other.xtx is not None else None
            self.xty = other.xty.copy() if other.xty is not None else None
            self.yty = other.yty
            self.n = other.n
        else:
            if other.xdim != self.xdim:
                raise ValueError("Dimensions must match")
            self.n += other.n
            self.yty += other.yty
            self.xtx += other.xtx
            self.xty += other.xty
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        self.yty = 0.0
        if self.xdim is not None:
            self.xtx = Matrix.zero(self.xdim, self.xdim)
            self.xty = Vector.zero(self.xdim)
        else:
            self.xtx = None
            self.xty = None
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def ols_coefficients(self) -> Vector:
        """Ordinary least squares coefficients."""
        if self.n == 0:
            return Vector.zero(self.xdim)
        return self.xtx.solve(self.xty)
    
    def residual_sum_of_squares(self, beta: Vector) -> float:
        """Residual sum of squares for given coefficients."""
        return self.yty - 2 * beta.dot(self.xty) + beta.dot(self.xtx @ beta)


class BetaBinomialSuf(SufStat):
    """Sufficient statistics for Beta-Binomial model."""
    
    def __init__(self):
        """Initialize sufficient statistics."""
        self.clear()
    
    def update(self, successes: int, trials: int):
        """Update with binomial data."""
        if successes < 0 or trials < 0:
            raise ValueError("Successes and trials must be non-negative")
        if successes > trials:
            raise ValueError("Successes cannot exceed trials")
        
        self.n += 1
        self.sum_successes += successes
        self.sum_trials += trials
        
        # Track individual observations for beta-binomial
        self.successes_list.append(successes)
        self.trials_list.append(trials)
    
    def combine(self, other: 'BetaBinomialSuf'):
        """Combine with another sufficient statistic."""
        self.n += other.n
        self.sum_successes += other.sum_successes
        self.sum_trials += other.sum_trials
        self.successes_list.extend(other.successes_list)
        self.trials_list.extend(other.trials_list)
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        self.sum_successes = 0
        self.sum_trials = 0
        self.successes_list = []
        self.trials_list = []
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def overall_proportion(self) -> float:
        """Overall success proportion."""
        if self.sum_trials == 0:
            return 0.0
        return self.sum_successes / self.sum_trials


class LogisticSuf(SufStat):
    """Sufficient statistics for logistic regression."""
    
    def __init__(self, xdim: Optional[int] = None):
        """Initialize logistic regression sufficient statistics.
        
        Args:
            xdim: Dimension of predictors (inferred if None)
        """
        self.xdim = xdim
        self.clear()
    
    def update(self, data: Union[RegressionData, tuple]):
        """Update with logistic regression data."""
        if isinstance(data, RegressionData):
            y = data.y
            x = data.x
        else:
            y, x = data
            x = Vector(x)
        
        # Validate binary response
        if y not in [0, 1]:
            raise ValueError(f"Response must be 0 or 1, got {y}")
        
        if self.xdim is None:
            self.xdim = len(x)
            self.sum_x = Vector.zero(self.xdim)
            self.sum_x_y1 = Vector.zero(self.xdim)
            self.sum_xx = Matrix.zero(self.xdim, self.xdim)
        elif len(x) != self.xdim:
            raise ValueError(f"Expected x dimension {self.xdim}, got {len(x)}")
        
        self.n += 1
        self.sum_y += y
        self.sum_x += x
        self.sum_xx += x.outer(x)
        
        if y == 1:
            self.sum_x_y1 += x
    
    def combine(self, other: 'LogisticSuf'):
        """Combine with another sufficient statistic."""
        if self.xdim is None:
            self.xdim = other.xdim
            self.sum_x = other.sum_x.copy() if other.sum_x is not None else None
            self.sum_x_y1 = other.sum_x_y1.copy() if other.sum_x_y1 is not None else None
            self.sum_xx = other.sum_xx.copy() if other.sum_xx is not None else None
            self.sum_y = other.sum_y
            self.n = other.n
        else:
            if other.xdim != self.xdim:
                raise ValueError("Dimensions must match")
            self.n += other.n
            self.sum_y += other.sum_y
            self.sum_x += other.sum_x
            self.sum_x_y1 += other.sum_x_y1
            self.sum_xx += other.sum_xx
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        self.sum_y = 0
        if self.xdim is not None:
            self.sum_x = Vector.zero(self.xdim)
            self.sum_x_y1 = Vector.zero(self.xdim)
            self.sum_xx = Matrix.zero(self.xdim, self.xdim)
        else:
            self.sum_x = None
            self.sum_x_y1 = None
            self.sum_xx = None
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.n == 0:
            return 0.0
        return self.sum_y / self.n


class PoissonSuf(SufStat):
    """Sufficient statistics for Poisson regression."""
    
    def __init__(self, xdim: Optional[int] = None):
        """Initialize Poisson regression sufficient statistics.
        
        Args:
            xdim: Dimension of predictors (inferred if None)
        """
        self.xdim = xdim
        self.clear()
    
    def update(self, data: Union[RegressionData, tuple]):
        """Update with Poisson regression data."""
        if isinstance(data, RegressionData):
            y = data.y
            x = data.x
        else:
            y, x = data
            x = Vector(x)
        
        # Validate non-negative integer response
        if not (isinstance(y, (int, np.integer, float, np.floating)) and y >= 0 and y == int(y)):
            raise ValueError(f"Response must be non-negative integer, got {y}")
        y = int(y)  # Convert to int
        
        if self.xdim is None:
            self.xdim = len(x)
            self.sum_x = Vector.zero(self.xdim)
            self.sum_x_y = Vector.zero(self.xdim)
            self.sum_xx = Matrix.zero(self.xdim, self.xdim)
        elif len(x) != self.xdim:
            raise ValueError(f"Expected x dimension {self.xdim}, got {len(x)}")
        
        self.n += 1
        self.sum_y += y
        self.sum_x += x
        self.sum_x_y += x * y
        self.sum_xx += x.outer(x)
    
    def combine(self, other: 'PoissonSuf'):
        """Combine with another sufficient statistic."""
        if self.xdim is None:
            self.xdim = other.xdim
            self.sum_x = other.sum_x.copy() if other.sum_x is not None else None
            self.sum_x_y = other.sum_x_y.copy() if other.sum_x_y is not None else None
            self.sum_xx = other.sum_xx.copy() if other.sum_xx is not None else None
            self.sum_y = other.sum_y
            self.n = other.n
        else:
            if other.xdim != self.xdim:
                raise ValueError("Dimensions must match")
            self.n += other.n
            self.sum_y += other.sum_y
            self.sum_x += other.sum_x
            self.sum_x_y += other.sum_x_y
            self.sum_xx += other.sum_xx
    
    def clear(self):
        """Clear sufficient statistics."""
        self.n = 0
        self.sum_y = 0
        if self.xdim is not None:
            self.sum_x = Vector.zero(self.xdim)
            self.sum_x_y = Vector.zero(self.xdim)
            self.sum_xx = Matrix.zero(self.xdim, self.xdim)
        else:
            self.sum_x = None
            self.sum_x_y = None
            self.sum_xx = None
    
    @property
    def sample_size(self) -> int:
        """Number of observations."""
        return self.n
    
    def mean_response(self) -> float:
        """Mean response (count rate)."""
        if self.n == 0:
            return 0.0
        return self.sum_y / self.n