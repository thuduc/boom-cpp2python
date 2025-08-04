"""Base classes for generalized linear models."""

import numpy as np
from typing import List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

from boom.models.base import Model, ConjugateModel
from boom.models.data import Data
from boom.models.sufstat import Sufstat
from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import RNG


class GlmData(Data):
    """Base class for GLM data with predictors and response."""
    
    def __init__(self, y: Union[float, int, bool], x: Union[List[float], np.ndarray, Vector]):
        """Initialize GLM data point.
        
        Args:
            y: Response value
            x: Predictor vector (design matrix row)
        """
        super().__init__()
        self._y = float(y)
        if isinstance(x, (list, np.ndarray)):
            self._x = Vector(x)
        elif isinstance(x, Vector):
            self._x = x.copy()
        else:
            raise ValueError(f"x must be list, ndarray, or Vector, got {type(x)}")
    
    def y(self) -> float:
        """Get response value."""
        return self._y
    
    def x(self) -> Vector:
        """Get predictor vector."""
        return self._x
    
    def set_y(self, y: float):
        """Set response value."""
        self._y = float(y)
    
    def set_x(self, x: Union[List[float], np.ndarray, Vector]):
        """Set predictor vector."""
        if isinstance(x, (list, np.ndarray)):
            self._x = Vector(x)
        elif isinstance(x, Vector):
            self._x = x.copy()
        else:
            raise ValueError(f"x must be list, ndarray, or Vector, got {type(x)}")
        
    
    def dim(self) -> int:
        """Get dimension of predictor vector."""
        return len(self._x)
    
    def clone(self) -> 'GlmData':
        """Create a copy of this data point."""
        return GlmData(self._y, self._x)
    
    def __repr__(self) -> str:
        return f"GlmData(y={self._y}, x={self._x.to_numpy()})"


class RegressionSufstat(Sufstat):
    """Sufficient statistics for regression models."""
    
    def __init__(self, xdim: int = 0):
        """Initialize regression sufficient statistics.
        
        Args:
            xdim: Dimension of predictor space
        """
        super().__init__()
        self._xdim = xdim
        self._n = 0.0
        self._yty = 0.0
        self._xty = Vector(np.zeros(xdim))
        self._xtx = SpdMatrix(np.zeros((xdim, xdim)))
        self._ysum = 0.0
        self._xsum = Vector(np.zeros(xdim))
    
    def clone(self) -> 'RegressionSufstat':
        """Create a copy of this sufstat."""
        cloned = RegressionSufstat(self._xdim)
        cloned._n = self._n
        cloned._yty = self._yty
        cloned._xty = self._xty.copy()
        cloned._xtx = self._xtx.copy()
        cloned._ysum = self._ysum
        cloned._xsum = self._xsum.copy()
        return cloned
    
    def vectorize(self) -> Vector:
        """Vectorize sufficient statistics (not typically used)."""
        # Return empty vector as sufstats are not typically vectorized
        return Vector([])
    
    def unvectorize(self, theta: Vector):
        """Set from vector (not typically used)."""
        # No-op as sufstats are not typically vectorized
        pass
    
    def update(self, data: GlmData):
        """Update sufficient statistics with new data point."""
        y = data.y()
        x = data.x()
        
        if len(x) != self._xdim and self._xdim > 0:
            raise ValueError(f"Data dimension {len(x)} doesn't match expected {self._xdim}")
        
        if self._xdim == 0:
            self._xdim = len(x)
            self._xty = Vector(np.zeros(self._xdim))
            self._xtx = SpdMatrix(np.zeros((self._xdim, self._xdim)))
            self._xsum = Vector(np.zeros(self._xdim))
        
        self._n += 1.0
        self._yty += y * y
        self._ysum += y
        
        x_array = x.to_numpy()
        self._xsum += x
        self._xty += x * y
        self._xtx += SpdMatrix(np.outer(x_array, x_array))
    
    def combine(self, other: 'RegressionSufstat'):
        """Combine with another sufstat."""
        if other._xdim != self._xdim:
            raise ValueError("Cannot combine sufstats with different dimensions")
        
        self._n += other._n
        self._yty += other._yty
        self._ysum += other._ysum
        self._xsum += other._xsum
        self._xty += other._xty
        self._xtx += other._xtx
    
    def clear(self):
        """Clear all statistics."""
        self._n = 0.0
        self._yty = 0.0
        self._ysum = 0.0
        if self._xdim > 0:
            self._xty = Vector(np.zeros(self._xdim))
            self._xtx = SpdMatrix(np.zeros((self._xdim, self._xdim)))
            self._xsum = Vector(np.zeros(self._xdim))
    
    def n(self) -> float:
        """Number of observations."""
        return self._n
    
    def yty(self) -> float:
        """Sum of y^2."""
        return self._yty
    
    def xty(self) -> Vector:
        """Sum of x_i * y_i."""
        return self._xty.copy()
    
    def xtx(self) -> SpdMatrix:
        """Sum of x_i * x_i^T."""
        return self._xtx.copy()
    
    def ybar(self) -> float:
        """Sample mean of y."""
        return self._ysum / self._n if self._n > 0 else 0.0
    
    def xbar(self) -> Vector:
        """Sample mean of x."""
        return self._xsum / self._n if self._n > 0 else Vector(np.zeros(self._xdim))
    
    def beta_hat(self) -> Vector:
        """Ordinary least squares estimate."""
        if self._n == 0:
            return Vector(np.zeros(self._xdim))
        
        try:
            # Regularize slightly to avoid numerical issues
            reg_xtx = self._xtx.to_numpy() + 1e-12 * np.eye(self._xdim)
            return Vector(np.linalg.solve(reg_xtx, self._xty.to_numpy()))
        except np.linalg.LinAlgError:
            # If still singular, use pseudoinverse
            return Vector(np.linalg.pinv(self._xtx.to_numpy()) @ self._xty.to_numpy())
    
    def sse(self) -> float:
        """Sum of squared errors for OLS estimate."""
        if self._n == 0:
            return 0.0
        
        beta_hat = self.beta_hat()
        # SSE = yty - 2 * beta^T * xty + beta^T * xtx * beta
        beta_array = beta_hat.to_numpy()
        sse = (self._yty - 2 * np.dot(beta_array, self._xty.to_numpy()) + 
               np.dot(beta_array, self._xtx.to_numpy() @ beta_array))
        return max(0.0, sse)  # Ensure non-negative due to numerical precision
    
    def sst(self) -> float:
        """Total sum of squares."""
        if self._n == 0:
            return 0.0
        
        ybar = self.ybar()
        return self._yty - self._n * ybar * ybar
    
    def r_squared(self) -> float:
        """R-squared coefficient of determination."""
        sst = self.sst()
        if sst == 0:
            return 1.0
        return 1.0 - self.sse() / sst


class GlmModel(Model, ABC):
    """Base class for generalized linear models."""
    
    def __init__(self, xdim: int):
        """Initialize GLM model.
        
        Args:
            xdim: Dimension of predictor space
        """
        super().__init__()
        self._xdim = xdim
        self._beta = Vector(np.zeros(xdim))
        self._data: List[GlmData] = []
        self._suf = RegressionSufstat(xdim)
    
    def beta(self) -> Vector:
        """Get regression coefficients."""
        return self._beta.copy()
    
    def set_beta(self, beta: Union[List[float], np.ndarray, Vector]):
        """Set regression coefficients."""
        if isinstance(beta, (list, np.ndarray)):
            beta_vec = Vector(beta)
        elif isinstance(beta, Vector):
            beta_vec = beta.copy()
        else:
            raise ValueError(f"beta must be list, ndarray, or Vector, got {type(beta)}")
        
        if len(beta_vec) != self._xdim:
            raise ValueError(f"beta dimension {len(beta_vec)} doesn't match model dimension {self._xdim}")
        
        self._beta = beta_vec
        self._notify_observers()
    
    def xdim(self) -> int:
        """Get predictor dimension."""
        return self._xdim
    
    def add_data(self, data: Union[GlmData, Tuple[float, Vector], List[Tuple[float, Vector]]]):
        """Add data to the model."""
        if isinstance(data, GlmData):
            self._add_single_data(data)
        elif isinstance(data, tuple) and len(data) == 2:
            y, x = data
            glm_data = GlmData(y, x)
            self._add_single_data(glm_data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, GlmData):
                    self._add_single_data(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    y, x = item
                    glm_data = GlmData(y, x)
                    self._add_single_data(glm_data)
                else:
                    raise ValueError(f"Invalid data item: {item}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    def _add_single_data(self, data: GlmData):
        """Add a single data point."""
        self._data.append(data)
        self._suf.update(data)
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._suf.clear()
    
    def suf(self) -> RegressionSufstat:
        """Get sufficient statistics."""
        return self._suf
    
    def linear_predictor(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Compute linear predictor beta^T * x."""
        if isinstance(x, (np.ndarray, list)):
            x_vec = Vector(x)
        elif isinstance(x, Vector):
            x_vec = x
        else:
            raise ValueError(f"x must be Vector, ndarray, or list, got {type(x)}")
        
        return float(np.dot(self._beta.to_numpy(), x_vec.to_numpy()))
    
    @abstractmethod
    def mean_function(self, linear_pred: float) -> float:
        """Transform linear predictor to mean parameter."""
        pass
    
    @abstractmethod
    def variance_function(self, mean: float) -> float:
        """Compute variance as function of mean."""
        pass
    
    def predict(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict response for given predictors."""
        linear_pred = self.linear_predictor(x)
        return self.mean_function(linear_pred)
    
    def predict_batch(self, X: Union[Matrix, np.ndarray, List[List[float]]]) -> Vector:
        """Predict responses for batch of predictors."""
        if isinstance(X, (np.ndarray, list)):
            X_array = np.array(X)
        elif isinstance(X, Matrix):
            X_array = X.to_numpy()
        else:
            raise ValueError(f"X must be Matrix, ndarray, or nested list, got {type(X)}")
        
        if X_array.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        
        if X_array.shape[1] != self._xdim:
            raise ValueError(f"X has {X_array.shape[1]} columns, expected {self._xdim}")
        
        predictions = []
        for i in range(X_array.shape[0]):
            pred = self.predict(X_array[i])
            predictions.append(pred)
        
        return Vector(predictions)
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Vectorize parameters for optimization."""
        return self._beta.copy()
    
    def unvectorize_params(self, theta: Vector):
        """Set parameters from vector."""
        self.set_beta(theta)
    
    def clone(self) -> 'GlmModel':
        """Create a copy of this model."""
        cloned = self.__class__(self._xdim)
        cloned.set_beta(self._beta)
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        return cloned
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(xdim={self._xdim}, beta={self._beta.to_numpy()}, data_points={len(self._data)})"