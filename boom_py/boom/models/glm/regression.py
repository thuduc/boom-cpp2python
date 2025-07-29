"""Linear regression models."""
import numpy as np
from typing import Union, List, Optional
from ..base import Model, VectorParameter, PositiveParameter
from ..data import RegressionData
from ..sufstat import RegressionSuf
from ...linalg import Vector, Matrix, SpdMatrix
from ...distributions.rng import GlobalRng
from ...distributions.continuous import Normal


class RegressionModel(Model):
    """Linear regression model: y = X*beta + epsilon, epsilon ~ N(0, sigma^2)."""
    
    def __init__(self, beta: Optional[Vector] = None, 
                 sigma: float = 1.0, 
                 xdim: Optional[int] = None):
        """Initialize regression model.
        
        Args:
            beta: Regression coefficients
            sigma: Error standard deviation
            xdim: Dimension of predictors (inferred if None)
        """
        super().__init__()
        
        if beta is not None:
            self.xdim = len(beta)
            self._params['beta'] = VectorParameter(Vector(beta), 'beta')
        elif xdim is not None:
            self.xdim = xdim
            self._params['beta'] = VectorParameter(Vector.zero(xdim), 'beta')
        else:
            self.xdim = None
            self._params['beta'] = None
        
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self._params['sigma'] = PositiveParameter(sigma, 'sigma')
        self._params['sigsq'] = PositiveParameter(sigma**2, 'sigsq')
        
        self._suf = RegressionSuf(self.xdim)
    
    @property
    def beta(self) -> Vector:
        """Get regression coefficients."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        return self._params['beta'].value
    
    @beta.setter
    def beta(self, value: Union[Vector, np.ndarray, List[float]]):
        """Set regression coefficients."""
        beta_vec = Vector(value)
        if self.xdim is None:
            self.xdim = len(beta_vec)
            self._suf = RegressionSuf(self.xdim)
            self._params['beta'] = VectorParameter(beta_vec, 'beta')
        elif len(beta_vec) != self.xdim:
            raise ValueError(f"Expected {self.xdim} coefficients, got {len(beta_vec)}")
        else:
            self._params['beta'].value = beta_vec
    
    @property
    def sigma(self) -> float:
        """Get error standard deviation."""
        return self._params['sigma'].value
    
    @sigma.setter
    def sigma(self, value: float):
        """Set error standard deviation."""
        if value <= 0:
            raise ValueError("Sigma must be positive")
        self._params['sigma'].value = value
        self._params['sigsq'].value = value ** 2
    
    @property
    def sigsq(self) -> float:
        """Get error variance."""
        return self._params['sigsq'].value
    
    @sigsq.setter
    def sigsq(self, value: float):
        """Set error variance."""
        if value <= 0:
            raise ValueError("Variance must be positive")
        self._params['sigsq'].value = value
        self._params['sigma'].value = np.sqrt(value)
    
    @property
    def xdim(self) -> Optional[int]:
        """Get dimension of predictors."""
        return self._xdim
    
    @xdim.setter
    def xdim(self, value: Optional[int]):
        """Set dimension of predictors."""
        self._xdim = value
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._suf.clear()
    
    def add_data(self, data: Union[RegressionData, tuple]):
        """Add regression data."""
        if not isinstance(data, RegressionData):
            y, x = data
            data = RegressionData(y, x)
        
        # Initialize beta if needed
        if self.xdim is None:
            self.xdim = data.xdim
            self._params['beta'] = VectorParameter(Vector.zero(self.xdim), 'beta')
            self._suf = RegressionSuf(self.xdim)
        elif data.xdim != self.xdim:
            raise ValueError(f"Expected {self.xdim} predictors, got {data.xdim}")
        
        self._data.append(data)
        self._suf.update(data)
    
    def set_data(self, data: List[tuple]):
        """Set data (replaces existing)."""
        self.clear_data()
        for item in data:
            self.add_data(item)
    
    def suf(self) -> RegressionSuf:
        """Get sufficient statistics."""
        return self._suf
    
    def predict(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict response for given predictors."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        x_vec = Vector(x)
        return float(self.beta.dot(x_vec))
    
    def predict_batch(self, X: Union[Matrix, np.ndarray]) -> Vector:
        """Predict responses for multiple observations."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        X_mat = Matrix(X)
        return X_mat @ self.beta
    
    def residuals(self) -> Vector:
        """Compute residuals for current data."""
        if not self._data:
            return Vector()
        
        resid = []
        for data_point in self._data:
            pred = self.predict(data_point.x)
            resid.append(data_point.y - pred)
        return Vector(resid)
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            if self._suf.n == 0:
                return 0.0
            
            rss = self._suf.residual_sum_of_squares(self.beta)
            n = self._suf.n
            
            return (-0.5 * n * np.log(2 * np.pi * self.sigsq) - 
                    0.5 * rss / self.sigsq)
        else:
            # Single observation
            if isinstance(data, RegressionData):
                y = data.y
                x = data.x
            else:
                y, x = data
                x = Vector(x)
            
            pred = self.predict(x)
            return Normal(pred, self.sigma).logpdf(y)
    
    def simulate(self, X: Union[Matrix, np.ndarray], 
                 rng: Optional[GlobalRng] = None) -> Union[RegressionData, List[RegressionData]]:
        """Simulate data from the model."""
        if rng is None:
            from ...distributions import rng as global_rng
            rng = global_rng
        
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        
        X_mat = Matrix(X)
        n = X_mat.nrow()
        
        # Generate predictions
        y_mean = X_mat @ self.beta
        
        # Add noise
        y_sim = []
        for i in range(n):
            y_i = rng.rnorm(y_mean[i], self.sigma)
            y_sim.append(y_i)
        
        # Create data objects
        data = []
        for i in range(n):
            data.append(RegressionData(y_sim[i], X_mat.row(i)))
        
        if n == 1:
            return data[0]
        return data
    
    def ols(self):
        """Ordinary least squares estimation."""
        if self._suf.n == 0:
            return
        
        # Estimate coefficients
        self.beta = self._suf.ols_coefficients()
        
        # Estimate error variance
        if self._suf.n > self.xdim:
            rss = self._suf.residual_sum_of_squares(self.beta)
            self.sigsq = rss / (self._suf.n - self.xdim)
    
    def mle(self):
        """Maximum likelihood estimation (same as OLS for normal errors)."""
        if self._suf.n == 0:
            return
        
        # MLE coefficients (same as OLS)
        self.beta = self._suf.ols_coefficients()
        
        # MLE variance (uses n instead of n-p)
        rss = self._suf.residual_sum_of_squares(self.beta)
        self.sigsq = rss / self._suf.n
    
    def coefficient_covariance(self) -> SpdMatrix:
        """Covariance matrix of coefficient estimates."""
        if self._suf.n == 0:
            return SpdMatrix.identity(self.xdim)
        
        xtx_inv = self._suf.xtx.inv()
        return SpdMatrix(self.sigsq * xtx_inv)
    
    def coefficient_standard_errors(self) -> Vector:
        """Standard errors of coefficient estimates."""
        cov = self.coefficient_covariance()
        return Vector(np.sqrt(cov.diag()))
    
    def r_squared(self) -> float:
        """R-squared (coefficient of determination)."""
        if self._suf.n == 0:
            return 0.0
        
        # Total sum of squares
        y_mean = np.mean([d.y for d in self._data])
        tss = sum((d.y - y_mean)**2 for d in self._data)
        
        if tss == 0:
            return 1.0
        
        # Residual sum of squares
        rss = self._suf.residual_sum_of_squares(self.beta)
        
        return 1 - rss / tss
    
    def adjusted_r_squared(self) -> float:
        """Adjusted R-squared."""
        if self._suf.n <= self.xdim:
            return 0.0
        
        r2 = self.r_squared()
        n = self._suf.n
        p = self.xdim
        
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def clone(self) -> 'RegressionModel':
        """Create a copy of the model."""
        if self._params['beta'] is None:
            model = RegressionModel(sigma=self.sigma, xdim=self.xdim)
        else:
            model = RegressionModel(self.beta, self.sigma)
        
        # Copy data
        data_tuples = [(d.y, d.x) for d in self._data]
        model.set_data(data_tuples)
        
        return model