"""Poisson regression models."""
import numpy as np
from typing import Union, List, Optional
from ..base import Model, VectorParameter
from ..data import RegressionData
from ..sufstat import PoissonSuf
from ...linalg import Vector, Matrix, SpdMatrix
from ...distributions.rng import GlobalRng
from ...distributions.discrete import Poisson


class PoissonRegressionModel(Model):
    """Poisson regression model: y ~ Poisson(exp(x'beta))."""
    
    def __init__(self, beta: Optional[Vector] = None, 
                 xdim: Optional[int] = None):
        """Initialize Poisson regression model.
        
        Args:
            beta: Regression coefficients
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
        
        self._suf = PoissonSuf(self.xdim)
    
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
            self._suf = PoissonSuf(self.xdim)
            self._params['beta'] = VectorParameter(beta_vec, 'beta')
        elif len(beta_vec) != self.xdim:
            raise ValueError(f"Expected {self.xdim} coefficients, got {len(beta_vec)}")
        else:
            self._params['beta'].value = beta_vec
    
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
        """Add Poisson regression data."""
        if not isinstance(data, RegressionData):
            y, x = data
            data = RegressionData(y, x)
        
        # Validate non-negative integer response
        y_val = data.y
        if not (isinstance(y_val, (int, np.integer, float, np.floating)) and y_val >= 0 and y_val == int(y_val)):
            raise ValueError(f"Response must be non-negative integer, got {data.y}")
        data.y = int(y_val)  # Convert to int
        
        # Initialize beta if needed
        if self.xdim is None:
            self.xdim = data.xdim
            self._params['beta'] = VectorParameter(Vector.zero(self.xdim), 'beta')
            self._suf = PoissonSuf(self.xdim)
        elif data.xdim != self.xdim:
            raise ValueError(f"Expected {self.xdim} predictors, got {data.xdim}")
        
        self._data.append(data)
        self._suf.update(data)
    
    def set_data(self, data: List[tuple]):
        """Set data (replaces existing)."""
        self.clear_data()
        for item in data:
            self.add_data(item)
    
    def suf(self) -> PoissonSuf:
        """Get sufficient statistics."""
        return self._suf
    
    def predict_rate(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict Poisson rate parameter lambda = exp(x'beta)."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        x_vec = Vector(x)
        linear_pred = self.beta.dot(x_vec)
        return float(np.exp(linear_pred))
    
    def predict_rate_batch(self, X: Union[Matrix, np.ndarray]) -> Vector:
        """Predict rates for multiple observations."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        X_mat = Matrix(X)
        linear_preds = X_mat @ self.beta
        return Vector([np.exp(pred) for pred in linear_preds])
    
    def predict(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict expected count (same as rate for Poisson)."""
        return self.predict_rate(x)
    
    def predict_batch(self, X: Union[Matrix, np.ndarray]) -> Vector:
        """Predict expected counts for multiple observations."""
        return self.predict_rate_batch(X)
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            if self._suf.n == 0:
                return 0.0
            
            loglike = 0.0
            for data_point in self._data:
                y = data_point.y
                rate = self.predict_rate(data_point.x)
                
                # Poisson log likelihood: y * log(lambda) - lambda - log(y!)
                loglike += y * np.log(rate) - rate
                # Skip log(y!) as it's constant for given data
            
            return loglike
        else:
            # Single observation
            if isinstance(data, RegressionData):
                y = data.y
                x = data.x
            else:
                y, x = data
                x = Vector(x)
            
            rate = self.predict_rate(x)
            return y * np.log(rate) - rate
    
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
        
        # Generate rates
        rates = self.predict_rate_batch(X_mat)
        
        # Generate Poisson counts
        y_sim = []
        for i in range(n):
            y_i = rng.rpois(rates[i])
            y_sim.append(y_i)
        
        # Create data objects
        data = []
        for i in range(n):
            data.append(RegressionData(y_sim[i], X_mat.row(i)))
        
        if n == 1:
            return data[0]
        return data
    
    def mle(self, max_iter: int = 100, tol: float = 1e-6):
        """Maximum likelihood estimation using Newton-Raphson."""
        if self._suf.n == 0:
            return
        
        # Initialize beta if needed
        if self._params['beta'] is None:
            self.beta = Vector.zero(self.xdim)
        
        # Newton-Raphson iterations
        for iter_count in range(max_iter):
            old_beta = self.beta.copy()
            
            # Compute gradient and Hessian
            gradient = Vector.zero(self.xdim)
            hessian = Matrix.zero(self.xdim, self.xdim)
            
            for data_point in self._data:
                x = data_point.x
                y = data_point.y
                rate = self.predict_rate(x)
                
                # Gradient contribution
                gradient += (y - rate) * x
                
                # Hessian contribution
                hessian += rate * np.outer(x, x)
            
            # Newton-Raphson update
            try:
                hessian_inv = SpdMatrix(hessian).inv()
                self.beta = old_beta + hessian_inv @ gradient
            except np.linalg.LinAlgError:
                # If Hessian is singular, use gradient ascent
                step_size = 0.01
                self.beta = old_beta + step_size * gradient
            
            # Check convergence
            if np.linalg.norm(self.beta - old_beta) < tol:
                break
    
    def coefficient_covariance(self) -> SpdMatrix:
        """Covariance matrix of coefficient estimates (Fisher information)."""
        if self._suf.n == 0:
            return SpdMatrix.identity(self.xdim)
        
        # Compute Fisher information matrix
        fisher_info = Matrix.zero(self.xdim, self.xdim)
        
        for data_point in self._data:
            x = data_point.x
            rate = self.predict_rate(x)
            fisher_info += rate * np.outer(x, x)
        
        try:
            return SpdMatrix(fisher_info).inv()
        except np.linalg.LinAlgError:
            return SpdMatrix.identity(self.xdim)
    
    def coefficient_standard_errors(self) -> Vector:
        """Standard errors of coefficient estimates."""
        cov = self.coefficient_covariance()
        return Vector(np.sqrt(cov.diag()))
    
    def deviance(self) -> float:
        """Deviance (-2 * log likelihood)."""
        return -2 * self.loglike()
    
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return 2 * self.xdim - 2 * self.loglike()
    
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return np.log(self._suf.n) * self.xdim - 2 * self.loglike()
    
    def pearson_residuals(self) -> Vector:
        """Pearson residuals for model diagnostics."""
        if not self._data:
            return Vector()
        
        residuals = []
        for data_point in self._data:
            y = data_point.y
            rate = self.predict_rate(data_point.x)
            # Pearson residual: (y - mu) / sqrt(mu)
            residual = (y - rate) / np.sqrt(rate) if rate > 0 else 0
            residuals.append(residual)
        
        return Vector(residuals)
    
    def deviance_residuals(self) -> Vector:
        """Deviance residuals for model diagnostics."""
        if not self._data:
            return Vector()
        
        residuals = []
        for data_point in self._data:
            y = data_point.y
            rate = self.predict_rate(data_point.x)
            
            if y == 0:
                dev_resid = np.sqrt(2 * rate)
            else:
                dev_resid = np.sqrt(2 * (y * np.log(y / rate) - (y - rate)))
            
            # Sign based on y vs rate
            if y < rate:
                dev_resid = -dev_resid
            
            residuals.append(dev_resid)
        
        return Vector(residuals)
    
    def pseudo_r_squared(self) -> float:
        """McFadden's pseudo R-squared."""
        if self._suf.n == 0:
            return 0.0
        
        # Null model (intercept only)
        y_values = [d.y for d in self._data]
        y_mean = np.mean(y_values)
        
        if y_mean <= 0:
            return 1.0
        
        null_loglike = sum(y * np.log(y_mean) - y_mean for y in y_values)
        full_loglike = self.loglike()
        
        return 1 - full_loglike / null_loglike
    
    def clone(self) -> 'PoissonRegressionModel':
        """Create a copy of the model."""
        if self._params['beta'] is None:
            model = PoissonRegressionModel(xdim=self.xdim)
        else:
            model = PoissonRegressionModel(self.beta)
        
        # Copy data
        data_tuples = [(d.y, d.x) for d in self._data]
        model.set_data(data_tuples)
        
        return model