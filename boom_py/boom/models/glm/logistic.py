"""Logistic regression models."""
import numpy as np
from typing import Union, List, Optional
from ..base import Model, VectorParameter
from ..data import RegressionData
from ..sufstat import LogisticSuf
from ...linalg import Vector, Matrix, SpdMatrix
from ...distributions.rng import GlobalRng
from ...distributions.continuous import Normal
from scipy.optimize import minimize
from scipy.special import expit, logit


class LogisticRegressionModel(Model):
    """Logistic regression model: P(y=1|x) = 1/(1 + exp(-x'beta))."""
    
    def __init__(self, beta: Optional[Vector] = None, 
                 xdim: Optional[int] = None):
        """Initialize logistic regression model.
        
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
        
        self._suf = LogisticSuf(self.xdim)
    
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
            self._suf = LogisticSuf(self.xdim)
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
        """Add logistic regression data."""
        if not isinstance(data, RegressionData):
            y, x = data
            data = RegressionData(y, x)
        
        # Validate binary response
        if data.y not in [0, 1]:
            raise ValueError(f"Response must be 0 or 1, got {data.y}")
        
        # Initialize beta if needed
        if self.xdim is None:
            self.xdim = data.xdim
            self._params['beta'] = VectorParameter(Vector.zero(self.xdim), 'beta')
            self._suf = LogisticSuf(self.xdim)
        elif data.xdim != self.xdim:
            raise ValueError(f"Expected {self.xdim} predictors, got {data.xdim}")
        
        self._data.append(data)
        self._suf.update(data)
    
    def set_data(self, data: List[tuple]):
        """Set data (replaces existing)."""
        self.clear_data()
        for item in data:
            self.add_data(item)
    
    def suf(self) -> LogisticSuf:
        """Get sufficient statistics."""
        return self._suf
    
    def predict_prob(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict probability P(y=1|x)."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        x_vec = Vector(x)
        linear_pred = self.beta.dot(x_vec)
        return float(expit(linear_pred))
    
    def predict_prob_batch(self, X: Union[Matrix, np.ndarray]) -> Vector:
        """Predict probabilities for multiple observations."""
        if self._params['beta'] is None:
            raise ValueError("Beta not initialized")
        X_mat = Matrix(X)
        linear_preds = X_mat @ self.beta
        return Vector([expit(pred) for pred in linear_preds])
    
    def predict(self, x: Union[Vector, np.ndarray, List[float]], 
                threshold: float = 0.5) -> int:
        """Predict binary response (0 or 1)."""
        prob = self.predict_prob(x)
        return 1 if prob >= threshold else 0
    
    def predict_batch(self, X: Union[Matrix, np.ndarray], 
                      threshold: float = 0.5) -> Vector:
        """Predict binary responses for multiple observations."""
        probs = self.predict_prob_batch(X)
        return Vector([1 if p >= threshold else 0 for p in probs])
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            if self._suf.n == 0:
                return 0.0
            
            loglike = 0.0
            for data_point in self._data:
                linear_pred = self.beta.dot(data_point.x)
                if data_point.y == 1:
                    loglike += linear_pred - np.log(1 + np.exp(linear_pred))
                else:
                    loglike += -np.log(1 + np.exp(linear_pred))
            
            return loglike
        else:
            # Single observation
            if isinstance(data, RegressionData):
                y = data.y
                x = data.x
            else:
                y, x = data
                x = Vector(x)
            
            linear_pred = self.beta.dot(x)
            if y == 1:
                return linear_pred - np.log(1 + np.exp(linear_pred))
            else:
                return -np.log(1 + np.exp(linear_pred))
    
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
        
        # Generate probabilities
        probs = self.predict_prob_batch(X_mat)
        
        # Generate binary responses
        y_sim = []
        for i in range(n):
            y_i = 1 if rng.runif() < probs[i] else 0
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
                linear_pred = self.beta.dot(x)
                prob = expit(linear_pred)
                
                # Gradient contribution
                gradient += (y - prob) * x
                
                # Hessian contribution
                weight = prob * (1 - prob)
                hessian += weight * np.outer(x, x)
            
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
            linear_pred = self.beta.dot(x)
            prob = expit(linear_pred)
            weight = prob * (1 - prob)
            fisher_info += weight * np.outer(x, x)
        
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
    
    def pseudo_r_squared(self) -> float:
        """McFadden's pseudo R-squared."""
        if self._suf.n == 0:
            return 0.0
        
        # Null model (intercept only)
        y_values = [d.y for d in self._data]
        p_bar = np.mean(y_values)
        
        if p_bar == 0 or p_bar == 1:
            return 1.0
        
        null_loglike = (np.sum(y_values) * np.log(p_bar) + 
                       (len(y_values) - np.sum(y_values)) * np.log(1 - p_bar))
        
        full_loglike = self.loglike()
        
        return 1 - full_loglike / null_loglike
    
    def clone(self) -> 'LogisticRegressionModel':
        """Create a copy of the model."""
        if self._params['beta'] is None:
            model = LogisticRegressionModel(xdim=self.xdim)
        else:
            model = LogisticRegressionModel(self.beta)
        
        # Copy data
        data_tuples = [(d.y, d.x) for d in self._data]
        model.set_data(data_tuples)
        
        return model