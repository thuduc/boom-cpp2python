"""Poisson regression model for count data."""

import numpy as np
from typing import List, Optional, Union, Tuple
from scipy import stats, optimize

from .base import GlmModel, GlmData
from boom.linalg import Vector, SpdMatrix
from boom.distributions import RNG
from boom.distributions.rmath import dpois, rpois
from boom.models.base import ConjugateModel


class PoissonRegressionData(GlmData):
    """Data for Poisson regression with count response."""
    
    def __init__(self, count: int, x: Union[List[float], np.ndarray, Vector],
                 exposure: float = 1.0):
        """Initialize Poisson regression data.
        
        Args:
            count: Observed count (non-negative integer)
            x: Predictor vector
            exposure: Exposure time/area (offset)
        """
        if count < 0:
            raise ValueError("Count must be non-negative")
        if exposure <= 0:
            raise ValueError("Exposure must be positive")
        
        super().__init__(float(count), x)
        self._count = int(count)
        self._exposure = float(exposure)
    
    def count(self) -> int:
        """Get observed count."""
        return self._count
    
    def exposure(self) -> float:
        """Get exposure."""
        return self._exposure
    
    def rate(self) -> float:
        """Get observed rate (count / exposure)."""
        return self._count / self._exposure
    
    def clone(self) -> 'PoissonRegressionData':
        """Create a copy of this data point."""
        return PoissonRegressionData(self._count, self.x(), self._exposure)
    
    def __repr__(self) -> str:
        return f"PoissonRegressionData(count={self._count}, exposure={self._exposure}, x={self.x().to_numpy()})"


class PoissonRegressionModel(GlmModel):
    """Poisson regression model for count data.
    
    Model: y_i ~ Poisson(lambda_i * exposure_i)
           log(lambda_i) = X_i^T * beta
    
    Where lambda_i is the rate parameter and exposure_i is the offset.
    """
    
    def __init__(self, xdim: int):
        """Initialize Poisson regression model.
        
        Args:
            xdim: Dimension of predictor space
        """
        super().__init__(xdim)
        
        # Prior parameters for beta (multivariate normal)
        self._beta_prior_mean = Vector(np.zeros(xdim))
        self._beta_prior_precision = SpdMatrix(1e-4 * np.eye(xdim))  # Diffuse prior
        self._has_prior = False
    
    def set_prior(self, beta_mean: Union[Vector, np.ndarray, List[float]],
                  beta_precision: Union[SpdMatrix, np.ndarray]):
        """Set normal prior for beta.
        
        Args:
            beta_mean: Prior mean for beta
            beta_precision: Prior precision matrix for beta
        """
        if isinstance(beta_mean, (np.ndarray, list)):
            self._beta_prior_mean = Vector(beta_mean)
        elif isinstance(beta_mean, Vector):
            self._beta_prior_mean = beta_mean.copy()
        else:
            raise ValueError(f"beta_mean must be Vector, ndarray, or list, got {type(beta_mean)}")
        
        if len(self._beta_prior_mean) != self._xdim:
            raise ValueError(f"beta_mean dimension doesn't match model dimension")
        
        if isinstance(beta_precision, np.ndarray):
            self._beta_prior_precision = SpdMatrix(beta_precision)
        elif isinstance(beta_precision, SpdMatrix):
            self._beta_prior_precision = beta_precision.copy()
        else:
            raise ValueError(f"beta_precision must be SpdMatrix or ndarray")
        
        if self._beta_prior_precision.nrow() != self._xdim:
            raise ValueError(f"beta_precision dimension doesn't match model dimension")
        
        self._has_prior = True
    
    def mean_function(self, linear_pred: float) -> float:
        """Exponential function: lambda = exp(eta)."""
        # Use stable computation to avoid overflow
        if linear_pred > 500:
            return np.exp(500)  # Cap at very large value
        elif linear_pred < -500:
            return np.exp(-500)  # Cap at very small value
        else:
            return np.exp(linear_pred)
    
    def variance_function(self, mean: float) -> float:
        """Poisson variance: Var(Y) = mean."""
        return mean
    
    def add_data(self, data: Union[PoissonRegressionData, Tuple[int, Vector], Tuple[int, Vector, float],
                                 List[Union[PoissonRegressionData, Tuple[int, Vector], Tuple[int, Vector, float]]]]):
        """Add Poisson regression data."""
        if isinstance(data, PoissonRegressionData):
            self._add_single_poisson_data(data)
        elif isinstance(data, tuple):
            if len(data) == 2:
                count, x = data
                poisson_data = PoissonRegressionData(count, x)
            elif len(data) == 3:
                count, x, exposure = data
                poisson_data = PoissonRegressionData(count, x, exposure)
            else:
                raise ValueError(f"Invalid tuple length: {len(data)}")
            self._add_single_poisson_data(poisson_data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, PoissonRegressionData):
                    self._add_single_poisson_data(item)
                elif isinstance(item, tuple):
                    if len(item) == 2:
                        count, x = item
                        poisson_data = PoissonRegressionData(count, x)
                    elif len(item) == 3:
                        count, x, exposure = item
                        poisson_data = PoissonRegressionData(count, x, exposure)
                    else:
                        raise ValueError(f"Invalid tuple length: {len(item)}")
                    self._add_single_poisson_data(poisson_data)
                else:
                    raise ValueError(f"Invalid data item: {item}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    def _add_single_poisson_data(self, data: PoissonRegressionData):
        """Add a single Poisson regression data point."""
        self._data.append(data)
        # Note: We don't use the standard regression sufstat for Poisson regression
    
    def predict_rate(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict rate parameter lambda for given predictors."""
        return self.predict(x)
    
    def predict_count(self, x: Union[Vector, np.ndarray, List[float]], exposure: float = 1.0) -> float:
        """Predict expected count for given predictors and exposure."""
        rate = self.predict_rate(x)
        return rate * exposure
    
    def log_likelihood(self, data: Optional[List[PoissonRegressionData]] = None) -> float:
        """Compute log likelihood."""
        if data is None:
            data = self._data
        
        log_lik = 0.0
        for data_point in data:
            if not isinstance(data_point, PoissonRegressionData):
                continue
                
            x = data_point.x()
            count = data_point.count()
            exposure = data_point.exposure()
            
            linear_pred = self.linear_predictor(x)
            rate = self.mean_function(linear_pred)
            expected_count = rate * exposure
            
            # Poisson log likelihood
            log_lik += dpois(count, expected_count, log=True)
        
        return log_lik
    
    def log_prior(self) -> float:
        """Compute log prior density."""
        if not self._has_prior:
            return 0.0
        
        beta_diff = self._beta - self._beta_prior_mean
        beta_diff_array = beta_diff.to_numpy()
        precision_array = self._beta_prior_precision.to_numpy()
        
        log_prior = (-0.5 * self._xdim * np.log(2 * np.pi) +
                    0.5 * np.log(np.linalg.det(precision_array)) -
                    0.5 * np.dot(beta_diff_array, precision_array @ beta_diff_array))
        
        return log_prior
    
    def log_posterior(self) -> float:
        """Compute log posterior density."""
        return self.log_likelihood() + self.log_prior()
    
    def _log_posterior_gradient(self, beta: np.ndarray) -> np.ndarray:
        """Compute gradient of log posterior."""
        old_beta = self._beta.copy()
        self.set_beta(Vector(beta))
        
        # Gradient from likelihood
        gradient = np.zeros(self._xdim)
        for data_point in self._data:
            if not isinstance(data_point, PoissonRegressionData):
                continue
                
            x_array = data_point.x().to_numpy()
            count = data_point.count()
            exposure = data_point.exposure()
            
            linear_pred = self.linear_predictor(data_point.x())
            rate = self.mean_function(linear_pred)
            expected_count = rate * exposure
            
            # Gradient contribution: (y - expected_count) * x
            gradient += (count - expected_count) * x_array
        
        # Gradient from prior
        if self._has_prior:
            beta_diff = beta - self._beta_prior_mean.to_numpy()
            precision_array = self._beta_prior_precision.to_numpy()
            gradient -= precision_array @ beta_diff
        
        # Restore original beta
        self.set_beta(old_beta)
        return gradient
    
    def _log_posterior_hessian(self, beta: np.ndarray) -> np.ndarray:
        """Compute Hessian of log posterior."""
        old_beta = self._beta.copy()
        self.set_beta(Vector(beta))
        
        # Hessian from likelihood (negative definite)
        hessian = np.zeros((self._xdim, self._xdim))
        for data_point in self._data:
            if not isinstance(data_point, PoissonRegressionData):
                continue
                
            x_array = data_point.x().to_numpy()
            exposure = data_point.exposure()
            
            linear_pred = self.linear_predictor(data_point.x())
            rate = self.mean_function(linear_pred)
            expected_count = rate * exposure
            
            # Hessian contribution: -expected_count * x * x^T
            hessian -= expected_count * np.outer(x_array, x_array)
        
        # Hessian from prior (negative definite)
        if self._has_prior:
            hessian -= self._beta_prior_precision.to_numpy()
        
        # Restore original beta
        self.set_beta(old_beta)
        return hessian
    
    def mle(self, max_iterations: int = 100, tolerance: float = 1e-8):
        """Compute maximum likelihood estimate using Newton-Raphson."""
        if len(self._data) == 0:
            return
        
        # Define objective function (negative log likelihood)
        def neg_log_likelihood(beta):
            old_beta = self._beta.copy()
            self.set_beta(Vector(beta))
            nll = -self.log_likelihood()
            self.set_beta(old_beta)
            return nll
        
        def neg_log_likelihood_grad(beta):
            return -self._log_posterior_gradient(beta)
        
        # Use scipy's optimizer
        result = optimize.minimize(
            neg_log_likelihood,
            x0=self._beta.to_numpy(),
            method='BFGS',
            jac=neg_log_likelihood_grad,
            options={'maxiter': max_iterations, 'gtol': tolerance}
        )
        
        if result.success:
            self.set_beta(Vector(result.x))
        else:
            print(f"Warning: MLE optimization did not converge: {result.message}")
    
    def map_estimate(self, max_iterations: int = 100, tolerance: float = 1e-8):
        """Compute maximum a posteriori estimate."""
        if len(self._data) == 0:
            return
        
        # Define objective function (negative log posterior)
        def neg_log_posterior(beta):
            old_beta = self._beta.copy()
            self.set_beta(Vector(beta))
            nlp = -self.log_posterior()
            self.set_beta(old_beta)
            return nlp
        
        def neg_log_posterior_grad(beta):
            return -self._log_posterior_gradient(beta)
        
        # Use scipy's optimizer
        result = optimize.minimize(
            neg_log_posterior,
            x0=self._beta.to_numpy(),
            method='BFGS',
            jac=neg_log_posterior_grad,
            options={'maxiter': max_iterations, 'gtol': tolerance}
        )
        
        if result.success:
            self.set_beta(Vector(result.x))
        else:
            print(f"Warning: MAP optimization did not converge: {result.message}")
    
    def laplace_approximation(self) -> Tuple[Vector, SpdMatrix]:
        """Compute Laplace approximation to posterior.
        
        Returns:
            Posterior mean and covariance matrix
        """
        # Find MAP estimate
        self.map_estimate()
        
        # Compute Hessian at MAP
        hessian = self._log_posterior_hessian(self._beta.to_numpy())
        
        try:
            # Posterior covariance is negative inverse of Hessian
            posterior_cov = SpdMatrix(-np.linalg.inv(hessian))
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudoinverse
            posterior_cov = SpdMatrix(-np.linalg.pinv(hessian))
        
        return self._beta.copy(), posterior_cov
    
    def sample_posterior_laplace(self, n: int = 1, rng: Optional[RNG] = None) -> List[Vector]:
        """Sample from Laplace approximation to posterior."""
        if rng is None:
            rng = RNG()
        
        posterior_mean, posterior_cov = self.laplace_approximation()
        
        samples = []
        for _ in range(n):
            sample = stats.multivariate_normal.rvs(
                mean=posterior_mean.to_numpy(),
                cov=posterior_cov.to_numpy(),
                random_state=rng._rng
            )
            samples.append(Vector(sample))
        
        return samples
    
    def deviance(self) -> float:
        """Compute deviance for model assessment."""
        deviance = 0.0
        for data_point in self._data:
            if not isinstance(data_point, PoissonRegressionData):
                continue
                
            x = data_point.x()
            count = data_point.count()
            exposure = data_point.exposure()
            
            linear_pred = self.linear_predictor(x)
            rate = self.mean_function(linear_pred)
            expected_count = rate * exposure
            
            # Deviance contribution: 2 * (y * log(y/mu) - (y - mu))
            if count > 0:
                deviance += 2 * (count * np.log(count / expected_count) - (count - expected_count))
            else:
                deviance += 2 * expected_count
        
        return deviance
    
    def aic(self) -> float:
        """Compute Akaike Information Criterion."""
        return -2 * self.log_likelihood() + 2 * self._xdim
    
    def bic(self) -> float:
        """Compute Bayesian Information Criterion."""
        n = len(self._data)
        return -2 * self.log_likelihood() + self._xdim * np.log(n)
    
    def simulate_data(self, n: int, X: Union[np.ndarray, List[List[float]]],
                     exposures: Union[float, List[float]] = 1.0,
                     rng: Optional[RNG] = None) -> List[PoissonRegressionData]:
        """Simulate data from the model.
        
        Args:
            n: Number of observations to simulate
            X: Design matrix (n x p)
            exposures: Exposure values (single value or list)
            rng: Random number generator
            
        Returns:
            List of simulated Poisson regression data
        """
        if rng is None:
            rng = RNG()
        
        X_array = np.array(X)
        if X_array.shape[0] != n:
            raise ValueError(f"X has {X_array.shape[0]} rows, expected {n}")
        if X_array.shape[1] != self._xdim:
            raise ValueError(f"X has {X_array.shape[1]} columns, expected {self._xdim}")
        
        if isinstance(exposures, (int, float)):
            exposure_list = [float(exposures)] * n
        else:
            exposure_list = list(exposures)
            if len(exposure_list) != n:
                raise ValueError(f"exposures has length {len(exposure_list)}, expected {n}")
        
        simulated_data = []
        beta_array = self._beta.to_numpy()
        
        for i in range(n):
            x_i = Vector(X_array[i])
            linear_pred = np.dot(beta_array, X_array[i])
            rate_i = self.mean_function(linear_pred)
            exposure_i = exposure_list[i]
            expected_count = rate_i * exposure_i
            
            count_i = rpois(expected_count, rng)
            simulated_data.append(PoissonRegressionData(count_i, x_i, exposure_i))
        
        return simulated_data
    
    def clone(self) -> 'PoissonRegressionModel':
        """Create a copy of this model."""
        cloned = PoissonRegressionModel(self._xdim)
        cloned.set_beta(self._beta)
        
        if self._has_prior:
            cloned.set_prior(self._beta_prior_mean, self._beta_prior_precision)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"PoissonRegressionModel(xdim={self._xdim}, beta={self._beta.to_numpy()}, "
                f"data_points={len(self._data)})")