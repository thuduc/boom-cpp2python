"""Logistic regression model with conjugate prior support."""

import numpy as np
from typing import List, Optional, Union, Tuple
from scipy import stats, optimize

from .base import GlmModel, GlmData
from boom.linalg import Vector, SpdMatrix
from boom.distributions import RNG
from boom.distributions.rmath import dbinom, rbinom
from boom.models.base import ConjugateModel


class LogisticRegressionData(GlmData):
    """Data for logistic regression with trials and successes."""
    
    def __init__(self, successes: int, trials: int, x: Union[List[float], np.ndarray, Vector]):
        """Initialize logistic regression data.
        
        Args:
            successes: Number of successes
            trials: Number of trials
            x: Predictor vector
        """
        if successes < 0 or trials < 0 or successes > trials:
            raise ValueError("Invalid successes/trials combination")
        
        # Store success proportion as response
        proportion = successes / trials if trials > 0 else 0.0
        super().__init__(proportion, x)
        
        self._successes = int(successes)
        self._trials = int(trials)
    
    def successes(self) -> int:
        """Get number of successes."""
        return self._successes
    
    def trials(self) -> int:
        """Get number of trials."""
        return self._trials
    
    def failures(self) -> int:
        """Get number of failures."""
        return self._trials - self._successes
    
    def proportion(self) -> float:
        """Get success proportion."""
        return self.y()
    
    def clone(self) -> 'LogisticRegressionData':
        """Create a copy of this data point."""
        return LogisticRegressionData(self._successes, self._trials, self.x())
    
    def __repr__(self) -> str:
        return f"LogisticRegressionData(successes={self._successes}, trials={self._trials}, x={self.x().to_numpy()})"


class LogisticRegressionModel(GlmModel):
    """Logistic regression model.
    
    Model: y_i ~ Binomial(n_i, p_i)
           logit(p_i) = X_i^T * beta
    
    Note: While full conjugacy is not available for logistic regression,
    we can use normal priors on beta and approximate posteriors.
    """
    
    def __init__(self, xdim: int):
        """Initialize logistic regression model.
        
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
        """Logistic function: p = exp(eta) / (1 + exp(eta))."""
        # Use stable computation to avoid overflow
        if linear_pred > 500:
            return 1.0
        elif linear_pred < -500:
            return 0.0
        else:
            exp_eta = np.exp(linear_pred)
            return exp_eta / (1.0 + exp_eta)
    
    def variance_function(self, mean: float) -> float:
        """Binomial variance: p * (1 - p)."""
        return mean * (1.0 - mean)
    
    def add_data(self, data: Union[LogisticRegressionData, Tuple[int, int, Vector], 
                                 List[Union[LogisticRegressionData, Tuple[int, int, Vector]]]]):
        """Add logistic regression data."""
        if isinstance(data, LogisticRegressionData):
            self._add_single_logistic_data(data)
        elif isinstance(data, tuple) and len(data) == 3:
            successes, trials, x = data
            logistic_data = LogisticRegressionData(successes, trials, x)
            self._add_single_logistic_data(logistic_data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, LogisticRegressionData):
                    self._add_single_logistic_data(item)
                elif isinstance(item, tuple) and len(item) == 3:
                    successes, trials, x = item
                    logistic_data = LogisticRegressionData(successes, trials, x)
                    self._add_single_logistic_data(logistic_data)
                else:
                    raise ValueError(f"Invalid data item: {item}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    def _add_single_logistic_data(self, data: LogisticRegressionData):
        """Add a single logistic regression data point."""
        self._data.append(data)
        # Note: We don't use the standard regression sufstat for logistic regression
    
    def predict_probability(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Predict probability for given predictors."""
        return self.predict(x)
    
    def predict_class(self, x: Union[Vector, np.ndarray, List[float]], threshold: float = 0.5) -> int:
        """Predict class (0 or 1) for given predictors."""
        prob = self.predict_probability(x)
        return 1 if prob >= threshold else 0
    
    def log_likelihood(self, data: Optional[List[LogisticRegressionData]] = None) -> float:
        """Compute log likelihood."""
        if data is None:
            data = self._data
        
        log_lik = 0.0
        for data_point in data:
            if not isinstance(data_point, LogisticRegressionData):
                continue
                
            x = data_point.x()
            successes = data_point.successes()
            trials = data_point.trials()
            
            linear_pred = self.linear_predictor(x)
            p = self.mean_function(linear_pred)
            
            # Binomial log likelihood
            log_lik += dbinom(successes, trials, p, log=True)
        
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
            if not isinstance(data_point, LogisticRegressionData):
                continue
                
            x_array = data_point.x().to_numpy()
            successes = data_point.successes()
            trials = data_point.trials()
            
            linear_pred = self.linear_predictor(data_point.x())
            p = self.mean_function(linear_pred)
            
            # Gradient contribution: (y - n*p) * x
            gradient += (successes - trials * p) * x_array
        
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
            if not isinstance(data_point, LogisticRegressionData):
                continue
                
            x_array = data_point.x().to_numpy()
            trials = data_point.trials()
            
            linear_pred = self.linear_predictor(data_point.x())
            p = self.mean_function(linear_pred)
            
            # Hessian contribution: -n * p * (1-p) * x * x^T
            weight = trials * p * (1 - p)
            hessian -= weight * np.outer(x_array, x_array)
        
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
        
        # Better initialization using simple heuristic
        self._initialize_beta_for_mle()
        
        # Define objective function (negative log likelihood)
        def neg_log_likelihood(beta):
            # Prevent extreme values that cause numerical issues
            beta = np.clip(beta, -10, 10)
            old_beta = self._beta.copy()
            try:
                self.set_beta(Vector(beta))
                nll = -self.log_likelihood()
                # Add small regularization to prevent divergence
                nll += 1e-8 * np.sum(beta**2)
                self.set_beta(old_beta)
                return nll
            except:
                self.set_beta(old_beta)
                return 1e10  # Return large value if computation fails
        
        def neg_log_likelihood_grad(beta):
            beta = np.clip(beta, -10, 10)
            try:
                grad = -self._log_posterior_gradient(beta)
                # Add small regularization
                grad += 2 * 1e-8 * beta
                return grad
            except:
                return np.zeros_like(beta)  # Return zero gradient if computation fails
        
        # Try different optimization methods
        methods = ['BFGS', 'L-BFGS-B']
        best_result = None
        best_obj = np.inf
        
        for method in methods:
            try:
                if method == 'L-BFGS-B':
                    result = optimize.minimize(
                        neg_log_likelihood,
                        x0=self._beta.to_numpy(),
                        method=method,
                        jac=neg_log_likelihood_grad,
                        bounds=[(-10, 10)] * self._xdim,  # Bound parameters
                        options={'maxiter': max_iterations, 'ftol': tolerance}
                    )
                else:
                    result = optimize.minimize(
                        neg_log_likelihood,
                        x0=self._beta.to_numpy(),
                        method=method,
                        jac=neg_log_likelihood_grad,
                        options={'maxiter': max_iterations, 'gtol': tolerance}
                    )
                
                if result.fun < best_obj:
                    best_result = result
                    best_obj = result.fun
                    
            except Exception as e:
                continue
        
        if best_result is not None and best_result.fun < 1e9:
            self.set_beta(Vector(np.clip(best_result.x, -10, 10)))
        else:
            print(f"Warning: MLE optimization did not converge: {best_result.message if best_result else 'All methods failed'}")
    
    def _initialize_beta_for_mle(self):
        """Initialize beta using simple heuristic."""
        if len(self._data) == 0:
            return
        
        # Simple initialization: use proportion-based logit values
        try:
            # Collect all data for initialization
            X_list = []
            y_list = []
            
            for data_point in self._data:
                if isinstance(data_point, LogisticRegressionData):
                    X_list.append(data_point.x().to_numpy())
                    # Use adjusted proportions to avoid log(0) or log(inf)
                    prop = data_point.proportion()
                    # Adjust extreme proportions
                    prop = max(0.01, min(0.99, prop))
                    y_list.append(prop)
            
            if len(X_list) > 0:
                X = np.array(X_list)
                y = np.array(y_list)
                
                # Simple linear regression on logit-transformed proportions
                logit_y = np.log(y / (1 - y))
                
                try:
                    # Use regularized least squares
                    XTX = X.T @ X + 1e-6 * np.eye(X.shape[1])
                    XTy = X.T @ logit_y
                    beta_init = np.linalg.solve(XTX, XTy)
                    
                    # Clip to reasonable range
                    beta_init = np.clip(beta_init, -5, 5)
                    self.set_beta(Vector(beta_init))
                except:
                    # If that fails, just use small random values
                    self.set_beta(Vector(np.random.normal(0, 0.1, self._xdim)))
                    
        except Exception as e:
            # Fallback: small random initialization
            self.set_beta(Vector(np.random.normal(0, 0.1, self._xdim)))
    
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
    
    def simulate_data(self, n: int, X: Union[np.ndarray, List[List[float]]],
                     trials_per_obs: Union[int, List[int]] = 1,
                     rng: Optional[RNG] = None) -> List[LogisticRegressionData]:
        """Simulate data from the model.
        
        Args:
            n: Number of observations to simulate
            X: Design matrix (n x p)
            trials_per_obs: Number of trials per observation (int or list)
            rng: Random number generator
            
        Returns:
            List of simulated logistic regression data
        """
        if rng is None:
            rng = RNG()
        
        X_array = np.array(X)
        if X_array.shape[0] != n:
            raise ValueError(f"X has {X_array.shape[0]} rows, expected {n}")
        if X_array.shape[1] != self._xdim:
            raise ValueError(f"X has {X_array.shape[1]} columns, expected {self._xdim}")
        
        if isinstance(trials_per_obs, int):
            trials_list = [trials_per_obs] * n
        else:
            trials_list = list(trials_per_obs)
            if len(trials_list) != n:
                raise ValueError(f"trials_per_obs has length {len(trials_list)}, expected {n}")
        
        simulated_data = []
        beta_array = self._beta.to_numpy()
        
        for i in range(n):
            x_i = Vector(X_array[i])
            linear_pred = np.dot(beta_array, X_array[i])
            p_i = self.mean_function(linear_pred)
            trials_i = trials_list[i]
            
            successes_i = rbinom(trials_i, p_i, rng)
            simulated_data.append(LogisticRegressionData(successes_i, trials_i, x_i))
        
        return simulated_data
    
    def clone(self) -> 'LogisticRegressionModel':
        """Create a copy of this model."""
        cloned = LogisticRegressionModel(self._xdim)
        cloned.set_beta(self._beta)
        
        if self._has_prior:
            cloned.set_prior(self._beta_prior_mean, self._beta_prior_precision)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"LogisticRegressionModel(xdim={self._xdim}, beta={self._beta.to_numpy()}, "
                f"data_points={len(self._data)})")