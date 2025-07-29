"""Gaussian mixture models with multivariate support."""
import numpy as np
from typing import Optional, Union, List
from .base import MixtureModel, MixtureComponent
from ..base import VectorParameter
from ...linalg import Vector, Matrix, SpdMatrix
from ...distributions.continuous import Normal


class MultivariateGaussianComponent(MixtureComponent):
    """Multivariate Gaussian mixture component."""
    
    def __init__(self, mean: Optional[Vector] = None, 
                 covariance: Optional[SpdMatrix] = None, 
                 weight: float = 1.0):
        """Initialize multivariate Gaussian component.
        
        Args:
            mean: Component mean vector
            covariance: Component covariance matrix
            weight: Component weight
        """
        super().__init__(weight)
        
        self.dim = len(mean) if mean is not None else None
        
        if mean is not None:
            self._mean_param = VectorParameter(Vector(mean), 'mean')
        else:
            self._mean_param = None
        
        if covariance is not None:
            self.covariance = SpdMatrix(covariance)
        else:
            self.covariance = None
    
    @property
    def mean(self) -> Vector:
        """Get mean vector."""
        if self._mean_param is None:
            raise ValueError("Mean not initialized")
        return self._mean_param.value
    
    @mean.setter
    def mean(self, value: Union[Vector, np.ndarray, List[float]]):
        """Set mean vector."""
        mean_vec = Vector(value)
        if self.dim is None:
            self.dim = len(mean_vec)
            self._mean_param = VectorParameter(mean_vec, 'mean')
            if self.covariance is None:
                self.covariance = SpdMatrix.identity(self.dim)
        elif len(mean_vec) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(mean_vec)}")
        else:
            self._mean_param.value = mean_vec
    
    def set_covariance(self, cov: Union[SpdMatrix, Matrix, np.ndarray]):
        """Set covariance matrix."""
        self.covariance = SpdMatrix(cov)
        if self.dim is not None and self.covariance.nrow() != self.dim:
            raise ValueError(f"Covariance dimension {self.covariance.nrow()} doesn't match mean dimension {self.dim}")
    
    def logpdf(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Log probability density function."""
        if self._mean_param is None or self.covariance is None:
            raise ValueError("Component not fully initialized")
        
        x_vec = Vector(x)
        return self._multivariate_normal_logpdf(x_vec, self.mean, self.covariance)
    
    def _multivariate_normal_logpdf(self, x: Vector, mean: Vector, cov: SpdMatrix) -> float:
        """Compute multivariate normal log PDF."""
        d = len(x)
        diff = x - mean
        
        # Compute (x-mu)^T Sigma^{-1} (x-mu)
        try:
            cov_inv = cov.inv()
            mahalanobis_sq = diff.dot(cov_inv @ diff)
        except np.linalg.LinAlgError:
            # Fallback for singular covariance
            return -np.inf
        
        # Compute log determinant
        try:
            log_det = np.log(np.linalg.det(cov))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
        
        # Multivariate normal log PDF
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis_sq)
    
    def pdf(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Probability density function."""
        return np.exp(self.logpdf(x))
    
    def sample(self, rng) -> Vector:
        """Sample from the component."""
        if self._mean_param is None or self.covariance is None:
            raise ValueError("Component not fully initialized")
        
        return Vector(rng.rmvn(self.mean, self.covariance))
    
    def fit(self, data: Matrix, weights: Vector):
        """Fit component to weighted data using MLE.
        
        Args:
            data: Data matrix (n_samples x n_features)
            weights: Sample weights
        """
        if data.nrow() != len(weights):
            raise ValueError("Data and weights must have same number of samples")
        
        if len(weights) == 0:
            return
        
        n, d = data.nrow(), data.ncol()
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            # Weighted mean
            weighted_sum = Vector.zero(d)
            for i in range(n):
                weighted_sum += weights[i] * data.row(i)
            self.mean = weighted_sum / weight_sum
            
            # Weighted covariance
            weighted_cov = Matrix.zero(d, d)
            for i in range(n):
                diff = data.row(i) - self.mean
                weighted_cov += weights[i] * np.outer(diff, diff)
            
            cov_matrix = weighted_cov / weight_sum
            
            # Add regularization to prevent singular matrices
            cov_matrix += 1e-6 * np.eye(d)
            
            self.covariance = SpdMatrix(cov_matrix)
    
    def clone(self) -> 'MultivariateGaussianComponent':
        """Create a copy of the component."""
        mean = self.mean.copy() if self._mean_param is not None else None
        cov = SpdMatrix(self.covariance) if self.covariance is not None else None
        return MultivariateGaussianComponent(mean, cov, self.weight)


class GaussianMixtureModel(MixtureModel):
    """Multivariate Gaussian mixture model."""
    
    def __init__(self, n_components: int, n_features: int):
        """Initialize Gaussian mixture model.
        
        Args:
            n_components: Number of mixture components
            n_features: Number of features (dimensionality)
        """
        super().__init__(n_components)
        self.n_features = n_features
        self._data_matrix = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components with default parameters."""
        self.components = []
        
        for i in range(self.n_components):
            # Initialize with different means
            mean = Vector.zero(self.n_features)
            mean[0] = i - (self.n_components - 1) / 2.0  # Spread along first dimension
            
            covariance = SpdMatrix.identity(self.n_features)
            component = MultivariateGaussianComponent(mean, covariance)
            self.components.append(component)
    
    def add_data(self, x: Union[Vector, np.ndarray, List[float]]):
        """Add a data vector."""
        x_vec = Vector(x)
        if len(x_vec) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x_vec)}")
        
        # Store as list first, convert to matrix when needed
        if not hasattr(self, '_data_list'):
            self._data_list = []
        self._data_list.append(x_vec)
        self._data_matrix = None  # Invalidate cached matrix
    
    def set_data(self, data: Union[Matrix, np.ndarray, List[List[float]]]):
        """Set all data."""
        if isinstance(data, (list, tuple)):
            # Convert list of lists to matrix
            data = Matrix(data)
        elif isinstance(data, np.ndarray):
            data = Matrix(data)
        
        if data.ncol() != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {data.ncol()}")
        
        self._data_matrix = data
        self._data_list = [data.row(i) for i in range(data.nrow())]
    
    def get_data_matrix(self) -> Matrix:
        """Get data as a matrix."""
        if self._data_matrix is None and hasattr(self, '_data_list'):
            if self._data_list:
                data_array = np.array([np.array(x) for x in self._data_list])
                self._data_matrix = Matrix(data_array)
            else:
                self._data_matrix = Matrix.zero(0, self.n_features)
        
        if self._data_matrix is not None:
            return self._data_matrix
        else:
            return Matrix.zero(0, self.n_features)
    
    def clear_data(self):
        """Clear all data."""
        self._data_list = []
        self._data_matrix = None
    
    def logpdf(self, x: Union[Vector, np.ndarray, List[float]]) -> float:
        """Log probability density of mixture."""
        if not self.components:
            raise ValueError("No components initialized")
        
        # Log-sum-exp trick for numerical stability
        log_densities = []
        for i, component in enumerate(self.components):
            log_weight = np.log(self.mixing_weights[i]) if self.mixing_weights[i] > 0 else -np.inf
            log_densities.append(log_weight + component.logpdf(x))
        
        max_log_density = max(log_densities)
        log_sum = max_log_density + np.log(sum(np.exp(ld - max_log_density) for ld in log_densities))
        return log_sum
    
    def loglike(self) -> float:
        """Log likelihood of data."""
        data_matrix = self.get_data_matrix()
        if data_matrix.nrow() == 0:
            return 0.0
        
        loglike = 0.0
        for i in range(data_matrix.nrow()):
            loglike += self.logpdf(data_matrix.row(i))
        
        return loglike
    
    def component_posteriors(self, x: Union[Vector, np.ndarray, List[float]]) -> Vector:
        """Posterior probabilities of components given x."""
        if not self.components:
            raise ValueError("No components initialized")
        
        log_posteriors = []
        for i, component in enumerate(self.components):
            log_weight = np.log(self.mixing_weights[i]) if self.mixing_weights[i] > 0 else -np.inf
            log_posteriors.append(log_weight + component.logpdf(x))
        
        # Normalize using log-sum-exp
        max_log_post = max(log_posteriors)
        exp_log_posts = [np.exp(lp - max_log_post) for lp in log_posteriors]
        sum_exp = sum(exp_log_posts)
        
        return Vector([elp / sum_exp for elp in exp_log_posts])
    
    def initialize_from_data(self, method: str = "kmeans"):
        """Initialize components from data."""
        data_matrix = self.get_data_matrix()
        if data_matrix.nrow() == 0:
            raise ValueError("No data available for initialization")
        
        n = data_matrix.nrow()
        
        if method == "random":
            # Random initialization
            from ...distributions import rng
            # Simple random sampling without replacement
            n_sample = min(self.n_components, n)
            indices = []
            available = list(range(n))
            for _ in range(n_sample):
                idx = int(rng.runif() * len(available))
                indices.append(available.pop(idx))
            
            # Compute global covariance for initialization
            overall_mean = Vector.zero(self.n_features)
            for i in range(n):
                overall_mean += data_matrix.row(i)
            overall_mean /= n
            
            overall_cov = Matrix.zero(self.n_features, self.n_features)
            for i in range(n):
                diff = data_matrix.row(i) - overall_mean
                overall_cov += np.outer(diff, diff)
            overall_cov = SpdMatrix(overall_cov / n + 1e-6 * np.eye(self.n_features))
            
            for i, component in enumerate(self.components):
                component.mean = data_matrix.row(indices[i])
                component.set_covariance(overall_cov)
        
        elif method == "kmeans":
            # Simple k-means style initialization
            # Use first principal component to spread means
            data_array = np.array([np.array(data_matrix.row(i)) for i in range(n)])
            
            # Compute mean and center data
            overall_mean = np.mean(data_array, axis=0)
            centered_data = data_array - overall_mean
            
            # Simple spread along first principal component
            if self.n_features > 0:
                # Use range along first dimension as proxy
                min_val = np.min(data_array[:, 0])
                max_val = np.max(data_array[:, 0])
                spread = np.linspace(min_val, max_val, self.n_components)
                
                # Compute overall covariance
                overall_cov = np.cov(data_array.T) + 1e-6 * np.eye(self.n_features)
                
                for i, component in enumerate(self.components):
                    mean = Vector(overall_mean)
                    if self.n_features > 0:
                        mean[0] = spread[i]
                    component.mean = mean
                    component.set_covariance(SpdMatrix(overall_cov))
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def expectation_step(self) -> Matrix:
        """E-step: compute component responsibilities."""
        data_matrix = self.get_data_matrix()
        n = data_matrix.nrow()
        
        if n == 0:
            return Matrix.zero(0, self.n_components)
        
        responsibilities = Matrix.zero(n, self.n_components)
        
        for i in range(n):
            x = data_matrix.row(i)
            posteriors = self.component_posteriors(x)
            responsibilities[i, :] = posteriors
        
        return responsibilities
    
    def maximization_step(self, responsibilities: Matrix):
        """M-step: update parameters."""
        data_matrix = self.get_data_matrix()
        n = data_matrix.nrow()
        
        if n == 0:
            return
        
        # Update mixing weights
        n_k = Vector([float(responsibilities[:, k].sum()) for k in range(self.n_components)])
        self.mixing_weights = n_k / n
        
        # Update component parameters
        for k, component in enumerate(self.components):
            weights = Vector([responsibilities[i, k] for i in range(n)])
            component.fit(data_matrix, weights)
    
    def fit(self, max_iter: int = 100, tol: float = 1e-6, init_method: str = "kmeans"):
        """Fit mixture model using EM algorithm."""
        data_matrix = self.get_data_matrix()
        if data_matrix.nrow() == 0:
            raise ValueError("No data to fit")
        
        # Initialize components
        self.initialize_from_data(init_method)
        
        prev_loglike = -np.inf
        
        for iteration in range(max_iter):
            # E-step
            responsibilities = self.expectation_step()
            
            # M-step
            self.maximization_step(responsibilities)
            
            # Check convergence
            current_loglike = self.loglike()
            
            if abs(current_loglike - prev_loglike) < tol:
                break
            
            prev_loglike = current_loglike
    
    def sample(self, rng) -> Vector:
        """Sample from the mixture."""
        if not self.components:
            raise ValueError("No components initialized")
        
        # Sample component
        component_idx = self.sample_component(rng)
        
        # Sample from that component
        return self.components[component_idx].sample(rng)
    
    def simulate(self, n: int, rng=None) -> Matrix:
        """Simulate n samples from the mixture."""
        if rng is None:
            from ...distributions import rng as global_rng
            rng = global_rng
        
        samples = Matrix.zero(n, self.n_features)
        for i in range(n):
            samples[i, :] = self.sample(rng)
        
        return samples
    
    def clone(self) -> 'GaussianMixtureModel':
        """Create a copy of the model."""
        model = GaussianMixtureModel(self.n_components, self.n_features)
        
        # Copy components
        model.components = [comp.clone() for comp in self.components]
        
        # Copy mixing weights
        model.mixing_weights = self.mixing_weights.copy()
        
        # Copy data
        if hasattr(self, '_data_list'):
            model._data_list = [x.copy() for x in self._data_list]
            model._data_matrix = None
        
        return model