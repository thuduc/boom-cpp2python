"""Multivariate probability distributions for BOOM."""
import numpy as np
from scipy import stats
from typing import Union, Optional, Tuple
from ..linalg import Vector, Matrix, SpdMatrix
from ..math.special_functions import lmultigamma, lgamma


class MultivariateDistribution:
    """Base class for multivariate distributions."""
    
    def pdf(self, x: np.ndarray) -> float:
        """Probability density function."""
        raise NotImplementedError
    
    def logpdf(self, x: np.ndarray) -> float:
        """Log probability density function."""
        return np.log(self.pdf(x))
    
    def mean(self) -> np.ndarray:
        """Mean vector."""
        raise NotImplementedError
    
    def covariance(self) -> np.ndarray:
        """Covariance matrix."""
        raise NotImplementedError
    
    def sample(self, size: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        """Sample from the distribution."""
        raise NotImplementedError


class MultivariateNormal(MultivariateDistribution):
    """Multivariate normal distribution."""
    
    def __init__(self, mean: Union[Vector, np.ndarray], cov: Union[SpdMatrix, np.ndarray]):
        """Initialize Multivariate Normal distribution.
        
        Args:
            mean: Mean vector
            cov: Covariance matrix (must be positive semi-definite)
        """
        self.mean_vec = np.asarray(mean)
        self.cov_mat = np.asarray(cov)
        self.dim = len(self.mean_vec)
        
        if self.cov_mat.shape != (self.dim, self.dim):
            raise ValueError("Covariance matrix dimensions must match mean vector")
        
        # Check positive semi-definite
        eigvals = np.linalg.eigvalsh(self.cov_mat)
        if np.any(eigvals < -1e-8):
            raise ValueError("Covariance matrix must be positive semi-definite")
    
    def pdf(self, x):
        return stats.multivariate_normal.pdf(x, mean=self.mean_vec, cov=self.cov_mat)
    
    def logpdf(self, x):
        return stats.multivariate_normal.logpdf(x, mean=self.mean_vec, cov=self.cov_mat)
    
    def mean(self):
        return self.mean_vec.copy()
    
    def covariance(self):
        return self.cov_mat.copy()
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.multivariate_normal(self.mean_vec, self.cov_mat, size=size)


class Dirichlet(MultivariateDistribution):
    """Dirichlet distribution."""
    
    def __init__(self, alpha: Union[Vector, np.ndarray]):
        """Initialize Dirichlet distribution.
        
        Args:
            alpha: Concentration parameters (all must be positive)
        """
        self.alpha = np.asarray(alpha)
        if np.any(self.alpha <= 0):
            raise ValueError("All concentration parameters must be positive")
        self.dim = len(self.alpha)
        self.alpha_sum = np.sum(self.alpha)
    
    def pdf(self, x):
        return stats.dirichlet.pdf(x.T, alpha=self.alpha)
    
    def logpdf(self, x):
        return stats.dirichlet.logpdf(x.T, alpha=self.alpha)
    
    def mean(self):
        return self.alpha / self.alpha_sum
    
    def covariance(self):
        mean = self.mean()
        cov = np.diag(mean * (1 - mean))
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                cov[i, j] = cov[j, i] = -mean[i] * mean[j]
        return cov / (self.alpha_sum + 1)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.dirichlet(self.alpha, size=size)


class Multinomial(MultivariateDistribution):
    """Multinomial distribution."""
    
    def __init__(self, n: int, probs: Union[Vector, np.ndarray]):
        """Initialize Multinomial distribution.
        
        Args:
            n: Number of trials
            probs: Event probabilities (must sum to 1)
        """
        if n < 0 or not isinstance(n, (int, np.integer)):
            raise ValueError("n must be non-negative integer")
        self.n = n
        self.probs = np.asarray(probs)
        
        if np.any(self.probs < 0) or np.any(self.probs > 1):
            raise ValueError("Probabilities must be in [0, 1]")
        if not np.allclose(np.sum(self.probs), 1.0):
            raise ValueError("Probabilities must sum to 1")
        
        self.dim = len(self.probs)
    
    def pmf(self, x):
        """Probability mass function."""
        return stats.multinomial.pmf(x, n=self.n, p=self.probs)
    
    def logpmf(self, x):
        """Log probability mass function."""
        return stats.multinomial.logpmf(x, n=self.n, p=self.probs)
    
    def mean(self):
        return self.n * self.probs
    
    def covariance(self):
        p = self.probs
        cov = -self.n * np.outer(p, p)
        np.fill_diagonal(cov, self.n * p * (1 - p))
        return cov
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.multinomial(self.n, self.probs, size=size)


class Wishart(MultivariateDistribution):
    """Wishart distribution."""
    
    def __init__(self, df: int, scale: Union[SpdMatrix, np.ndarray]):
        """Initialize Wishart distribution.
        
        Args:
            df: Degrees of freedom (must be > dim - 1)
            scale: Scale matrix (must be positive definite)
        """
        self.scale = np.asarray(scale)
        self.dim = self.scale.shape[0]
        
        if self.scale.shape != (self.dim, self.dim):
            raise ValueError("Scale must be square matrix")
        if df <= self.dim - 1:
            raise ValueError(f"Degrees of freedom must be > {self.dim - 1}")
        
        self.df = df
        
        # Check positive definite
        try:
            self.scale_chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError:
            raise ValueError("Scale matrix must be positive definite")
    
    def pdf(self, X):
        return stats.wishart.pdf(X, df=self.df, scale=self.scale)
    
    def logpdf(self, X):
        return stats.wishart.logpdf(X, df=self.df, scale=self.scale)
    
    def mean(self):
        return self.df * self.scale
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return stats.wishart.rvs(df=self.df, scale=self.scale, size=size, random_state=rng)


class InverseWishart(MultivariateDistribution):
    """Inverse Wishart distribution."""
    
    def __init__(self, df: int, scale: Union[SpdMatrix, np.ndarray]):
        """Initialize Inverse Wishart distribution.
        
        Args:
            df: Degrees of freedom (must be > dim + 1)
            scale: Scale matrix (must be positive definite)
        """
        self.scale = np.asarray(scale)
        self.dim = self.scale.shape[0]
        
        if self.scale.shape != (self.dim, self.dim):
            raise ValueError("Scale must be square matrix")
        if df <= self.dim + 1:
            raise ValueError(f"Degrees of freedom must be > {self.dim + 1}")
        
        self.df = df
        
        # Check positive definite
        try:
            self.scale_chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError:
            raise ValueError("Scale matrix must be positive definite")
    
    def pdf(self, X):
        return stats.invwishart.pdf(X, df=self.df, scale=self.scale)
    
    def logpdf(self, X):
        return stats.invwishart.logpdf(X, df=self.df, scale=self.scale)
    
    def mean(self):
        if self.df > self.dim + 1:
            return self.scale / (self.df - self.dim - 1)
        else:
            return None  # Mean doesn't exist
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return stats.invwishart.rvs(df=self.df, scale=self.scale, size=size, random_state=rng)


class MatrixNormal(MultivariateDistribution):
    """Matrix Normal distribution."""
    
    def __init__(self, mean: Union[Matrix, np.ndarray], 
                 row_cov: Union[SpdMatrix, np.ndarray],
                 col_cov: Union[SpdMatrix, np.ndarray]):
        """Initialize Matrix Normal distribution.
        
        Args:
            mean: Mean matrix (n x p)
            row_cov: Row covariance matrix (n x n)
            col_cov: Column covariance matrix (p x p)
        """
        self.mean_mat = np.asarray(mean)
        self.row_cov = np.asarray(row_cov)
        self.col_cov = np.asarray(col_cov)
        
        self.n_rows, self.n_cols = self.mean_mat.shape
        
        if self.row_cov.shape != (self.n_rows, self.n_rows):
            raise ValueError("Row covariance dimensions don't match mean")
        if self.col_cov.shape != (self.n_cols, self.n_cols):
            raise ValueError("Column covariance dimensions don't match mean")
    
    def mean(self):
        return self.mean_mat.copy()
    
    def vec_covariance(self):
        """Covariance of vec(X)."""
        return np.kron(self.col_cov, self.row_cov)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        
        # Sample from matrix normal using the relationship:
        # X ~ MN(M, U, V) iff vec(X) ~ N(vec(M), V ⊗ U)
        if size is None:
            Z = rng.standard_normal((self.n_rows, self.n_cols))
            L_row = np.linalg.cholesky(self.row_cov)
            L_col = np.linalg.cholesky(self.col_cov)
            return self.mean_mat + L_row @ Z @ L_col.T
        else:
            samples = []
            for _ in range(size):
                Z = rng.standard_normal((self.n_rows, self.n_cols))
                L_row = np.linalg.cholesky(self.row_cov)
                L_col = np.linalg.cholesky(self.col_cov)
                samples.append(self.mean_mat + L_row @ Z @ L_col.T)
            return np.array(samples)