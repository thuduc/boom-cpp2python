"""Integration tests for basic BOOM workflows."""
import pytest
import numpy as np
from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import rng, seed_rng


class TestBasicWorkflow:
    """Test basic workflows combining multiple components."""
    
    def test_linear_regression_data_generation(self):
        """Test generating data for linear regression."""
        seed_rng(42)
        
        # Set up true parameters
        n = 100
        p = 3
        true_beta = Vector([1.5, -2.0, 0.5])
        sigma = 0.5
        
        # Generate predictors
        X = Matrix((n, p))
        for i in range(n):
            for j in range(p):
                X[i, j] = rng.rnorm()
        
        # Generate response
        y = Vector(n)
        for i in range(n):
            mean = X.row(i).dot(true_beta)
            y[i] = rng.rnorm(mean, sigma)
        
        # Compute OLS estimate
        XtX = Matrix(X.T @ X)
        Xty = X.T @ y
        beta_hat = XtX.solve(Xty)
        
        # Should be close to true values
        assert isinstance(beta_hat, Vector)
        assert len(beta_hat) == p
        assert np.allclose(beta_hat, true_beta, atol=0.2)
    
    def test_multivariate_normal_sampling(self):
        """Test sampling from multivariate normal with covariance structure."""
        seed_rng(123)
        
        # Create correlation matrix
        p = 4
        corr = SpdMatrix.identity(p)
        corr[0, 1] = corr[1, 0] = 0.5
        corr[0, 2] = corr[2, 0] = 0.3
        corr[1, 2] = corr[2, 1] = 0.6
        
        # Create covariance from correlation and standard deviations
        sd = Vector([1.0, 2.0, 1.5, 0.5])
        cov = SpdMatrix.from_correlation(corr, sd)
        
        # Sample from multivariate normal
        mean = Vector.zero(p)
        n_samples = 10000
        samples = []
        
        for _ in range(n_samples):
            samples.append(rng.rmvn(mean, cov))
        
        samples = Matrix(samples)
        
        # Check sample statistics
        sample_mean = samples.colsums() / n_samples
        assert np.allclose(sample_mean, mean, atol=0.05)
        
        # Compute sample covariance
        centered = Matrix((n_samples, p))
        for i in range(n_samples):
            for j in range(p):
                centered[i, j] = samples[i, j] - sample_mean[j]
        
        sample_cov = (centered.T @ centered) / (n_samples - 1)
        # Check non-zero entries with relative tolerance
        for i in range(p):
            for j in range(p):
                if abs(cov[i, j]) > 1e-10:  # Non-zero entries
                    assert np.allclose(sample_cov[i, j], cov[i, j], rtol=0.1)
                else:  # Zero entries - check they're small
                    assert abs(sample_cov[i, j]) < 0.05
    
    def test_matrix_operations_workflow(self):
        """Test workflow involving various matrix operations."""
        seed_rng(456)
        
        # Generate random positive definite matrix
        n = 5
        A = Matrix((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = rng.rnorm()
        
        # Make it symmetric positive definite
        S = SpdMatrix(A.T @ A)
        S = S.add_to_diag(0.1)  # Ensure positive definite
        
        # Cholesky decomposition
        L = S.chol()
        assert np.allclose(L @ L.T, S)
        
        # Solve system S x = b
        b = Vector([rng.rnorm() for _ in range(n)])
        x = S.solve(b)
        assert np.allclose(S @ x, b)
        
        # Eigendecomposition
        eigvals, eigvecs = S.eig()
        assert np.all(eigvals > 0)  # All positive for SPD matrix
        
        # Quadratic form
        q = S.quad_form(x)
        assert q > 0  # Positive for SPD matrix and non-zero x
        
        # Condition number
        cond = S.condition_number()
        assert cond > 1
    
    def test_bayesian_update_workflow(self):
        """Test Bayesian updating workflow."""
        seed_rng(789)
        
        # Prior for mean of normal distribution
        prior_mean = 0.0
        prior_var = 10.0
        
        # Known variance
        sigma2 = 1.0
        
        # Generate data
        true_mu = 2.5
        n = 20
        data = Vector([rng.rnorm(true_mu, np.sqrt(sigma2)) for _ in range(n)])
        
        # Compute posterior
        data_mean = data.mean()
        
        # Posterior precision = prior precision + n/sigma2
        prior_precision = 1.0 / prior_var
        data_precision = n / sigma2
        post_precision = prior_precision + data_precision
        post_var = 1.0 / post_precision
        
        # Posterior mean is precision-weighted average
        post_mean = (prior_precision * prior_mean + data_precision * data_mean) / post_precision
        
        # Sample from posterior
        posterior_samples = Vector([rng.rnorm(post_mean, np.sqrt(post_var)) 
                                   for _ in range(1000)])
        
        # Check posterior samples
        assert abs(posterior_samples.mean() - post_mean) < 0.1
        assert abs(posterior_samples.var() - post_var) < 0.05
    
    def test_random_matrix_generation(self):
        """Test generating various random matrices."""
        seed_rng(999)
        
        # Random correlation matrix using vine method
        p = 4
        partial_corr = Matrix((p-1, p-1))
        
        # Generate partial correlations
        for i in range(p-1):
            for j in range(i+1):
                partial_corr[i, j] = 2 * rng.rbeta(1, 1) - 1
        
        # Convert to correlation matrix (simplified version)
        R = SpdMatrix.identity(p)
        
        # Random Wishart matrix
        df = 10
        scale = SpdMatrix.identity(3)
        W = SpdMatrix(rng.rwish(df, scale))
        
        # Should be positive definite
        assert W.is_pos_def()
        
        # Random orthogonal matrix via QR
        n = 4
        A = Matrix([[rng.rnorm() for _ in range(n)] for _ in range(n)])
        Q, R = np.linalg.qr(A)
        Q = Matrix(Q)
        
        # Check orthogonality
        QtQ = Q.T @ Q
        assert np.allclose(QtQ, Matrix.identity(n))
    
    def test_error_propagation(self):
        """Test that errors are properly handled."""
        # Non-positive definite matrix
        bad_spd = SpdMatrix([[1, 2], [2, 1]])
        
        with pytest.raises(ValueError):
            bad_spd.chol()
        
        # Singular matrix
        singular = Matrix([[1, 2], [2, 4]])
        with pytest.raises(np.linalg.LinAlgError):
            singular.inv()
        
        # Dimension mismatch
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([1, 2, 3])
        
        with pytest.raises(ValueError):
            m @ v  # 2x2 times 3x1 should fail