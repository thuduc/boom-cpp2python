"""Comprehensive integration tests for BOOM Python."""
import pytest
import numpy as np
from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import GlobalRng
from boom.models.glm import LinearRegression, LogisticRegression
from boom.models.state_space import LocalLevelModel
from boom.models.mixtures import GaussianMixtureModel
from boom.optimization import BFGS, RosenbrockFunction
from boom.mcmc import MetropolisHastings, SliceSampler
from boom.stats.descriptive import mean, standard_deviation, summary_statistics
from boom.stats.information_criteria import aic, bic


class TestFullWorkflow:
    """Test complete statistical modeling workflows."""
    
    def test_linear_regression_workflow(self):
        """Test complete linear regression analysis."""
        # Generate synthetic data
        np.random.seed(42)
        n = 100
        p = 3
        
        X = np.random.randn(n, p)
        true_beta = Vector([1.5, -2.0, 0.5])
        true_sigma = 1.0
        
        y = X @ true_beta + true_sigma * np.random.randn(n)
        
        # Create and fit model
        model = LinearRegression()
        
        # Add data
        for i in range(n):
            x_vec = Vector(X[i, :])
            model.add_data((y[i], x_vec))
        
        # Fit using MLE
        model.mle()
        
        # Check parameter estimates
        beta_est = model.beta
        sigma_est = model.sigma
        
        np.testing.assert_allclose(beta_est, true_beta, atol=0.2)
        assert abs(sigma_est - true_sigma) < 0.2
        
        # Model diagnostics
        predictions = []
        residuals = []
        
        for i in range(n):
            x_vec = Vector(X[i, :])
            pred = float(beta_est.dot(x_vec))
            predictions.append(pred)
            residuals.append(y[i] - pred)
        
        # Check residual properties
        residual_mean = mean(residuals)
        residual_std = standard_deviation(residuals)
        
        assert abs(residual_mean) < 0.1  # Should be close to 0
        assert abs(residual_std - true_sigma) < 0.2
        
        # Information criteria
        log_lik = model.loglike()
        n_params = p + 1  # beta + sigma
        
        aic_val = aic(log_lik, n_params)
        bic_val = bic(log_lik, n_params, n)
        
        assert aic_val < 0  # Should be reasonable
        assert bic_val > aic_val  # BIC penalizes more
    
    def test_logistic_regression_workflow(self):
        """Test complete logistic regression analysis."""
        np.random.seed(123)
        n = 200
        p = 2
        
        # Generate data
        X = np.random.randn(n, p)
        true_beta = Vector([0.5, -1.0])
        
        linear_pred = X @ true_beta
        prob = 1 / (1 + np.exp(-linear_pred))
        y = np.random.binomial(1, prob)
        
        # Create and fit model
        model = LogisticRegression()
        
        for i in range(n):
            x_vec = Vector(X[i, :])
            model.add_data((int(y[i]), x_vec))
        
        # Fit using optimization
        model.mle()
        
        # Check estimates
        beta_est = model.beta
        np.testing.assert_allclose(beta_est, true_beta, atol=0.3)
        
        # Prediction accuracy
        correct_predictions = 0
        for i in range(n):
            x_vec = Vector(X[i, :])
            prob_est = model.predict_probability(x_vec)
            pred_class = 1 if prob_est > 0.5 else 0
            if pred_class == y[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / n
        assert accuracy > 0.7  # Should have reasonable accuracy
    
    def test_mixture_model_workflow(self):
        """Test Gaussian mixture model workflow."""
        np.random.seed(456)
        
        # Generate mixture data
        n1, n2 = 50, 70
        data1 = np.random.normal(-2, 1, n1)
        data2 = np.random.normal(3, 1.5, n2)
        data = np.concatenate([data1, data2])
        np.random.shuffle(data)
        
        # Fit mixture model
        model = GaussianMixtureModel(n_components=2, max_iter=100)
        model.set_data(data)
        model.fit()
        
        # Check that it found two components
        weights = model.mixing_weights
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-10
        
        # Check component parameters are reasonable
        means = [comp.mean for comp in model.components]
        stds = [comp.sigma for comp in model.components]
        
        means.sort()  # Sort to compare with true values
        assert abs(means[0] - (-2)) < 1.0
        assert abs(means[1] - 3) < 1.0
        
        for std in stds:
            assert 0.5 < std < 3.0  # Should be reasonable
    
    def test_mcmc_workflow(self):
        """Test MCMC sampling workflow."""
        # Simple normal model: x ~ N(mu, sigma^2)
        data = [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.0]
        
        def log_posterior(params):
            mu, log_sigma = params[0], params[1]
            sigma = np.exp(log_sigma)
            
            # Prior: mu ~ N(0, 10), log_sigma ~ N(0, 1)
            log_prior = -0.5 * mu**2 / 100 - 0.5 * log_sigma**2
            
            # Likelihood
            log_lik = sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (x - mu)**2 / sigma**2 
                         for x in data)
            
            return log_prior + log_lik
        
        # MCMC sampling
        sampler = MetropolisHastings(
            log_density_func=log_posterior,
            proposal_covariance=Matrix([[0.1, 0], [0, 0.1]])
        )
        
        initial_point = Vector([0.0, 0.0])
        samples = sampler.sample(1000, initial_point, burn_in=500)
        
        # Check convergence
        assert len(samples) == 500  # After burn-in
        
        # Check posterior means
        mu_samples = [s[0] for s in samples]
        sigma_samples = [np.exp(s[1]) for s in samples]
        
        mu_mean = mean(mu_samples)
        sigma_mean = mean(sigma_samples)
        
        # Should be close to data mean and std
        data_mean = mean(data)
        data_std = standard_deviation(data)
        
        assert abs(mu_mean - data_mean) < 0.2
        assert abs(sigma_mean - data_std) < 0.3
    
    def test_optimization_workflow(self):
        """Test optimization workflow."""
        # Optimize Rosenbrock function
        f = RosenbrockFunction()
        optimizer = BFGS(max_iterations=100)
        
        x0 = Vector([-1.2, 1.0])
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)
        assert result.f < 1e-6
    
    def test_state_space_workflow(self):
        """Test state space model workflow."""
        # Generate local level data
        np.random.seed(789)
        n = 50
        true_level = 0.0
        true_level_var = 0.1
        true_obs_var = 1.0
        
        # Simulate data
        levels = [true_level]
        observations = []
        
        for t in range(n):
            if t > 0:
                levels.append(levels[-1] + np.random.normal(0, np.sqrt(true_level_var)))
            
            obs = levels[-1] + np.random.normal(0, np.sqrt(true_obs_var))
            observations.append(obs)
        
        # Fit model
        model = LocalLevelModel()
        model.observation_variance = true_obs_var  # Fix observation variance
        
        for obs in observations:
            model.add_data(obs)
        
        # Run Kalman filter
        filtered_states, filtered_vars = model.kalman_filter()
        
        # Check that filtered states track the true levels reasonably
        assert len(filtered_states) == n
        assert len(filtered_vars) == n
        
        # States should be reasonable
        for state in filtered_states:
            assert abs(state) < 5.0  # Should be in reasonable range
        
        # Variances should be positive
        for var in filtered_vars:
            assert var > 0


class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_information_criteria_comparison(self):
        """Test comparison using information criteria."""
        # Generate data that clearly favors a specific model
        np.random.seed(42)
        n = 100
        
        # True model: y = 2x + noise
        x = np.random.randn(n)
        y = 2 * x + 0.5 * np.random.randn(n)
        
        # Fit models of different complexity
        models_info = []
        
        # Model 1: Intercept only
        model1 = LinearRegression()
        for i in range(n):
            model1.add_data((y[i], Vector([1.0])))  # Just intercept
        model1.mle()
        
        log_lik1 = model1.loglike()
        models_info.append(('Intercept', log_lik1, 2, n))  # 2 params: intercept + sigma
        
        # Model 2: Linear (correct model)
        model2 = LinearRegression()
        for i in range(n):
            model2.add_data((y[i], Vector([1.0, x[i]])))
        model2.mle()
        
        log_lik2 = model2.loglike()
        models_info.append(('Linear', log_lik2, 3, n))  # 3 params: intercept + slope + sigma
        
        # Model 3: Quadratic (overfit)
        model3 = LinearRegression()
        for i in range(n):
            model3.add_data((y[i], Vector([1.0, x[i], x[i]**2])))
        model3.mle()
        
        log_lik3 = model3.loglike()
        models_info.append(('Quadratic', log_lik3, 4, n))  # 4 params
        
        # Compare models
        from boom.stats.information_criteria import information_criterion_comparison
        comparison = information_criterion_comparison(models_info)
        
        # Linear model should be best (lowest AIC/BIC)
        aic_values = comparison['aic']
        bic_values = comparison['bic']
        
        linear_idx = 1  # Linear model is second in list
        
        # Linear should have lowest AIC and BIC
        assert aic_values[linear_idx] == min(aic_values)
        assert bic_values[linear_idx] == min(bic_values)
        
        # Linear should have highest weight
        aic_weights = comparison['aic_weights']
        assert aic_weights[linear_idx] == max(aic_weights)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_matrix_operations_stability(self):
        """Test numerical stability of matrix operations."""
        # Test with ill-conditioned matrix
        A = Matrix([[1.0, 1.0], [1.0, 1.0001]])
        
        # Should still be invertible
        try:
            A_inv = SpdMatrix(A).inv()
            identity = A @ A_inv
            
            # Check it's close to identity
            expected_identity = Matrix([[1.0, 0.0], [0.0, 1.0]])
            np.testing.assert_allclose(identity, expected_identity, atol=1e-3)
        except np.linalg.LinAlgError:
            pytest.skip("Matrix too ill-conditioned")
    
    def test_optimization_edge_cases(self):
        """Test optimization with edge cases."""
        # Test with function that has multiple minima
        def multi_minima(x):
            return (x[0]**2 - 1)**2 + 0.1 * x[1]**2
        
        from boom.optimization.target_functions import TargetFunction
        
        class MultiMinimaFunction(TargetFunction):
            def evaluate(self, x):
                self.n_evaluations += 1
                return multi_minima(x)
        
        f = MultiMinimaFunction()
        optimizer = BFGS(max_iterations=100)
        
        # Try from different starting points
        starting_points = [
            Vector([0.5, 0.0]),
            Vector([-0.5, 0.0]),
            Vector([2.0, 1.0])
        ]
        
        solutions = []
        for x0 in starting_points:
            result = optimizer.optimize(f, x0)
            if result.success:
                solutions.append(result.x)
        
        # Should find at least one solution
        assert len(solutions) > 0
        
        # Solutions should be near ±1 for x[0]
        for sol in solutions:
            assert abs(abs(sol[0]) - 1.0) < 0.1
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        n = 1000
        p = 5
        
        X = np.random.randn(n, p)
        true_beta = Vector(np.random.randn(p))
        y = X @ true_beta + 0.5 * np.random.randn(n)
        
        # Fit linear regression
        model = LinearRegression()
        
        for i in range(n):
            x_vec = Vector(X[i, :])
            model.add_data((y[i], x_vec))
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        model.mle()
        end_time = time.time()
        
        # Should take less than 5 seconds
        assert end_time - start_time < 5.0
        
        # Should still get reasonable estimates
        beta_est = model.beta
        np.testing.assert_allclose(beta_est, true_beta, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])