"""Tests for mixture models."""
import numpy as np
from boom.models.mixtures import (
    FiniteMixtureModel, GaussianMixtureModel, DirichletProcessMixtureModel
)
from boom.models.mixtures.finite_mixture import GaussianComponent
from boom.linalg import Vector, Matrix
from boom.distributions import rng


class TestGaussianComponent:
    """Test Gaussian mixture component."""
    
    def test_construction(self):
        """Test component construction."""
        comp = GaussianComponent(mean=2.0, sigma=0.5, weight=0.3)
        assert comp.mean == 2.0
        assert comp.sigma == 0.5
        assert comp.weight == 0.3
    
    def test_pdf_logpdf(self):
        """Test PDF and log PDF."""
        comp = GaussianComponent(mean=0.0, sigma=1.0)
        
        # At mean, should be maximum
        pdf_at_mean = comp.pdf(0.0)
        logpdf_at_mean = comp.logpdf(0.0)
        
        assert pdf_at_mean > 0
        assert np.isclose(np.log(pdf_at_mean), logpdf_at_mean)
        
        # Test known value
        # N(0,1) at x=0 has pdf = 1/sqrt(2π) ≈ 0.3989
        expected_pdf = 1.0 / np.sqrt(2 * np.pi)
        assert np.isclose(pdf_at_mean, expected_pdf, rtol=1e-6)
    
    def test_fitting(self):
        """Test component fitting."""
        comp = GaussianComponent()
        
        # Simple data
        data = Vector([1.0, 2.0, 3.0])
        weights = Vector([1.0, 1.0, 1.0])
        
        comp.fit(data, weights)
        
        # Should fit to sample mean and std
        assert np.isclose(comp.mean, 2.0)  # Mean of [1,2,3]
        expected_std = np.sqrt(2.0/3.0)  # Sample std
        assert np.isclose(comp.sigma, expected_std, rtol=1e-6)
    
    def test_weighted_fitting(self):
        """Test weighted fitting."""
        comp = GaussianComponent()
        
        data = Vector([1.0, 3.0])
        weights = Vector([3.0, 1.0])  # Weight first point more
        
        comp.fit(data, weights)
        
        # Weighted mean: (3*1 + 1*3) / 4 = 1.5
        assert np.isclose(comp.mean, 1.5)


class TestFiniteMixtureModel:
    """Test finite mixture model."""
    
    def test_construction(self):
        """Test model construction."""
        model = FiniteMixtureModel(n_components=2)
        assert model.n_components == 2
        assert len(model.components) == 2
        assert len(model.mixing_weights) == 2
        
        # Should start with equal weights
        np.testing.assert_allclose(model.mixing_weights, [0.5, 0.5])
    
    def test_adding_data(self):
        """Test data management."""
        model = FiniteMixtureModel(n_components=2)
        
        # Add data points
        model.add_data(1.0)
        model.add_data(2.0)
        model.add_data(3.0)
        
        data = model.get_data()
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0])
    
    def test_pdf_evaluation(self):
        """Test PDF evaluation."""
        model = FiniteMixtureModel(n_components=2)
        
        # Set specific parameters
        model.components[0].mean = 0.0
        model.components[0].sigma = 1.0
        model.components[1].mean = 2.0
        model.components[1].sigma = 1.0
        model.mixing_weights = Vector([0.6, 0.4])
        
        # PDF should be mixture of two Gaussians
        pdf_val = model.pdf(1.0)  # Point between the means
        assert pdf_val > 0
        
        # Log PDF should be finite
        logpdf_val = model.logpdf(1.0)
        assert np.isfinite(logpdf_val)
    
    def test_em_fitting(self):
        """Test EM algorithm fitting."""
        rng.seed(123)
        model = FiniteMixtureModel(n_components=2)
        
        # Generate bimodal data
        data = []
        for _ in range(50):
            if rng.runif() < 0.6:
                data.append(rng.rnorm(-1.0, 0.5))  # First mode
            else:
                data.append(rng.rnorm(1.0, 0.5))   # Second mode
        
        model.set_data(data)
        
        # Fit using EM
        model.fit(max_iter=50)
        
        # Check that components found the modes
        means = [comp.mean for comp in model.components]
        means.sort()
        
        # Should be close to -1 and 1
        assert means[0] < 0  # First component should be negative
        assert means[1] > 0  # Second component should be positive
        
        # Mixing weights should be reasonable
        assert all(0.1 < w < 0.9 for w in model.mixing_weights)
    
    def test_bic_aic(self):
        """Test model selection criteria."""
        model = FiniteMixtureModel(n_components=2)
        model.set_data([1.0, 2.0, 3.0])
        
        bic = model.bic()
        aic = model.aic()
        
        assert np.isfinite(bic)
        assert np.isfinite(aic)
        assert bic > aic  # BIC penalizes complexity more


class TestGaussianMixtureModel:
    """Test multivariate Gaussian mixture model."""
    
    def test_construction(self):
        """Test model construction."""
        model = GaussianMixtureModel(n_components=2, n_features=3)
        assert model.n_components == 2
        assert model.n_features == 3
        assert len(model.components) == 2
        
        # Each component should have 3D mean and covariance
        for comp in model.components:
            assert comp.dim == 3
            assert len(comp.mean) == 3
            assert comp.covariance.nrow() == 3
    
    def test_data_management(self):
        """Test multivariate data management."""
        model = GaussianMixtureModel(n_components=2, n_features=2)
        
        # Add 2D data points
        model.add_data([1.0, 2.0])
        model.add_data([3.0, 4.0])
        
        data_matrix = model.get_data_matrix()
        assert data_matrix.shape == (2, 2)
        np.testing.assert_array_equal(data_matrix[0, :], [1.0, 2.0])
        np.testing.assert_array_equal(data_matrix[1, :], [3.0, 4.0])
    
    def test_multivariate_pdf(self):
        """Test multivariate PDF evaluation."""
        model = GaussianMixtureModel(n_components=1, n_features=2)
        
        # Set specific parameters for the single component
        comp = model.components[0]
        comp.mean = Vector([0.0, 0.0])
        comp.set_covariance(Matrix([[1.0, 0.0], [0.0, 1.0]]))  # Identity covariance
        
        # PDF at mean should be maximum
        pdf_at_mean = model.pdf([0.0, 0.0])
        pdf_away = model.pdf([1.0, 1.0])
        
        assert pdf_at_mean > pdf_away
        assert pdf_at_mean > 0
    
    def test_multivariate_em(self):
        """Test EM fitting for multivariate case."""
        rng.seed(456)
        model = GaussianMixtureModel(n_components=2, n_features=2)
        
        # Generate 2D bimodal data
        data = []
        for _ in range(100):
            if rng.runif() < 0.5:
                # First cluster around (0, 0)
                x = rng.rnorm(0.0, 0.5)
                y = rng.rnorm(0.0, 0.5)
                data.append([x, y])
            else:
                # Second cluster around (2, 2)
                x = rng.rnorm(2.0, 0.5)
                y = rng.rnorm(2.0, 0.5)
                data.append([x, y])
        
        model.set_data(data)
        model.fit(max_iter=30)
        
        # Check that components found the clusters
        means = [comp.mean for comp in model.components]
        
        # Should have one mean near (0,0) and one near (2,2)
        distances_to_origin = [np.linalg.norm(mean) for mean in means]
        distances_to_target = [np.linalg.norm(mean - Vector([2.0, 2.0])) for mean in means]
        
        # At least one should be close to each target
        assert min(distances_to_origin) < 1.0
        assert min(distances_to_target) < 1.0
    
    def test_simulation(self):
        """Test simulation from fitted model."""
        model = GaussianMixtureModel(n_components=1, n_features=2)
        
        # Set parameters
        comp = model.components[0]
        comp.mean = Vector([1.0, -1.0])
        comp.set_covariance(Matrix([[0.5, 0.0], [0.0, 0.5]]))
        
        # Simulate samples
        rng.seed(789)
        samples = model.simulate(10, rng)
        
        assert samples.shape == (10, 2)
        
        # Samples should be roughly centered around the mean
        sample_mean = Vector([np.mean(samples[:, 0]), np.mean(samples[:, 1])])
        distance = np.linalg.norm(sample_mean - comp.mean)
        assert distance < 1.0  # Should be reasonably close with 10 samples


class TestDirichletProcessMixtureModel:
    """Test Dirichlet Process mixture model."""
    
    def test_construction(self):
        """Test model construction."""
        model = DirichletProcessMixtureModel(max_components=10, alpha=1.0)
        assert model.max_components == 10
        assert model.alpha == 1.0
        assert len(model.components) == 10
        
        # Should have stick-breaking weights
        assert len(model.mixing_weights) == 10
        assert abs(np.sum(model.mixing_weights) - 1.0) < 1e-10
    
    def test_effective_components(self):
        """Test effective number of components."""
        model = DirichletProcessMixtureModel(max_components=10, alpha=1.0)
        
        # Initially should have some effective components
        n_eff = model.effective_n_components()
        assert 1 <= n_eff <= 10
    
    def test_alpha_parameter(self):
        """Test alpha parameter setting."""
        model = DirichletProcessMixtureModel(max_components=5, alpha=0.5)
        assert model.alpha == 0.5
        
        # Change alpha
        model.alpha = 2.0
        assert model.alpha == 2.0
    
    def test_component_assignment(self):
        """Test component assignment sampling."""
        model = DirichletProcessMixtureModel(max_components=5, alpha=1.0)
        model.add_data(1.0)
        
        # Should be able to assign component
        rng.seed(111)
        assignment = model.sample_component_assignment(1.0, rng)
        assert 0 <= assignment < 5
    
    def test_simple_fitting(self):
        """Test simple Gibbs fitting."""
        rng.seed(222)
        model = DirichletProcessMixtureModel(max_components=5, alpha=1.0)
        
        # Add simple unimodal data
        data = [rng.rnorm(0.0, 1.0) for _ in range(20)]
        model.set_data(data)
        
        # Short fitting run
        model.fit(n_iter=10, burn_in=5)
        
        # Should still be functional
        loglike = model.loglike()
        assert np.isfinite(loglike)
        
        # Should have some significant components
        significant = model.get_significant_components()
        assert len(significant) >= 1


def run_mixture_tests():
    """Run all mixture model tests."""
    print("Testing GaussianComponent...")
    test_gc = TestGaussianComponent()
    test_gc.test_construction()
    test_gc.test_pdf_logpdf()
    test_gc.test_fitting()
    test_gc.test_weighted_fitting()
    print("GaussianComponent tests passed!")
    
    print("Testing FiniteMixtureModel...")
    test_fm = TestFiniteMixtureModel()
    test_fm.test_construction()
    test_fm.test_adding_data()
    test_fm.test_pdf_evaluation()
    test_fm.test_em_fitting()
    test_fm.test_bic_aic()
    print("FiniteMixtureModel tests passed!")
    
    print("Testing GaussianMixtureModel...")
    test_gmm = TestGaussianMixtureModel()
    test_gmm.test_construction()
    test_gmm.test_data_management()
    test_gmm.test_multivariate_pdf()
    test_gmm.test_multivariate_em()
    test_gmm.test_simulation()
    print("GaussianMixtureModel tests passed!")
    
    print("Testing DirichletProcessMixtureModel...")
    test_dpm = TestDirichletProcessMixtureModel()
    test_dpm.test_construction()
    test_dpm.test_effective_components()
    test_dpm.test_alpha_parameter()
    test_dpm.test_component_assignment()
    test_dpm.test_simple_fitting()
    print("DirichletProcessMixtureModel tests passed!")
    
    print("All mixture model tests passed successfully!")


if __name__ == "__main__":
    run_mixture_tests()