"""
Tests for statistics utilities.

This module tests descriptive statistics, hypothesis tests, and 
model selection utilities.
"""

import pytest
import numpy as np
import sys
import os

# Add the impl-python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from boom.stats import (
    DescriptiveStats, compute_summary_stats,
    TTest, ChiSquareTest, KolmogorovSmirnovTest,
    AIC, BIC, compute_ic,
    ModelComparison, CrossValidator,
    RegressionDiagnostics, residual_analysis
)
from boom.linalg import Vector, Matrix


class TestDescriptiveStats:
    """Test descriptive statistics functionality."""
    
    def test_univariate_stats(self):
        """Test univariate descriptive statistics."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        desc = DescriptiveStats(data)
        stats = desc.compute()
        
        assert stats['n'] == 10
        assert stats['mean'] == pytest.approx(5.5)
        assert stats['median'] == pytest.approx(5.5)
        assert stats['min'] == 1
        assert stats['max'] == 10
        assert stats['std'] > 0
        
    def test_multivariate_stats(self):
        """Test multivariate descriptive statistics."""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        desc = DescriptiveStats(data)
        stats = desc.compute()
        
        assert stats['n_observations'] == 4
        assert stats['n_variables'] == 2
        assert len(stats['means']) == 2
        assert stats['covariance'].shape == (2, 2)
        assert stats['correlation'].shape == (2, 2)
        
    def test_stats_with_missing_values(self):
        """Test statistics with missing values."""
        data = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        desc = DescriptiveStats(data)
        stats = desc.compute()
        
        assert stats['n'] == 10
        assert stats['n_missing'] == 2
        assert stats['n_valid'] == 8
        assert not np.isnan(stats['mean'])
        
    def test_summary_generation(self):
        """Test summary report generation."""
        data = np.random.normal(0, 1, 100)
        desc = DescriptiveStats(data)
        summary = desc.summary()
        
        assert isinstance(summary, str)
        assert 'Mean:' in summary
        assert 'Std Dev:' in summary
        assert 'Skewness:' in summary
        
    def test_convenience_function(self):
        """Test convenience function for computing stats."""
        data = [1, 2, 3, 4, 5]
        stats = compute_summary_stats(data)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['n'] == 5


class TestHypothesisTesting:
    """Test hypothesis testing functionality."""
    
    def test_one_sample_t_test(self):
        """Test one-sample t-test."""
        # Generate data with known mean
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, 30)
        
        ttest = TTest()
        result = ttest.one_sample(data, mu0=5.0, alternative='two-sided')
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'confidence_interval' in result
        assert result['sample_size'] == 30
        assert result['hypothesized_mean'] == 5.0
        
        # Should not reject null hypothesis (mean = 5.0)
        assert result['p_value'] > 0.01  # Not significant at 1% level
        
    def test_two_sample_t_test(self):
        """Test two-sample t-test."""
        np.random.seed(42)
        data1 = np.random.normal(5.0, 1.0, 25)
        data2 = np.random.normal(5.2, 1.0, 25)  # Slightly different mean
        
        ttest = TTest()
        result = ttest.two_sample(data1, data2, equal_var=True)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'mean_diff' in result
        assert result['sample_size1'] == 25
        assert result['sample_size2'] == 25
        
    def test_paired_t_test(self):
        """Test paired t-test."""
        np.random.seed(42)
        before = np.random.normal(5.0, 1.0, 20)
        after = before + np.random.normal(0.5, 0.3, 20)  # Small improvement
        
        ttest = TTest()
        result = ttest.paired(before, after)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert isinstance(result['significant'], bool)
        
    def test_chi_square_goodness_of_fit(self):
        """Test chi-square goodness of fit test."""
        # Test against uniform distribution
        observed = [18, 22, 16, 14, 12, 18]  # Roughly uniform
        expected = [20, 20, 20, 20, 20, 20]  # Exactly uniform
        
        chi_test = ChiSquareTest()
        result = chi_test.goodness_of_fit(observed, expected)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'degrees_freedom' in result
        assert result['degrees_freedom'] == 5
        
    def test_chi_square_independence(self):
        """Test chi-square test of independence."""
        # 2x2 contingency table
        table = [[10, 20], [15, 25]]
        
        chi_test = ChiSquareTest()
        result = chi_test.independence(table)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'degrees_freedom' in result
        assert result['degrees_freedom'] == 1  # (2-1)*(2-1)
        
    def test_kolmogorov_smirnov_one_sample(self):
        """Test one-sample KS test."""
        # Test against normal distribution
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)
        
        def standard_normal_cdf(x):
            from scipy import stats
            return stats.norm.cdf(x)
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.one_sample(data, standard_normal_cdf)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert result['sample_size'] == 50
        
        # Should not reject null (data is from normal distribution)
        assert result['p_value'] > 0.01
        
    def test_kolmogorov_smirnov_two_sample(self):
        """Test two-sample KS test."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 30)
        data2 = np.random.normal(0, 1, 35)  # Same distribution
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.two_sample(data1, data2)
        
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert result['sample_size1'] == 30
        assert result['sample_size2'] == 35
        
        # Should not reject null (same distribution)
        assert result['p_value'] > 0.01


class TestInformationCriteria:
    """Test information criteria calculations."""
    
    def test_aic_calculation(self):
        """Test AIC calculation."""
        log_likelihood = -100.0
        n_parameters = 5
        
        aic = AIC(log_likelihood, n_parameters)
        expected_aic = -2 * (-100.0) + 2 * 5
        
        assert aic == expected_aic
        
    def test_bic_calculation(self):
        """Test BIC calculation."""
        log_likelihood = -100.0
        n_parameters = 5
        n_observations = 50
        
        bic = BIC(log_likelihood, n_parameters, n_observations)
        expected_bic = -2 * (-100.0) + np.log(50) * 5
        
        assert bic == expected_bic
        
    def test_compute_ic_function(self):
        """Test compute_ic convenience function."""
        class MockModel:
            def log_likelihood(self):
                return -100.0
                
            def n_parameters(self):
                return 3
                
            def n_observations(self):
                return 100
        
        model = MockModel()
        results = compute_ic(model, ['aic', 'bic'])
        
        assert 'AIC' in results
        assert 'BIC' in results
        assert results['AIC'] == AIC(-100.0, 3)
        assert results['BIC'] == BIC(-100.0, 3, 100)


class TestModelComparison:
    """Test model comparison utilities."""
    
    def test_model_comparison_creation(self):
        """Test creating model comparison."""
        class MockModel:
            def __init__(self, ll, n_params):
                self._ll = ll
                self._n_params = n_params
                
            def log_likelihood(self):
                return self._ll
                
            def n_parameters(self):
                return self._n_params
                
            def n_observations(self):
                return 100
        
        models = [MockModel(-50, 2), MockModel(-45, 4)]
        names = ['Simple', 'Complex']
        
        comparison = ModelComparison(models, names)
        
        assert len(comparison._models) == 2
        assert len(comparison._names) == 2
        
    def test_likelihood_comparison(self):
        """Test likelihood comparison."""
        class MockModel:
            def __init__(self, ll):
                self._ll = ll
                
            def log_likelihood(self):
                return self._ll
        
        models = [MockModel(-50), MockModel(-45)]
        comparison = ModelComparison(models)
        
        ll_results = comparison.compare_likelihood()
        
        assert len(ll_results) == 2
        assert 'Model_1' in ll_results
        assert 'Model_2' in ll_results
        
    def test_summary_table(self):
        """Test summary table generation."""
        class MockModel:
            def __init__(self, ll, n_params):
                self._ll = ll
                self._n_params = n_params
                
            def log_likelihood(self):
                return self._ll
                
            def n_parameters(self):
                return self._n_params
                
            def n_observations(self):
                return 100
        
        models = [MockModel(-50, 2)]
        comparison = ModelComparison(models)
        
        summary = comparison.summary_table()
        
        assert isinstance(summary, str)
        assert 'Model Comparison' in summary


class TestRegressionDiagnostics:
    """Test regression diagnostics."""
    
    def test_diagnostics_creation(self):
        """Test creating regression diagnostics."""
        np.random.seed(42)
        n = 50
        
        # Generate synthetic regression data
        x = np.random.normal(0, 1, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        
        # Simple linear regression to get fitted values and residuals
        fitted_values = 2 + 3 * x  # True relationship
        residuals = y - fitted_values
        
        diagnostics = RegressionDiagnostics(
            fitted_values=Vector(fitted_values),
            residuals=Vector(residuals)
        )
        
        results = diagnostics.compute_all_diagnostics()
        
        assert 'residual_mean' in results
        assert 'residual_std' in results
        assert 'shapiro_wilk' in results or 'jarque_bera' in results
        
    def test_normality_tests(self):
        """Test normality tests in diagnostics."""
        # Generate normal residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        fitted_values = np.random.normal(0, 1, 100)
        
        diagnostics = RegressionDiagnostics(
            fitted_values=Vector(fitted_values),
            residuals=Vector(residuals)
        )
        
        results = diagnostics.compute_all_diagnostics()
        
        # Should pass normality tests
        if 'shapiro_wilk' in results and 'error' not in results['shapiro_wilk']:
            assert results['shapiro_wilk']['normal']
            
    def test_outlier_detection(self):
        """Test outlier detection."""
        np.random.seed(42)
        n = 50
        
        # Generate data with outliers
        residuals = np.random.normal(0, 1, n)
        residuals[0] = 5.0  # Add outlier
        
        fitted_values = np.random.normal(0, 1, n)
        
        diagnostics = RegressionDiagnostics(
            fitted_values=Vector(fitted_values),
            residuals=Vector(residuals)
        )
        
        results = diagnostics.compute_all_diagnostics()
        
        # Should detect the outlier
        if 'standardized_outliers' in results:
            assert results['standardized_outliers']['count'] > 0
            
    def test_summary_report(self):
        """Test diagnostic summary report."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 30)
        fitted_values = np.random.normal(0, 1, 30)
        
        diagnostics = RegressionDiagnostics(
            fitted_values=Vector(fitted_values),
            residuals=Vector(residuals)
        )
        
        summary = diagnostics.summary_report()
        
        assert isinstance(summary, str)
        assert 'Regression Diagnostics' in summary
        assert 'Residual Statistics' in summary
        
    def test_residual_analysis_function(self):
        """Test convenience function for residual analysis."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 40)
        fitted_values = np.random.normal(0, 1, 40)
        
        diagnostics = residual_analysis(
            fitted_values=Vector(fitted_values),
            residuals=Vector(residuals)
        )
        
        assert isinstance(diagnostics, RegressionDiagnostics)
        
        # Should have computed diagnostics
        results = diagnostics.get_all_stats()
        assert len(results) > 0


class TestStatsIntegration:
    """Integration tests for statistics functionality."""
    
    def test_complete_statistical_analysis(self):
        """Test complete statistical analysis workflow."""
        # Generate two samples from different distributions
        np.random.seed(42)
        sample1 = np.random.normal(5.0, 1.0, 50)
        sample2 = np.random.normal(5.5, 1.2, 45)
        
        # Descriptive statistics
        desc1 = DescriptiveStats(sample1)
        desc2 = DescriptiveStats(sample2)
        
        stats1 = desc1.compute()
        stats2 = desc2.compute()
        
        # Hypothesis test
        ttest = TTest()
        test_result = ttest.two_sample(sample1, sample2)
        
        # Check that all components work together
        assert stats1['mean'] != stats2['mean']
        assert 'p_value' in test_result
        
        # Generate summary
        summary1 = desc1.summary()
        summary2 = desc2.summary()
        
        assert isinstance(summary1, str)
        assert isinstance(summary2, str)
        
    def test_data_type_compatibility(self):
        """Test compatibility with different data types."""
        # Test with different input types
        data_list = [1, 2, 3, 4, 5]
        data_array = np.array([1, 2, 3, 4, 5])
        data_vector = Vector(np.array([1, 2, 3, 4, 5]))
        
        # All should produce same results
        desc_list = DescriptiveStats(data_list)
        desc_array = DescriptiveStats(data_array)
        desc_vector = DescriptiveStats(data_vector)
        
        stats_list = desc_list.compute()
        stats_array = desc_array.compute()
        stats_vector = desc_vector.compute()
        
        assert stats_list['mean'] == stats_array['mean'] == stats_vector['mean']
        assert stats_list['std'] == stats_array['std'] == stats_vector['std']


if __name__ == '__main__':
    pytest.main([__file__])