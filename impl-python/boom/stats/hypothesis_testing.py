"""
Hypothesis testing utilities.

This module provides various statistical hypothesis tests
for data analysis and model validation.
"""

import numpy as np
import scipy.stats as stats
from typing import Optional, Union, Tuple, Dict, Any, List
from ..linalg import Vector


class HypothesisTest:
    """Base class for hypothesis tests."""
    
    def __init__(self, name: str):
        """Initialize hypothesis test."""
        self._name = name
        self._test_statistic = None
        self._p_value = None
        self._critical_value = None
        self._confidence_level = 0.95
        self._performed = False
    
    @property
    def name(self) -> str:
        """Get test name."""
        return self._name
    
    @property
    def test_statistic(self) -> Optional[float]:
        """Get test statistic."""
        return self._test_statistic
    
    @property
    def p_value(self) -> Optional[float]:
        """Get p-value."""
        return self._p_value
    
    @property
    def critical_value(self) -> Optional[float]:
        """Get critical value."""
        return self._critical_value
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        if self._p_value is None:
            return False
        return bool(self._p_value < alpha)
    
    def summary(self) -> str:
        """Generate test summary."""
        if not self._performed:
            return f"{self._name}: Test not performed"
        
        return f"""
{self._name} Test Results
{'=' * (len(self._name) + 13)}
Test Statistic: {self._test_statistic:.6f}
P-value: {self._p_value:.6f}
Critical Value: {self._critical_value:.6f}
Significant at α=0.05: {self.is_significant()}
"""


class TTest(HypothesisTest):
    """
    T-test for means.
    
    Supports one-sample, two-sample (independent), and paired t-tests.
    """
    
    def __init__(self):
        """Initialize t-test."""
        super().__init__("T-Test")
    
    def one_sample(self, data: Union[Vector, np.ndarray, List],
                   mu0: float, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        One-sample t-test.
        
        Tests H0: μ = μ0 vs H1: μ ≠ μ0 (or μ > μ0, μ < μ0)
        
        Args:
            data: Sample data
            mu0: Hypothesized population mean
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data, Vector):
            sample = data.to_numpy()
        else:
            sample = np.asarray(data)
        
        # Remove missing values
        sample = sample[~np.isnan(sample)]
        
        if len(sample) == 0:
            raise ValueError("No valid data points")
        
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        if sample_std == 0:
            raise ValueError("Sample standard deviation is zero")
        
        # Compute t-statistic
        self._test_statistic = (sample_mean - mu0) / (sample_std / np.sqrt(n))
        
        # Degrees of freedom
        df = n - 1
        
        # Compute p-value and critical value
        if alternative == 'two-sided':
            self._p_value = 2 * (1 - stats.t.cdf(abs(self._test_statistic), df))
            self._critical_value = stats.t.ppf(1 - 0.025, df)  # Two-tailed
        elif alternative == 'greater':
            self._p_value = 1 - stats.t.cdf(self._test_statistic, df)
            self._critical_value = stats.t.ppf(1 - 0.05, df)
        elif alternative == 'less':
            self._p_value = stats.t.cdf(self._test_statistic, df)
            self._critical_value = stats.t.ppf(0.05, df)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
        
        self._performed = True
        
        # Confidence interval for mean
        margin_error = self._critical_value * (sample_std / np.sqrt(n))
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'degrees_freedom': df,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'sample_size': n,
            'hypothesized_mean': mu0,
            'alternative': alternative,
            'confidence_interval': (ci_lower, ci_upper),
            'significant': self.is_significant()
        }
    
    def two_sample(self, data1: Union[Vector, np.ndarray, List],
                   data2: Union[Vector, np.ndarray, List],
                   equal_var: bool = True,
                   alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Two-sample t-test.
        
        Tests H0: μ1 = μ2 vs H1: μ1 ≠ μ2 (or μ1 > μ2, μ1 < μ2)
        
        Args:
            data1: First sample
            data2: Second sample
            equal_var: Assume equal variances (Welch's t-test if False)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data1, Vector):
            sample1 = data1.to_numpy()
        else:
            sample1 = np.asarray(data1)
        
        if isinstance(data2, Vector):
            sample2 = data2.to_numpy()
        else:
            sample2 = np.asarray(data2)
        
        # Remove missing values
        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]
        
        if len(sample1) == 0 or len(sample2) == 0:
            raise ValueError("No valid data points in one or both samples")
        
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        if equal_var:
            # Pooled t-test
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(var1/n1 + var2/n2)
            # Welch-Satterthwaite equation for degrees of freedom
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        self._test_statistic = (mean1 - mean2) / se
        
        # Compute p-value and critical value
        if alternative == 'two-sided':
            self._p_value = 2 * (1 - stats.t.cdf(abs(self._test_statistic), df))
            self._critical_value = stats.t.ppf(1 - 0.025, df)
        elif alternative == 'greater':
            self._p_value = 1 - stats.t.cdf(self._test_statistic, df)
            self._critical_value = stats.t.ppf(1 - 0.05, df)
        elif alternative == 'less':
            self._p_value = stats.t.cdf(self._test_statistic, df)
            self._critical_value = stats.t.ppf(0.05, df)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
        
        self._performed = True
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'degrees_freedom': df,
            'mean1': mean1,
            'mean2': mean2,
            'mean_diff': mean1 - mean2,
            'std_error': se,
            'sample_size1': n1,
            'sample_size2': n2,
            'equal_variances': equal_var,
            'alternative': alternative,
            'significant': self.is_significant()
        }
    
    def paired(self, data1: Union[Vector, np.ndarray, List],
               data2: Union[Vector, np.ndarray, List],
               alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Paired t-test.
        
        Tests H0: μd = 0 vs H1: μd ≠ 0 where d = data1 - data2
        
        Args:
            data1: First paired sample
            data2: Second paired sample
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data1, Vector):
            sample1 = data1.to_numpy()
        else:
            sample1 = np.asarray(data1)
        
        if isinstance(data2, Vector):
            sample2 = data2.to_numpy()
        else:
            sample2 = np.asarray(data2)
        
        if len(sample1) != len(sample2):
            raise ValueError("Paired samples must have same length")
        
        # Compute differences
        differences = sample1 - sample2
        
        # Remove missing differences
        differences = differences[~np.isnan(differences)]
        
        if len(differences) == 0:
            raise ValueError("No valid paired differences")
        
        # Perform one-sample t-test on differences
        return self.one_sample(differences, mu0=0.0, alternative=alternative)


class ChiSquareTest(HypothesisTest):
    """
    Chi-square tests.
    
    Supports goodness-of-fit and independence tests.
    """
    
    def __init__(self):
        """Initialize chi-square test."""
        super().__init__("Chi-Square Test")
    
    def goodness_of_fit(self, observed: Union[Vector, np.ndarray, List],
                       expected: Optional[Union[Vector, np.ndarray, List]] = None) -> Dict[str, Any]:
        """
        Chi-square goodness-of-fit test.
        
        Tests H0: Data follows expected distribution
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)
            
        Returns:
            Dictionary with test results
        """
        if isinstance(observed, Vector):
            obs = observed.to_numpy()
        else:
            obs = np.asarray(observed)
        
        if expected is None:
            # Uniform distribution
            exp = np.full_like(obs, np.sum(obs) / len(obs), dtype=float)
        else:
            if isinstance(expected, Vector):
                exp = expected.to_numpy()
            else:
                exp = np.asarray(expected)
        
        if len(obs) != len(exp):
            raise ValueError("Observed and expected must have same length")
        
        if np.any(exp <= 0):
            raise ValueError("Expected frequencies must be positive")
        
        # Chi-square statistic
        self._test_statistic = np.sum((obs - exp)**2 / exp)
        
        # Degrees of freedom
        df = len(obs) - 1
        
        # P-value
        self._p_value = 1 - stats.chi2.cdf(self._test_statistic, df)
        
        # Critical value
        self._critical_value = stats.chi2.ppf(0.95, df)
        
        self._performed = True
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'degrees_freedom': df,
            'observed': obs,
            'expected': exp,
            'residuals': (obs - exp) / np.sqrt(exp),
            'significant': self.is_significant()
        }
    
    def independence(self, contingency_table: Union[np.ndarray, List[List]]) -> Dict[str, Any]:
        """
        Chi-square test of independence.
        
        Tests H0: Variables are independent
        
        Args:
            contingency_table: 2D contingency table
            
        Returns:
            Dictionary with test results
        """
        table = np.asarray(contingency_table)
        
        if table.ndim != 2:
            raise ValueError("Contingency table must be 2-dimensional")
        
        if np.any(table < 0):
            raise ValueError("All entries must be non-negative")
        
        # Row and column totals
        row_totals = np.sum(table, axis=1)
        col_totals = np.sum(table, axis=0)
        total = np.sum(table)
        
        if total == 0:
            raise ValueError("Contingency table is empty")
        
        # Expected frequencies under independence
        expected = np.outer(row_totals, col_totals) / total
        
        # Avoid division by zero
        if np.any(expected == 0):
            raise ValueError("Expected frequencies contain zeros")
        
        # Chi-square statistic
        self._test_statistic = np.sum((table - expected)**2 / expected)
        
        # Degrees of freedom
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        
        # P-value
        self._p_value = 1 - stats.chi2.cdf(self._test_statistic, df)
        
        # Critical value
        self._critical_value = stats.chi2.ppf(0.95, df)
        
        self._performed = True
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'degrees_freedom': df,
            'observed': table,
            'expected': expected,
            'residuals': (table - expected) / np.sqrt(expected),
            'significant': self.is_significant()
        }


class KolmogorovSmirnovTest(HypothesisTest):
    """
    Kolmogorov-Smirnov tests.
    
    Tests for distributional assumptions.
    """
    
    def __init__(self):
        """Initialize KS test."""
        super().__init__("Kolmogorov-Smirnov Test")
    
    def one_sample(self, data: Union[Vector, np.ndarray, List],
                   cdf_func: callable) -> Dict[str, Any]:
        """
        One-sample KS test.
        
        Tests H0: Data follows specified distribution
        
        Args:
            data: Sample data
            cdf_func: Cumulative distribution function
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data, Vector):
            sample = data.to_numpy()
        else:
            sample = np.asarray(data)
        
        # Remove missing values and sort
        sample = sample[~np.isnan(sample)]
        sample = np.sort(sample)
        
        if len(sample) == 0:
            raise ValueError("No valid data points")
        
        n = len(sample)
        
        # Empirical CDF values
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Theoretical CDF values
        theoretical_cdf = np.array([cdf_func(x) for x in sample])
        
        # KS statistic (maximum difference)
        d_plus = np.max(empirical_cdf - theoretical_cdf)
        d_minus = np.max(theoretical_cdf - empirical_cdf + 1/n)
        self._test_statistic = max(d_plus, d_minus)
        
        # Approximate p-value (Kolmogorov distribution)
        self._p_value = stats.ksone.sf(self._test_statistic, n)
        
        # Critical value at α = 0.05
        self._critical_value = stats.ksone.ppf(0.95, n)
        
        self._performed = True
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'sample_size': n,
            'd_plus': d_plus,
            'd_minus': d_minus,
            'significant': self.is_significant()
        }
    
    def two_sample(self, data1: Union[Vector, np.ndarray, List],
                   data2: Union[Vector, np.ndarray, List]) -> Dict[str, Any]:
        """
        Two-sample KS test.
        
        Tests H0: Both samples come from same distribution
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data1, Vector):
            sample1 = data1.to_numpy()
        else:
            sample1 = np.asarray(data1)
        
        if isinstance(data2, Vector):
            sample2 = data2.to_numpy()
        else:
            sample2 = np.asarray(data2)
        
        # Remove missing values
        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]
        
        if len(sample1) == 0 or len(sample2) == 0:
            raise ValueError("No valid data points in one or both samples")
        
        # Use scipy's implementation
        self._test_statistic, self._p_value = stats.ks_2samp(sample1, sample2)
        
        # Approximate critical value
        n1, n2 = len(sample1), len(sample2)
        c_alpha = 1.36  # For α = 0.05
        self._critical_value = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))
        
        self._performed = True
        
        return {
            'test_statistic': self._test_statistic,
            'p_value': self._p_value,
            'critical_value': self._critical_value,
            'sample_size1': n1,
            'sample_size2': n2,
            'significant': self.is_significant()
        }