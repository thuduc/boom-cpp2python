"""Hypothesis testing functions."""
import numpy as np
from typing import Union, Tuple
from scipy import stats
from ..linalg import Vector


def t_test_one_sample(data: Union[Vector, np.ndarray], 
                     mu0: float = 0.0) -> Tuple[float, float]:
    """One-sample t-test.
    
    Args:
        data: Sample data
        mu0: Null hypothesis mean
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    data = np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    t_stat = (sample_mean - mu0) / (sample_std / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_value


def t_test_two_sample(data1: Union[Vector, np.ndarray], 
                     data2: Union[Vector, np.ndarray], 
                     equal_var: bool = True) -> Tuple[float, float]:
    """Two-sample t-test.
    
    Args:
        data1: First sample
        data2: Second sample
        equal_var: Assume equal variances
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    return float(t_stat), float(p_value)


def paired_t_test(data1: Union[Vector, np.ndarray], 
                 data2: Union[Vector, np.ndarray]) -> Tuple[float, float]:
    """Paired t-test.
    
    Args:
        data1: First sample
        data2: Second sample (paired)
        
    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(data1, data2)
    return float(t_stat), float(p_value)


def chi_square_test(observed: np.ndarray, 
                   expected: np.ndarray = None) -> Tuple[float, float]:
    """Chi-square goodness of fit test.
    
    Args:
        observed: Observed frequencies 
        expected: Expected frequencies (if None, assume uniform)
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    chi2_stat, p_value = stats.chisquare(observed, expected)
    return float(chi2_stat), float(p_value)


def kolmogorov_smirnov_test(data: Union[Vector, np.ndarray], 
                           cdf_func=None) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test for normality (default) or custom distribution.
    
    Args:
        data: Sample data
        cdf_func: CDF function to test against (default: standard normal)
        
    Returns:
        Tuple of (ks_statistic, p_value)
    """
    if cdf_func is None:
        # Test against standard normal
        data_std = (np.array(data) - np.mean(data)) / np.std(data, ddof=1)
        ks_stat, p_value = stats.kstest(data_std, 'norm')
    else:
        ks_stat, p_value = stats.kstest(data, cdf_func)
    
    return float(ks_stat), float(p_value)


def anderson_darling_test(data: Union[Vector, np.ndarray], 
                         dist: str = 'norm') -> Tuple[float, np.ndarray, np.ndarray]:
    """Anderson-Darling test for distribution fit.
    
    Args:
        data: Sample data
        dist: Distribution to test ('norm', 'expon', 'logistic', etc.)
        
    Returns:
        Tuple of (ad_statistic, critical_values, significance_levels)
    """
    result = stats.anderson(data, dist=dist)
    return result.statistic, result.critical_values, result.significance_level


def shapiro_wilk_test(data: Union[Vector, np.ndarray]) -> Tuple[float, float]:
    """Shapiro-Wilk test for normality.
    
    Args:
        data: Sample data
        
    Returns:
        Tuple of (w_statistic, p_value)
    """
    w_stat, p_value = stats.shapiro(data)
    return float(w_stat), float(p_value)


def jarque_bera_test(data: Union[Vector, np.ndarray]) -> Tuple[float, float]:
    """Jarque-Bera test for normality based on skewness and kurtosis.
    
    Args:
        data: Sample data
        
    Returns:
        Tuple of (jb_statistic, p_value)
    """
    jb_stat, p_value = stats.jarque_bera(data)
    return float(jb_stat), float(p_value)