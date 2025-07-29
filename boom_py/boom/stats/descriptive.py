"""Descriptive statistics functions."""
import numpy as np
from typing import Union, Optional, List
from ..linalg import Vector, Matrix


def mean(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute arithmetic mean."""
    return float(np.mean(data))


def variance(data: Union[Vector, np.ndarray, List[float]], 
            ddof: int = 1) -> float:
    """Compute sample variance (default) or population variance.
    
    Args:
        data: Input data
        ddof: Delta degrees of freedom (1 for sample, 0 for population)
    """
    return float(np.var(data, ddof=ddof))


def standard_deviation(data: Union[Vector, np.ndarray, List[float]], 
                      ddof: int = 1) -> float:
    """Compute sample standard deviation (default) or population std."""
    return float(np.std(data, ddof=ddof))


def skewness(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute sample skewness."""
    data = np.array(data)
    n = len(data)
    if n < 3:
        return np.nan
    
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    if sigma == 0:
        return np.nan
    
    # Sample skewness with bias correction
    skew = np.mean(((data - mu) / sigma) ** 3)
    return float(skew * n * (n - 1) / ((n - 1) * (n - 2)))


def kurtosis(data: Union[Vector, np.ndarray, List[float]], 
            excess: bool = True) -> float:
    """Compute sample kurtosis.
    
    Args:
        data: Input data
        excess: If True, return excess kurtosis (kurtosis - 3)
    """
    data = np.array(data)
    n = len(data)
    if n < 4:
        return np.nan
    
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    if sigma == 0:
        return np.nan
    
    # Sample kurtosis with bias correction
    kurt = np.mean(((data - mu) / sigma) ** 4)
    kurt = kurt * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
    kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    if not excess:
        kurt += 3
    
    return float(kurt)


def quantile(data: Union[Vector, np.ndarray, List[float]], 
            q: Union[float, List[float]]) -> Union[float, np.ndarray]:
    """Compute quantiles of data.
    
    Args:
        data: Input data
        q: Quantile(s) to compute (between 0 and 1)
    """
    result = np.quantile(data, q)
    if np.isscalar(result):
        return float(result)
    return result


def median(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute median."""
    return float(np.median(data))


def mode(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute mode (most frequent value)."""
    data = np.array(data)
    unique_values, counts = np.unique(data, return_counts=True)
    mode_idx = np.argmax(counts)
    return float(unique_values[mode_idx])


def range_stat(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute range (max - min)."""
    data = np.array(data)
    return float(np.max(data) - np.min(data))


def iqr(data: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute interquartile range (Q3 - Q1)."""
    q75, q25 = np.percentile(data, [75, 25])
    return float(q75 - q25)


def covariance(x: Union[Vector, np.ndarray, List[float]], 
              y: Union[Vector, np.ndarray, List[float]], 
              ddof: int = 1) -> float:
    """Compute sample covariance between x and y."""
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    cov_matrix = np.cov(x, y, ddof=ddof)
    return float(cov_matrix[0, 1])


def correlation(x: Union[Vector, np.ndarray, List[float]], 
               y: Union[Vector, np.ndarray, List[float]]) -> float:
    """Compute Pearson correlation coefficient between x and y."""
    corr_matrix = np.corrcoef(x, y)
    return float(corr_matrix[0, 1])


def autocorrelation(data: Union[Vector, np.ndarray, List[float]], 
                   max_lags: Optional[int] = None) -> np.ndarray:
    """Compute sample autocorrelation function.
    
    Args:
        data: Time series data
        max_lags: Maximum number of lags (default: len(data) - 1)
    
    Returns:
        Array of autocorrelations at lags 0, 1, ..., max_lags
    """
    data = np.array(data)
    n = len(data)
    
    if max_lags is None:
        max_lags = n - 1
    
    max_lags = min(max_lags, n - 1)
    
    # Center the data
    data_centered = data - np.mean(data)
    
    # Compute autocorrelations
    autocorr = np.zeros(max_lags + 1)
    
    for lag in range(max_lags + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            # Sample autocorrelation at lag k
            numerator = np.sum(data_centered[:-lag] * data_centered[lag:])
            denominator = np.sum(data_centered ** 2)
            autocorr[lag] = numerator / denominator if denominator > 0 else 0.0
    
    return autocorr


def cross_correlation(x: Union[Vector, np.ndarray, List[float]], 
                     y: Union[Vector, np.ndarray, List[float]], 
                     max_lags: Optional[int] = None) -> np.ndarray:
    """Compute sample cross-correlation function between x and y.
    
    Args:
        x: First time series
        y: Second time series
        max_lags: Maximum number of lags
    
    Returns:
        Array of cross-correlations at lags -max_lags, ..., 0, ..., max_lags
    """
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    n = len(x)
    if max_lags is None:
        max_lags = n - 1
    
    max_lags = min(max_lags, n - 1)
    
    # Center the data
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Compute cross-correlations
    cross_corr = np.zeros(2 * max_lags + 1)
    
    for i, lag in enumerate(range(-max_lags, max_lags + 1)):
        if lag == 0:
            numerator = np.sum(x_centered * y_centered)
        elif lag > 0:
            numerator = np.sum(x_centered[:-lag] * y_centered[lag:])
        else:  # lag < 0
            numerator = np.sum(x_centered[-lag:] * y_centered[:lag])
        
        # Normalize by standard deviations
        denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
        cross_corr[i] = numerator / denominator if denominator > 0 else 0.0
    
    return cross_corr


def summary_statistics(data: Union[Vector, np.ndarray, List[float]]) -> dict:
    """Compute comprehensive summary statistics.
    
    Returns:
        Dictionary with various statistics
    """
    data = np.array(data)
    
    stats = {
        'count': len(data),
        'mean': mean(data),
        'std': standard_deviation(data),
        'min': float(np.min(data)),
        'q25': quantile(data, 0.25),
        'median': median(data),
        'q75': quantile(data, 0.75),
        'max': float(np.max(data)),
        'range': range_stat(data),
        'iqr': iqr(data),
        'skewness': skewness(data),
        'kurtosis': kurtosis(data),
        'variance': variance(data)
    }
    
    return stats


def moments(data: Union[Vector, np.ndarray, List[float]], 
           n_moments: int = 4) -> List[float]:
    """Compute central moments of data.
    
    Args:
        data: Input data
        n_moments: Number of moments to compute
    
    Returns:
        List of central moments [1st, 2nd, 3rd, ..., nth]
    """
    data = np.array(data)
    mu = np.mean(data)
    
    moments_list = []
    for k in range(1, n_moments + 1):
        moment_k = np.mean((data - mu) ** k)
        moments_list.append(float(moment_k))
    
    return moments_list


def standardize(data: Union[Vector, np.ndarray, List[float]]) -> np.ndarray:
    """Standardize data to have mean 0 and std 1."""
    data = np.array(data)
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    if sigma == 0:
        return np.zeros_like(data)
    
    return (data - mu) / sigma


def normalize(data: Union[Vector, np.ndarray, List[float]], 
             method: str = 'minmax') -> np.ndarray:
    """Normalize data using specified method.
    
    Args:
        data: Input data
        method: 'minmax' (0-1 scaling) or 'zscore' (standardization)
    """
    data = np.array(data)
    
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        return standardize(data)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def winsorize(data: Union[Vector, np.ndarray, List[float]], 
             limits: tuple = (0.05, 0.05)) -> np.ndarray:
    """Winsorize data by clipping extreme values.
    
    Args:
        data: Input data
        limits: Tuple of (lower_percentile, upper_percentile) to clip
    """
    data = np.array(data)
    lower_limit, upper_limit = limits
    
    lower_val = np.percentile(data, 100 * lower_limit)
    upper_val = np.percentile(data, 100 * (1 - upper_limit))
    
    return np.clip(data, lower_val, upper_val)