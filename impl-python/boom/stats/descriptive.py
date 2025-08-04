"""
Descriptive statistics utilities.

This module provides functions for computing descriptive statistics
and summary measures for data analysis.
"""

import numpy as np
import scipy.stats as stats
from typing import Optional, Union, Dict, Any, List
from ..linalg import Vector, Matrix


class DescriptiveStats:
    """
    Class for computing and storing descriptive statistics.
    
    Provides comprehensive summary statistics for univariate and
    multivariate data including central tendency, dispersion,
    and shape measures.
    """
    
    def __init__(self, data: Union[Vector, Matrix, np.ndarray, List]):
        """
        Initialize with data.
        
        Args:
            data: Input data (Vector, Matrix, array, or list)
        """
        if isinstance(data, Vector):
            self._data = data.to_numpy().reshape(-1)
            self._is_multivariate = False
        elif isinstance(data, Matrix):
            self._data = data.to_numpy()
            self._is_multivariate = True if self._data.ndim > 1 and self._data.shape[1] > 1 else False
        elif isinstance(data, list):
            self._data = np.array(data)
            self._is_multivariate = self._data.ndim > 1 and self._data.shape[1] > 1
        else:
            self._data = np.asarray(data)
            self._is_multivariate = self._data.ndim > 1 and self._data.shape[1] > 1
        
        # Ensure 2D for multivariate case
        if self._is_multivariate and self._data.ndim == 1:
            self._data = self._data.reshape(-1, 1)
            self._is_multivariate = False
        elif not self._is_multivariate and self._data.ndim > 1:
            self._data = self._data.flatten()
        
        self._computed = False
        self._stats = {}
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute all descriptive statistics.
        
        Returns:
            Dictionary containing all computed statistics
        """
        if self._is_multivariate:
            self._compute_multivariate_stats()
        else:
            self._compute_univariate_stats()
        
        self._computed = True
        return self._stats.copy()
    
    def _compute_univariate_stats(self) -> None:
        """Compute statistics for univariate data."""
        data = self._data
        n = len(data)
        
        if n == 0:
            self._stats = {'n': 0}
            return
        
        # Basic counts
        self._stats['n'] = n
        self._stats['n_missing'] = np.sum(np.isnan(data))
        self._stats['n_valid'] = n - self._stats['n_missing']
        
        # Remove missing values for further calculations
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            self._stats.update({
                'mean': np.nan, 'median': np.nan, 'mode': np.nan,
                'std': np.nan, 'var': np.nan, 'min': np.nan, 'max': np.nan,
                'range': np.nan, 'iqr': np.nan, 'skewness': np.nan, 'kurtosis': np.nan
            })
            return
        
        # Central tendency
        self._stats['mean'] = np.mean(valid_data)
        self._stats['median'] = np.median(valid_data)
        
        # Mode (most frequent value for discrete data)
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        mode_idx = np.argmax(counts)
        self._stats['mode'] = unique_vals[mode_idx]
        self._stats['mode_count'] = counts[mode_idx]
        
        # Dispersion
        self._stats['std'] = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
        self._stats['var'] = np.var(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
        self._stats['min'] = np.min(valid_data)
        self._stats['max'] = np.max(valid_data)
        self._stats['range'] = self._stats['max'] - self._stats['min']
        
        # Quantiles
        self._stats['q25'] = np.percentile(valid_data, 25)
        self._stats['q50'] = np.percentile(valid_data, 50)  # Same as median
        self._stats['q75'] = np.percentile(valid_data, 75)
        self._stats['iqr'] = self._stats['q75'] - self._stats['q25']
        
        # Additional percentiles
        for p in [1, 5, 10, 90, 95, 99]:
            self._stats[f'p{p}'] = np.percentile(valid_data, p)
        
        # Shape measures
        if len(valid_data) > 2:
            self._stats['skewness'] = stats.skew(valid_data)
            self._stats['kurtosis'] = stats.kurtosis(valid_data)  # Excess kurtosis
            self._stats['kurtosis_raw'] = stats.kurtosis(valid_data, fisher=False)  # Raw kurtosis
        else:
            self._stats['skewness'] = np.nan
            self._stats['kurtosis'] = np.nan
            self._stats['kurtosis_raw'] = np.nan
        
        # Robust measures
        self._stats['mad'] = stats.median_abs_deviation(valid_data)  # Median absolute deviation
        self._stats['trimmed_mean_10'] = stats.trim_mean(valid_data, 0.1)  # 10% trimmed mean
        self._stats['trimmed_mean_20'] = stats.trim_mean(valid_data, 0.2)  # 20% trimmed mean
        
        # Standard error of mean
        self._stats['sem'] = self._stats['std'] / np.sqrt(len(valid_data))
        
        # Coefficient of variation
        if self._stats['mean'] != 0:
            self._stats['cv'] = self._stats['std'] / abs(self._stats['mean'])
        else:
            self._stats['cv'] = np.inf if self._stats['std'] > 0 else np.nan
    
    def _compute_multivariate_stats(self) -> None:
        """Compute statistics for multivariate data."""
        data = self._data
        n_obs, n_vars = data.shape
        
        self._stats['n_observations'] = n_obs
        self._stats['n_variables'] = n_vars
        
        # Per-variable statistics
        var_stats = []
        for j in range(n_vars):
            var_data = data[:, j]
            var_desc = DescriptiveStats(var_data)
            var_stats.append(var_desc.compute())
        
        self._stats['variable_stats'] = var_stats
        
        # Multivariate statistics
        # Remove rows with any missing values for multivariate calculations
        complete_rows = ~np.any(np.isnan(data), axis=1)
        complete_data = data[complete_rows]
        
        if len(complete_data) == 0:
            self._stats['n_complete'] = 0
            return
        
        self._stats['n_complete'] = len(complete_data)
        
        # Means and covariance
        self._stats['means'] = np.mean(complete_data, axis=0)
        if len(complete_data) > 1:
            self._stats['covariance'] = np.cov(complete_data, rowvar=False, ddof=1)
            self._stats['correlation'] = np.corrcoef(complete_data, rowvar=False)
            
            # Eigenanalysis of correlation matrix
            if n_vars > 1:
                eigenvals, eigenvecs = np.linalg.eigh(self._stats['correlation'])
                # Sort by eigenvalue (descending)
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                self._stats['eigenvalues'] = eigenvals
                self._stats['eigenvectors'] = eigenvecs
                self._stats['condition_number'] = eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 1e-10 else np.inf
        else:
            self._stats['covariance'] = np.zeros((n_vars, n_vars))
            self._stats['correlation'] = np.eye(n_vars)
        
        # Mahalanobis distances (if possible)
        if len(complete_data) > n_vars and n_vars > 1:
            try:
                inv_cov = np.linalg.inv(self._stats['covariance'])
                centered_data = complete_data - self._stats['means']
                mahal_distances = np.array([
                    np.sqrt(np.dot(row, np.dot(inv_cov, row)))
                    for row in centered_data
                ])
                self._stats['mahalanobis_distances'] = mahal_distances
                self._stats['mahalanobis_mean'] = np.mean(mahal_distances)
                self._stats['mahalanobis_std'] = np.std(mahal_distances)
            except np.linalg.LinAlgError:
                # Singular covariance matrix
                self._stats['mahalanobis_distances'] = None
    
    def summary(self) -> str:
        """
        Generate a formatted summary of the statistics.
        
        Returns:
            Formatted string summary
        """
        if not self._computed:
            self.compute()
        
        if self._is_multivariate:
            return self._multivariate_summary()
        else:
            return self._univariate_summary()
    
    def _univariate_summary(self) -> str:
        """Generate summary for univariate data."""
        s = self._stats
        
        summary = f"""
Descriptive Statistics Summary
=============================
Sample size (n): {s['n']}
Missing values: {s['n_missing']}
Valid values: {s['n_valid']}

Central Tendency:
  Mean:   {s['mean']:.6f}
  Median: {s['median']:.6f}
  Mode:   {s['mode']:.6f} (count: {s['mode_count']})

Dispersion:
  Std Dev:  {s['std']:.6f}
  Variance: {s['var']:.6f}
  Range:    {s['range']:.6f}
  IQR:      {s['iqr']:.6f}
  MAD:      {s['mad']:.6f}
  CV:       {s['cv']:.6f}

Quantiles:
  Min:  {s['min']:.6f}
  Q25:  {s['q25']:.6f}
  Q50:  {s['q50']:.6f}
  Q75:  {s['q75']:.6f}
  Max:  {s['max']:.6f}

Shape:
  Skewness: {s['skewness']:.6f}
  Kurtosis: {s['kurtosis']:.6f}

Robust Measures:
  10% Trimmed Mean: {s['trimmed_mean_10']:.6f}
  20% Trimmed Mean: {s['trimmed_mean_20']:.6f}
  SEM: {s['sem']:.6f}
"""
        return summary
    
    def _multivariate_summary(self) -> str:
        """Generate summary for multivariate data."""
        s = self._stats
        
        summary = f"""
Multivariate Descriptive Statistics
===================================
Observations: {s['n_observations']}
Variables: {s['n_variables']}
Complete cases: {s['n_complete']}

Variable Means:
"""
        for i, mean in enumerate(s['means']):
            summary += f"  Var {i+1}: {mean:.6f}\n"
        
        if 'condition_number' in s:
            summary += f"\nCorrelation Matrix Condition Number: {s['condition_number']:.2f}\n"
        
        return summary
    
    def get_statistic(self, name: str) -> Any:
        """Get a specific statistic by name."""
        if not self._computed:
            self.compute()
        return self._stats.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all computed statistics."""
        if not self._computed:
            self.compute()
        return self._stats.copy()


def compute_summary_stats(data: Union[Vector, Matrix, np.ndarray, List]) -> Dict[str, Any]:
    """
    Convenience function to compute summary statistics.
    
    Args:
        data: Input data
        
    Returns:
        Dictionary of computed statistics
    """
    desc = DescriptiveStats(data)
    return desc.compute()


def compute_correlation_matrix(data: Union[Matrix, np.ndarray], 
                             method: str = 'pearson') -> Matrix:
    """
    Compute correlation matrix.
    
    Args:
        data: Input data matrix
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    if isinstance(data, Matrix):
        data_array = data.to_numpy()
    else:
        data_array = np.asarray(data)
    
    if method == 'pearson':
        corr = np.corrcoef(data_array, rowvar=False)
    elif method == 'spearman':
        corr = stats.spearmanr(data_array, axis=0)[0]
    elif method == 'kendall':
        n_vars = data_array.shape[1]
        corr = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    tau, _ = stats.kendalltau(data_array[:, i], data_array[:, j])
                    corr[i, j] = tau
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return Matrix(corr)


def compute_covariance_matrix(data: Union[Matrix, np.ndarray]) -> Matrix:
    """
    Compute covariance matrix.
    
    Args:
        data: Input data matrix
        
    Returns:
        Covariance matrix
    """
    if isinstance(data, Matrix):
        data_array = data.to_numpy()
    else:
        data_array = np.asarray(data)
    
    cov = np.cov(data_array, rowvar=False, ddof=1)
    return Matrix(cov)