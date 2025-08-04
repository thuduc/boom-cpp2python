"""
Regression diagnostics utilities.

This module provides diagnostic tools for regression models
including residual analysis and model validation.
"""

import numpy as np
import scipy.stats as stats
from typing import Optional, Dict, Any, Tuple, List
from ..linalg import Vector, Matrix


class RegressionDiagnostics:
    """
    Comprehensive regression diagnostics.
    
    Provides tools for analyzing residuals, detecting outliers,
    and validating model assumptions.
    """
    
    def __init__(self, fitted_values: Vector, residuals: Vector,
                 standardized_residuals: Optional[Vector] = None,
                 leverage: Optional[Vector] = None):
        """
        Initialize regression diagnostics.
        
        Args:
            fitted_values: Model fitted values
            residuals: Model residuals
            standardized_residuals: Standardized residuals (optional)
            leverage: Leverage values (optional)
        """
        self._fitted = fitted_values.to_numpy()
        self._residuals = residuals.to_numpy()
        
        if len(self._fitted) != len(self._residuals):
            raise ValueError("Fitted values and residuals must have same length")
        
        self._n = len(self._residuals)
        
        # Compute standardized residuals if not provided
        if standardized_residuals is not None:
            self._std_residuals = standardized_residuals.to_numpy()
        else:
            residual_std = np.std(self._residuals)
            self._std_residuals = self._residuals / residual_std if residual_std > 0 else self._residuals
        
        # Store leverage if provided
        self._leverage = leverage.to_numpy() if leverage is not None else None
        
        # Computed diagnostics
        self._diagnostics = {}
    
    def compute_all_diagnostics(self) -> Dict[str, Any]:
        """
        Compute all available diagnostics.
        
        Returns:
            Dictionary with all diagnostic results
        """
        self._diagnostics = {}
        
        # Basic residual statistics
        self._diagnostics.update(self._basic_residual_stats())
        
        # Normality tests
        self._diagnostics.update(self._normality_tests())
        
        # Homoscedasticity tests
        self._diagnostics.update(self._homoscedasticity_tests())
        
        # Outlier detection
        self._diagnostics.update(self._outlier_detection())
        
        # Autocorrelation tests
        self._diagnostics.update(self._autocorrelation_tests())
        
        return self._diagnostics.copy()
    
    def _basic_residual_stats(self) -> Dict[str, Any]:
        """Compute basic residual statistics."""
        return {
            'residual_mean': np.mean(self._residuals),
            'residual_std': np.std(self._residuals),
            'residual_var': np.var(self._residuals),
            'residual_min': np.min(self._residuals),
            'residual_max': np.max(self._residuals),
            'residual_median': np.median(self._residuals),
            'residual_iqr': np.percentile(self._residuals, 75) - np.percentile(self._residuals, 25)
        }
    
    def _normality_tests(self) -> Dict[str, Any]:
        """Test normality of residuals."""
        results = {}
        
        # Shapiro-Wilk test
        if self._n <= 5000:  # Shapiro-Wilk has sample size limitations
            try:
                stat, p_value = stats.shapiro(self._residuals)
                results['shapiro_wilk'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > 0.05
                }
            except Exception:
                results['shapiro_wilk'] = {'error': 'Test failed'}
        
        # Jarque-Bera test
        try:
            stat, p_value = stats.jarque_bera(self._residuals)
            results['jarque_bera'] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        except Exception:
            results['jarque_bera'] = {'error': 'Test failed'}
        
        # Anderson-Darling test
        try:
            result = stats.anderson(self._residuals, dist='norm')
            results['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values,
                'significance_levels': result.significance_level,
                'normal': result.statistic < result.critical_values[2]  # 5% significance level
            }
        except Exception:
            results['anderson_darling'] = {'error': 'Test failed'}
        
        return results
    
    def _homoscedasticity_tests(self) -> Dict[str, Any]:
        """Test homoscedasticity of residuals."""
        results = {}
        
        # Breusch-Pagan test (simplified version)
        try:
            # Regress squared residuals on fitted values
            y = self._residuals**2
            X = np.column_stack([np.ones(self._n), self._fitted])
            
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            
            # LM statistic
            ssr = np.sum((y - y_pred)**2)
            lm_stat = self._n - ssr / np.var(y)
            p_value = 1 - stats.chi2.cdf(lm_stat, 1)
            
            results['breusch_pagan'] = {
                'statistic': lm_stat,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05
            }
        except Exception:
            results['breusch_pagan'] = {'error': 'Test failed'}
        
        # White test (simplified)
        try:
            # Test correlation between squared residuals and fitted values
            corr, p_value = stats.pearsonr(self._residuals**2, self._fitted)
            
            results['white_test'] = {
                'correlation': corr,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05
            }
        except Exception:
            results['white_test'] = {'error': 'Test failed'}
        
        return results
    
    def _outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using various methods."""
        results = {}
        
        # Standardized residuals outliers (|z| > 2.5)
        outlier_threshold = 2.5
        outliers_std = np.abs(self._std_residuals) > outlier_threshold
        
        results['standardized_outliers'] = {
            'indices': np.where(outliers_std)[0].tolist(),
            'count': np.sum(outliers_std),
            'proportion': np.mean(outliers_std),
            'threshold': outlier_threshold
        }
        
        # Cook's distance (if leverage available)
        if self._leverage is not None:
            try:
                # Simplified Cook's distance calculation
                p = 1  # Number of predictors (simplified)
                cooks_d = (self._std_residuals**2 / p) * (self._leverage / (1 - self._leverage)**2)
                
                cook_threshold = 4 / self._n
                outliers_cook = cooks_d > cook_threshold
                
                results['cooks_distance'] = {
                    'values': cooks_d.tolist(),
                    'outliers': np.where(outliers_cook)[0].tolist(),
                    'count': np.sum(outliers_cook),
                    'threshold': cook_threshold
                }
            except Exception:
                results['cooks_distance'] = {'error': 'Calculation failed'}
        
        # DFFITS (if leverage available)
        if self._leverage is not None:
            try:
                dffits = self._std_residuals * np.sqrt(self._leverage / (1 - self._leverage))
                dffits_threshold = 2 * np.sqrt(2 / self._n)  # Simplified threshold
                outliers_dffits = np.abs(dffits) > dffits_threshold
                
                results['dffits'] = {
                    'values': dffits.tolist(),
                    'outliers': np.where(outliers_dffits)[0].tolist(),
                    'count': np.sum(outliers_dffits),
                    'threshold': dffits_threshold
                }
            except Exception:
                results['dffits'] = {'error': 'Calculation failed'}
        
        return results
    
    def _autocorrelation_tests(self) -> Dict[str, Any]:
        """Test for autocorrelation in residuals."""
        results = {}
        
        # Durbin-Watson test
        try:
            # Compute Durbin-Watson statistic
            diff_residuals = np.diff(self._residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(self._residuals**2)
            
            results['durbin_watson'] = {
                'statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except Exception:
            results['durbin_watson'] = {'error': 'Test failed'}
        
        # Ljung-Box test
        try:
            # Simple version - test first-order autocorrelation
            if self._n > 1:
                autocorr = np.corrcoef(self._residuals[:-1], self._residuals[1:])[0, 1]
                lb_stat = self._n * (self._n + 2) * autocorr**2 / (self._n - 1)
                p_value = 1 - stats.chi2.cdf(lb_stat, 1)
                
                results['ljung_box'] = {
                    'statistic': lb_stat,
                    'p_value': p_value,
                    'autocorrelation': autocorr,
                    'no_autocorr': p_value > 0.05
                }
        except Exception:
            results['ljung_box'] = {'error': 'Test failed'}
        
        return results
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation likely"
        elif dw_stat > 2.5:
            return "Negative autocorrelation likely"
        else:
            return "No strong evidence of autocorrelation"
    
    def get_diagnostic(self, name: str) -> Any:
        """Get specific diagnostic result."""
        return self._diagnostics.get(name)
    
    def summary_report(self) -> str:
        """Generate comprehensive diagnostics report."""
        if not self._diagnostics:
            self.compute_all_diagnostics()
        
        lines = []
        lines.append("Regression Diagnostics Report")
        lines.append("=" * 40)
        lines.append("")
        
        # Basic statistics
        lines.append("Residual Statistics:")
        lines.append(f"  Mean: {self._diagnostics['residual_mean']:.6f}")
        lines.append(f"  Std Dev: {self._diagnostics['residual_std']:.6f}")
        lines.append(f"  Min: {self._diagnostics['residual_min']:.6f}")
        lines.append(f"  Max: {self._diagnostics['residual_max']:.6f}")
        lines.append("")
        
        # Normality tests
        lines.append("Normality Tests:")
        if 'shapiro_wilk' in self._diagnostics:
            sw = self._diagnostics['shapiro_wilk']
            if 'error' not in sw:
                lines.append(f"  Shapiro-Wilk: p = {sw['p_value']:.4f} ({'Normal' if sw['normal'] else 'Non-normal'})")
        
        if 'jarque_bera' in self._diagnostics:
            jb = self._diagnostics['jarque_bera']
            if 'error' not in jb:
                lines.append(f"  Jarque-Bera: p = {jb['p_value']:.4f} ({'Normal' if jb['normal'] else 'Non-normal'})")
        lines.append("")
        
        # Homoscedasticity
        lines.append("Homoscedasticity Tests:")
        if 'breusch_pagan' in self._diagnostics:
            bp = self._diagnostics['breusch_pagan']
            if 'error' not in bp:
                lines.append(f"  Breusch-Pagan: p = {bp['p_value']:.4f} ({'Homoscedastic' if bp['homoscedastic'] else 'Heteroscedastic'})")
        lines.append("")
        
        # Outliers
        lines.append("Outlier Detection:")
        if 'standardized_outliers' in self._diagnostics:
            std_out = self._diagnostics['standardized_outliers']
            lines.append(f"  Standardized residual outliers: {std_out['count']} ({std_out['proportion']:.2%})")
        lines.append("")
        
        # Autocorrelation
        lines.append("Autocorrelation Tests:")
        if 'durbin_watson' in self._diagnostics:
            dw = self._diagnostics['durbin_watson']
            if 'error' not in dw:
                lines.append(f"  Durbin-Watson: {dw['statistic']:.4f} ({dw['interpretation']})")
        
        return "\n".join(lines)
    
    def diagnostics(self) -> Dict[str, Any]:
        """Get all computed diagnostics."""
        return self._diagnostics.copy()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all computed statistics (alias for diagnostics)."""
        return self.diagnostics()


def residual_analysis(fitted_values: Vector, residuals: Vector,
                     leverage: Optional[Vector] = None) -> RegressionDiagnostics:
    """
    Convenience function for residual analysis.
    
    Args:
        fitted_values: Model fitted values
        residuals: Model residuals
        leverage: Leverage values (optional)
        
    Returns:
        RegressionDiagnostics object with computed diagnostics
    """
    diagnostics = RegressionDiagnostics(fitted_values, residuals, leverage=leverage)
    diagnostics.compute_all_diagnostics()
    return diagnostics