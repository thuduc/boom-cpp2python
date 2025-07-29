"""Regression diagnostics and model checking functions."""
import numpy as np
from typing import Union, Tuple, Optional
from ..linalg import Vector, Matrix


def residual_analysis(observed: Union[Vector, np.ndarray], 
                     predicted: Union[Vector, np.ndarray]) -> dict:
    """Compute residual statistics.
    
    Args:
        observed: Observed values
        predicted: Predicted values
        
    Returns:
        Dictionary with residual statistics
    """
    observed = np.array(observed)
    predicted = np.array(predicted)
    residuals = observed - predicted
    
    return {
        'residuals': residuals,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals, ddof=1),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals))
    }


def durbin_watson(residuals: Union[Vector, np.ndarray]) -> float:
    """Compute Durbin-Watson statistic for autocorrelation in residuals.
    
    Args:
        residuals: Regression residuals
        
    Returns:
        Durbin-Watson statistic (values near 2 indicate no autocorrelation)
    """
    residuals = np.array(residuals)
    diff_residuals = np.diff(residuals)
    dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
    return float(dw)


def breusch_pagan(residuals: Union[Vector, np.ndarray], 
                 fitted_values: Union[Vector, np.ndarray]) -> Tuple[float, float]:
    """Breusch-Pagan test for heteroscedasticity.
    
    Args:
        residuals: Regression residuals
        fitted_values: Fitted values from regression
        
    Returns:
        Tuple of (bp_statistic, p_value)
    """
    residuals = np.array(residuals)
    fitted_values = np.array(fitted_values)
    
    # Square residuals and regress on fitted values
    squared_residuals = residuals**2
    
    # Simple linear regression of squared residuals on fitted values
    n = len(residuals)
    X = np.column_stack([np.ones(n), fitted_values])
    
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ squared_residuals)
        predicted_sq_res = X @ beta
        
        # R-squared from auxiliary regression
        ss_total = np.sum((squared_residuals - np.mean(squared_residuals))**2)
        ss_res = np.sum((squared_residuals - predicted_sq_res)**2)
        r_squared = 1 - ss_res / ss_total if ss_total > 0 else 0
        
        # BP statistic ~ Chi-square(1)
        bp_stat = n * r_squared
        
        # Approximate p-value (would need chi-square distribution)
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(bp_stat, df=1)
        
        return float(bp_stat), float(p_value)
        
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def white_test(residuals: Union[Vector, np.ndarray], 
              X: Union[Matrix, np.ndarray]) -> Tuple[float, float]:
    """White test for heteroscedasticity.
    
    Args:
        residuals: Regression residuals
        X: Design matrix (including intercept)
        
    Returns:
        Tuple of (white_statistic, p_value)
    """
    residuals = np.array(residuals)
    X = np.array(X)
    n, k = X.shape
    
    # Create auxiliary regression matrix with squares and cross-products
    X_aux = []
    
    # Add original variables
    for j in range(k):
        X_aux.append(X[:, j])
    
    # Add squares
    for j in range(1, k):  # Skip intercept
        X_aux.append(X[:, j]**2)
    
    # Add cross-products
    for j in range(1, k):
        for l in range(j+1, k):
            X_aux.append(X[:, j] * X[:, l])
    
    X_aux = np.column_stack(X_aux)
    
    # Regression of squared residuals on auxiliary variables
    squared_residuals = residuals**2
    
    try:
        beta = np.linalg.solve(X_aux.T @ X_aux, X_aux.T @ squared_residuals)
        predicted_sq_res = X_aux @ beta
        
        # R-squared
        ss_total = np.sum((squared_residuals - np.mean(squared_residuals))**2)
        ss_res = np.sum((squared_residuals - predicted_sq_res)**2)
        r_squared = 1 - ss_res / ss_total if ss_total > 0 else 0
        
        # White statistic
        white_stat = n * r_squared
        df = X_aux.shape[1] - 1
        
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(white_stat, df=df)
        
        return float(white_stat), float(p_value)
        
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def cook_distance(residuals: Union[Vector, np.ndarray], 
                 hat_matrix: Union[Matrix, np.ndarray], 
                 mse: float) -> np.ndarray:
    """Compute Cook's distance for each observation.
    
    Args:
        residuals: Regression residuals
        hat_matrix: Hat matrix (X(X'X)^(-1)X')
        mse: Mean squared error
        
    Returns:
        Array of Cook's distances
    """
    residuals = np.array(residuals)
    hat_matrix = np.array(hat_matrix)
    n, p = hat_matrix.shape[0], np.trace(hat_matrix)
    
    leverage_values = np.diag(hat_matrix)
    
    cook_d = (residuals**2 / (p * mse)) * (leverage_values / (1 - leverage_values)**2)
    
    return cook_d


def leverage(X: Union[Matrix, np.ndarray]) -> np.ndarray:
    """Compute leverage values (diagonal of hat matrix).
    
    Args:
        X: Design matrix
        
    Returns:
        Array of leverage values
    """
    X = np.array(X)
    
    try:
        hat_matrix = X @ np.linalg.solve(X.T @ X, X.T)
        return np.diag(hat_matrix)
    except np.linalg.LinAlgError:
        return np.full(X.shape[0], np.nan)


def studentized_residuals(residuals: Union[Vector, np.ndarray], 
                         leverage_values: Union[Vector, np.ndarray], 
                         mse: float) -> np.ndarray:
    """Compute studentized residuals.
    
    Args:
        residuals: Raw residuals
        leverage_values: Leverage values for each observation
        mse: Mean squared error
        
    Returns:
        Array of studentized residuals
    """
    residuals = np.array(residuals)
    leverage_values = np.array(leverage_values)
    
    # Standard errors of residuals
    residual_se = np.sqrt(mse * (1 - leverage_values))
    
    # Avoid division by zero
    residual_se = np.where(residual_se > 1e-15, residual_se, np.nan)
    
    return residuals / residual_se


def vif(X: Union[Matrix, np.ndarray]) -> np.ndarray:
    """Compute Variance Inflation Factors.
    
    Args:
        X: Design matrix (without intercept)
        
    Returns:
        Array of VIF values for each predictor
    """
    X = np.array(X)
    n, k = X.shape
    vif_values = np.zeros(k)
    
    for i in range(k):
        # Regress X[i] on all other X variables
        X_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        # Add intercept to other variables
        X_others_with_intercept = np.column_stack([np.ones(n), X_others])
        
        try:
            beta = np.linalg.solve(X_others_with_intercept.T @ X_others_with_intercept, 
                                 X_others_with_intercept.T @ X_i)
            predicted = X_others_with_intercept @ beta
            
            # R-squared
            ss_total = np.sum((X_i - np.mean(X_i))**2)
            ss_res = np.sum((X_i - predicted)**2)
            r_squared = 1 - ss_res / ss_total if ss_total > 0 else 0
            
            # VIF = 1 / (1 - R^2)
            vif_values[i] = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
            
        except np.linalg.LinAlgError:
            vif_values[i] = np.inf
    
    return vif_values


def condition_index(X: Union[Matrix, np.ndarray]) -> np.ndarray:
    """Compute condition indices for collinearity diagnosis.
    
    Args:
        X: Design matrix
        
    Returns:
        Array of condition indices
    """
    X = np.array(X)
    
    try:
        # SVD of X
        _, singular_values, _ = np.linalg.svd(X)
        
        # Condition indices
        max_sv = np.max(singular_values)
        condition_indices = max_sv / singular_values
        
        return condition_indices
        
    except np.linalg.LinAlgError:
        return np.array([np.inf])


def outlier_detection(residuals: Union[Vector, np.ndarray], 
                     leverage_values: Union[Vector, np.ndarray],
                     threshold_residual: float = 2.0,
                     threshold_leverage: float = None) -> dict:
    """Detect outliers based on residuals and leverage.
    
    Args:
        residuals: Studentized residuals
        leverage_values: Leverage values
        threshold_residual: Threshold for residual outliers
        threshold_leverage: Threshold for leverage outliers (default: 2*p/n)
        
    Returns:
        Dictionary with outlier information
    """
    residuals = np.array(residuals)
    leverage_values = np.array(leverage_values)
    n = len(residuals)
    
    if threshold_leverage is None:
        # Default threshold: 2 * (p / n) where p is number of parameters
        # Approximate p as average leverage * n
        p_estimate = np.mean(leverage_values) * n
        threshold_leverage = 2 * p_estimate / n
    
    # Identify outliers
    residual_outliers = np.abs(residuals) > threshold_residual
    leverage_outliers = leverage_values > threshold_leverage
    
    return {
        'residual_outliers': np.where(residual_outliers)[0],
        'leverage_outliers': np.where(leverage_outliers)[0],
        'influential_outliers': np.where(residual_outliers & leverage_outliers)[0],
        'n_residual_outliers': np.sum(residual_outliers),
        'n_leverage_outliers': np.sum(leverage_outliers),
        'n_influential_outliers': np.sum(residual_outliers & leverage_outliers)
    }