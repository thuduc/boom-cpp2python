"""Information criteria for model selection."""
import numpy as np
from typing import Union, Optional
from ..models.base import Model


def aic(log_likelihood: float, n_parameters: int) -> float:
    """Compute Akaike Information Criterion.
    
    AIC = -2 * log_likelihood + 2 * k
    
    Args:
        log_likelihood: Log likelihood of the model
        n_parameters: Number of parameters in the model
        
    Returns:
        AIC value (lower is better)
    """
    return -2 * log_likelihood + 2 * n_parameters


def aicc(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute corrected Akaike Information Criterion.
    
    AICc = AIC + 2k(k+1)/(n-k-1)
    
    Args:
        log_likelihood: Log likelihood of the model
        n_parameters: Number of parameters in the model
        n_observations: Number of observations
        
    Returns:
        AICc value (lower is better)
    """
    aic_val = aic(log_likelihood, n_parameters)
    
    if n_observations - n_parameters - 1 <= 0:
        return np.inf  # Correction term undefined
    
    correction = (2 * n_parameters * (n_parameters + 1)) / (n_observations - n_parameters - 1)
    return aic_val + correction


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute Bayesian Information Criterion.
    
    BIC = -2 * log_likelihood + k * log(n)
    
    Args:
        log_likelihood: Log likelihood of the model
        n_parameters: Number of parameters in the model
        n_observations: Number of observations
        
    Returns:
        BIC value (lower is better)
    """
    return -2 * log_likelihood + n_parameters * np.log(n_observations)


def dic(log_likelihood_samples: np.ndarray, 
        log_likelihood_mean: float) -> tuple:
    """Compute Deviance Information Criterion.
    
    DIC = D_bar + p_D
    where D_bar is the posterior mean deviance and p_D is the effective number of parameters
    
    Args:
        log_likelihood_samples: Array of log likelihood values from MCMC samples
        log_likelihood_mean: Log likelihood at posterior mean parameters
        
    Returns:
        Tuple of (DIC, p_D, D_bar)
    """
    # Deviance = -2 * log likelihood
    deviance_samples = -2 * log_likelihood_samples
    deviance_mean_params = -2 * log_likelihood_mean
    
    # Posterior mean deviance
    D_bar = np.mean(deviance_samples)
    
    # Effective number of parameters
    p_D = D_bar - deviance_mean_params
    
    # DIC
    dic_val = D_bar + p_D
    
    return dic_val, p_D, D_bar


def waic(log_likelihood_matrix: np.ndarray) -> tuple:
    """Compute Widely Applicable Information Criterion.
    
    WAIC = -2 * (lppd - p_WAIC)
    where lppd is the log pointwise predictive density
    
    Args:
        log_likelihood_matrix: Matrix of log likelihoods with shape (n_samples, n_observations)
        
    Returns:
        Tuple of (WAIC, p_WAIC, lppd)
    """
    n_samples, n_obs = log_likelihood_matrix.shape
    
    # Log pointwise predictive density
    # For each observation, compute log(mean(exp(log_lik)))
    lppd_contributions = []
    p_waic_contributions = []
    
    for i in range(n_obs):
        log_lik_i = log_likelihood_matrix[:, i]
        
        # Log pointwise predictive density for observation i
        # Use log-sum-exp trick for numerical stability
        max_log_lik = np.max(log_lik_i)
        lppd_i = max_log_lik + np.log(np.mean(np.exp(log_lik_i - max_log_lik)))
        lppd_contributions.append(lppd_i)
        
        # Effective number of parameters for observation i
        p_waic_i = np.var(log_lik_i)
        p_waic_contributions.append(p_waic_i)
    
    lppd = np.sum(lppd_contributions)
    p_waic = np.sum(p_waic_contributions)
    
    waic_val = -2 * (lppd - p_waic)
    
    return waic_val, p_waic, lppd


def log_marginal_likelihood_harmonic_mean(log_likelihood_samples: np.ndarray) -> float:
    """Estimate log marginal likelihood using harmonic mean estimator.
    
    Note: This estimator is known to be unstable and is included for completeness.
    
    Args:
        log_likelihood_samples: Array of log likelihood values from MCMC samples
        
    Returns:
        Log marginal likelihood estimate
    """
    # Use log-sum-exp trick for stability
    neg_log_lik = -log_likelihood_samples
    max_neg_log_lik = np.max(neg_log_lik)
    
    harmonic_mean = -max_neg_log_lik - np.log(np.mean(np.exp(neg_log_lik - max_neg_log_lik)))
    
    return harmonic_mean


def log_marginal_likelihood_bridge_sampling(log_likelihood_prior: np.ndarray,
                                           log_likelihood_posterior: np.ndarray,
                                           max_iter: int = 1000,
                                           tol: float = 1e-10) -> float:
    """Estimate log marginal likelihood using bridge sampling.
    
    Args:
        log_likelihood_prior: Log likelihoods evaluated at draws from prior
        log_likelihood_posterior: Log likelihoods evaluated at draws from posterior
        max_iter: Maximum iterations for iterative scheme
        tol: Convergence tolerance
        
    Returns:
        Log marginal likelihood estimate
    """
    n1 = len(log_likelihood_prior)
    n2 = len(log_likelihood_posterior)
    
    # Initialize log marginal likelihood estimate
    log_ml = 0.0
    
    for iteration in range(max_iter):
        # Compute weights
        log_weights_1 = log_likelihood_prior - np.log(n1 * np.exp(log_likelihood_prior - log_ml) + n2)
        log_weights_2 = log_likelihood_posterior - np.log(n1 * np.exp(log_likelihood_posterior - log_ml) + n2)
        
        # Update estimate using log-sum-exp
        max_log_w1 = np.max(log_weights_1)
        max_log_w2 = np.max(log_weights_2)
        
        numerator = max_log_w1 + np.log(np.sum(np.exp(log_weights_1 - max_log_w1)))
        denominator = max_log_w2 + np.log(np.sum(np.exp(log_weights_2 - max_log_w2)))
        
        log_ml_new = numerator - denominator
        
        # Check convergence
        if abs(log_ml_new - log_ml) < tol:
            break
        
        log_ml = log_ml_new
    
    return log_ml


def model_weights(information_criterion_values: np.ndarray) -> np.ndarray:
    """Compute model weights from information criterion values.
    
    Args:
        information_criterion_values: Array of IC values (AIC, BIC, etc.)
        
    Returns:
        Array of model weights (probabilities)
    """
    # Convert to relative likelihood
    min_ic = np.min(information_criterion_values)
    delta_ic = information_criterion_values - min_ic
    
    # Akaike weights
    rel_likelihood = np.exp(-0.5 * delta_ic)
    weights = rel_likelihood / np.sum(rel_likelihood)
    
    return weights


def information_criterion_comparison(models_data: list) -> dict:
    """Compare models using multiple information criteria.
    
    Args:
        models_data: List of tuples (model_name, log_likelihood, n_parameters, n_observations)
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        'models': [],
        'aic': [],
        'aicc': [],
        'bic': [],
        'aic_weights': [],
        'bic_weights': []
    }
    
    for model_name, log_lik, n_params, n_obs in models_data:
        results['models'].append(model_name)
        
        aic_val = aic(log_lik, n_params)
        aicc_val = aicc(log_lik, n_params, n_obs)
        bic_val = bic(log_lik, n_params, n_obs)
        
        results['aic'].append(aic_val)
        results['aicc'].append(aicc_val)
        results['bic'].append(bic_val)
    
    # Compute model weights
    results['aic_weights'] = model_weights(np.array(results['aic']))
    results['bic_weights'] = model_weights(np.array(results['bic']))
    
    return results


def cross_validation_score(model: Model, data: list, k_folds: int = 5) -> float:
    """Compute k-fold cross-validation score.
    
    Args:
        model: Model to evaluate
        data: List of data points
        k_folds: Number of folds
        
    Returns:
        Average log likelihood across folds
    """
    n = len(data)
    fold_size = n // k_folds
    
    log_likelihoods = []
    
    for fold in range(k_folds):
        # Split data
        start_idx = fold * fold_size
        if fold == k_folds - 1:
            end_idx = n  # Include remaining data in last fold
        else:
            end_idx = (fold + 1) * fold_size
        
        test_data = data[start_idx:end_idx]
        train_data = data[:start_idx] + data[end_idx:]
        
        # Clone model and train on training data
        model_copy = model.clone()
        model_copy.clear_data()
        for datum in train_data:
            model_copy.add_data(datum)
        
        # Fit model (this depends on the specific model implementation)
        if hasattr(model_copy, 'fit'):
            model_copy.fit()
        
        # Evaluate on test data
        test_log_lik = 0.0
        for datum in test_data:
            test_log_lik += model_copy.logpdf(datum)
        
        log_likelihoods.append(test_log_lik)
    
    return np.mean(log_likelihoods)


def leave_one_out_cv(model: Model, data: list) -> float:
    """Compute leave-one-out cross-validation score.
    
    Args:
        model: Model to evaluate
        data: List of data points
        
    Returns:
        Average log likelihood
    """
    n = len(data)
    log_likelihoods = []
    
    for i in range(n):
        # Leave out data point i
        train_data = data[:i] + data[i+1:]
        test_datum = data[i]
        
        # Clone model and train
        model_copy = model.clone()
        model_copy.clear_data()
        for datum in train_data:
            model_copy.add_data(datum)
        
        if hasattr(model_copy, 'fit'):
            model_copy.fit()
        
        # Evaluate on left-out point
        log_lik = model_copy.logpdf(test_datum)
        log_likelihoods.append(log_lik)
    
    return np.mean(log_likelihoods)