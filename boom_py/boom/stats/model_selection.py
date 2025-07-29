"""Model selection and validation functions."""
import numpy as np
from typing import List, Tuple, Union, Callable, Optional
from ..models.base import Model


def cross_validation(model: Model, data: List, k_folds: int = 5, 
                    metric: str = 'log_likelihood') -> dict:
    """Perform k-fold cross-validation.
    
    Args:
        model: Model to validate
        data: List of data points
        k_folds: Number of folds
        metric: Evaluation metric ('log_likelihood', 'mse', 'mae')
        
    Returns:
        Dictionary with CV results
    """
    n = len(data)
    fold_size = n // k_folds
    
    fold_scores = []
    
    for fold in range(k_folds):
        # Create train/test split
        start_idx = fold * fold_size
        if fold == k_folds - 1:
            end_idx = n
        else:
            end_idx = (fold + 1) * fold_size
        
        test_data = data[start_idx:end_idx]
        train_data = data[:start_idx] + data[end_idx:]
        
        # Train model
        model_copy = model.clone()
        model_copy.clear_data()
        for datum in train_data:
            model_copy.add_data(datum)
        
        if hasattr(model_copy, 'fit'):
            model_copy.fit()
        
        # Evaluate on test set
        if metric == 'log_likelihood':
            score = sum(model_copy.logpdf(datum) for datum in test_data)
        elif metric == 'mse':
            predictions = [model_copy.predict(datum) for datum in test_data]
            actuals = [model_copy.extract_response(datum) for datum in test_data]
            score = -np.mean([(p - a)**2 for p, a in zip(predictions, actuals)])
        elif metric == 'mae':
            predictions = [model_copy.predict(datum) for datum in test_data]
            actuals = [model_copy.extract_response(datum) for datum in test_data]
            score = -np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        fold_scores.append(score)
    
    return {
        'fold_scores': fold_scores,
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores, ddof=1),
        'n_folds': k_folds
    }


def bootstrap(data: List, statistic_func: Callable, n_bootstrap: int = 1000,
             confidence_level: float = 0.95) -> dict:
    """Bootstrap confidence intervals for a statistic.
    
    Args:
        data: Original data
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with bootstrap results
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = [data[np.random.randint(n)] for _ in range(n)]
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return {
        'bootstrap_samples': bootstrap_stats,
        'mean': np.mean(bootstrap_stats),
        'std': np.std(bootstrap_stats, ddof=1),
        'confidence_interval': (ci_lower, ci_upper),
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap
    }


def jackknife(data: List, statistic_func: Callable) -> dict:
    """Jackknife estimation for bias and standard error.
    
    Args:
        data: Original data
        statistic_func: Function to compute statistic
        
    Returns:
        Dictionary with jackknife results
    """
    n = len(data)
    original_stat = statistic_func(data)
    
    jackknife_stats = []
    for i in range(n):
        # Leave-one-out sample
        loo_sample = data[:i] + data[i+1:]
        stat = statistic_func(loo_sample)
        jackknife_stats.append(stat)
    
    jackknife_stats = np.array(jackknife_stats)
    
    # Jackknife estimates
    jackknife_mean = np.mean(jackknife_stats)
    
    # Bias estimate
    bias = (n - 1) * (jackknife_mean - original_stat)
    
    # Bias-corrected estimate
    bias_corrected = original_stat - bias
    
    # Standard error
    jackknife_se = np.sqrt((n - 1) * np.mean((jackknife_stats - jackknife_mean)**2))
    
    return {
        'original_statistic': original_stat,
        'jackknife_samples': jackknife_stats,
        'jackknife_mean': jackknife_mean,
        'bias_estimate': bias,
        'bias_corrected_estimate': bias_corrected,
        'standard_error': jackknife_se
    }


def model_comparison(models: List[Model], data: List, 
                    method: str = 'aic') -> dict:
    """Compare multiple models using information criteria.
    
    Args:
        models: List of models to compare
        data: Data for comparison
        method: Comparison method ('aic', 'bic', 'cross_validation')
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        'models': [],
        'scores': [],
        'ranks': [],
        'weights': []
    }
    
    for i, model in enumerate(models):
        # Fit model
        model_copy = model.clone()
        model_copy.clear_data()
        for datum in data:
            model_copy.add_data(datum)
        
        if hasattr(model_copy, 'fit'):
            model_copy.fit()
        
        # Compute score
        if method == 'aic':
            log_lik = model_copy.loglike()
            n_params = len(model_copy._params)
            score = -2 * log_lik + 2 * n_params
        elif method == 'bic':
            log_lik = model_copy.loglike()
            n_params = len(model_copy._params)
            n_obs = len(data)
            score = -2 * log_lik + n_params * np.log(n_obs)
        elif method == 'cross_validation':
            cv_result = cross_validation(model, data)
            score = -cv_result['mean_score']  # Negative for minimization
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results['models'].append(f"Model_{i}")
        results['scores'].append(score)
    
    # Compute ranks (1 = best)
    scores_array = np.array(results['scores'])
    ranks = np.argsort(np.argsort(scores_array)) + 1
    results['ranks'] = ranks.tolist()
    
    # Compute model weights (for AIC/BIC)
    if method in ['aic', 'bic']:
        min_score = np.min(scores_array)
        delta_scores = scores_array - min_score
        weights = np.exp(-0.5 * delta_scores)
        weights = weights / np.sum(weights)
        results['weights'] = weights.tolist()
    
    return results


def bayes_factor(model1_log_marginal: float, model2_log_marginal: float) -> float:
    """Compute Bayes factor comparing two models.
    
    Args:
        model1_log_marginal: Log marginal likelihood of model 1
        model2_log_marginal: Log marginal likelihood of model 2
        
    Returns:
        Bayes factor B12 = p(data|M1) / p(data|M2)
    """
    return np.exp(model1_log_marginal - model2_log_marginal)


def model_averaging(models: List[Model], weights: List[float], 
                   prediction_points: List) -> dict:
    """Perform Bayesian model averaging for predictions.
    
    Args:
        models: List of fitted models
        weights: Model weights (should sum to 1)
        prediction_points: Points at which to make predictions
        
    Returns:
        Dictionary with averaged predictions
    """
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize
    
    predictions = []
    prediction_variances = []
    
    for point in prediction_points:
        # Get predictions from all models
        model_predictions = []
        for model in models:
            if hasattr(model, 'predict'):
                pred = model.predict(point)
                model_predictions.append(pred)
            else:
                # Fallback: use mean of distribution
                model_predictions.append(model.logpdf(point))
        
        model_predictions = np.array(model_predictions)
        
        # Weighted average
        avg_prediction = np.sum(weights * model_predictions)
        
        # Prediction variance (including between-model uncertainty)
        within_model_var = 0.0  # Would need model-specific variance estimates
        between_model_var = np.sum(weights * (model_predictions - avg_prediction)**2)
        total_var = within_model_var + between_model_var
        
        predictions.append(avg_prediction)
        prediction_variances.append(total_var)
    
    return {
        'predictions': predictions,
        'prediction_variances': prediction_variances,
        'model_weights': weights
    }


def nested_model_test(restricted_model: Model, full_model: Model, 
                     data: List) -> dict:
    """Likelihood ratio test for nested models.
    
    Args:
        restricted_model: Nested (restricted) model
        full_model: Full model
        data: Data for testing
        
    Returns:
        Dictionary with test results
    """
    # Fit both models
    for model in [restricted_model, full_model]:
        model_copy = model.clone()
        model_copy.clear_data()
        for datum in data:
            model_copy.add_data(datum)
        if hasattr(model_copy, 'fit'):
            model_copy.fit()
    
    # Log likelihoods
    ll_restricted = restricted_model.loglike()
    ll_full = full_model.loglike()
    
    # Test statistic
    lr_statistic = 2 * (ll_full - ll_restricted)
    
    # Degrees of freedom
    df = len(full_model._params) - len(restricted_model._params)
    
    # P-value (would need chi-square distribution)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(lr_statistic, df=df)
    
    return {
        'lr_statistic': lr_statistic,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'll_restricted': ll_restricted,
        'll_full': ll_full
    }


def forward_selection(candidate_models: List[Model], data: List,
                     criterion: str = 'aic', alpha: float = 0.05) -> dict:
    """Forward selection of variables/models.
    
    Args:
        candidate_models: List of models to consider
        data: Data for selection
        criterion: Selection criterion ('aic', 'bic', 'p_value')
        alpha: Significance level (for p_value criterion)
        
    Returns:
        Dictionary with selection results
    """
    selected_models = []
    remaining_models = candidate_models.copy()
    
    current_score = np.inf
    
    while remaining_models:
        best_model = None
        best_score = np.inf
        best_idx = -1
        
        # Try adding each remaining model
        for i, model in enumerate(remaining_models):
            # Fit model
            model_copy = model.clone()
            model_copy.clear_data()
            for datum in data:
                model_copy.add_data(datum)
            
            if hasattr(model_copy, 'fit'):
                model_copy.fit()
            
            # Compute criterion
            if criterion == 'aic':
                log_lik = model_copy.loglike()
                n_params = len(model_copy._params)
                score = -2 * log_lik + 2 * n_params
            elif criterion == 'bic':
                log_lik = model_copy.loglike()
                n_params = len(model_copy._params)
                score = -2 * log_lik + n_params * np.log(len(data))
            else:
                score = 0  # Placeholder for p_value criterion
            
            if score < best_score:
                best_score = score
                best_model = model
                best_idx = i
        
        # Check if improvement is significant
        if best_score < current_score:
            selected_models.append(best_model)
            remaining_models.pop(best_idx)
            current_score = best_score
        else:
            break
    
    return {
        'selected_models': selected_models,
        'selection_path': [f"Model_{i}" for i in range(len(selected_models))],
        'final_score': current_score
    }