"""
Information criteria for model selection.

This module provides various information criteria (AIC, BIC, etc.)
for comparing and selecting statistical models.
"""

import numpy as np
from typing import Union, Dict, Any, List
from ..models.base import Model


def AIC(log_likelihood: float, n_parameters: int) -> float:
    """
    Compute Akaike Information Criterion.
    
    AIC = -2 * log_likelihood + 2 * n_parameters
    
    Args:
        log_likelihood: Log likelihood value
        n_parameters: Number of model parameters
        
    Returns:
        AIC value (lower is better)
    """
    return -2 * log_likelihood + 2 * n_parameters


def BIC(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """
    Compute Bayesian Information Criterion.
    
    BIC = -2 * log_likelihood + log(n) * n_parameters
    
    Args:
        log_likelihood: Log likelihood value
        n_parameters: Number of model parameters
        n_observations: Number of observations
        
    Returns:
        BIC value (lower is better)
    """
    return -2 * log_likelihood + np.log(n_observations) * n_parameters


def compute_ic(model: Model, criteria: List[str] = ['aic', 'bic']) -> Dict[str, float]:
    """
    Compute information criteria for a model.
    
    Args:
        model: Fitted model with log_likelihood and n_parameters methods
        criteria: List of criteria to compute
        
    Returns:
        Dictionary with computed criteria
    """
    if not hasattr(model, 'log_likelihood'):
        raise AttributeError("Model must have log_likelihood method")
    
    if not hasattr(model, 'n_parameters'):
        raise AttributeError("Model must have n_parameters method")
    
    log_lik = model.log_likelihood()
    n_params = model.n_parameters()
    
    results = {}
    
    for criterion in criteria:
        if criterion.lower() == 'aic':
            results['AIC'] = AIC(log_lik, n_params)
        elif criterion.lower() == 'bic':
            if hasattr(model, 'n_observations'):
                n_obs = model.n_observations()
            elif hasattr(model, '_data') and model._data is not None:
                n_obs = len(model._data)
            else:
                raise ValueError("Cannot determine number of observations for BIC")
            results['BIC'] = BIC(log_lik, n_params, n_obs)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    return results