"""
Posterior and likelihood target functions.

This module provides target functions for Bayesian inference
including log posteriors and log likelihoods.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from ..linalg import Vector
from ..models.base import Model
from .base import LogTargetFunction


class LogLikelihoodTarget(LogTargetFunction):
    """
    Log likelihood target function for maximum likelihood estimation.
    """
    
    def __init__(self, model: Model, name: str = "LogLikelihood"):
        """
        Initialize log likelihood target.
        
        Args:
            model: Statistical model with log_likelihood method
            name: Function name
        """
        super().__init__(name)
        self._model = model
    
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate log likelihood.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Log likelihood value
        """
        self._evaluation_count += 1
        
        try:
            # Set model parameters
            self._set_model_parameters(parameters)
            return self._model.log_likelihood()
            
        except Exception:
            # Return very negative value for invalid parameters
            return -1e10
    
    def _set_model_parameters(self, parameters: Vector) -> None:
        """Set model parameters from vector."""
        if hasattr(self._model, 'unvectorize_params'):
            self._model.unvectorize_params(parameters)
        elif hasattr(self._model, 'set_parameters'):
            param_dict = self._vector_to_params(parameters)
            self._model.set_parameters(param_dict)
        else:
            raise AttributeError("Model must have unvectorize_params or set_parameters method")
    
    def _vector_to_params(self, parameters: Vector) -> Dict[str, Any]:
        """Convert parameter vector to model parameter dictionary."""
        # This is a placeholder - specific models should override
        return {'parameters': parameters.to_numpy()}
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """
        Compute gradient of log likelihood.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Gradient vector
        """
        # Check if model provides analytical gradient
        if hasattr(self._model, 'log_likelihood_gradient'):
            try:
                self._set_model_parameters(parameters)
                return self._model.log_likelihood_gradient()
            except Exception:
                pass
        
        # Fall back to numerical gradient
        return self._numerical_gradient(parameters)
    
    @property
    def model(self) -> Model:
        """Get the underlying model."""
        return self._model


class LogPosteriorTarget(LogTargetFunction):
    """
    Log posterior target function for Bayesian parameter estimation.
    
    Combines log likelihood and log prior for MAP estimation or MCMC sampling.
    """
    
    def __init__(self, model: Model,
                 log_prior: Optional[Callable[[Vector], float]] = None,
                 name: str = "LogPosterior"):
        """
        Initialize log posterior target.
        
        Args:
            model: Statistical model with log_likelihood method
            log_prior: Log prior function (uniform if None)
            name: Function name
        """
        super().__init__(name)
        self._model = model
        self._log_prior = log_prior
    
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate log posterior.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Log posterior value
        """
        self._evaluation_count += 1
        
        try:
            # Set model parameters
            self._set_model_parameters(parameters)
            
            # Compute log likelihood
            log_lik = self._model.log_likelihood()
            
            # Add log prior if specified
            if self._log_prior is not None:
                log_prior = self._log_prior(parameters)
            else:
                log_prior = 0.0  # Uniform prior
            
            return log_lik + log_prior
            
        except Exception:
            # Return very negative value for invalid parameters
            return -1e10
    
    def _set_model_parameters(self, parameters: Vector) -> None:
        """Set model parameters from vector."""
        if hasattr(self._model, 'unvectorize_params'):
            self._model.unvectorize_params(parameters)
        elif hasattr(self._model, 'set_parameters'):
            param_dict = self._vector_to_params(parameters)
            self._model.set_parameters(param_dict)
        else:
            raise AttributeError("Model must have unvectorize_params or set_parameters method")
    
    def _vector_to_params(self, parameters: Vector) -> Dict[str, Any]:
        """Convert parameter vector to model parameter dictionary."""
        return {'parameters': parameters.to_numpy()}
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """
        Compute gradient of log posterior.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Gradient vector
        """
        try:
            # Likelihood gradient
            if hasattr(self._model, 'log_likelihood_gradient'):
                self._set_model_parameters(parameters)
                grad_lik = self._model.log_likelihood_gradient()
            else:
                # Numerical gradient of likelihood
                log_lik_target = LogLikelihoodTarget(self._model)
                grad_lik = log_lik_target._numerical_gradient(parameters)
            
            # Prior gradient
            if self._log_prior is not None:
                if hasattr(self, '_log_prior_gradient'):
                    grad_prior = self._log_prior_gradient(parameters)
                else:
                    # Numerical gradient of prior
                    grad_prior = self._numerical_prior_gradient(parameters)
                
                # Combine gradients
                total_grad = grad_lik.to_numpy() + grad_prior.to_numpy()
                return Vector(total_grad)
            else:
                return grad_lik
                
        except Exception:
            # Fall back to numerical gradient
            return self._numerical_gradient(parameters)
    
    def _numerical_prior_gradient(self, parameters: Vector,
                                 epsilon: float = 1e-8) -> Vector:
        """Compute numerical gradient of log prior."""
        if self._log_prior is None:
            return Vector(np.zeros(len(parameters.to_numpy())))
        
        x = parameters.to_numpy()
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = self._log_prior(Vector(x_plus))
            f_minus = self._log_prior(Vector(x_minus))
            
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return Vector(grad)
    
    def log_likelihood_component(self, parameters: Vector) -> float:
        """Get log likelihood component of posterior."""
        try:
            self._set_model_parameters(parameters)
            return self._model.log_likelihood()
        except Exception:
            return -1e10
    
    def log_prior_component(self, parameters: Vector) -> float:
        """Get log prior component of posterior."""
        if self._log_prior is not None:
            try:
                return self._log_prior(parameters)
            except Exception:
                return -1e10
        else:
            return 0.0
    
    @property
    def model(self) -> Model:
        """Get the underlying model."""
        return self._model
    
    @property
    def log_prior(self) -> Optional[Callable[[Vector], float]]:
        """Get the log prior function."""
        return self._log_prior


class LogPriorTarget(LogTargetFunction):
    """
    Log prior target function.
    """
    
    def __init__(self, log_prior: Callable[[Vector], float],
                 name: str = "LogPrior"):
        """
        Initialize log prior target.
        
        Args:
            log_prior: Log prior function
            name: Function name
        """
        super().__init__(name)
        self._log_prior = log_prior
    
    def evaluate(self, parameters: Vector) -> float:
        """Evaluate log prior."""
        self._evaluation_count += 1
        
        try:
            return self._log_prior(parameters)
        except Exception:
            return -1e10
    
    @property
    def log_prior_function(self) -> Callable[[Vector], float]:
        """Get the log prior function."""
        return self._log_prior


class ConditionalLogPosteriorTarget(LogTargetFunction):
    """
    Conditional log posterior for specific parameter subsets.
    
    Useful for block sampling in MCMC where only a subset of parameters
    are updated while others are held fixed.
    """
    
    def __init__(self, full_posterior: LogPosteriorTarget,
                 parameter_indices: list,
                 fixed_parameters: Vector,
                 name: str = "ConditionalLogPosterior"):
        """
        Initialize conditional log posterior.
        
        Args:
            full_posterior: Full log posterior target
            parameter_indices: Indices of parameters to vary
            fixed_parameters: Values of fixed parameters
            name: Function name
        """
        super().__init__(name)
        self._full_posterior = full_posterior
        self._parameter_indices = list(parameter_indices)
        self._fixed_parameters = fixed_parameters.copy()
        
        # Validate indices
        n_total = len(self._fixed_parameters.to_numpy())
        if any(idx >= n_total or idx < 0 for idx in self._parameter_indices):
            raise ValueError("Parameter indices out of range")
    
    def evaluate(self, subset_parameters: Vector) -> float:
        """
        Evaluate conditional log posterior.
        
        Args:
            subset_parameters: Values for varying parameters
            
        Returns:
            Conditional log posterior value
        """
        self._evaluation_count += 1
        
        # Construct full parameter vector
        full_params = self._fixed_parameters.to_numpy().copy()
        subset_array = subset_parameters.to_numpy()
        
        if len(subset_array) != len(self._parameter_indices):
            raise ValueError("Subset parameters length must match number of varying indices")
        
        for i, idx in enumerate(self._parameter_indices):
            full_params[idx] = subset_array[i]
        
        # Evaluate full posterior
        return self._full_posterior.evaluate(Vector(full_params))
    
    def _compute_gradient(self, subset_parameters: Vector) -> Optional[Vector]:
        """Compute gradient with respect to varying parameters."""
        # Construct full parameter vector
        full_params = self._fixed_parameters.to_numpy().copy()
        subset_array = subset_parameters.to_numpy()
        
        for i, idx in enumerate(self._parameter_indices):
            full_params[idx] = subset_array[i]
        
        # Get full gradient
        full_grad = self._full_posterior.gradient(Vector(full_params))
        if full_grad is None:
            return None
        
        # Extract subset gradient
        full_grad_array = full_grad.to_numpy()
        subset_grad = np.array([full_grad_array[idx] for idx in self._parameter_indices])
        
        return Vector(subset_grad)
    
    def update_fixed_parameters(self, fixed_parameters: Vector) -> None:
        """Update the fixed parameter values."""
        if len(fixed_parameters.to_numpy()) != len(self._fixed_parameters.to_numpy()):
            raise ValueError("Fixed parameters must have same length as original")
        
        self._fixed_parameters = fixed_parameters.copy()
        self.clear_cache()  # Clear any cached values
    
    @property
    def full_posterior(self) -> LogPosteriorTarget:
        """Get the full posterior target."""
        return self._full_posterior
    
    @property
    def parameter_indices(self) -> list:
        """Get the varying parameter indices."""
        return self._parameter_indices.copy()
    
    @property
    def fixed_parameters(self) -> Vector:
        """Get the fixed parameter values."""
        return self._fixed_parameters.copy()