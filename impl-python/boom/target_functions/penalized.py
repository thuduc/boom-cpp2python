"""
Penalized target functions for regularized optimization.

This module provides penalized target functions with various
regularization terms (Ridge, Lasso, Elastic Net, etc.).
"""

import numpy as np
from typing import Optional
from ..linalg import Vector
from .base import TargetFunction


class PenalizedTarget(TargetFunction):
    """
    Base class for penalized target functions.
    
    Combines a base target function with penalty terms
    for regularized optimization.
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 name: Optional[str] = None):
        """
        Initialize penalized target function.
        
        Args:
            base_target: Base target function (e.g., log likelihood)
            penalty_lambda: Penalty strength parameter
            name: Function name
        """
        if name is None:
            name = f"Penalized_{base_target.name}"
        
        super().__init__(name)
        self._base_target = base_target
        self._penalty_lambda = penalty_lambda
        
        if penalty_lambda < 0:
            raise ValueError("Penalty lambda must be non-negative")
    
    def evaluate(self, parameters: Vector) -> float:
        """
        Evaluate penalized target function.
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Base target value minus penalty term
        """
        self._evaluation_count += 1
        
        base_value = self._base_target.evaluate(parameters)
        penalty_value = self._penalty_lambda * self._penalty_function(parameters)
        
        return base_value - penalty_value  # Subtract penalty (for maximization)
    
    def _penalty_function(self, parameters: Vector) -> float:
        """
        Compute penalty function (to be implemented by subclasses).
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Penalty value
        """
        raise NotImplementedError("Subclasses must implement _penalty_function")
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """
        Compute gradient of penalty function (to be implemented by subclasses).
        
        Args:
            parameters: Parameter vector
            
        Returns:
            Penalty gradient vector
        """
        raise NotImplementedError("Subclasses must implement _penalty_gradient")
    
    def _compute_gradient(self, parameters: Vector) -> Optional[Vector]:
        """Compute gradient of penalized target."""
        # Base gradient
        base_grad = self._base_target.gradient(parameters)
        if base_grad is None:
            return self._numerical_gradient(parameters)
        
        # Penalty gradient
        try:
            penalty_grad = self._penalty_gradient(parameters)
            
            # Combine gradients
            total_grad = base_grad.to_numpy() - self._penalty_lambda * penalty_grad.to_numpy()
            return Vector(total_grad)
            
        except NotImplementedError:
            # Fall back to numerical gradient
            return self._numerical_gradient(parameters)
    
    def evaluate_unpenalized(self, parameters: Vector) -> float:
        """Evaluate base target without penalty."""
        return self._base_target.evaluate(parameters)
    
    def evaluate_penalty(self, parameters: Vector) -> float:
        """Evaluate penalty term only."""
        return self._penalty_lambda * self._penalty_function(parameters)
    
    @property
    def base_target(self) -> TargetFunction:
        """Get base target function."""
        return self._base_target
    
    @property
    def penalty_lambda(self) -> float:
        """Get penalty strength parameter."""
        return self._penalty_lambda
    
    @penalty_lambda.setter
    def penalty_lambda(self, value: float) -> None:
        """Set penalty strength parameter."""
        if value < 0:
            raise ValueError("Penalty lambda must be non-negative")
        self._penalty_lambda = value


class Ridge(PenalizedTarget):
    """
    Ridge (L2) penalized target function.
    
    Adds L2 penalty: λ * ||θ||²₂
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 exclude_indices: Optional[list] = None,
                 name: str = "Ridge"):
        """
        Initialize Ridge penalized target.
        
        Args:
            base_target: Base target function
            penalty_lambda: L2 penalty strength
            exclude_indices: Parameter indices to exclude from penalty
            name: Function name
        """
        super().__init__(base_target, penalty_lambda, name)
        self._exclude_indices = set(exclude_indices) if exclude_indices else set()
    
    def _penalty_function(self, parameters: Vector) -> float:
        """Compute L2 penalty."""
        params = parameters.to_numpy()
        
        # Apply penalty only to non-excluded parameters
        penalty = 0.0
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                penalty += param**2
        
        return penalty
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """Compute gradient of L2 penalty."""
        params = parameters.to_numpy()
        grad = np.zeros_like(params)
        
        # Gradient is 2*θ for included parameters
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                grad[i] = 2 * param
        
        return Vector(grad)


class Lasso(PenalizedTarget):
    """
    Lasso (L1) penalized target function.
    
    Adds L1 penalty: λ * ||θ||₁
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 exclude_indices: Optional[list] = None,
                 name: str = "Lasso"):
        """
        Initialize Lasso penalized target.
        
        Args:
            base_target: Base target function
            penalty_lambda: L1 penalty strength
            exclude_indices: Parameter indices to exclude from penalty
            name: Function name
        """
        super().__init__(base_target, penalty_lambda, name)
        self._exclude_indices = set(exclude_indices) if exclude_indices else set()
    
    def _penalty_function(self, parameters: Vector) -> float:
        """Compute L1 penalty."""
        params = parameters.to_numpy()
        
        # Apply penalty only to non-excluded parameters
        penalty = 0.0
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                penalty += abs(param)
        
        return penalty
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """Compute gradient of L1 penalty (subgradient)."""
        params = parameters.to_numpy()
        grad = np.zeros_like(params)
        
        # Subgradient is sign(θ) for non-zero parameters
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                if param > 0:
                    grad[i] = 1.0
                elif param < 0:
                    grad[i] = -1.0
                else:
                    # At zero, subgradient is in [-1, 1] - use 0
                    grad[i] = 0.0
        
        return Vector(grad)


class ElasticNet(PenalizedTarget):
    """
    Elastic Net penalized target function.
    
    Combines L1 and L2 penalties: λ * (α * ||θ||₁ + (1-α) * ||θ||²₂)
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 alpha: float = 0.5,
                 exclude_indices: Optional[list] = None,
                 name: str = "ElasticNet"):
        """
        Initialize Elastic Net penalized target.
        
        Args:
            base_target: Base target function
            penalty_lambda: Overall penalty strength
            alpha: Mixing parameter (0 = Ridge, 1 = Lasso)
            exclude_indices: Parameter indices to exclude from penalty
            name: Function name
        """
        super().__init__(base_target, penalty_lambda, name)
        
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self._alpha = alpha
        self._exclude_indices = set(exclude_indices) if exclude_indices else set()
    
    def _penalty_function(self, parameters: Vector) -> float:
        """Compute Elastic Net penalty."""
        params = parameters.to_numpy()
        
        l1_penalty = 0.0
        l2_penalty = 0.0
        
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                l1_penalty += abs(param)
                l2_penalty += param**2
        
        return self._alpha * l1_penalty + (1 - self._alpha) * l2_penalty
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """Compute gradient of Elastic Net penalty."""
        params = parameters.to_numpy()
        grad = np.zeros_like(params)
        
        for i, param in enumerate(params):
            if i not in self._exclude_indices:
                # L1 component (subgradient)
                if param > 0:
                    l1_grad = 1.0
                elif param < 0:
                    l1_grad = -1.0
                else:
                    l1_grad = 0.0
                
                # L2 component
                l2_grad = 2 * param
                
                # Combined gradient
                grad[i] = self._alpha * l1_grad + (1 - self._alpha) * l2_grad
        
        return Vector(grad)
    
    @property
    def alpha(self) -> float:
        """Get mixing parameter."""
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set mixing parameter."""
        if not 0 <= value <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self._alpha = value


class GroupLasso(PenalizedTarget):
    """
    Group Lasso penalized target function.
    
    Applies L2 penalty within groups and L1 penalty between groups.
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 groups: Optional[list] = None,
                 name: str = "GroupLasso"):
        """
        Initialize Group Lasso penalized target.
        
        Args:
            base_target: Base target function
            penalty_lambda: Penalty strength
            groups: List of parameter index groups
            name: Function name
        """
        super().__init__(base_target, penalty_lambda, name)
        
        if groups is None:
            # Default: each parameter is its own group (equivalent to Lasso)
            n_params = 1  # This should be determined from first evaluation
            self._groups = [[i] for i in range(n_params)]
        else:
            self._groups = [list(group) for group in groups]
        
        # Validate groups
        all_indices = []
        for group in self._groups:
            all_indices.extend(group)
        
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Parameter indices cannot appear in multiple groups")
    
    def _penalty_function(self, parameters: Vector) -> float:
        """Compute Group Lasso penalty."""
        params = parameters.to_numpy()
        
        # Update groups if needed (first evaluation)
        if not self._groups:
            self._groups = [[i] for i in range(len(params))]
        
        penalty = 0.0
        
        for group in self._groups:
            # L2 norm within group
            group_norm = np.sqrt(sum(params[i]**2 for i in group))
            penalty += group_norm
        
        return penalty
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """Compute gradient of Group Lasso penalty."""
        params = parameters.to_numpy()
        grad = np.zeros_like(params)
        
        for group in self._groups:
            # L2 norm within group
            group_norm = np.sqrt(sum(params[i]**2 for i in group))
            
            if group_norm > 0:
                # Gradient is θ_g / ||θ_g||₂ for each parameter in group
                for i in group:
                    grad[i] = params[i] / group_norm
            # If group_norm == 0, gradient is 0 (subgradient)
        
        return Vector(grad)
    
    @property
    def groups(self) -> list:
        """Get parameter groups."""
        return [group.copy() for group in self._groups]


class AdaptiveLasso(PenalizedTarget):
    """
    Adaptive Lasso penalized target function.
    
    Uses adaptive weights for different parameters: λ * Σ w_j |θ_j|
    """
    
    def __init__(self, base_target: TargetFunction,
                 penalty_lambda: float = 1.0,
                 weights: Optional[Vector] = None,
                 name: str = "AdaptiveLasso"):
        """
        Initialize Adaptive Lasso penalized target.
        
        Args:
            base_target: Base target function
            penalty_lambda: Overall penalty strength
            weights: Adaptive weights for each parameter
            name: Function name
        """
        super().__init__(base_target, penalty_lambda, name)
        self._weights = weights
    
    def _penalty_function(self, parameters: Vector) -> float:
        """Compute adaptive Lasso penalty."""
        params = parameters.to_numpy()
        
        # Use uniform weights if not provided
        if self._weights is None:
            weights = np.ones_like(params)
        else:
            weights = self._weights.to_numpy()
            
        if len(weights) != len(params):
            raise ValueError("Weights must have same length as parameters")
        
        penalty = np.sum(weights * np.abs(params))
        return penalty
    
    def _penalty_gradient(self, parameters: Vector) -> Vector:
        """Compute gradient of adaptive Lasso penalty."""
        params = parameters.to_numpy()
        
        # Use uniform weights if not provided
        if self._weights is None:
            weights = np.ones_like(params)
        else:
            weights = self._weights.to_numpy()
        
        grad = np.zeros_like(params)
        
        for i, (param, weight) in enumerate(zip(params, weights)):
            if param > 0:
                grad[i] = weight
            elif param < 0:
                grad[i] = -weight
            else:
                grad[i] = 0.0  # Subgradient at zero
        
        return Vector(grad)
    
    def set_weights(self, weights: Vector) -> None:
        """Update adaptive weights."""
        self._weights = weights.copy()
    
    @property
    def weights(self) -> Optional[Vector]:
        """Get adaptive weights."""
        return self._weights.copy() if self._weights is not None else None