"""
Parameter transformation utilities for target functions.

This module provides transformations (log, logit, etc.) and 
transformed target functions for constrained optimization.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from ..linalg import Vector
from .base import TargetFunction


class ParameterTransform(ABC):
    """
    Abstract base class for parameter transformations.
    
    Transforms parameters from constrained to unconstrained space
    and vice versa, with Jacobian computation.
    """
    
    @abstractmethod
    def transform(self, constrained: Vector) -> Vector:
        """
        Transform from constrained to unconstrained space.
        
        Args:
            constrained: Parameters in constrained space
            
        Returns:
            Parameters in unconstrained space
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, unconstrained: Vector) -> Vector:
        """
        Transform from unconstrained to constrained space.
        
        Args:
            unconstrained: Parameters in unconstrained space
            
        Returns:
            Parameters in constrained space
        """
        pass
    
    @abstractmethod
    def log_jacobian(self, constrained: Vector) -> float:
        """
        Compute log absolute determinant of Jacobian.
        
        Args:
            constrained: Parameters in constrained space
            
        Returns:
            Log |det(J)| where J is Jacobian of transformation
        """
        pass


class LogTransform(ParameterTransform):
    """
    Log transformation for positive parameters.
    
    Maps (0, ∞) → (-∞, ∞) using log transformation.
    """
    
    def transform(self, constrained: Vector) -> Vector:
        """Transform positive parameters to unconstrained space."""
        x = constrained.to_numpy()
        
        if np.any(x <= 0):
            raise ValueError("All parameters must be positive for log transform")
        
        return Vector(np.log(x))
    
    def inverse_transform(self, unconstrained: Vector) -> Vector:
        """Transform unconstrained parameters to positive space."""
        y = unconstrained.to_numpy()
        return Vector(np.exp(y))
    
    def log_jacobian(self, constrained: Vector) -> float:
        """Compute log Jacobian determinant."""
        x = constrained.to_numpy()
        
        if np.any(x <= 0):
            raise ValueError("All parameters must be positive for log transform")
        
        # For log transform: J = diag(1/x), so |det(J)| = prod(1/x) = 1/prod(x)
        # log|det(J)| = -sum(log(x))
        return -np.sum(np.log(x))


class LogitTransform(ParameterTransform):
    """
    Logit transformation for parameters in (0, 1).
    
    Maps (0, 1) → (-∞, ∞) using logit transformation.
    """
    
    def transform(self, constrained: Vector) -> Vector:
        """Transform (0,1) parameters to unconstrained space."""
        p = constrained.to_numpy()
        
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("All parameters must be in (0, 1) for logit transform")
        
        return Vector(np.log(p / (1 - p)))
    
    def inverse_transform(self, unconstrained: Vector) -> Vector:
        """Transform unconstrained parameters to (0,1) space."""
        y = unconstrained.to_numpy()
        
        # Use numerically stable computation
        exp_y = np.exp(y)
        p = exp_y / (1 + exp_y)
        
        # Handle extreme values
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        return Vector(p)
    
    def log_jacobian(self, constrained: Vector) -> float:
        """Compute log Jacobian determinant."""
        p = constrained.to_numpy()
        
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("All parameters must be in (0, 1) for logit transform")
        
        # For logit transform: J = diag(1/(p*(1-p)))
        # log|det(J)| = sum(log(1/(p*(1-p)))) = -sum(log(p) + log(1-p))
        return -np.sum(np.log(p) + np.log(1 - p))


class BoundedTransform(ParameterTransform):
    """
    Transformation for parameters with finite bounds.
    
    Maps (lower, upper) → (-∞, ∞) using scaled logit transformation.
    """
    
    def __init__(self, lower: float, upper: float):
        """
        Initialize bounded transform.
        
        Args:
            lower: Lower bound
            upper: Upper bound
        """
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        
        self._lower = lower
        self._upper = upper
        self._range = upper - lower
    
    def transform(self, constrained: Vector) -> Vector:
        """Transform bounded parameters to unconstrained space."""
        x = constrained.to_numpy()
        
        if np.any(x <= self._lower) or np.any(x >= self._upper):
            raise ValueError(f"All parameters must be in ({self._lower}, {self._upper})")
        
        # Scale to (0, 1) then apply logit
        p = (x - self._lower) / self._range
        return Vector(np.log(p / (1 - p)))
    
    def inverse_transform(self, unconstrained: Vector) -> Vector:
        """Transform unconstrained parameters to bounded space."""
        y = unconstrained.to_numpy()
        
        # Apply inverse logit then scale to (lower, upper)
        exp_y = np.exp(y)
        p = exp_y / (1 + exp_y)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        x = self._lower + self._range * p
        return Vector(x)
    
    def log_jacobian(self, constrained: Vector) -> float:
        """Compute log Jacobian determinant."""
        x = constrained.to_numpy()
        
        if np.any(x <= self._lower) or np.any(x >= self._upper):
            raise ValueError(f"All parameters must be in ({self._lower}, {self._upper})")
        
        # Scale to (0, 1)
        p = (x - self._lower) / self._range
        
        # Jacobian includes logit jacobian and scaling factor
        # J = diag(1/range * 1/(p*(1-p)))
        # log|det(J)| = n*log(1/range) - sum(log(p) + log(1-p))
        n = len(x)
        return n * np.log(1 / self._range) - np.sum(np.log(p) + np.log(1 - p))


class CompositeTransform(ParameterTransform):
    """
    Composite transformation for mixed parameter types.
    
    Applies different transformations to different parameter subsets.
    """
    
    def __init__(self, transforms: list, indices: list):
        """
        Initialize composite transform.
        
        Args:
            transforms: List of parameter transforms
            indices: List of parameter indices for each transform
        """
        if len(transforms) != len(indices):
            raise ValueError("Number of transforms must match number of index sets")
        
        self._transforms = transforms
        self._indices = indices
        
        # Check for overlapping indices
        all_indices = []
        for index_set in indices:
            all_indices.extend(index_set)
        
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Parameter indices cannot overlap")
    
    def transform(self, constrained: Vector) -> Vector:
        """Transform mixed constrained parameters."""
        x = constrained.to_numpy()
        y = x.copy()
        
        for transform, idx_set in zip(self._transforms, self._indices):
            if idx_set:  # Non-empty index set
                subset = Vector(x[idx_set])
                transformed_subset = transform.transform(subset)
                y[idx_set] = transformed_subset.to_numpy()
        
        return Vector(y)
    
    def inverse_transform(self, unconstrained: Vector) -> Vector:
        """Transform mixed unconstrained parameters."""
        y = unconstrained.to_numpy()
        x = y.copy()
        
        for transform, idx_set in zip(self._transforms, self._indices):
            if idx_set:  # Non-empty index set
                subset = Vector(y[idx_set])
                transformed_subset = transform.inverse_transform(subset)
                x[idx_set] = transformed_subset.to_numpy()
        
        return Vector(x)
    
    def log_jacobian(self, constrained: Vector) -> float:
        """Compute composite log Jacobian determinant."""
        x = constrained.to_numpy()
        total_log_jac = 0.0
        
        for transform, idx_set in zip(self._transforms, self._indices):
            if idx_set:  # Non-empty index set
                subset = Vector(x[idx_set])
                total_log_jac += transform.log_jacobian(subset)
        
        return total_log_jac


class TransformedTarget(TargetFunction):
    """
    Target function with parameter transformation.
    
    Allows optimization/sampling in unconstrained space while
    the underlying target operates in constrained space.
    """
    
    def __init__(self, base_target: TargetFunction,
                 transform: ParameterTransform,
                 name: Optional[str] = None):
        """
        Initialize transformed target function.
        
        Args:
            base_target: Base target function (operates in constrained space)
            transform: Parameter transformation
            name: Function name (default: transformed version of base name)
        """
        if name is None:
            name = f"Transformed_{base_target.name}"
        
        super().__init__(name)
        self._base_target = base_target
        self._transform = transform
    
    def evaluate(self, unconstrained_parameters: Vector) -> float:
        """
        Evaluate target in unconstrained space.
        
        Args:
            unconstrained_parameters: Parameters in unconstrained space
            
        Returns:
            Target function value with Jacobian correction
        """
        self._evaluation_count += 1
        
        try:
            # Transform to constrained space
            constrained_params = self._transform.inverse_transform(unconstrained_parameters)
            
            # Evaluate base target
            base_value = self._base_target.evaluate(constrained_params)
            
            # Add log Jacobian for change of variables
            log_jac = self._transform.log_jacobian(constrained_params)
            
            return base_value + log_jac
            
        except Exception:
            return -1e10
    
    def _compute_gradient(self, unconstrained_parameters: Vector) -> Optional[Vector]:
        """Compute gradient in unconstrained space."""
        try:
            # Transform to constrained space
            constrained_params = self._transform.inverse_transform(unconstrained_parameters)
            
            # Get gradient in constrained space
            constrained_grad = self._base_target.gradient(constrained_params)
            if constrained_grad is None:
                return self._numerical_gradient(unconstrained_parameters)
            
            # Transform gradient using chain rule
            # This is a simplified implementation - full version requires
            # computing the Jacobian of the inverse transformation
            return self._transform_gradient(constrained_grad, unconstrained_parameters)
            
        except Exception:
            return self._numerical_gradient(unconstrained_parameters)
    
    def _transform_gradient(self, constrained_grad: Vector,
                           unconstrained_params: Vector) -> Vector:
        """Transform gradient from constrained to unconstrained space."""
        # This is a simplified implementation
        # Full implementation would compute the Jacobian of inverse transformation
        # and apply chain rule properly
        
        # For now, use numerical gradient as fallback
        return self._numerical_gradient(unconstrained_params)
    
    def evaluate_constrained(self, constrained_parameters: Vector) -> float:
        """Evaluate base target in constrained space (without Jacobian)."""
        return self._base_target.evaluate(constrained_parameters)
    
    @property
    def base_target(self) -> TargetFunction:
        """Get base target function."""
        return self._base_target
    
    @property
    def transform(self) -> ParameterTransform:
        """Get parameter transformation."""
        return self._transform