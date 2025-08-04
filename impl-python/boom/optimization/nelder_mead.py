"""
Nelder-Mead simplex optimizer implementation.

This module provides the Nelder-Mead downhill simplex optimization algorithm,
which is a gradient-free method suitable for noisy or discontinuous functions.
"""

import numpy as np
from typing import Optional, Callable, List
from ..linalg import Vector
from .base import Optimizer, OptimizationResult


class NelderMeadOptimizer(Optimizer):
    """
    Nelder-Mead simplex optimizer.
    
    A direct search method that maintains a simplex of n+1 points
    in n-dimensional space and iteratively improves it through
    reflection, expansion, contraction, and shrinkage operations.
    """
    
    def __init__(self, max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 initial_simplex_scale: float = 1.0,
                 reflection_coeff: float = 1.0,
                 expansion_coeff: float = 2.0,
                 contraction_coeff: float = 0.5,
                 shrinkage_coeff: float = 0.5):
        """
        Initialize Nelder-Mead optimizer.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            initial_simplex_scale: Scale for initial simplex
            reflection_coeff: Reflection coefficient (α)
            expansion_coeff: Expansion coefficient (γ)
            contraction_coeff: Contraction coefficient (β)
            shrinkage_coeff: Shrinkage coefficient (δ)
        """
        super().__init__(max_iterations, tolerance)
        self._initial_scale = initial_simplex_scale
        self._alpha = reflection_coeff
        self._gamma = expansion_coeff
        self._beta = contraction_coeff
        self._delta = shrinkage_coeff
    
    def minimize(self, objective: Callable[[Vector], float],
                 x0: Vector,
                 gradient: Optional[Callable[[Vector], Vector]] = None,
                 hessian: Optional[Callable[[Vector], np.ndarray]] = None,
                 bounds: Optional[list] = None,
                 constraints: Optional[list] = None) -> OptimizationResult:
        """
        Minimize objective function using Nelder-Mead algorithm.
        
        Args:
            objective: Function to minimize
            x0: Initial parameter values
            gradient: Gradient function (ignored - not used by Nelder-Mead)
            hessian: Hessian function (ignored - not used by Nelder-Mead)
            bounds: Parameter bounds (optional)
            constraints: Optimization constraints (not supported)
            
        Returns:
            OptimizationResult containing optimization results
        """
        if constraints is not None:
            raise NotImplementedError("Constraints not supported in Nelder-Mead optimizer")
        
        self._reset_tracking()
        
        # Initialize simplex
        simplex = self._create_initial_simplex(x0)
        
        # Evaluate function at all simplex vertices
        values = []
        for vertex in simplex:
            if bounds is not None:
                from .base import ConstraintHandler
                vertex = ConstraintHandler.project_bounds(vertex, bounds)
            value = self._evaluate_function(objective, vertex)
            values.append(value)
        
        # Sort simplex by function values
        sorted_indices = np.argsort(values)
        simplex = [simplex[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Main optimization loop
        for iteration in range(self._max_iterations):
            self._iteration_count = iteration + 1
            
            # Check convergence
            if self._check_simplex_convergence(simplex, values):
                return OptimizationResult(
                    x=simplex[0], fun=values[0], success=True,
                    message="Simplex converged", nit=self._iteration_count,
                    nfev=self._function_evaluations
                )
            
            # Nelder-Mead operations
            n = len(simplex) - 1  # dimension
            
            # Compute centroid of all points except worst
            centroid = self._compute_centroid(simplex[:-1])
            
            # Reflection
            reflected = self._reflect(simplex[-1], centroid)
            if bounds is not None:
                from .base import ConstraintHandler
                reflected = ConstraintHandler.project_bounds(reflected, bounds)
            
            f_reflected = self._evaluate_function(objective, reflected)
            
            if values[0] <= f_reflected < values[-2]:
                # Accept reflection
                simplex[-1] = reflected
                values[-1] = f_reflected
            
            elif f_reflected < values[0]:
                # Try expansion
                expanded = self._expand(reflected, centroid)
                if bounds is not None:
                    from .base import ConstraintHandler
                    expanded = ConstraintHandler.project_bounds(expanded, bounds)
                
                f_expanded = self._evaluate_function(objective, expanded)
                
                if f_expanded < f_reflected:
                    # Accept expansion
                    simplex[-1] = expanded
                    values[-1] = f_expanded
                else:
                    # Accept reflection
                    simplex[-1] = reflected
                    values[-1] = f_reflected
            
            else:
                # Try contraction
                if f_reflected < values[-1]:
                    # Outside contraction
                    contracted = self._contract_outside(reflected, centroid)
                    if bounds is not None:
                        from .base import ConstraintHandler
                        contracted = ConstraintHandler.project_bounds(contracted, bounds)
                    
                    f_contracted = self._evaluate_function(objective, contracted)
                    
                    if f_contracted < f_reflected:
                        simplex[-1] = contracted
                        values[-1] = f_contracted
                    else:
                        # Shrink simplex
                        simplex, values = self._shrink_simplex(objective, simplex, values, bounds)
                else:
                    # Inside contraction
                    contracted = self._contract_inside(simplex[-1], centroid)
                    if bounds is not None:
                        from .base import ConstraintHandler
                        contracted = ConstraintHandler.project_bounds(contracted, bounds)
                    
                    f_contracted = self._evaluate_function(objective, contracted)
                    
                    if f_contracted < values[-1]:
                        simplex[-1] = contracted
                        values[-1] = f_contracted
                    else:
                        # Shrink simplex
                        simplex, values = self._shrink_simplex(objective, simplex, values, bounds)
            
            # Sort simplex by function values
            sorted_indices = np.argsort(values)
            simplex = [simplex[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
        
        # Maximum iterations reached
        return OptimizationResult(
            x=simplex[0], fun=values[0], success=False,
            message="Maximum iterations reached",
            nit=self._iteration_count, nfev=self._function_evaluations
        )
    
    def _create_initial_simplex(self, x0: Vector) -> List[Vector]:
        """Create initial simplex around starting point."""
        n = len(x0.to_numpy())
        simplex = [x0.copy()]  # Include starting point
        
        x0_array = x0.to_numpy()
        
        # Create n additional vertices
        for i in range(n):
            vertex = x0_array.copy()
            if vertex[i] != 0:
                vertex[i] *= (1 + self._initial_scale)
            else:
                vertex[i] = self._initial_scale
            simplex.append(Vector(vertex))
        
        return simplex
    
    def _compute_centroid(self, vertices: List[Vector]) -> Vector:
        """Compute centroid of vertices."""
        if not vertices:
            raise ValueError("Cannot compute centroid of empty vertex list")
        
        centroid = np.zeros(len(vertices[0].to_numpy()))
        for vertex in vertices:
            centroid += vertex.to_numpy()
        centroid /= len(vertices)
        
        return Vector(centroid)
    
    def _reflect(self, worst: Vector, centroid: Vector) -> Vector:
        """Reflect worst point through centroid."""
        worst_array = worst.to_numpy()
        centroid_array = centroid.to_numpy()
        
        reflected = centroid_array + self._alpha * (centroid_array - worst_array)
        return Vector(reflected)
    
    def _expand(self, reflected: Vector, centroid: Vector) -> Vector:
        """Expand beyond reflected point."""
        reflected_array = reflected.to_numpy()
        centroid_array = centroid.to_numpy()
        
        expanded = centroid_array + self._gamma * (reflected_array - centroid_array)
        return Vector(expanded)
    
    def _contract_outside(self, reflected: Vector, centroid: Vector) -> Vector:
        """Contract outside (between centroid and reflected point)."""
        reflected_array = reflected.to_numpy()
        centroid_array = centroid.to_numpy()
        
        contracted = centroid_array + self._beta * (reflected_array - centroid_array)
        return Vector(contracted)
    
    def _contract_inside(self, worst: Vector, centroid: Vector) -> Vector:
        """Contract inside (between centroid and worst point)."""
        worst_array = worst.to_numpy()
        centroid_array = centroid.to_numpy()
        
        contracted = centroid_array + self._beta * (worst_array - centroid_array)
        return Vector(contracted)
    
    def _shrink_simplex(self, objective: Callable[[Vector], float],
                       simplex: List[Vector], values: List[float],
                       bounds: Optional[list]) -> tuple:
        """Shrink entire simplex toward best point."""
        best = simplex[0]
        new_simplex = [best]
        new_values = [values[0]]
        
        best_array = best.to_numpy()
        
        for i in range(1, len(simplex)):
            vertex_array = simplex[i].to_numpy()
            shrunk_array = best_array + self._delta * (vertex_array - best_array)
            shrunk = Vector(shrunk_array)
            
            if bounds is not None:
                from .base import ConstraintHandler
                shrunk = ConstraintHandler.project_bounds(shrunk, bounds)
            
            value = self._evaluate_function(objective, shrunk)
            new_simplex.append(shrunk)
            new_values.append(value)
        
        return new_simplex, new_values
    
    def _check_simplex_convergence(self, simplex: List[Vector], 
                                  values: List[float]) -> bool:
        """Check if simplex has converged."""
        # Function value convergence
        f_range = max(values) - min(values)
        if f_range < self._tolerance:
            return True
        
        # Simplex size convergence
        best = simplex[0].to_numpy()
        max_distance = 0.0
        
        for vertex in simplex[1:]:
            distance = np.linalg.norm(vertex.to_numpy() - best)
            max_distance = max(max_distance, distance)
        
        if max_distance < self._tolerance:
            return True
        
        return False