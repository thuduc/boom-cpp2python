"""
Tests for optimization algorithms and target functions.

This module tests the optimization frameworks including BFGS, Nelder-Mead,
and various target functions.
"""

import pytest
import numpy as np
import sys
import os

# Add the impl-python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from boom.optimization import (
    BfgsOptimizer, NelderMeadOptimizer, LineSearchOptimizer,
    OptimizationResult
)
from boom.optimization.target_functions import (
    QuadraticTarget, RosenbrockTarget, TargetFunction
)
from boom.linalg import Vector


class SimpleQuadratic(TargetFunction):
    """Simple quadratic for testing: f(x) = x^2"""
    
    def evaluate(self, parameters: Vector) -> float:
        self._evaluation_count += 1
        x = parameters.to_numpy()[0]
        return x**2
    
    def _compute_gradient(self, parameters: Vector) -> Vector:
        x = parameters.to_numpy()[0]
        return Vector(np.array([2*x]))


class TestOptimizationResult:
    """Test optimization result class."""
    
    def test_result_creation(self):
        """Test creating optimization results."""
        x = Vector(np.array([1.0, 2.0]))
        result = OptimizationResult(x=x, fun=0.5, success=True, nit=10, nfev=20)
        
        assert np.allclose(result.x.to_numpy(), [1.0, 2.0])
        assert result.fun == 0.5
        assert result.success is True
        assert result.nit == 10
        assert result.nfev == 20
        
    def test_result_string_representation(self):
        """Test string representation of results."""
        x = Vector(np.array([1.0]))
        result = OptimizationResult(x=x, fun=0.5, success=True)
        
        str_repr = str(result)
        assert "success=True" in str_repr
        assert "fun=0.500000" in str_repr


class TestTargetFunctions:
    """Test target function implementations."""
    
    def test_quadratic_target(self):
        """Test quadratic target function."""
        Q = np.array([[2.0, 0.0], [0.0, 1.0]])
        center = Vector(np.array([1.0, 2.0]))
        target = QuadraticTarget(Q, center)
        
        # Test evaluation at center (should be minimum)
        value_at_center = target.evaluate(center)
        assert value_at_center == pytest.approx(0.0)
        
        # Test evaluation away from center
        x = Vector(np.array([2.0, 3.0]))
        value = target.evaluate(x)
        assert value > 0
        
        # Test gradient
        grad = target.gradient(center)
        np.testing.assert_array_almost_equal(grad.to_numpy(), [0.0, 0.0])
        
        # Test Hessian
        hess = target.hessian(center)
        np.testing.assert_array_almost_equal(hess, Q)
        
    def test_rosenbrock_target(self):
        """Test Rosenbrock target function."""
        target = RosenbrockTarget(a=1.0, b=100.0)
        
        # Test evaluation at global minimum (1, 1)
        minimum = Vector(np.array([1.0, 1.0]))
        value_at_min = target.evaluate(minimum)
        assert value_at_min == pytest.approx(0.0, abs=1e-10)
        
        # Test evaluation at other points
        x = Vector(np.array([0.0, 0.0]))
        value = target.evaluate(x)
        assert value > 0
        
        # Test gradient at minimum
        grad = target.gradient(minimum)
        np.testing.assert_array_almost_equal(grad.to_numpy(), [0.0, 0.0], decimal=10)
        
    def test_simple_quadratic(self):
        """Test simple quadratic function."""
        target = SimpleQuadratic()
        
        # Test evaluation
        x = Vector(np.array([3.0]))
        value = target.evaluate(x)
        assert value == pytest.approx(9.0)
        
        # Test gradient
        grad = target.gradient(x)
        assert grad.to_numpy()[0] == pytest.approx(6.0)
        
        # Test evaluation counting
        target.reset_counters()
        target.evaluate(x)
        target.evaluate(x)
        counts = target.get_evaluation_counts()
        assert counts['function_evaluations'] == 2


class TestBfgsOptimizer:
    """Test BFGS optimizer."""
    
    def test_bfgs_quadratic_optimization(self):
        """Test BFGS on simple quadratic."""
        optimizer = BfgsOptimizer(max_iterations=100, tolerance=1e-8)
        target = SimpleQuadratic()
        
        # Start away from minimum
        x0 = Vector(np.array([5.0]))
        
        result = optimizer.minimize(target.evaluate, x0, target.gradient)
        
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(0.0, abs=1e-6)
        assert result.fun == pytest.approx(0.0, abs=1e-10)
        
    def test_bfgs_rosenbrock_optimization(self):
        """Test BFGS on Rosenbrock function."""
        optimizer = BfgsOptimizer(max_iterations=1000, tolerance=1e-6)
        target = RosenbrockTarget()
        
        # Start at a typical starting point
        x0 = Vector(np.array([-1.2, 1.0]))
        
        result = optimizer.minimize(target.evaluate, x0, target.gradient)
        
        # Should find the global minimum at (1, 1)
        assert result.success
        np.testing.assert_array_almost_equal(result.x.to_numpy(), [1.0, 1.0], decimal=4)
        assert result.fun == pytest.approx(0.0, abs=1e-6)
        
    def test_bfgs_numerical_gradient(self):
        """Test BFGS with numerical gradient."""
        optimizer = BfgsOptimizer(max_iterations=100)
        target = SimpleQuadratic()
        
        x0 = Vector(np.array([3.0]))
        
        # Don't provide gradient - should use numerical
        result = optimizer.minimize(target.evaluate, x0)
        
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(0.0, abs=1e-4)
        
    def test_bfgs_with_bounds(self):
        """Test BFGS with bound constraints."""
        optimizer = BfgsOptimizer(max_iterations=100)
        target = SimpleQuadratic()
        
        x0 = Vector(np.array([5.0]))
        bounds = [(1.0, 10.0)]  # Constrain to be >= 1
        
        result = optimizer.minimize(target.evaluate, x0, bounds=bounds)
        
        # Should find constrained minimum at x=1
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(1.0, abs=1e-6)


class TestNelderMeadOptimizer:
    """Test Nelder-Mead optimizer."""
    
    def test_nelder_mead_quadratic(self):
        """Test Nelder-Mead on simple quadratic."""
        optimizer = NelderMeadOptimizer(max_iterations=200, tolerance=1e-6)
        target = SimpleQuadratic()
        
        x0 = Vector(np.array([4.0]))
        
        result = optimizer.minimize(target.evaluate, x0)
        
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(0.0, abs=1e-4)
        
    def test_nelder_mead_rosenbrock(self):
        """Test Nelder-Mead on Rosenbrock function."""
        optimizer = NelderMeadOptimizer(max_iterations=1000, tolerance=1e-6)
        target = RosenbrockTarget()
        
        x0 = Vector(np.array([0.0, 0.0]))
        
        result = optimizer.minimize(target.evaluate, x0)
        
        # Nelder-Mead might not get as close as BFGS
        assert result.success
        np.testing.assert_array_almost_equal(result.x.to_numpy(), [1.0, 1.0], decimal=2)


class TestLineSearchOptimizer:
    """Test line search optimizer."""
    
    def test_line_search_quadratic(self):
        """Test line search optimizer on quadratic."""
        optimizer = LineSearchOptimizer(max_iterations=100)
        target = SimpleQuadratic()
        
        x0 = Vector(np.array([3.0]))
        
        result = optimizer.minimize(target.evaluate, x0, target.gradient)
        
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(0.0, abs=1e-4)


class TestOptimizerComparison:
    """Compare different optimizers on the same problems."""
    
    def test_optimizer_comparison_quadratic(self):
        """Compare optimizers on quadratic function."""
        target = QuadraticTarget(np.array([[1.0]]), Vector(np.array([2.0])))
        x0 = Vector(np.array([0.0]))
        
        optimizers = [
            BfgsOptimizer(max_iterations=50),
            NelderMeadOptimizer(max_iterations=100),
            LineSearchOptimizer(max_iterations=50)
        ]
        
        results = []
        for optimizer in optimizers:
            if isinstance(optimizer, (BfgsOptimizer, LineSearchOptimizer)):
                result = optimizer.minimize(target.evaluate, x0, target.gradient)
            else:
                result = optimizer.minimize(target.evaluate, x0)
            results.append(result)
        
        # All should find the minimum at x=2
        for result in results:
            assert result.success
            assert result.x.to_numpy()[0] == pytest.approx(2.0, abs=1e-3)
            
    def test_maximization(self):
        """Test maximization functionality."""
        optimizer = BfgsOptimizer(max_iterations=50)
        
        # Maximize -(x-2)^2, which has maximum at x=2
        def objective(x):
            return -(x.to_numpy()[0] - 2.0)**2
        
        def gradient(x):
            return Vector(np.array([-2 * (x.to_numpy()[0] - 2.0)]))
        
        x0 = Vector(np.array([0.0]))
        
        result = optimizer.maximize(objective, x0, gradient)
        
        assert result.success
        assert result.x.to_numpy()[0] == pytest.approx(2.0, abs=1e-6)
        assert result.fun == pytest.approx(0.0, abs=1e-10)  # Maximum value


class TestOptimizerRobustness:
    """Test optimizer robustness and edge cases."""
    
    def test_optimizer_with_invalid_function(self):
        """Test optimizer behavior with function that returns invalid values."""
        class ProblematicFunction(TargetFunction):
            def evaluate(self, parameters: Vector) -> float:
                x = parameters.to_numpy()[0]
                if x < 0:
                    return np.nan
                return x**2
        
        optimizer = BfgsOptimizer(max_iterations=50)
        target = ProblematicFunction()
        
        x0 = Vector(np.array([1.0]))  # Start in valid region
        
        result = optimizer.minimize(target.evaluate, x0)
        
        # Should handle invalid values gracefully
        assert isinstance(result, OptimizationResult)
        
    def test_optimizer_convergence_tracking(self):
        """Test optimization history tracking."""
        optimizer = BfgsOptimizer(max_iterations=20)
        target = SimpleQuadratic()
        
        x0 = Vector(np.array([5.0]))
        
        result = optimizer.minimize(target.evaluate, x0, target.gradient)
        
        history = optimizer.get_optimization_history()
        
        assert 'function_values' in history
        assert 'parameters' in history
        assert len(history['function_values']) > 0
        assert len(history['parameters']) > 0
        
        # Function values should generally decrease
        func_vals = history['function_values']
        assert func_vals[-1] <= func_vals[0]


if __name__ == '__main__':
    pytest.main([__file__])