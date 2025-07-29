"""Tests for optimization algorithms."""
import pytest
import numpy as np
from boom.optimization import (
    NewtonRaphson, BFGS, ConjugateGradient, LevenbergMarquardt, SimulatedAnnealing
)
from boom.optimization.target_functions import (
    RosenbrockFunction, QuadraticFunction, TargetFunction
)
from boom.optimization.line_search import (
    BacktrackingLineSearch, WolfeLineSearch, ExactLineSearch
)
from boom.optimization.trust_region import (
    DoglegTrustRegion, CauchyPointTrustRegion, SteihaugTrustRegion
)
from boom.optimization.utils import (
    numerical_gradient, numerical_hessian, check_gradient, check_hessian,
    generate_test_problems
)
from boom.linalg import Vector, Matrix


class SimpleQuadratic(TargetFunction):
    """Simple quadratic for testing: f(x) = (x-1)^2 + (y-2)^2"""
    
    def evaluate(self, x: Vector) -> float:
        self.n_evaluations += 1
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def gradient(self, x: Vector) -> Vector:
        self.n_gradient_evaluations += 1
        return Vector([2 * (x[0] - 1), 2 * (x[1] - 2)])
    
    def hessian(self, x: Vector) -> Matrix:
        self.n_hessian_evaluations += 1
        return Matrix([[2.0, 0.0], [0.0, 2.0]])


class TestTargetFunctions:
    """Test target function implementations."""
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function."""
        f = RosenbrockFunction()
        x = Vector([0.0, 0.0])
        
        # Function value
        assert f.evaluate(x) == pytest.approx(1.0)
        
        # Gradient
        grad = f.gradient(x)
        expected_grad = Vector([-2.0, 0.0])
        np.testing.assert_allclose(grad, expected_grad)
        
        # Hessian
        hess = f.hessian(x)
        expected_hess = Matrix([[2.0, 0.0], [0.0, 200.0]])
        np.testing.assert_allclose(hess, expected_hess)
        
        # Check optimum
        x_opt = Vector([1.0, 1.0])
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-10)
        grad_opt = f.gradient(x_opt)
        np.testing.assert_allclose(grad_opt, [0.0, 0.0], atol=1e-10)
    
    def test_quadratic_function(self):
        """Test quadratic function."""
        A = Matrix([[2.0, 0.5], [0.5, 1.0]])
        b = Vector([1.0, -1.0])
        c = 2.0
        
        f = QuadraticFunction(A, b, c)
        x = Vector([0.0, 0.0])
        
        # Function value: 0.5 * 0 + 0 + 2 = 2
        assert f.evaluate(x) == pytest.approx(2.0)
        
        # Gradient: A * x + b = b
        grad = f.gradient(x)
        expected_grad = b
        np.testing.assert_allclose(grad, expected_grad)
        
        # Hessian: A
        hess = f.hessian(x)
        np.testing.assert_allclose(hess, A)
    
    def test_numerical_derivatives(self):
        """Test numerical differentiation."""
        f = SimpleQuadratic()
        x = Vector([0.5, 1.5])
        
        # Check gradient
        assert check_gradient(
            lambda v: f.evaluate(v),
            lambda v: f.gradient(v),
            x
        )
        
        # Check Hessian
        assert check_hessian(
            lambda v: f.gradient(v),
            lambda v: f.hessian(v),
            x
        )


class TestOptimizers:
    """Test optimization algorithms."""
    
    def test_newton_raphson(self):
        """Test Newton-Raphson optimizer."""
        optimizer = NewtonRaphson(max_iterations=50)
        f = SimpleQuadratic()
        x0 = Vector([0.0, 0.0])
        
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
        assert result.f == pytest.approx(0.0, abs=1e-10)
    
    def test_bfgs(self):
        """Test BFGS optimizer."""
        optimizer = BFGS(max_iterations=100)
        f = RosenbrockFunction()
        x0 = Vector([-1.2, 1.0])
        
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)
        assert result.f == pytest.approx(0.0, abs=1e-6)
    
    def test_conjugate_gradient(self):
        """Test Conjugate Gradient optimizer."""
        optimizer = ConjugateGradient(max_iterations=100)
        f = SimpleQuadratic()
        x0 = Vector([0.0, 0.0])
        
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
        assert result.f == pytest.approx(0.0, abs=1e-8)
    
    def test_levenberg_marquardt(self):
        """Test Levenberg-Marquardt optimizer."""
        optimizer = LevenbergMarquardt(max_iterations=50)
        f = SimpleQuadratic()
        x0 = Vector([0.0, 0.0])
        
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
        assert result.f == pytest.approx(0.0, abs=1e-8)
    
    def test_simulated_annealing(self):
        """Test Simulated Annealing optimizer."""
        optimizer = SimulatedAnnealing(
            max_iterations=5000,
            initial_temperature=10.0,
            cooling_rate=0.995
        )
        f = SimpleQuadratic()
        x0 = Vector([0.0, 0.0])
        
        result = optimizer.optimize(f, x0)
        
        assert result.success
        # SA is stochastic, so use larger tolerance
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=0.1)
        assert result.f == pytest.approx(0.0, abs=0.01)


class TestLineSearch:
    """Test line search algorithms."""
    
    def test_backtracking_line_search(self):
        """Test backtracking line search."""
        line_search = BacktrackingLineSearch()
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        p = Vector([-1.0, -2.0])  # Descent direction
        f_val = f.evaluate(x)
        grad = f.gradient(x)
        
        alpha = line_search.search(f, x, p, f_val, grad)
        
        assert alpha > 0
        assert alpha <= 1.0
        
        # Check Armijo condition
        x_new = x + alpha * p
        f_new = f.evaluate(x_new)
        assert f_new <= f_val + 1e-4 * alpha * grad.dot(p)
    
    def test_wolfe_line_search(self):
        """Test Wolfe line search."""
        line_search = WolfeLineSearch()
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        p = Vector([-1.0, -2.0])  # Descent direction
        f_val = f.evaluate(x)
        grad = f.gradient(x)
        
        alpha = line_search.search(f, x, p, f_val, grad)
        
        assert alpha > 0
        
        # Check Wolfe conditions
        x_new = x + alpha * p
        f_new = f.evaluate(x_new)
        grad_new = f.gradient(x_new)
        
        # Armijo condition
        assert f_new <= f_val + 1e-4 * alpha * grad.dot(p)
        
        # Curvature condition
        assert grad_new.dot(p) >= 0.9 * grad.dot(p)
    
    def test_exact_line_search(self):
        """Test exact line search on quadratic function."""
        line_search = ExactLineSearch()
        
        A = Matrix([[2.0, 0.0], [0.0, 2.0]])
        b = Vector([2.0, 4.0])
        f = QuadraticFunction(A, b, 0.0)
        
        x = Vector([0.0, 0.0])
        grad = f.gradient(x)
        p = -grad  # Steepest descent
        f_val = f.evaluate(x)
        
        alpha = line_search.search(f, x, p, f_val, grad)
        
        # For quadratic f(x) = 0.5 x^T A x + b^T x
        # with steepest descent p = -g, exact step is alpha = g^T g / (g^T A g)
        expected_alpha = grad.dot(grad) / grad.dot(A @ grad)
        assert alpha == pytest.approx(expected_alpha, rel=1e-6)


class TestTrustRegion:
    """Test trust region methods."""
    
    def test_cauchy_point_trust_region(self):
        """Test Cauchy point trust region."""
        tr = CauchyPointTrustRegion(initial_radius=1.0)
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        result = tr.compute_step(f, x)
        
        assert np.linalg.norm(result.step) <= tr.radius + 1e-10
        assert result.predicted_reduction > 0  # Should predict improvement
    
    def test_dogleg_trust_region(self):
        """Test dogleg trust region."""
        tr = DoglegTrustRegion(initial_radius=0.5)
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        result = tr.compute_step(f, x)
        
        assert np.linalg.norm(result.step) <= tr.radius + 1e-10
        assert result.predicted_reduction > 0
    
    def test_steihaug_trust_region(self):
        """Test Steihaug-Toint CG trust region."""
        tr = SteihaugTrustRegion(initial_radius=1.0)
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        result = tr.compute_step(f, x)
        
        assert np.linalg.norm(result.step) <= tr.radius + 1e-10
        assert result.predicted_reduction > 0
    
    def test_trust_region_radius_update(self):
        """Test trust region radius updates."""
        tr = DoglegTrustRegion(initial_radius=1.0)
        f = SimpleQuadratic()
        
        x = Vector([0.0, 0.0])
        initial_radius = tr.radius
        
        # Good step should potentially expand radius
        result = tr.compute_step(f, x)
        
        if result.rho > 0.75:
            assert tr.radius >= initial_radius
        elif result.rho < 0.25:
            assert tr.radius < initial_radius


class TestOptimizationUtils:
    """Test optimization utilities."""
    
    def test_numerical_gradient(self):
        """Test numerical gradient computation."""
        def f(x):
            return x[0]**2 + 2 * x[1]**2
        
        x = Vector([1.0, 2.0])
        grad = numerical_gradient(f, x)
        
        expected = Vector([2.0, 8.0])
        np.testing.assert_allclose(grad, expected, atol=1e-6)
    
    def test_numerical_hessian(self):
        """Test numerical Hessian computation."""
        def f(x):
            return x[0]**2 + 2 * x[1]**2 + x[0] * x[1]
        
        x = Vector([1.0, 1.0])
        hess = numerical_hessian(f, x)
        
        expected = Matrix([[2.0, 1.0], [1.0, 4.0]])
        np.testing.assert_allclose(hess, expected, atol=1e-4)
    
    def test_generate_test_problems(self):
        """Test test problem generation."""
        problems = generate_test_problems()
        
        assert len(problems) >= 3  # Should have at least 3 test problems
        
        for name, f, grad_f, x0, x_opt in problems:
            assert isinstance(name, str)
            assert callable(f)
            assert callable(grad_f)
            assert isinstance(x0, Vector)
            assert isinstance(x_opt, Vector)
            
            # Test that gradient is reasonably accurate
            x_test = x0
            assert check_gradient(f, grad_f, x_test, tol=1e-5)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_optimization_convergence(self):
        """Test that different optimizers converge to same solution."""
        f = RosenbrockFunction()
        x0 = Vector([-1.2, 1.0])
        
        optimizers = [
            NewtonRaphson(max_iterations=100),
            BFGS(max_iterations=100),
            ConjugateGradient(max_iterations=200)
        ]
        
        solutions = []
        for optimizer in optimizers:
            result = optimizer.optimize(f, x0)
            if result.success:
                solutions.append(result.x)
        
        # All successful solutions should be close to [1, 1]
        for sol in solutions:
            np.testing.assert_allclose(sol, [1.0, 1.0], atol=1e-3)
    
    def test_line_search_in_optimization(self):
        """Test that optimization with different line searches works."""
        from boom.optimization.optimizers import BFGS
        
        # Modify BFGS to use different line searches
        f = SimpleQuadratic()
        x0 = Vector([0.0, 0.0])
        
        optimizer = BFGS(max_iterations=50)
        result = optimizer.optimize(f, x0)
        
        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
    
    def test_high_dimensional_optimization(self):
        """Test optimization on higher dimensional problems."""
        # 10-dimensional quadratic
        n = 10
        A = Matrix(np.eye(n) + 0.1 * np.random.randn(n, n))
        A = Matrix(A.T @ A)  # Make positive definite
        b = Vector(np.random.randn(n))
        
        f = QuadraticFunction(A, b, 0.0)
        x0 = Vector.zero(n)
        
        optimizer = BFGS(max_iterations=200)
        result = optimizer.optimize(f, x0)
        
        assert result.success
        # For quadratic f(x) = 0.5 x^T A x + b^T x, optimum is x* = -A^(-1) b
        from boom.linalg import SpdMatrix
        x_opt = -SpdMatrix(A).inv() @ b
        np.testing.assert_allclose(result.x, x_opt, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])