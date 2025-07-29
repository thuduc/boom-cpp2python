#!/usr/bin/env python3
"""Simple tests for BOOM Python that work with current implementation."""

import sys
import numpy as np
from datetime import datetime

# Test results
results = {"passed": 0, "failed": 0, "total": 0}

def test(name, condition, error_msg=""):
    """Simple test assertion."""
    results["total"] += 1
    if condition:
        results["passed"] += 1
        print(f"✓ {name}")
    else:
        results["failed"] += 1
        print(f"✗ {name}: {error_msg}")

print("BOOM Python Test Suite")
print("=" * 60)
print(f"Started: {datetime.now()}\n")

# Test 1: Linear Algebra
print("Testing Linear Algebra...")
from boom.linalg import Vector, Matrix, SpdMatrix

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
test("Vector dot product", v1.dot(v2) == 32.0)
test("Vector addition", np.allclose(v1 + v2, [5, 7, 9]))
test("Vector norm", abs(v1.norm() - 3.741657) < 0.0001)

A = Matrix([[1, 2], [3, 4]])
test("Matrix determinant", abs(A.det() - (-2.0)) < 1e-10)
test("Matrix shape", A.shape == (2, 2))

S = SpdMatrix([[4, 1], [1, 3]])
S_inv = S.inv()
identity = S @ S_inv
test("SPD inverse", np.allclose(identity, np.eye(2), atol=1e-10))

# Test 2: Distributions
print("\nTesting Distributions...")
from boom.distributions import GlobalRng

rng = GlobalRng(seed=42)
test("Uniform in [0,1]", 0 <= rng.runif() <= 1)
test("Poisson >= 0", rng.rpois(5) >= 0)
test("Beta in [0,1]", 0 <= rng.rbeta(2, 2) <= 1)

# Normal samples
normal_samples = [rng.rnorm(0, 1) for _ in range(1000)]
mean_est = np.mean(normal_samples)
std_est = np.std(normal_samples)
test("Normal mean", abs(mean_est) < 0.1, f"mean={mean_est:.3f}")
test("Normal std", abs(std_est - 1.0) < 0.1, f"std={std_est:.3f}")

# Test 3: Regression
print("\nTesting Regression Models...")
from boom.models.glm import RegressionModel

np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
true_beta = Vector([1.5, -2.0])
y = X @ true_beta + 0.5 * np.random.randn(n)

model = RegressionModel()
for i in range(n):
    model.add_data((y[i], Vector(X[i, :])))

model.mle()
beta_est = model.beta
error = (beta_est - true_beta).norm()
test("Regression estimation", error < 0.2, f"error={error:.3f}")
test("Sigma positive", model.sigma > 0)

# Test 4: Optimization
print("\nTesting Optimization...")
from boom.optimization import BFGS, RosenbrockFunction

f = RosenbrockFunction()
optimizer = BFGS(max_iterations=200)
x0 = Vector([-1.2, 1.0])

result = optimizer.optimize(f, x0)
test("Optimization success", result.success)
test("Function value reasonable", result.f < 10.0, f"f={result.f:.2e}")

# Test 5: Simple MCMC
print("\nTesting MCMC (simplified)...")
from boom.samplers import MetropolisHastings

def log_density(x):
    return -0.5 * float(x.dot(x))  # Standard normal

sampler = MetropolisHastings(
    target_log_density=log_density,
    proposal_cov=np.eye(2) * 0.5
)

sampler.set_initial_value(Vector([0.0, 0.0]))
samples = []
for _ in range(100):
    samples.append(sampler.draw())

test("MCMC samples generated", len(samples) == 100)
test("MCMC draws are vectors", all(isinstance(s, np.ndarray) for s in samples))

# Test 6: State Space Components
print("\nTesting State Space Components...")
from boom.models.state_space import LocalLevelModel

component = LocalLevelModel(sigma=0.5)
test("Component dimension", component.state_dimension() == 1)
test("Component sigma", component.sigma == 0.5)

T = component.transition_matrix(0)
test("Transition matrix shape", T.shape == (1, 1))
test("Transition matrix value", T[0, 0] == 1.0)

# Test 7: Mixture Model Structure
print("\nTesting Mixture Models...")
from boom.models.mixtures import FiniteMixtureModel

mix_model = FiniteMixtureModel(n_components=2)
test("Mixture components", len(mix_model.components) == 2)
test("Mixing weights", len(mix_model.mixing_weights) == 2)
test("Weights sum to 1", abs(sum(mix_model.mixing_weights) - 1.0) < 1e-10)

# Test 8: Statistics
print("\nTesting Statistics Functions...")
from boom.stats.descriptive import mean, standard_deviation
from boom.stats.information_criteria import aic, bic

data = [1, 2, 3, 4, 5]
test("Mean calculation", mean(data) == 3.0)
test("Std calculation", standard_deviation(data) > 0)

log_lik = -100.0
n_params = 5
test("AIC calculation", aic(log_lik, n_params) == 210.0)
test("BIC > AIC", bic(log_lik, n_params, 100) > 210.0)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Total Tests: {results['total']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Success Rate: {results['passed']/results['total']*100:.1f}%")

if results['failed'] == 0:
    print("\nAll tests PASSED! ✓")
else:
    print(f"\n{results['failed']} tests failed.")

print(f"\nCompleted: {datetime.now()}")

sys.exit(0 if results['failed'] == 0 else 1)