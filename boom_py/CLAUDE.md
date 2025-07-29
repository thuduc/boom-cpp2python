# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BOOM Python (boom_py) is a complete Python implementation of the BOOM (Bayesian Object Oriented Modeling) library, migrated from the original C++ codebase. It provides statistical models, MCMC samplers, and Bayesian inference tools with an R compatibility layer.

## Development Commands

### Installation

```bash
# Install for development (editable mode)
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=boom tests/

# Run specific test file
pytest tests/test_distributions.py -v

# Run only fast tests (skip slow ones)
pytest -m "not slow"

# Run benchmarks
pytest benchmarks/ -v
```

### Code Quality

```bash
# Format code with black
black boom/ tests/

# Sort imports
isort boom/ tests/

# Lint with flake8
flake8 boom/ tests/

# Type checking
mypy boom/
```

### R Package Testing

```bash
# Test R interface (requires R with testthat installed)
Rscript test_r_interface.R

# Or from R:
# testthat::test_dir("boompy_r/tests/testthat/")
```

## Architecture

### Core Components

1. **boom/linalg/**: Linear algebra foundations
   - Vector, Matrix, SpdMatrix (symmetric positive definite)
   - Built on NumPy with BOOM-specific methods
   - Cholesky decomposition, eigenvalues, QR

2. **boom/distributions/**: Probability distributions
   - GlobalRng singleton for random generation
   - Continuous: Normal, Gamma, Beta, Student-t, etc.
   - Discrete: Binomial, Poisson, Multinomial, etc.
   - Multivariate: MultivariateNormal, Wishart, Dirichlet

3. **boom/models/**: Statistical models
   - Base model classes with parameter management
   - Sufficient statistics for efficiency
   - GLM models: Linear, Logistic, Poisson regression
   - Mixture models: Finite mixtures, Dirichlet process
   - State space models: Kalman filter, trend/seasonal components

4. **boom/samplers/**: MCMC algorithms
   - Metropolis-Hastings with adaptive proposals
   - Slice sampling with step-out procedures
   - Move accounting for diagnostics

5. **boom/optimization/**: Numerical optimization
   - Newton-Raphson, BFGS, Conjugate Gradient
   - Line search with Wolfe conditions
   - Trust region methods

6. **boom/stats/**: Statistical utilities
   - Descriptive statistics, hypothesis tests
   - Information criteria (AIC, BIC, DIC, WAIC)
   - Model selection and diagnostics

### Design Patterns

- **NumPy Integration**: All array operations use NumPy for performance
- **Type Hints**: Full typing for better IDE support and error catching
- **Sufficient Statistics**: Memory-efficient incremental computation
- **Parameter Classes**: Type-safe parameter management with constraints
- **ABC Classes**: Abstract base classes define model interfaces

### R Interface (boompy_r/)

The R package provides backward compatibility:
- Model wrappers: boom_lm(), boom_glm(), boom_bsts()
- S3 methods: print(), summary(), plot(), predict()
- Uses reticulate for Python-R bridge
- Automatic data type conversion

### Key Implementation Notes

- **Random Number Generation**: Thread-safe GlobalRng singleton
- **Matrix Operations**: Column-major storage for R compatibility
- **Missing Data**: Handled via masked arrays
- **Numerical Stability**: Joseph form updates in Kalman filter
- **Memory Management**: Python GC handles all allocations