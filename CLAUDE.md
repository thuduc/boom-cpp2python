# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BOOM (Bayesian Object Oriented Modeling) is a C++ library for Bayesian modeling, primarily through Markov chain Monte Carlo (MCMC). The library provides implementations of various statistical models and sampling algorithms, with bindings for R and experimental Python support.

## Build Commands

The primary build system is Bazel:

```bash
# Build the library
bazel build boom
bazel build -c opt boom  # Optimized build

# Run all tests
./testall
# Or directly with bazel:
bazel test Models/... cpputil/... LinAlg/... Samplers/... stats/... distributions/...

# Run specific tests
bazel test //LinAlg/tests:array_test
bazel test //Models/tests:gaussian_test

# Generate compile commands for IDE integration
bazel run @hedron_compile_commands//:refresh_all
```

## R Package Building

To build and install R packages:

```bash
# Build and install Boom R package
./install/create_boom_rpackage -i

# Build and install dependent R packages (after Boom is installed)
./install/boom_spike_slab -i
./install/bsts -i

# Check package for CRAN submission
./install/create_boom_rpackage -C
```

## Code Architecture

### Core Components

1. **LinAlg/**: Linear algebra utilities
   - Matrix, Vector, SpdMatrix (symmetric positive definite)
   - Decompositions: Cholesky, LU, QR, SVD
   - Eigen integration

2. **distributions/**: Probability distributions
   - Random number generation (rng.hpp)
   - R math functions (Rmath_dist.hpp)
   - Specialized samplers (truncated, bounded distributions)

3. **Models/**: Statistical models
   - Base model interfaces and policies
   - Glm/: Generalized linear models
   - StateSpace/: State space models
   - HMM/: Hidden Markov models
   - Mixtures/: Mixture models
   - TimeSeries/: Time series models
   - Hierarchical/: Hierarchical models

4. **Samplers/**: MCMC sampling algorithms
   - Metropolis-Hastings variants
   - Slice samplers
   - Adaptive rejection sampling (ARMS)

5. **Models/PosteriorSamplers/**: Model-specific posterior samplers
   - Conjugate samplers
   - Custom MCMC implementations

6. **TargetFun/**: Target function utilities for optimization
   - Log posterior calculations
   - Transformations (log, logit)

7. **numopt/**: Numerical optimization
   - Nelder-Mead, Powell, BFGS
   - Numerical derivatives

8. **stats/**: Statistical utilities
   - Data structures (DataTable, Design matrices)
   - Statistical tests and summaries

### Key Design Patterns

- **Policy-based design**: Models use policy classes for data storage, parameters, and priors
- **Visitor pattern**: Used in some model hierarchies
- **Smart pointers**: Extensive use of Ptr<> (custom smart pointer) for memory management
- **Sufstat**: Sufficient statistics classes for efficient computation

### Testing

- Tests use Google Test framework
- Test files follow pattern: `*_test.cc`
- Each major component has its own test directory with BUILD file
- Common test utilities in `//:boom_test_utils`

### Platform Notes

- Primary development on macOS and Linux
- Windows support may require adjustments to linker flags
- Thread support requires `-lpthread` on Linux (already configured in BUILD)