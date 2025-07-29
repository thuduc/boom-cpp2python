# boompy: R Interface to BOOM Python

[![R Version](https://img.shields.io/badge/R-%3E%3D%203.5.0-blue.svg)](https://cran.r-project.org/)
[![Python Version](https://img.shields.io/badge/Python-%3E%3D%203.7-green.svg)](https://www.python.org/)

## Overview

The `boompy` package provides an R interface to the BOOM (Bayesian Object Oriented Modeling) Python library. This package allows R users to access BOOM's comprehensive suite of Bayesian statistical models through familiar R syntax.

## Features

- **Linear Models**: Bayesian linear regression with `boom_lm()`
- **GLMs**: Logistic and Poisson regression with `boom_glm()`
- **Time Series**: State space models with `boom_bsts()`
- **Mixture Models**: Finite mixtures with `boom_mixture()`
- **MCMC**: Flexible sampling with `boom_mcmc()`
- **Model Comparison**: Information criteria and model weights

## Installation

### Prerequisites

1. Python >= 3.7
2. BOOM Python package installed in your Python environment
3. R >= 3.5.0

### Install from Source

```r
# Install required R packages
install.packages(c("reticulate", "devtools"))

# Install boompy
devtools::install("path/to/boompy_r")
```

### Setup

```r
library(boompy)

# Check installation
boom_check_installation()

# Setup Python backend
boom_setup()
```

## Quick Start

### Linear Regression

```r
# Generate data
data <- data.frame(
  x = rnorm(100),
  y = 2 + 3 * x + rnorm(100)
)

# Fit model
model <- boom_lm(y ~ x, data)
summary(model)
plot(model)
```

### Logistic Regression

```r
# Binary outcome data
data <- data.frame(
  x = rnorm(200),
  y = rbinom(200, 1, prob = plogis(1 + 2*x))
)

# Fit logistic regression
model <- boom_logit(y ~ x, data)
predict(model, type = "response")
```

### Mixture Models

```r
# Mixture data
y <- c(rnorm(100, -2), rnorm(150, 3))

# Fit 2-component mixture
model <- boom_mixture(y, k = 2)
plot(model)
```

### MCMC Sampling

```r
# Define target distribution
log_density <- function(x) {
  -0.5 * sum(x^2)  # Standard normal
}

# Sample
samples <- boom_mcmc(log_density, initial = c(0, 0), niter = 1000)
plot(samples, type = "pairs")
```

## Documentation

- **Vignettes**: See `vignette("introduction", package = "boompy")`
- **Help**: Use `?boom_lm`, `?boom_glm`, etc.
- **Examples**: Each function includes executable examples

## Python-R Data Exchange

The package uses `reticulate` for seamless data exchange:

```r
# R to Python
r_vector <- c(1, 2, 3)
py_vector <- boom_env$Vector(r_vector)

# Python to R
r_result <- as.numeric(py_vector)
```

## Performance

- Computations leverage optimized NumPy/BLAS operations
- Large datasets are efficiently handled through Python backend
- Memory overhead from R-Python data transfer is minimized

## Troubleshooting

### Python Not Found

```r
# Specify Python path
boom_setup(python = "/path/to/python")
```

### Module Import Errors

```r
# Add BOOM to Python path
reticulate::py_run_string("
import sys
sys.path.append('/path/to/boom_py')
")
```

### Check Configuration

```r
# View setup
boom_version()
reticulate::py_config()
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

LGPL-2.1 (same as original BOOM)

## Authors

- **R Interface**: Claude Assistant (Anthropic)
- **Original BOOM**: Steven L. Scott

## Citation

```
@Manual{boompy,
  title = {boompy: R Interface to BOOM Python},
  author = {Claude Assistant},
  year = {2024},
  note = {R package version 1.0.0},
  url = {https://github.com/steve-the-bayesian/BOOM}
}
```

## Acknowledgments

This R interface was created as part of the BOOM C++ to Python migration project, providing backward compatibility for R users while leveraging the modern Python implementation.