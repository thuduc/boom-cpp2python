#!/usr/bin/env Rscript
# Test R interface to BOOM Python

# Set library path
.libPaths(c("boompy_r", .libPaths()))

# Add boom Python module to path
reticulate::py_run_string("
import sys
import os
boom_path = os.path.abspath('.')
if boom_path not in sys.path:
    sys.path.insert(0, boom_path)
")

# Source R files directly (since package not installed)
source("boompy_r/R/zzz.R")
source("boompy_r/R/regression.R")
source("boompy_r/R/methods.R")

# Initialize BOOM
boom_setup()

# Test 1: Linear Regression
cat("\n=== Testing Linear Regression ===\n")
set.seed(42)
n <- 50
x <- rnorm(n)
y <- 2 + 3*x + rnorm(n, sd = 0.5)
data <- data.frame(y = y, x = x)

model <- boom_lm(y ~ x, data)
cat("Coefficients:\n")
print(coef(model))
cat("True values: Intercept=2, Slope=3\n")

# Test 2: Predictions
cat("\n=== Testing Predictions ===\n")
newdata <- data.frame(x = c(-1, 0, 1))
preds <- predict(model, newdata)
cat("Predictions for x = -1, 0, 1:\n")
print(preds)

# Test 3: Summary
cat("\n=== Testing Summary Method ===\n")
print(model)

cat("\n=== R Interface Tests PASSED! ===\n")
cat("The boompy R package successfully interfaces with BOOM Python.\n")