#' BOOM Python R Interface Demo
#' ============================
#' 
#' This demo showcases the boompy package functionality

# Load the package
library(boompy)

# Check installation
cat("\n=== Checking BOOM Installation ===\n")
if (boom_check_installation()) {
  cat("BOOM Python backend is properly installed!\n")
  print(boom_version())
} else {
  stop("BOOM installation check failed. Please ensure Python and BOOM are installed.")
}

# 1. Linear Regression Demo
cat("\n=== Linear Regression Demo ===\n")
set.seed(42)
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- 1 + 2*x1 - 1.5*x2 + rnorm(n, sd = 0.5)
data <- data.frame(y = y, x1 = x1, x2 = x2)

# Fit model
lm_model <- boom_lm(y ~ x1 + x2, data)
cat("\nLinear Regression Results:\n")
print(summary(lm_model))

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm_model)
par(mfrow = c(1, 1))

# 2. Logistic Regression Demo
cat("\n=== Logistic Regression Demo ===\n")
set.seed(123)
n <- 200
x <- rnorm(n)
prob <- plogis(-0.5 + 1.5*x)
y <- rbinom(n, 1, prob)
binary_data <- data.frame(y = y, x = x)

logit_model <- boom_logit(y ~ x, binary_data)
cat("\nLogistic Regression Results:\n")
print(logit_model)

# ROC curve
probs <- predict(logit_model, type = "response")
plot(probs[y == 0], probs[y == 1], 
     xlab = "P(Y=1|X) for Y=0", 
     ylab = "P(Y=1|X) for Y=1",
     main = "Classification Performance")
abline(0, 1, lty = 2)

# 3. Mixture Model Demo
cat("\n=== Mixture Model Demo ===\n")
set.seed(456)
# Generate two-component mixture
component1 <- rnorm(150, mean = -2, sd = 0.8)
component2 <- rnorm(100, mean = 3, sd = 1.2)
mixture_data <- c(component1, component2)

# Fit mixture model
mix_model <- boom_mixture(mixture_data, k = 2)
cat("\nMixture Model Results:\n")
print(summary(mix_model))

# Visualize
plot(mix_model)

# 4. State Space Model Demo
cat("\n=== State Space Model Demo ===\n")
set.seed(789)
# Generate time series with trend
n_time <- 100
trend <- cumsum(rnorm(n_time, mean = 0.1, sd = 0.1))
observed <- trend + rnorm(n_time, sd = 0.5)

# Fit state space model
ss_model <- boom_bsts(observed, 
                      state.specification = list(type = "local_level"))
cat("\nState Space Model fitted\n")

# Plot results
plot(ss_model)

# 5. MCMC Demo
cat("\n=== MCMC Sampling Demo ===\n")
# Sample from a bivariate normal distribution
log_posterior <- function(theta) {
  # Bivariate normal with correlation
  mu <- c(0, 0)
  Sigma <- matrix(c(1, 0.7, 0.7, 1), 2, 2)
  
  # Compute log density
  diff <- theta - mu
  -0.5 * sum(diff * solve(Sigma, diff))
}

# Run MCMC
mcmc_samples <- boom_mcmc(log_posterior, 
                         initial = c(0, 0), 
                         niter = 2000, 
                         burn = 1000)
cat("\nMCMC Results:\n")
print(summary(mcmc_samples))

# Visualize samples
plot(mcmc_samples, type = "pairs")

# 6. Model Comparison Demo
cat("\n=== Model Comparison Demo ===\n")
# Fit models of increasing complexity
model1 <- boom_lm(y ~ 1, data)              # Intercept only
model2 <- boom_lm(y ~ x1, data)             # One predictor
model3 <- boom_lm(y ~ x1 + x2, data)        # Two predictors
model4 <- boom_lm(y ~ x1 * x2, data)        # With interaction

# Compare models
comparison <- boom_compare_models(model1, model2, model3, model4)
cat("\nModel Comparison Results:\n")
print(comparison)

# Plot AIC values
barplot(comparison$AIC, 
        names.arg = comparison$Model,
        main = "Model Comparison by AIC",
        ylab = "AIC",
        col = "lightblue")

cat("\n=== Demo Complete ===\n")
cat("The boompy package successfully demonstrates:\n")
cat("- Linear and logistic regression\n")
cat("- Mixture models\n")
cat("- State space models\n")
cat("- MCMC sampling\n")
cat("- Model comparison\n")
cat("\nBOOM Python backend is fully functional through R!\n")