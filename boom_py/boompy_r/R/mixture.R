#' BOOM Mixture Model
#' 
#' Fit a finite mixture model using the BOOM Python backend.
#' 
#' @param y Numeric vector of observations
#' @param k Number of mixture components
#' @param model Type of mixture model ("gaussian" or "poisson")
#' @param max_iter Maximum number of EM iterations
#' @param tol Convergence tolerance
#' @param ... Additional arguments
#' 
#' @return An object of class "boom_mixture" containing the fitted model
#' @export
#' 
#' @examples
#' \dontrun{
#' # Generate mixture data
#' y1 <- rnorm(100, mean = -2, sd = 1)
#' y2 <- rnorm(150, mean = 3, sd = 1.5)
#' y <- c(y1, y2)
#' 
#' # Fit mixture model
#' model <- boom_mixture(y, k = 2)
#' summary(model)
#' plot(model)
#' }
boom_mixture <- function(y, k = 2, model = "gaussian", 
                        max_iter = 100, tol = 1e-6, ...) {
  # Ensure BOOM is loaded
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  # Create mixture model
  py_model <- boom_env$boom$models$mixtures$FiniteMixtureModel(
    n_components = as.integer(k)
  )
  
  # Convert y to Python list/array
  py_model$set_data(y)
  
  # Fit model
  py_model$fit(max_iter = as.integer(max_iter), tol = tol)
  
  # Extract results
  weights <- as.numeric(py_model$mixing_weights)
  
  # Extract component parameters
  components <- list()
  for (i in seq_len(k)) {
    comp <- py_model$components[[i-1]]  # Python 0-indexed
    components[[i]] <- list(
      mean = comp$mean,
      sd = comp$sigma,
      weight = weights[i]
    )
  }
  
  # Create R object
  result <- structure(
    list(
      py_model = py_model,
      model_type = "mixture",
      k = k,
      y = y,
      components = components,
      weights = weights,
      loglik = py_model$loglike(),
      call = match.call()
    ),
    class = c("boom_mixture", "boom_model")
  )
  
  # Compute posterior probabilities for each observation
  result$posterior <- matrix(0, nrow = length(y), ncol = k)
  for (i in seq_along(y)) {
    probs <- as.numeric(py_model$component_posteriors(y[i]))
    result$posterior[i, ] <- probs
  }
  
  # Classify observations
  result$classification <- apply(result$posterior, 1, which.max)
  
  result
}

#' @export
print.boom_mixture <- function(x, ...) {
  cat("\nCall:\n")
  print(x$call)
  
  cat("\nMixture components:\n")
  for (i in seq_len(x$k)) {
    comp <- x$components[[i]]
    cat(sprintf("Component %d: mean = %.3f, sd = %.3f, weight = %.3f\n",
                i, comp$mean, comp$sd, comp$weight))
  }
  
  cat("\nLog-likelihood:", x$loglik, "\n")
  
  invisible(x)
}

#' @export
summary.boom_mixture <- function(object, ...) {
  cat("\nCall:\n")
  print(object$call)
  
  cat("\nNumber of observations:", length(object$y), "\n")
  cat("Number of components:", object$k, "\n")
  
  cat("\nMixing proportions:\n")
  print(object$weights)
  
  cat("\nComponent parameters:\n")
  params <- do.call(rbind, lapply(object$components, function(comp) {
    c(mean = comp$mean, sd = comp$sd)
  }))
  rownames(params) <- paste("Component", seq_len(object$k))
  print(params)
  
  cat("\nClassification table:\n")
  print(table(object$classification))
  
  # Information criteria
  n_params <- object$k * 3 - 1  # means, sds, weights (minus 1 for constraint)
  aic_val <- -2 * object$loglik + 2 * n_params
  bic_val <- -2 * object$loglik + n_params * log(length(object$y))
  
  cat("\nInformation criteria:\n")
  cat("  AIC:", aic_val, "\n")
  cat("  BIC:", bic_val, "\n")
  
  invisible(object)
}

#' @export
plot.boom_mixture <- function(x, ...) {
  # Histogram with fitted density
  hist(x$y, breaks = 30, freq = FALSE,
       main = "Mixture Model Fit",
       xlab = "Value", ylab = "Density",
       col = "lightgray")
  
  # Plot overall density
  xx <- seq(min(x$y), max(x$y), length.out = 200)
  overall_density <- numeric(length(xx))
  
  # Add component densities
  colors <- c("red", "blue", "green", "orange", "purple")
  for (i in seq_len(x$k)) {
    comp <- x$components[[i]]
    comp_density <- comp$weight * dnorm(xx, comp$mean, comp$sd)
    overall_density <- overall_density + comp_density
    
    lines(xx, comp_density, col = colors[i %% length(colors) + 1], 
          lty = 2, lwd = 2)
  }
  
  # Plot overall density
  lines(xx, overall_density, col = "black", lwd = 3)
  
  # Add legend
  legend("topright", 
         legend = c("Overall", paste("Component", seq_len(x$k))),
         col = c("black", colors[seq_len(x$k) %% length(colors) + 1]),
         lty = c(1, rep(2, x$k)),
         lwd = c(3, rep(2, x$k)))
  
  invisible(x)
}