#' BOOM MCMC Sampling
#' 
#' Run MCMC sampling using the BOOM Python backend.
#' 
#' @param log_density Function that computes log density
#' @param initial Initial parameter values
#' @param niter Number of MCMC iterations
#' @param burn Burn-in period
#' @param method MCMC method ("metropolis" or "slice")
#' @param ... Additional arguments passed to the sampler
#' 
#' @return An object of class "boom_mcmc" containing the samples
#' @export
#' 
#' @examples
#' \dontrun{
#' # Sample from a bivariate normal
#' log_density <- function(x) {
#'   -0.5 * sum(x^2)  # Standard normal
#' }
#' 
#' samples <- boom_mcmc(log_density, initial = c(0, 0), niter = 1000)
#' plot(samples)
#' }
boom_mcmc <- function(log_density, initial, niter = 1000, burn = 500,
                     method = "metropolis", ...) {
  # Ensure BOOM is loaded
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  # Convert R function to Python-compatible function
  py_log_density <- reticulate::py_func(function(x) {
    log_density(as.numeric(x))
  })
  
  # Convert initial values
  initial_vec <- boom_env$Vector(initial)
  
  # Create sampler based on method
  if (method == "metropolis") {
    # Default proposal covariance
    d <- length(initial)
    proposal_cov <- boom_env$Matrix(diag(0.1, d))
    
    sampler <- boom_env$boom$samplers$MetropolisHastings(
      log_density_func = py_log_density,
      proposal_covariance = proposal_cov
    )
  } else if (method == "slice") {
    sampler <- boom_env$boom$samplers$SliceSampler(
      log_density_func = py_log_density
    )
  } else {
    stop("Unknown MCMC method: ", method)
  }
  
  # Run sampling
  samples_list <- sampler$sample(
    n_samples = as.integer(niter),
    initial_point = initial_vec,
    burn_in = as.integer(burn)
  )
  
  # Convert samples to R matrix
  n_kept <- niter - burn
  d <- length(initial)
  samples <- matrix(0, nrow = n_kept, ncol = d)
  
  for (i in seq_len(n_kept)) {
    samples[i, ] <- as.numeric(samples_list[[i-1]])  # Python 0-indexed
  }
  
  colnames(samples) <- paste0("param", seq_len(d))
  
  # Create result object
  result <- structure(
    list(
      samples = samples,
      niter = niter,
      burn = burn,
      method = method,
      acceptance_rate = sampler$acceptance_rate,
      call = match.call()
    ),
    class = "boom_mcmc"
  )
  
  result
}

#' @export
print.boom_mcmc <- function(x, ...) {
  cat("\nBOOM MCMC Results\n")
  cat("Method:", x$method, "\n")
  cat("Iterations:", x$niter, "(burn-in:", x$burn, ")\n")
  cat("Samples:", nrow(x$samples), "x", ncol(x$samples), "\n")
  
  if (!is.null(x$acceptance_rate)) {
    cat("Acceptance rate:", round(x$acceptance_rate, 3), "\n")
  }
  
  cat("\nPosterior means:\n")
  print(colMeans(x$samples))
  
  invisible(x)
}

#' @export
summary.boom_mcmc <- function(object, ...) {
  # Compute posterior summaries
  posterior_mean <- colMeans(object$samples)
  posterior_sd <- apply(object$samples, 2, sd)
  posterior_quantiles <- t(apply(object$samples, 2, quantile, 
                                probs = c(0.025, 0.25, 0.5, 0.75, 0.975)))
  
  # Effective sample size (simple version)
  ess <- numeric(ncol(object$samples))
  for (i in seq_len(ncol(object$samples))) {
    # Compute autocorrelation
    acf_vals <- acf(object$samples[, i], plot = FALSE)$acf
    # Simple ESS estimate
    ess[i] <- nrow(object$samples) / (1 + 2 * sum(acf_vals[-1]))
  }
  
  # Create summary table
  summary_table <- cbind(
    Mean = posterior_mean,
    SD = posterior_sd,
    posterior_quantiles,
    ESS = round(ess)
  )
  
  cat("\nBOOM MCMC Summary\n")
  cat("================\n")
  print(summary_table)
  
  invisible(list(
    summary = summary_table,
    samples = object$samples
  ))
}

#' @export
plot.boom_mcmc <- function(x, type = c("trace", "density", "pairs"), ...) {
  type <- match.arg(type)
  
  n_params <- ncol(x$samples)
  
  if (type == "trace") {
    # Trace plots
    oldpar <- par(mfrow = c(n_params, 1))
    on.exit(par(oldpar))
    
    for (i in seq_len(n_params)) {
      plot(x$samples[, i], type = "l",
           main = paste("Trace plot:", colnames(x$samples)[i]),
           xlab = "Iteration", ylab = "Value")
    }
    
  } else if (type == "density") {
    # Density plots
    oldpar <- par(mfrow = c(n_params, 1))
    on.exit(par(oldpar))
    
    for (i in seq_len(n_params)) {
      plot(density(x$samples[, i]),
           main = paste("Density:", colnames(x$samples)[i]),
           xlab = "Value")
    }
    
  } else if (type == "pairs") {
    # Pairs plot
    if (n_params > 1) {
      pairs(x$samples, ...)
    } else {
      hist(x$samples[, 1], main = colnames(x$samples)[1], xlab = "Value")
    }
  }
  
  invisible(x)
}

#' BOOM Slice Sampler
#' 
#' Convenience function for slice sampling.
#' 
#' @param log_density Log density function
#' @param initial Initial values
#' @param niter Number of iterations
#' @param burn Burn-in period
#' @param ... Additional arguments
#' 
#' @return boom_mcmc object
#' @export
boom_slice_sampler <- function(log_density, initial, niter = 1000, 
                              burn = 500, ...) {
  boom_mcmc(log_density, initial, niter, burn, method = "slice", ...)
}