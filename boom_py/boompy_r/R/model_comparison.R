#' Compare BOOM Models
#' 
#' Compare multiple BOOM models using information criteria.
#' 
#' @param ... boom_model objects to compare
#' @param criterion Criterion to use ("aic", "bic", or "both")
#' 
#' @return Data frame with comparison results
#' @export
#' 
#' @examples
#' \dontrun{
#' model1 <- boom_lm(y ~ x1, data)
#' model2 <- boom_lm(y ~ x1 + x2, data)
#' model3 <- boom_lm(y ~ x1 + x2 + x3, data)
#' 
#' boom_compare_models(model1, model2, model3)
#' }
boom_compare_models <- function(..., criterion = "both") {
  models <- list(...)
  n_models <- length(models)
  
  # Check that all are boom models
  if (!all(sapply(models, inherits, "boom_model"))) {
    stop("All arguments must be boom_model objects")
  }
  
  # Extract model names
  model_names <- as.character(match.call())[-1]
  if (length(model_names) > n_models) {
    model_names <- model_names[seq_len(n_models)]
  }
  
  # Initialize results
  results <- data.frame(
    Model = model_names,
    LogLik = numeric(n_models),
    Df = integer(n_models),
    AIC = numeric(n_models),
    BIC = numeric(n_models),
    stringsAsFactors = FALSE
  )
  
  # Get sample size (assume all models have same n)
  n_obs <- length(models[[1]]$residuals)
  
  # Compute criteria for each model
  for (i in seq_len(n_models)) {
    model <- models[[i]]
    
    # Log likelihood
    loglik <- logLik(model)
    results$LogLik[i] <- as.numeric(loglik)
    results$Df[i] <- attr(loglik, "df")
    
    # AIC and BIC
    results$AIC[i] <- boom_aic(model)
    results$BIC[i] <- boom_bic(model)
  }
  
  # Add delta AIC and BIC
  results$dAIC <- results$AIC - min(results$AIC)
  results$dBIC <- results$BIC - min(results$BIC)
  
  # Add Akaike weights
  results$AIC_weight <- exp(-0.5 * results$dAIC) / sum(exp(-0.5 * results$dAIC))
  
  # Sort by AIC
  results <- results[order(results$AIC), ]
  
  # Print based on criterion
  if (criterion == "aic") {
    results <- results[, c("Model", "LogLik", "Df", "AIC", "dAIC", "AIC_weight")]
  } else if (criterion == "bic") {
    results <- results[, c("Model", "LogLik", "Df", "BIC", "dBIC")]
  }
  
  class(results) <- c("boom_model_comparison", "data.frame")
  results
}

#' @export
print.boom_model_comparison <- function(x, digits = 3, ...) {
  cat("\nModel Comparison\n")
  cat("================\n")
  print.data.frame(x, digits = digits, row.names = FALSE)
  
  # Identify best model
  if ("AIC" %in% names(x)) {
    best_model <- x$Model[1]  # Already sorted by AIC
    cat("\nBest model by AIC:", best_model, "\n")
  }
  
  invisible(x)
}

#' Calculate AIC for BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return AIC value
#' @export
boom_aic <- function(object, ...) {
  UseMethod("boom_aic")
}

#' @export
boom_aic.boom_model <- function(object, ...) {
  loglik <- logLik(object)
  -2 * as.numeric(loglik) + 2 * attr(loglik, "df")
}

#' Calculate BIC for BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return BIC value
#' @export
boom_bic <- function(object, ...) {
  UseMethod("boom_bic")
}

#' @export
boom_bic.boom_model <- function(object, ...) {
  loglik <- logLik(object)
  n <- attr(loglik, "nobs")
  -2 * as.numeric(loglik) + log(n) * attr(loglik, "df")
}

#' Extract AIC from boom_model
#' @export
AIC.boom_model <- function(object, ...) {
  boom_aic(object)
}

#' Extract BIC from boom_model
#' @export
BIC.boom_model <- function(object, ...) {
  boom_bic(object)
}