#' Print Method for BOOM Models
#' 
#' @param x A boom_model object
#' @param ... Additional arguments (ignored)
#' @return Invisible NULL
#' @export
print.boom_model <- function(x, ...) {
  cat("\nCall:\n")
  print(x$call)
  
  cat("\nCoefficients:\n")
  print(x$coefficients)
  
  if (x$model_type == "linear") {
    cat("\nResidual standard error:", round(x$sigma, 4), 
        "on", x$df.residual, "degrees of freedom\n")
  }
  
  invisible(x)
}

#' Summary Method for BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return An object of class "summary.boom_model"
#' @export
summary.boom_model <- function(object, ...) {
  # Calculate standard errors (would need Hessian from Python model)
  # For now, use approximate standard errors
  n <- length(object$residuals)
  p <- length(object$coefficients)
  
  if (object$model_type == "linear") {
    # For linear regression, calculate standard errors
    X <- model.matrix(object$formula, object$data)
    XtX_inv <- solve(t(X) %*% X)
    se <- sqrt(diag(XtX_inv) * object$sigma^2)
    
    t_values <- object$coefficients / se
    p_values <- 2 * pt(abs(t_values), df = object$df.residual, lower.tail = FALSE)
    
    coefficients <- cbind(
      Estimate = object$coefficients,
      `Std. Error` = se,
      `t value` = t_values,
      `Pr(>|t|)` = p_values
    )
  } else {
    # For GLMs, we would need the Hessian
    coefficients <- cbind(
      Estimate = object$coefficients
    )
  }
  
  # Calculate R-squared for linear models
  if (object$model_type == "linear") {
    y <- model.response(model.frame(object$formula, object$data))
    ss_tot <- sum((y - mean(y))^2)
    ss_res <- sum(object$residuals^2)
    r_squared <- 1 - ss_res / ss_tot
    adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p)
  } else {
    r_squared <- adj_r_squared <- NULL
  }
  
  result <- list(
    call = object$call,
    model_type = object$model_type,
    coefficients = coefficients,
    sigma = object$sigma,
    df = c(p, object$df.residual, p),
    r.squared = r_squared,
    adj.r.squared = adj_r_squared,
    residuals = object$residuals
  )
  
  class(result) <- "summary.boom_model"
  result
}

#' @export
print.summary.boom_model <- function(x, ...) {
  cat("\nCall:\n")
  print(x$call)
  
  cat("\nResiduals:\n")
  print(summary(x$residuals))
  
  cat("\nCoefficients:\n")
  printCoefmat(x$coefficients)
  
  if (x$model_type == "linear") {
    cat("\nResidual standard error:", round(x$sigma, 4), 
        "on", x$df[2], "degrees of freedom\n")
    
    if (!is.null(x$r.squared)) {
      cat("Multiple R-squared: ", round(x$r.squared, 4), 
          ",\tAdjusted R-squared: ", round(x$adj.r.squared, 4), "\n", sep = "")
    }
  }
  
  invisible(x)
}

#' Plot Method for BOOM Models
#' 
#' @param x A boom_model object
#' @param which Which plots to produce (1-4)
#' @param ... Additional graphical parameters
#' @export
plot.boom_model <- function(x, which = 1:4, ...) {
  # Set up plotting area
  if (length(which) > 1) {
    oldpar <- par(mfrow = c(2, 2))
    on.exit(par(oldpar))
  }
  
  # Residuals vs Fitted
  if (1 %in% which) {
    plot(x$fitted.values, x$residuals,
         xlab = "Fitted values", ylab = "Residuals",
         main = "Residuals vs Fitted")
    abline(h = 0, lty = 2, col = "gray")
    lines(lowess(x$fitted.values, x$residuals), col = "red")
  }
  
  # Q-Q plot
  if (2 %in% which) {
    qqnorm(x$residuals, main = "Normal Q-Q")
    qqline(x$residuals, col = "red")
  }
  
  # Scale-Location
  if (3 %in% which) {
    sqrt_abs_resid <- sqrt(abs(x$residuals))
    plot(x$fitted.values, sqrt_abs_resid,
         xlab = "Fitted values", ylab = expression(sqrt("|Residuals|")),
         main = "Scale-Location")
    lines(lowess(x$fitted.values, sqrt_abs_resid), col = "red")
  }
  
  # Residuals vs Leverage (would need hat matrix)
  if (4 %in% which && x$model_type == "linear") {
    # Simple histogram of residuals for now
    hist(x$residuals, main = "Histogram of Residuals", 
         xlab = "Residuals", probability = TRUE)
    curve(dnorm(x, mean = 0, sd = sd(x$residuals)), 
          add = TRUE, col = "red", lwd = 2)
  }
  
  invisible(x)
}

#' Predict Method for BOOM Models
#' 
#' @param object A boom_model object
#' @param newdata New data frame for predictions
#' @param type Type of prediction ("response" or "link")
#' @param ... Additional arguments
#' @return Vector of predictions
#' @export
predict.boom_model <- function(object, newdata = NULL, type = "response", ...) {
  # If no new data, return fitted values
  if (is.null(newdata)) {
    if (type == "response") {
      return(object$fitted.values)
    } else {
      # For link, need to transform back
      if (object$model_type == "logistic") {
        return(qlogis(object$fitted.values))
      } else if (object$model_type == "poisson") {
        return(log(object$fitted.values))
      } else {
        return(object$fitted.values)
      }
    }
  }
  
  # Prepare new data
  Terms <- delete.response(object$terms)
  X_new <- model.matrix(Terms, newdata)
  n_new <- nrow(X_new)
  
  predictions <- numeric(n_new)
  
  # Make predictions based on model type
  if (object$model_type == "linear") {
    for (i in seq_len(n_new)) {
      predictions[i] <- sum(object$coefficients * X_new[i, ])
    }
  } else if (object$model_type == "logistic") {
    for (i in seq_len(n_new)) {
      x_vec <- boom_env$Vector(X_new[i, ])
      if (type == "response") {
        predictions[i] <- object$py_model$predict_probability(x_vec)
      } else {
        linear_pred <- sum(object$coefficients * X_new[i, ])
        predictions[i] <- linear_pred
      }
    }
  } else if (object$model_type == "poisson") {
    for (i in seq_len(n_new)) {
      linear_pred <- sum(object$coefficients * X_new[i, ])
      if (type == "response") {
        predictions[i] <- exp(linear_pred)
      } else {
        predictions[i] <- linear_pred
      }
    }
  }
  
  predictions
}

#' Extract Coefficients from BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return Named vector of coefficients
#' @export
coef.boom_model <- function(object, ...) {
  object$coefficients
}

#' Extract Residuals from BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return Vector of residuals
#' @export
residuals.boom_model <- function(object, ...) {
  object$residuals
}

#' Extract Log-Likelihood from BOOM Models
#' 
#' @param object A boom_model object
#' @param ... Additional arguments
#' @return Log-likelihood value with degrees of freedom attribute
#' @export
logLik.boom_model <- function(object, ...) {
  # Get log-likelihood from Python model
  loglik <- object$py_model$loglike()
  
  # Number of parameters
  npar <- length(object$coefficients)
  if (object$model_type == "linear") {
    npar <- npar + 1  # Add sigma parameter
  }
  
  # Return with attributes
  structure(loglik, 
            df = npar,
            nobs = length(object$residuals),
            class = "logLik")
}