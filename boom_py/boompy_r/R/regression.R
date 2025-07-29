#' BOOM Linear Regression
#' 
#' Fit a linear regression model using the BOOM Python backend.
#' 
#' @param formula An R formula specifying the model
#' @param data A data frame containing the variables
#' @param prior Prior specification (currently unused, for future implementation)
#' @param ... Additional arguments passed to the model
#' 
#' @return An object of class "boom_model" containing the fitted model
#' @export
#' 
#' @examples
#' \dontrun{
#' # Generate example data
#' n <- 100
#' x <- rnorm(n)
#' y <- 2 + 3*x + rnorm(n)
#' data <- data.frame(y = y, x = x)
#' 
#' # Fit model
#' model <- boom_lm(y ~ x, data)
#' summary(model)
#' }
boom_lm <- function(formula, data, prior = NULL, ...) {
  # Ensure BOOM is loaded
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  # Parse formula and prepare data
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)
  
  # Create Python model
  py_model <- boom_env$boom$models$glm$RegressionModel()
  
  # Add data to model
  n <- nrow(X)
  for (i in seq_len(n)) {
    x_vec <- boom_env$Vector(X[i, ])
    py_model$add_data(reticulate::tuple(y[i], x_vec))
  }
  
  # Fit model using MLE
  py_model$mle()
  
  # Extract results
  coefficients <- as.numeric(py_model$beta)
  names(coefficients) <- colnames(X)
  
  # Create R object
  result <- structure(
    list(
      py_model = py_model,
      model_type = "linear",
      formula = formula,
      terms = terms(formula),
      data = data,
      coefficients = coefficients,
      sigma = py_model$sigma,
      df.residual = n - ncol(X),
      fitted.values = numeric(n),
      residuals = numeric(n),
      call = match.call()
    ),
    class = c("boom_lm", "boom_model")
  )
  
  # Calculate fitted values and residuals
  for (i in seq_len(n)) {
    x_vec <- boom_env$Vector(X[i, ])
    result$fitted.values[i] <- sum(coefficients * X[i, ])
    result$residuals[i] <- y[i] - result$fitted.values[i]
  }
  
  result
}

#' BOOM Generalized Linear Model
#' 
#' Fit a generalized linear model using the BOOM Python backend.
#' 
#' @param formula An R formula specifying the model
#' @param family A family object or character string naming the family
#' @param data A data frame containing the variables
#' @param ... Additional arguments
#' 
#' @return An object of class "boom_model" containing the fitted model
#' @export
boom_glm <- function(formula, family = "gaussian", data, ...) {
  # Parse family
  if (is.character(family)) {
    family_name <- family
  } else if (inherits(family, "family")) {
    family_name <- family$family
  } else {
    stop("family must be a character string or family object")
  }
  
  # Dispatch to appropriate model
  if (family_name == "gaussian") {
    boom_lm(formula, data, ...)
  } else if (family_name == "binomial") {
    boom_logit(formula, data, ...)
  } else if (family_name == "poisson") {
    boom_poisson(formula, data, ...)
  } else {
    stop("Unsupported family: ", family_name)
  }
}

#' BOOM Logistic Regression
#' 
#' Fit a logistic regression model using the BOOM Python backend.
#' 
#' @param formula An R formula specifying the model
#' @param data A data frame containing the variables
#' @param ... Additional arguments
#' 
#' @return An object of class "boom_model" containing the fitted model
#' @export
boom_logit <- function(formula, data, ...) {
  # Ensure BOOM is loaded
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  # Parse formula and prepare data
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)
  
  # Convert y to 0/1 if necessary
  if (is.factor(y)) {
    y <- as.numeric(y) - 1
  }
  
  # Create Python model
  py_model <- boom_env$boom$models$glm$LogisticRegressionModel()
  
  # Add data to model
  n <- nrow(X)
  for (i in seq_len(n)) {
    x_vec <- boom_env$Vector(X[i, ])
    # Python expects integer for binary outcome
    py_model$add_data(reticulate::tuple(as.integer(y[i]), x_vec))
  }
  
  # Fit model using MLE
  py_model$mle()
  
  # Extract results
  coefficients <- as.numeric(py_model$beta)
  names(coefficients) <- colnames(X)
  
  # Create R object
  result <- structure(
    list(
      py_model = py_model,
      model_type = "logistic",
      formula = formula,
      terms = terms(formula),
      data = data,
      coefficients = coefficients,
      df.residual = n - ncol(X),
      fitted.values = numeric(n),
      residuals = numeric(n),
      call = match.call()
    ),
    class = c("boom_logit", "boom_model")
  )
  
  # Calculate fitted values (probabilities)
  for (i in seq_len(n)) {
    x_vec <- boom_env$Vector(X[i, ])
    result$fitted.values[i] <- py_model$predict_probability(x_vec)
    result$residuals[i] <- y[i] - result$fitted.values[i]
  }
  
  result
}

#' BOOM Poisson Regression
#' 
#' Fit a Poisson regression model using the BOOM Python backend.
#' 
#' @param formula An R formula specifying the model
#' @param data A data frame containing the variables
#' @param ... Additional arguments
#' 
#' @return An object of class "boom_model" containing the fitted model
#' @export
boom_poisson <- function(formula, data, ...) {
  # Ensure BOOM is loaded
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  # Parse formula and prepare data
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, data)
  
  # Create Python model
  py_model <- boom_env$boom$models$glm$PoissonRegressionModel()
  
  # Add data to model
  n <- nrow(X)
  for (i in seq_len(n)) {
    x_vec <- boom_env$Vector(X[i, ])
    py_model$add_data(reticulate::tuple(as.integer(y[i]), x_vec))
  }
  
  # Fit model using MLE
  py_model$mle()
  
  # Extract results
  coefficients <- as.numeric(py_model$beta)
  names(coefficients) <- colnames(X)
  
  # Create R object
  result <- structure(
    list(
      py_model = py_model,
      model_type = "poisson",
      formula = formula,
      terms = terms(formula),
      data = data,
      coefficients = coefficients,
      df.residual = n - ncol(X),
      fitted.values = numeric(n),
      residuals = numeric(n),
      call = match.call()
    ),
    class = c("boom_poisson", "boom_model")
  )
  
  # Calculate fitted values
  for (i in seq_len(n)) {
    linear_pred <- sum(coefficients * X[i, ])
    result$fitted.values[i] <- exp(linear_pred)
    result$residuals[i] <- y[i] - result$fitted.values[i]
  }
  
  result
}