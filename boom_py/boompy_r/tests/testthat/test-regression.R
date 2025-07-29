test_that("boom_lm fits linear regression correctly", {
  # Skip if Python/BOOM not available
  skip_if_not(boom_check_installation())
  
  # Generate test data
  set.seed(123)
  n <- 100
  x <- rnorm(n)
  y <- 2 + 3*x + rnorm(n, sd = 0.5)
  data <- data.frame(y = y, x = x)
  
  # Fit model
  model <- boom_lm(y ~ x, data)
  
  # Check model structure
  expect_s3_class(model, "boom_model")
  expect_s3_class(model, "boom_lm")
  
  # Check coefficients are reasonable
  coefs <- coef(model)
  expect_equal(length(coefs), 2)
  expect_true(abs(coefs[1] - 2) < 0.5)  # Intercept near 2
  expect_true(abs(coefs[2] - 3) < 0.5)  # Slope near 3
  
  # Check residuals
  resids <- residuals(model)
  expect_equal(length(resids), n)
  expect_true(abs(mean(resids)) < 0.1)  # Mean near 0
  
  # Check predictions
  preds <- predict(model)
  expect_equal(length(preds), n)
})

test_that("boom_logit fits logistic regression correctly", {
  skip_if_not(boom_check_installation())
  
  # Generate test data
  set.seed(456)
  n <- 200
  x <- rnorm(n)
  prob <- 1 / (1 + exp(-(1 + 2*x)))
  y <- rbinom(n, 1, prob)
  data <- data.frame(y = y, x = x)
  
  # Fit model
  model <- boom_logit(y ~ x, data)
  
  # Check model structure
  expect_s3_class(model, "boom_model")
  expect_s3_class(model, "boom_logit")
  
  # Check coefficients are positive (should be near 1 and 2)
  coefs <- coef(model)
  expect_true(coefs[1] > 0)  # Intercept
  expect_true(coefs[2] > 0)  # Slope
  
  # Check fitted values are probabilities
  fitted <- model$fitted.values
  expect_true(all(fitted >= 0 & fitted <= 1))
})

test_that("boom_glm dispatches correctly", {
  skip_if_not(boom_check_installation())
  
  # Generate test data
  set.seed(789)
  n <- 50
  x <- rnorm(n)
  y <- 2 + 3*x + rnorm(n)
  data <- data.frame(y = y, x = x)
  
  # Test Gaussian family
  model_gaussian <- boom_glm(y ~ x, family = "gaussian", data)
  expect_s3_class(model_gaussian, "boom_lm")
  
  # Test with family object
  model_gaussian2 <- boom_glm(y ~ x, family = gaussian(), data)
  expect_s3_class(model_gaussian2, "boom_lm")
})

test_that("predict method works with new data", {
  skip_if_not(boom_check_installation())
  
  # Generate test data
  set.seed(321)
  n <- 80
  x <- rnorm(n)
  y <- 1 + 2*x + rnorm(n, sd = 0.3)
  data <- data.frame(y = y, x = x)
  
  # Fit model
  model <- boom_lm(y ~ x, data)
  
  # New data for prediction
  newdata <- data.frame(x = c(-1, 0, 1))
  preds <- predict(model, newdata)
  
  expect_equal(length(preds), 3)
  # Check predictions are ordered correctly (since slope is positive)
  expect_true(preds[1] < preds[2])
  expect_true(preds[2] < preds[3])
})

test_that("summary method provides correct output", {
  skip_if_not(boom_check_installation())
  
  # Generate test data
  set.seed(654)
  n <- 100
  x <- rnorm(n)
  y <- 2 + 3*x + rnorm(n)
  data <- data.frame(y = y, x = x)
  
  # Fit model and get summary
  model <- boom_lm(y ~ x, data)
  s <- summary(model)
  
  expect_s3_class(s, "summary.boom_model")
  expect_true("coefficients" %in% names(s))
  expect_true("r.squared" %in% names(s))
  expect_true(s$r.squared > 0.5)  # Should have decent R-squared
})