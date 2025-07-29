\name{boom_lm}
\alias{boom_lm}
\title{BOOM Linear Regression}
\description{
  Fit a linear regression model using the BOOM Python backend.
}
\usage{
boom_lm(formula, data, prior = NULL, ...)
}
\arguments{
  \item{formula}{An R formula specifying the model.}
  \item{data}{A data frame containing the variables.}
  \item{prior}{Prior specification (currently unused, for future implementation).}
  \item{...}{Additional arguments passed to the model.}
}
\value{
  An object of class \code{"boom_model"} containing:
  \item{py_model}{The underlying Python model object}
  \item{coefficients}{Named vector of regression coefficients}
  \item{sigma}{Residual standard deviation}
  \item{fitted.values}{Fitted values}
  \item{residuals}{Residuals}
  \item{formula}{The model formula}
  \item{call}{The matched call}
}
\details{
  \code{boom_lm} provides an interface to BOOM's Python implementation of
  Bayesian linear regression. The model is fitted using maximum likelihood
  estimation (MLE) by default, with future support planned for full
  Bayesian inference with customizable priors.
  
  The model assumes:
  \deqn{y = X\beta + \epsilon}
  where \eqn{\epsilon \sim N(0, \sigma^2)}.
}
\examples{
\dontrun{
# Generate example data
set.seed(123)
n <- 100
x <- rnorm(n)
y <- 2 + 3*x + rnorm(n)
data <- data.frame(y = y, x = x)

# Fit model
model <- boom_lm(y ~ x, data)
summary(model)

# Diagnostic plots
plot(model)

# Predictions
newdata <- data.frame(x = c(-1, 0, 1))
predict(model, newdata)

# Model comparison
model1 <- boom_lm(y ~ 1, data)
model2 <- boom_lm(y ~ x, data)
boom_compare_models(model1, model2)
}
}
\seealso{
  \code{\link{boom_glm}}, \code{\link{boom_logit}}, \code{\link{summary.boom_model}},
  \code{\link{predict.boom_model}}, \code{\link{plot.boom_model}}
}
\author{
  Claude Assistant
}