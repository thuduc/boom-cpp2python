"""Statistical functions and utilities for BOOM."""

from .descriptive import (
    mean, variance, standard_deviation, skewness, kurtosis,
    quantile, median, mode, range_stat, iqr,
    covariance, correlation, autocorrelation
)
from .hypothesis_testing import (
    t_test_one_sample, t_test_two_sample, paired_t_test,
    chi_square_test, kolmogorov_smirnov_test, anderson_darling_test,
    shapiro_wilk_test, jarque_bera_test
)
from .regression_diagnostics import (
    residual_analysis, durbin_watson, breusch_pagan, white_test,
    cook_distance, leverage, studentized_residuals,
    vif, condition_index
)
from .information_criteria import (
    aic, bic, aicc, dic, waic, 
    log_marginal_likelihood_harmonic_mean, log_marginal_likelihood_bridge_sampling,
    model_weights, information_criterion_comparison, cross_validation_score, leave_one_out_cv
)
from .model_selection import (
    cross_validation, bootstrap, jackknife,
    model_comparison, bayes_factor
)

__all__ = [
    # Descriptive statistics
    "mean", "variance", "standard_deviation", "skewness", "kurtosis",
    "quantile", "median", "mode", "range_stat", "iqr",
    "covariance", "correlation", "autocorrelation",
    
    # Hypothesis testing
    "t_test_one_sample", "t_test_two_sample", "paired_t_test",
    "chi_square_test", "kolmogorov_smirnov_test", "anderson_darling_test",
    "shapiro_wilk_test", "jarque_bera_test",
    
    # Regression diagnostics
    "residual_analysis", "durbin_watson", "breusch_pagan", "white_test",
    "cook_distance", "leverage", "studentized_residuals",
    "vif", "condition_index",
    
    # Information criteria
    "aic", "bic", "aicc", "dic", "waic", 
    "log_marginal_likelihood_harmonic_mean", "log_marginal_likelihood_bridge_sampling",
    "model_weights", "information_criterion_comparison", "cross_validation_score", "leave_one_out_cv",
    
    # Model selection
    "cross_validation", "bootstrap", "jackknife",
    "model_comparison", "bayes_factor"
]