"""R Math functions using SciPy - Python equivalent of Rmath_dist.hpp."""

from typing import Optional, Union, List
import numpy as np
import scipy.stats as stats
import scipy.special as special
from .rng import RNG, GlobalRng


# ============================================================================
# Normal Distribution
# ============================================================================

def dnorm(x: float, mu: float = 0.0, sig: float = 1.0, log: bool = False) -> float:
    """Normal density function."""
    if log:
        return stats.norm.logpdf(x, loc=mu, scale=sig)
    else:
        return stats.norm.pdf(x, loc=mu, scale=sig)


def pnorm(x: float, mu: float = 0.0, sig: float = 1.0, 
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Normal cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.norm.logcdf(x, loc=mu, scale=sig)
        else:
            return stats.norm.cdf(x, loc=mu, scale=sig)
    else:
        if log_p:
            return stats.norm.logsf(x, loc=mu, scale=sig)
        else:
            return stats.norm.sf(x, loc=mu, scale=sig)


def qnorm(p: float, mu: float = 0.0, sig: float = 1.0,
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Normal quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.norm.ppf(p, loc=mu, scale=sig)
    else:
        return stats.norm.isf(p, loc=mu, scale=sig)


def rnorm(mu: float = 0.0, sig: float = 1.0, rng: Optional[RNG] = None) -> float:
    """Generate normal random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.normal(mu, sig)


# ============================================================================
# Uniform Distribution  
# ============================================================================

def dunif(x: float, lo: float = 0.0, hi: float = 1.0, log: bool = False) -> float:
    """Uniform density function."""
    if log:
        return stats.uniform.logpdf(x, loc=lo, scale=hi-lo)
    else:
        return stats.uniform.pdf(x, loc=lo, scale=hi-lo)


def punif(x: float, lo: float = 0.0, hi: float = 1.0,
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Uniform cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.uniform.logcdf(x, loc=lo, scale=hi-lo)
        else:
            return stats.uniform.cdf(x, loc=lo, scale=hi-lo)
    else:
        if log_p:
            return stats.uniform.logsf(x, loc=lo, scale=hi-lo)
        else:
            return stats.uniform.sf(x, loc=lo, scale=hi-lo)


def qunif(p: float, lo: float = 0.0, hi: float = 1.0,
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Uniform quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.uniform.ppf(p, loc=lo, scale=hi-lo)
    else:
        return stats.uniform.isf(p, loc=lo, scale=hi-lo)


def runif(lo: float = 0.0, hi: float = 1.0, rng: Optional[RNG] = None) -> float:
    """Generate uniform random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.uniform(lo, hi)


# ============================================================================
# Gamma Distribution
# ============================================================================

def dgamma(x: float, shape: float, scale: float = 1.0, log: bool = False) -> float:
    """Gamma density function."""
    if log:
        return stats.gamma.logpdf(x, a=shape, scale=scale)
    else:
        return stats.gamma.pdf(x, a=shape, scale=scale)


def pgamma(x: float, shape: float, scale: float = 1.0,
           lower_tail: bool = True, log_p: bool = False) -> float:
    """Gamma cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.gamma.logcdf(x, a=shape, scale=scale)
        else:
            return stats.gamma.cdf(x, a=shape, scale=scale)
    else:
        if log_p:
            return stats.gamma.logsf(x, a=shape, scale=scale)
        else:
            return stats.gamma.sf(x, a=shape, scale=scale)


def qgamma(p: float, shape: float, scale: float = 1.0,
           lower_tail: bool = True, log_p: bool = False) -> float:
    """Gamma quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.gamma.ppf(p, a=shape, scale=scale)
    else:
        return stats.gamma.isf(p, a=shape, scale=scale)


def rgamma(shape: float, scale: float = 1.0, rng: Optional[RNG] = None) -> float:
    """Generate gamma random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.gamma(shape, scale)


# ============================================================================
# Beta Distribution
# ============================================================================

def dbeta(x: float, alpha: float, beta: float, log: bool = False) -> float:
    """Beta density function."""
    if log:
        return stats.beta.logpdf(x, a=alpha, b=beta)
    else:
        return stats.beta.pdf(x, a=alpha, b=beta)


def pbeta(x: float, alpha: float, beta: float,
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Beta cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.beta.logcdf(x, a=alpha, b=beta)
        else:
            return stats.beta.cdf(x, a=alpha, b=beta)
    else:
        if log_p:
            return stats.beta.logsf(x, a=alpha, b=beta)
        else:
            return stats.beta.sf(x, a=alpha, b=beta)


def qbeta(p: float, alpha: float, beta: float,
          lower_tail: bool = True, log_p: bool = False) -> float:
    """Beta quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.beta.ppf(p, a=alpha, b=beta)
    else:
        return stats.beta.isf(p, a=alpha, b=beta)


def rbeta(alpha: float, beta: float, rng: Optional[RNG] = None) -> float:
    """Generate beta random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.beta(alpha, beta)


# ============================================================================
# Chi-square Distribution
# ============================================================================

def dchisq(x: float, df: float, log: bool = False) -> float:
    """Chi-square density function."""
    if log:
        return stats.chi2.logpdf(x, df=df)
    else:
        return stats.chi2.pdf(x, df=df)


def pchisq(x: float, df: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Chi-square cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.chi2.logcdf(x, df=df)
        else:
            return stats.chi2.cdf(x, df=df)
    else:
        if log_p:
            return stats.chi2.logsf(x, df=df)
        else:
            return stats.chi2.sf(x, df=df)


def qchisq(p: float, df: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Chi-square quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.chi2.ppf(p, df=df)
    else:
        return stats.chi2.isf(p, df=df)


def rchisq(df: float, rng: Optional[RNG] = None) -> float:
    """Generate chi-square random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.chi_square(df)


# ============================================================================
# Student's t Distribution
# ============================================================================

def dt(x: float, df: float, log: bool = False) -> float:
    """Student's t density function."""
    if log:
        return stats.t.logpdf(x, df=df)
    else:
        return stats.t.pdf(x, df=df)


def pt(x: float, df: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Student's t cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.t.logcdf(x, df=df)
        else:
            return stats.t.cdf(x, df=df)
    else:
        if log_p:
            return stats.t.logsf(x, df=df)
        else:
            return stats.t.sf(x, df=df)


def qt(p: float, df: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Student's t quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.t.ppf(p, df=df)
    else:
        return stats.t.isf(p, df=df)


def rt(df: float, rng: Optional[RNG] = None) -> float:
    """Generate Student's t random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.student_t(df)


# ============================================================================
# F Distribution
# ============================================================================

def df_dist(x: float, dfn: float, dfd: float, log: bool = False) -> float:
    """F density function."""
    if log:
        return stats.f.logpdf(x, dfn=dfn, dfd=dfd)
    else:
        return stats.f.pdf(x, dfn=dfn, dfd=dfd)


def pf(x: float, dfn: float, dfd: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """F cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.f.logcdf(x, dfn=dfn, dfd=dfd)
        else:
            return stats.f.cdf(x, dfn=dfn, dfd=dfd)
    else:
        if log_p:
            return stats.f.logsf(x, dfn=dfn, dfd=dfd)
        else:
            return stats.f.sf(x, dfn=dfn, dfd=dfd)


def qf(p: float, dfn: float, dfd: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """F quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return stats.f.ppf(p, dfn=dfn, dfd=dfd)
    else:
        return stats.f.isf(p, dfn=dfn, dfd=dfd)


def rf(dfn: float, dfd: float, rng: Optional[RNG] = None) -> float:
    """Generate F random number."""
    if rng is None:
        rng = GlobalRng.rng
    return stats.f.rvs(dfn=dfn, dfd=dfd, random_state=rng.generator)


# ============================================================================
# Binomial Distribution
# ============================================================================

def dbinom(x: int, n: int, p: float, log: bool = False) -> float:
    """Binomial probability mass function."""
    if log:
        return stats.binom.logpmf(x, n=n, p=p)
    else:
        return stats.binom.pmf(x, n=n, p=p)


def pbinom(x: int, n: int, p: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Binomial cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.binom.logcdf(x, n=n, p=p)
        else:
            return stats.binom.cdf(x, n=n, p=p)
    else:
        if log_p:
            return stats.binom.logsf(x, n=n, p=p)
        else:
            return stats.binom.sf(x, n=n, p=p)


def qbinom(prob: float, n: int, p: float, lower_tail: bool = True, log_p: bool = False) -> int:
    """Binomial quantile function."""
    if log_p:
        prob = np.exp(prob)
    
    if lower_tail:
        return int(stats.binom.ppf(prob, n=n, p=p))
    else:
        return int(stats.binom.isf(prob, n=n, p=p))


def rbinom(n: int, p: float, rng: Optional[RNG] = None) -> int:
    """Generate binomial random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.binomial(n, p)


# ============================================================================
# Poisson Distribution
# ============================================================================

def dpois(x: int, lam: float, log: bool = False) -> float:
    """Poisson probability mass function."""
    if log:
        return stats.poisson.logpmf(x, mu=lam)
    else:
        return stats.poisson.pmf(x, mu=lam)


def ppois(x: int, lam: float, lower_tail: bool = True, log_p: bool = False) -> float:
    """Poisson cumulative distribution function."""
    if lower_tail:
        if log_p:
            return stats.poisson.logcdf(x, mu=lam)
        else:
            return stats.poisson.cdf(x, mu=lam)
    else:
        if log_p:
            return stats.poisson.logsf(x, mu=lam)
        else:
            return stats.poisson.sf(x, mu=lam)


def qpois(p: float, lam: float, lower_tail: bool = True, log_p: bool = False) -> int:
    """Poisson quantile function."""
    if log_p:
        p = np.exp(p)
    
    if lower_tail:
        return int(stats.poisson.ppf(p, mu=lam))
    else:
        return int(stats.poisson.isf(p, mu=lam))


def rpois(lam: float, rng: Optional[RNG] = None) -> int:
    """Generate Poisson random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.poisson(lam)


# ============================================================================
# Exponential Distribution
# ============================================================================

def dexp(x: float, rate: float = 1.0, log: bool = False) -> float:
    """Exponential density function."""
    scale = 1.0 / rate
    if log:
        return stats.expon.logpdf(x, scale=scale)
    else:
        return stats.expon.pdf(x, scale=scale)


def pexp(x: float, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False) -> float:
    """Exponential cumulative distribution function."""
    scale = 1.0 / rate
    if lower_tail:
        if log_p:
            return stats.expon.logcdf(x, scale=scale)
        else:
            return stats.expon.cdf(x, scale=scale)
    else:
        if log_p:
            return stats.expon.logsf(x, scale=scale)
        else:
            return stats.expon.sf(x, scale=scale)


def qexp(p: float, rate: float = 1.0, lower_tail: bool = True, log_p: bool = False) -> float:
    """Exponential quantile function."""
    if log_p:
        p = np.exp(p)
    
    scale = 1.0 / rate
    if lower_tail:
        return stats.expon.ppf(p, scale=scale)
    else:
        return stats.expon.isf(p, scale=scale)


def rexp(rate: float = 1.0, rng: Optional[RNG] = None) -> float:
    """Generate exponential random number."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.exponential(1.0 / rate)


# ============================================================================
# Multinomial Distribution
# ============================================================================

def rmultinom(n: int, probs: List[float], rng: Optional[RNG] = None) -> np.ndarray:
    """Generate multinomial random sample."""
    if rng is None:
        rng = GlobalRng.rng
    return rng.multinomial(n, probs)


# ============================================================================
# Gamma and Related Functions
# ============================================================================

def gamma_func(x: float) -> float:
    """Gamma function."""
    return special.gamma(x)


def lgamma_func(x: float) -> float:
    """Log gamma function."""
    return special.loggamma(x)


def digamma_func(x: float) -> float:
    """Digamma (psi) function."""
    return special.digamma(x)


def trigamma_func(x: float) -> float:
    """Trigamma function."""
    return float(special.polygamma(1, x))


def beta_func(a: float, b: float) -> float:
    """Beta function."""
    return special.beta(a, b)


def lbeta_func(a: float, b: float) -> float:
    """Log beta function."""
    return special.betaln(a, b)


def choose_func(n: float, k: float) -> float:
    """Binomial coefficient (n choose k)."""
    if k < 0 or k > n:
        return 0.0
    return special.comb(n, k, exact=False)


def lchoose_func(n: float, k: float) -> float:
    """Log binomial coefficient."""
    if k < 0 or k > n:
        return -np.inf
    return np.log(special.comb(n, k, exact=False))


# ============================================================================
# Utility Functions  
# ============================================================================

def sign(x: float) -> float:
    """Sign function."""
    return np.sign(x)


def ftrunc(x: float) -> float:
    """Truncate to integer."""
    return np.trunc(x)


# Export all functions
__all__ = [
    # Normal
    'dnorm', 'pnorm', 'qnorm', 'rnorm',
    # Uniform  
    'dunif', 'punif', 'qunif', 'runif',
    # Gamma
    'dgamma', 'pgamma', 'qgamma', 'rgamma',
    # Beta
    'dbeta', 'pbeta', 'qbeta', 'rbeta',
    # Chi-square
    'dchisq', 'pchisq', 'qchisq', 'rchisq',
    # Student's t
    'dt', 'pt', 'qt', 'rt',
    # F distribution
    'df_dist', 'pf', 'qf', 'rf',
    # Binomial
    'dbinom', 'pbinom', 'qbinom', 'rbinom',
    # Poisson
    'dpois', 'ppois', 'qpois', 'rpois',
    # Exponential
    'dexp', 'pexp', 'qexp', 'rexp',
    # Multinomial
    'rmultinom',
    # Special functions
    'gamma_func', 'lgamma_func', 'digamma_func', 'trigamma_func',
    'beta_func', 'lbeta_func', 'choose_func', 'lchoose_func',
    # Utilities
    'sign', 'ftrunc'
]