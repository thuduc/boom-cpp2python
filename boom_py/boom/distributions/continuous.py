"""Continuous probability distributions for BOOM."""
import numpy as np
from scipy import stats
from typing import Union, Optional
from ..math.special_functions import lgamma, lbeta, log_sum_exp


class Distribution:
    """Base class for probability distributions."""
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        raise NotImplementedError
    
    def logpdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Log probability density function."""
        return np.log(self.pdf(x))
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        raise NotImplementedError
    
    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function (inverse CDF)."""
        raise NotImplementedError
    
    def mean(self) -> float:
        """Mean of the distribution."""
        raise NotImplementedError
    
    def variance(self) -> float:
        """Variance of the distribution."""
        raise NotImplementedError
    
    def sample(self, size: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        """Sample from the distribution."""
        raise NotImplementedError


class Normal(Distribution):
    """Normal (Gaussian) distribution."""
    
    def __init__(self, mean: float = 0.0, sd: float = 1.0):
        """Initialize Normal distribution.
        
        Args:
            mean: Mean parameter
            sd: Standard deviation (must be positive)
        """
        if sd <= 0:
            raise ValueError("Standard deviation must be positive")
        self.mean_param = mean
        self.sd = sd
        self.variance_param = sd ** 2
    
    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mean_param, scale=self.sd)
    
    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.mean_param, scale=self.sd)
    
    def cdf(self, x):
        return stats.norm.cdf(x, loc=self.mean_param, scale=self.sd)
    
    def quantile(self, p):
        return stats.norm.ppf(p, loc=self.mean_param, scale=self.sd)
    
    def mean(self):
        return self.mean_param
    
    def variance(self):
        return self.variance_param
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.normal(self.mean_param, self.sd, size=size)


class Gamma(Distribution):
    """Gamma distribution."""
    
    def __init__(self, shape: float, scale: float = 1.0):
        """Initialize Gamma distribution.
        
        Args:
            shape: Shape parameter (alpha, must be positive)
            scale: Scale parameter (must be positive)
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be positive")
        self.shape = shape
        self.scale = scale
    
    def pdf(self, x):
        return stats.gamma.pdf(x, a=self.shape, scale=self.scale)
    
    def logpdf(self, x):
        return stats.gamma.logpdf(x, a=self.shape, scale=self.scale)
    
    def cdf(self, x):
        return stats.gamma.cdf(x, a=self.shape, scale=self.scale)
    
    def quantile(self, p):
        return stats.gamma.ppf(p, a=self.shape, scale=self.scale)
    
    def mean(self):
        return self.shape * self.scale
    
    def variance(self):
        return self.shape * self.scale ** 2
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.gamma(self.shape, self.scale, size=size)


class Beta(Distribution):
    """Beta distribution."""
    
    def __init__(self, a: float, b: float):
        """Initialize Beta distribution.
        
        Args:
            a: First shape parameter (must be positive)
            b: Second shape parameter (must be positive)
        """
        if a <= 0 or b <= 0:
            raise ValueError("Shape parameters must be positive")
        self.a = a
        self.b = b
    
    def pdf(self, x):
        return stats.beta.pdf(x, a=self.a, b=self.b)
    
    def logpdf(self, x):
        return stats.beta.logpdf(x, a=self.a, b=self.b)
    
    def cdf(self, x):
        return stats.beta.cdf(x, a=self.a, b=self.b)
    
    def quantile(self, p):
        return stats.beta.ppf(p, a=self.a, b=self.b)
    
    def mean(self):
        return self.a / (self.a + self.b)
    
    def variance(self):
        return (self.a * self.b) / ((self.a + self.b) ** 2 * (self.a + self.b + 1))
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.beta(self.a, self.b, size=size)


class StudentT(Distribution):
    """Student's t distribution."""
    
    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        """Initialize Student's t distribution.
        
        Args:
            df: Degrees of freedom (must be positive)
            loc: Location parameter
            scale: Scale parameter (must be positive)
        """
        if df <= 0 or scale <= 0:
            raise ValueError("Degrees of freedom and scale must be positive")
        self.df = df
        self.loc = loc
        self.scale = scale
    
    def pdf(self, x):
        return stats.t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)
    
    def logpdf(self, x):
        return stats.t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)
    
    def cdf(self, x):
        return stats.t.cdf(x, df=self.df, loc=self.loc, scale=self.scale)
    
    def quantile(self, p):
        return stats.t.ppf(p, df=self.df, loc=self.loc, scale=self.scale)
    
    def mean(self):
        if self.df > 1:
            return self.loc
        else:
            return np.nan
    
    def variance(self):
        if self.df > 2:
            return self.scale ** 2 * self.df / (self.df - 2)
        elif self.df > 1:
            return np.inf
        else:
            return np.nan
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return self.loc + self.scale * rng.standard_t(self.df, size=size)


class Exponential(Distribution):
    """Exponential distribution."""
    
    def __init__(self, rate: float):
        """Initialize Exponential distribution.
        
        Args:
            rate: Rate parameter (lambda, must be positive)
        """
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self.rate = rate
        self.scale = 1 / rate
    
    def pdf(self, x):
        return stats.expon.pdf(x, scale=self.scale)
    
    def logpdf(self, x):
        return stats.expon.logpdf(x, scale=self.scale)
    
    def cdf(self, x):
        return stats.expon.cdf(x, scale=self.scale)
    
    def quantile(self, p):
        return stats.expon.ppf(p, scale=self.scale)
    
    def mean(self):
        return self.scale
    
    def variance(self):
        return self.scale ** 2
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.exponential(self.scale, size=size)


class ChiSquare(Distribution):
    """Chi-square distribution."""
    
    def __init__(self, df: float):
        """Initialize Chi-square distribution.
        
        Args:
            df: Degrees of freedom (must be positive)
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
    
    def pdf(self, x):
        return stats.chi2.pdf(x, df=self.df)
    
    def logpdf(self, x):
        return stats.chi2.logpdf(x, df=self.df)
    
    def cdf(self, x):
        return stats.chi2.cdf(x, df=self.df)
    
    def quantile(self, p):
        return stats.chi2.ppf(p, df=self.df)
    
    def mean(self):
        return self.df
    
    def variance(self):
        return 2 * self.df
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.chisquare(self.df, size=size)


class F(Distribution):
    """F distribution."""
    
    def __init__(self, df1: float, df2: float):
        """Initialize F distribution.
        
        Args:
            df1: First degrees of freedom (must be positive)
            df2: Second degrees of freedom (must be positive)
        """
        if df1 <= 0 or df2 <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df1 = df1
        self.df2 = df2
    
    def pdf(self, x):
        return stats.f.pdf(x, dfn=self.df1, dfd=self.df2)
    
    def logpdf(self, x):
        return stats.f.logpdf(x, dfn=self.df1, dfd=self.df2)
    
    def cdf(self, x):
        return stats.f.cdf(x, dfn=self.df1, dfd=self.df2)
    
    def quantile(self, p):
        return stats.f.ppf(p, dfn=self.df1, dfd=self.df2)
    
    def mean(self):
        if self.df2 > 2:
            return self.df2 / (self.df2 - 2)
        else:
            return np.nan
    
    def variance(self):
        if self.df2 > 4:
            num = 2 * self.df2 ** 2 * (self.df1 + self.df2 - 2)
            den = self.df1 * (self.df2 - 2) ** 2 * (self.df2 - 4)
            return num / den
        else:
            return np.nan
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.f(self.df1, self.df2, size=size)


class Uniform(Distribution):
    """Uniform distribution."""
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        """Initialize Uniform distribution.
        
        Args:
            low: Lower bound
            high: Upper bound (must be > low)
        """
        if high <= low:
            raise ValueError("Upper bound must be greater than lower bound")
        self.low = low
        self.high = high
        self.width = high - low
    
    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.low, scale=self.width)
    
    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.low, scale=self.width)
    
    def cdf(self, x):
        return stats.uniform.cdf(x, loc=self.low, scale=self.width)
    
    def quantile(self, p):
        return stats.uniform.ppf(p, loc=self.low, scale=self.width)
    
    def mean(self):
        return (self.low + self.high) / 2
    
    def variance(self):
        return self.width ** 2 / 12
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.uniform(self.low, self.high, size=size)


class Lognormal(Distribution):
    """Lognormal distribution."""
    
    def __init__(self, meanlog: float = 0.0, sdlog: float = 1.0):
        """Initialize Lognormal distribution.
        
        Args:
            meanlog: Mean of log (location parameter)
            sdlog: SD of log (scale parameter, must be positive)
        """
        if sdlog <= 0:
            raise ValueError("SD of log must be positive")
        self.meanlog = meanlog
        self.sdlog = sdlog
    
    def pdf(self, x):
        return stats.lognorm.pdf(x, s=self.sdlog, scale=np.exp(self.meanlog))
    
    def logpdf(self, x):
        return stats.lognorm.logpdf(x, s=self.sdlog, scale=np.exp(self.meanlog))
    
    def cdf(self, x):
        return stats.lognorm.cdf(x, s=self.sdlog, scale=np.exp(self.meanlog))
    
    def quantile(self, p):
        return stats.lognorm.ppf(p, s=self.sdlog, scale=np.exp(self.meanlog))
    
    def mean(self):
        return np.exp(self.meanlog + self.sdlog ** 2 / 2)
    
    def variance(self):
        mean_val = self.mean()
        return mean_val ** 2 * (np.exp(self.sdlog ** 2) - 1)
    
    def sample(self, size=None, rng=None):
        if rng is None:
            rng = np.random
        return rng.lognormal(self.meanlog, self.sdlog, size=size)