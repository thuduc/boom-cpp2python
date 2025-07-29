"""Special mathematical functions for BOOM."""
import numpy as np
from scipy import special
from typing import Union, Optional


def lgamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Log gamma function."""
    return special.loggamma(x)


def lbeta(a: float, b: float) -> float:
    """Log beta function."""
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def lchoose(n: float, k: float) -> float:
    """Log binomial coefficient."""
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def lmultigamma(x: float, d: int) -> float:
    """Log multivariate gamma function."""
    result = d * (d - 1) / 4 * np.log(np.pi)
    for i in range(d):
        result += lgamma(x - i / 2)
    return result


def digamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Digamma (psi) function - derivative of log gamma."""
    return special.digamma(x)


def trigamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Trigamma function - derivative of digamma."""
    return special.polygamma(1, x)


def tetragamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Tetragamma function - second derivative of digamma."""
    return special.polygamma(2, x)


def beta_function(a: float, b: float) -> float:
    """Beta function."""
    return np.exp(lbeta(a, b))


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Incomplete beta function."""
    return special.betainc(a, b, x)


def incomplete_gamma(a: float, x: float) -> float:
    """Incomplete gamma function."""
    return special.gammainc(a, x)


def erf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Error function."""
    return special.erf(x)


def erfc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Complementary error function."""
    return special.erfc(x)


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Logit function."""
    return np.log(p / (1 - p))


def invlogit(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse logit (logistic) function."""
    return 1 / (1 + np.exp(-x))


def expit(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Alias for inverse logit."""
    return special.expit(x)


def log1p(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Accurate log(1 + x) for small x."""
    return np.log1p(x)


def expm1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Accurate exp(x) - 1 for small x."""
    return np.expm1(x)


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Compute log(sum(exp(x))) in a numerically stable way."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + np.squeeze(x_max, axis=axis)


def soft_max(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Softmax function - numerically stable."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_factorial(n: int) -> float:
    """Log factorial."""
    return lgamma(n + 1)


def pochhammer(x: float, n: float) -> float:
    """Pochhammer symbol (rising factorial)."""
    return special.poch(x, n)


def bessel_i(nu: float, x: float) -> float:
    """Modified Bessel function of the first kind."""
    return special.iv(nu, x)


def bessel_k(nu: float, x: float) -> float:
    """Modified Bessel function of the second kind."""
    return special.kv(nu, x)


def dawson(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Dawson's integral."""
    return special.dawsn(x)


def zeta(x: float) -> float:
    """Riemann zeta function."""
    return special.zeta(x)


# Constants
EULERS_CONSTANT = 0.57721566490153286060651209008240243104215933593992
LOG_2PI = np.log(2 * np.pi)
LOG_PI = np.log(np.pi)
LOG_SQRT_2PI = 0.5 * LOG_2PI
SQRT_2_OVER_PI = np.sqrt(2 / np.pi)