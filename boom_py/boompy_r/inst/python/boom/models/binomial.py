"""Binomial models for BOOM."""
import numpy as np
from typing import Union, List, Optional
from .base import Model, PriorModel, Parameter
from .data import BinomialData
from .sufstat import BetaBinomialSuf
from ..distributions.rng import GlobalRng
from ..distributions.continuous import Beta
from ..distributions.discrete import Binomial
from ..math.special_functions import lgamma, lbeta


class BinomialModel(Model):
    """Binomial model with fixed number of trials."""
    
    def __init__(self, n: int, p: float = 0.5):
        """Initialize Binomial model.
        
        Args:
            n: Number of trials (fixed)
            p: Success probability
        """
        super().__init__()
        if n < 0:
            raise ValueError("Number of trials must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("Probability must be in [0, 1]")
        
        self.n = n
        self._params['p'] = Parameter(p, 'p')
        self._successes = []
        self._dist = Binomial(n, p)
    
    @property
    def p(self) -> float:
        """Get success probability."""
        return self._params['p'].value
    
    @p.setter
    def p(self, value: float):
        """Set success probability."""
        if not 0 <= value <= 1:
            raise ValueError("Probability must be in [0, 1]")
        self._params['p'].value = value
        self._dist.p = value
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._successes.clear()
    
    def add_data(self, data: Union[int, BinomialData]):
        """Add single observation."""
        if isinstance(data, BinomialData):
            if data.trials != self.n:
                raise ValueError(f"Expected {self.n} trials, got {data.trials}")
            successes = data.successes
        else:
            successes = int(data)
            if successes < 0 or successes > self.n:
                raise ValueError(f"Successes must be in [0, {self.n}]")
            data = BinomialData(successes, self.n)
        
        self._data.append(data)
        self._successes.append(successes)
    
    def set_data(self, data: Union[List[int], np.ndarray]):
        """Set data (replaces existing)."""
        self.clear_data()
        for x in data:
            self.add_data(x)
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            total_ll = 0.0
            for s in self._successes:
                total_ll += self._dist.logpmf(s)
            return total_ll
        else:
            # Single observation
            if isinstance(data, BinomialData):
                s = data.successes
            else:
                s = int(data)
            return self._dist.logpmf(s)
    
    def pmf(self, k: int) -> float:
        """Probability mass function."""
        return self._dist.pmf(k)
    
    def logpmf(self, k: int) -> float:
        """Log probability mass function."""
        return self._dist.logpmf(k)
    
    def simulate(self, n: int = 1, rng: Optional[GlobalRng] = None) -> Union[BinomialData, List[BinomialData]]:
        """Simulate data from the model."""
        if rng is None:
            from ..distributions import rng as global_rng
            rng = global_rng
        
        successes = rng.rbinom_vec(n, self.n, self.p)
        data = [BinomialData(s, self.n) for s in successes]
        
        if n == 1:
            return data[0]
        return data
    
    def mle(self):
        """Maximum likelihood estimation."""
        if len(self._successes) > 0:
            self.p = np.mean(self._successes) / self.n
    
    def clone(self) -> 'BinomialModel':
        """Create a copy of the model."""
        model = BinomialModel(self.n, self.p)
        model.set_data(self._successes)
        return model


class BetaBinomialModel(Model):
    """Beta-Binomial model for varying trial sizes."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Beta-Binomial model.
        
        Args:
            alpha: First shape parameter of Beta prior
            beta: Second shape parameter of Beta prior
        """
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError("Shape parameters must be positive")
        
        self._params['alpha'] = Parameter(alpha, 'alpha')
        self._params['beta'] = Parameter(beta, 'beta')
        self._suf = BetaBinomialSuf()
        self._beta_dist = Beta(alpha, beta)
    
    @property
    def alpha(self) -> float:
        """Get alpha parameter."""
        return self._params['alpha'].value
    
    @alpha.setter
    def alpha(self, value: float):
        """Set alpha parameter."""
        if value <= 0:
            raise ValueError("Alpha must be positive")
        self._params['alpha'].value = value
        self._beta_dist.a = value
    
    @property
    def beta(self) -> float:
        """Get beta parameter."""
        return self._params['beta'].value
    
    @beta.setter
    def beta(self, value: float):
        """Set beta parameter."""
        if value <= 0:
            raise ValueError("Beta must be positive")
        self._params['beta'].value = value
        self._beta_dist.b = value
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._suf.clear()
    
    def add_data(self, data: Union[BinomialData, tuple]):
        """Add single observation."""
        if isinstance(data, BinomialData):
            successes = data.successes
            trials = data.trials
        else:
            successes, trials = data
            data = BinomialData(successes, trials)
        
        self._data.append(data)
        self._suf.update(successes, trials)
    
    def set_data(self, data: List[tuple]):
        """Set data (replaces existing)."""
        self.clear_data()
        for item in data:
            self.add_data(item)
    
    def suf(self) -> BetaBinomialSuf:
        """Get sufficient statistics."""
        return self._suf
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            total_ll = 0.0
            for i in range(self._suf.n):
                s = self._suf.successes_list[i]
                n = self._suf.trials_list[i]
                total_ll += self._logpmf(s, n)
            return total_ll
        else:
            # Single observation
            if isinstance(data, BinomialData):
                s = data.successes
                n = data.trials
            else:
                s, n = data
            return self._logpmf(s, n)
    
    def _logpmf(self, successes: int, trials: int) -> float:
        """Log PMF for beta-binomial."""
        from ..math.special_functions import lchoose, lbeta
        
        if successes < 0 or successes > trials:
            return -np.inf
        
        return (lchoose(trials, successes) + 
                lbeta(successes + self.alpha, trials - successes + self.beta) -
                lbeta(self.alpha, self.beta))
    
    def pmf(self, successes: int, trials: int) -> float:
        """Probability mass function."""
        return np.exp(self._logpmf(successes, trials))
    
    def simulate(self, trial_sizes: List[int], rng: Optional[GlobalRng] = None) -> List[BinomialData]:
        """Simulate data from the model."""
        if rng is None:
            from ..distributions import rng as global_rng
            rng = global_rng
        
        # First draw p from Beta(alpha, beta)
        p = rng.rbeta(self.alpha, self.beta)
        
        # Then draw binomial observations
        data = []
        for n in trial_sizes:
            s = rng.rbinom(n, p)
            data.append(BinomialData(s, n))
        
        return data
    
    def posterior_mean_p(self) -> float:
        """Posterior mean of success probability."""
        return (self.alpha + self._suf.sum_successes) / \
               (self.alpha + self.beta + self._suf.sum_trials)
    
    def clone(self) -> 'BetaBinomialModel':
        """Create a copy of the model."""
        model = BetaBinomialModel(self.alpha, self.beta)
        data_tuples = [(self._suf.successes_list[i], self._suf.trials_list[i]) 
                      for i in range(self._suf.n)]
        model.set_data(data_tuples)
        return model


class BetaModel(Model, PriorModel):
    """Beta distribution model."""
    
    def __init__(self, a: float = 1.0, b: float = 1.0):
        """Initialize Beta model.
        
        Args:
            a: First shape parameter
            b: Second shape parameter
        """
        super().__init__()
        if a <= 0 or b <= 0:
            raise ValueError("Shape parameters must be positive")
        
        self._params['a'] = Parameter(a, 'a')
        self._params['b'] = Parameter(b, 'b')
        self._values = []
        self._dist = Beta(a, b)
    
    @property
    def a(self) -> float:
        """Get first shape parameter."""
        return self._params['a'].value
    
    @a.setter
    def a(self, value: float):
        """Set first shape parameter."""
        if value <= 0:
            raise ValueError("a must be positive")
        self._params['a'].value = value
        self._dist.a = value
    
    @property
    def b(self) -> float:
        """Get second shape parameter."""
        return self._params['b'].value
    
    @b.setter
    def b(self, value: float):
        """Set second shape parameter."""
        if value <= 0:
            raise ValueError("b must be positive")
        self._params['b'].value = value
        self._dist.b = value
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
        self._values.clear()
    
    def add_data(self, data: Union[float, DoubleData]):
        """Add single observation."""
        if isinstance(data, DoubleData):
            value = data.value
        else:
            value = float(data)
            data = DoubleData(value)
        
        if not 0 <= value <= 1:
            raise ValueError("Beta data must be in [0, 1]")
        
        self._data.append(data)
        self._values.append(value)
    
    def set_data(self, data: Union[List[float], np.ndarray]):
        """Set data (replaces existing)."""
        self.clear_data()
        for x in data:
            self.add_data(x)
    
    def loglike(self, data=None) -> float:
        """Log likelihood."""
        if data is None:
            # Use stored data
            total_ll = 0.0
            for x in self._values:
                total_ll += self._dist.logpdf(x)
            return total_ll
        else:
            # Single observation
            if isinstance(data, DoubleData):
                x = data.value
            else:
                x = float(data)
            return self._dist.logpdf(x)
    
    def pdf(self, x: float) -> float:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def logpdf(self, x: float) -> float:
        """Log probability density function."""
        return self._dist.logpdf(x)
    
    def simulate(self, n: int = 1, rng: Optional[GlobalRng] = None) -> Union[DoubleData, List[DoubleData]]:
        """Simulate data from the model."""
        if rng is None:
            from ..distributions import rng as global_rng
            rng = global_rng
        
        values = rng.rbeta_vec(n, self.a, self.b)
        data = [DoubleData(x) for x in values]
        
        if n == 1:
            return data[0]
        return data
    
    def mean(self) -> float:
        """Mean of the distribution."""
        return self._dist.mean()
    
    def variance(self) -> float:
        """Variance of the distribution."""
        return self._dist.variance()
    
    def clone(self) -> 'BetaModel':
        """Create a copy of the model."""
        model = BetaModel(self.a, self.b)
        model.set_data(self._values)
        return model
    
    # PriorModel interface
    def logp(self, theta: Union[float, np.ndarray]) -> float:
        """Log prior density (for use as prior)."""
        return self.logpdf(theta)