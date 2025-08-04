"""Multinomial model implementation."""

from typing import List, Optional, Union
import numpy as np
from scipy.special import gammaln
from ..linalg import Vector
from ..distributions import RNG
from ..distributions.custom import Dirichlet
from .base import Model, Data, LoglikeModel, ConjugateModel
from .params import VectorParams
from .data import MultinomialData


class MultinomialSuf:
    """Sufficient statistics for Multinomial distribution.
    
    Sufficient statistics are:
    - n: total number of trials
    - counts: vector of counts for each category
    """
    
    def __init__(self, k: int = 0):
        """Initialize multinomial sufficient statistics.
        
        Args:
            k: Number of categories. If 0, will be inferred from first data.
        """
        self._k = k
        self.clear()
    
    def clear(self):
        """Reset to initial state."""
        self._n = 0
        if self._k > 0:
            self._counts = np.zeros(self._k, dtype=int)
        else:
            self._counts = None
    
    def update(self, data: Union[MultinomialData, tuple, dict]):
        """Update with new data.
        
        Args:
            data: Data in one of these formats:
                - MultinomialData object
                - tuple: (n_trials, counts_array)
                - dict: {'trials': n, 'counts': counts_array}
        """
        if isinstance(data, MultinomialData):
            n_trials = data.trials()
            counts = data.counts()
        elif isinstance(data, tuple) and len(data) == 2:
            n_trials, counts = data
            counts = np.array(counts, dtype=int)
        elif isinstance(data, dict):
            n_trials = data.get('trials', 0)
            counts = np.array(data.get('counts', []), dtype=int)
        else:
            raise TypeError(f"Cannot update MultinomialSuf with data of type {type(data)}")
        
        # Initialize if needed
        if self._k == 0:
            self._k = len(counts)
            self._counts = np.zeros(self._k, dtype=int)
        elif len(counts) != self._k:
            raise ValueError(f"Data dimension {len(counts)} doesn't match expected {self._k}")
        
        self._n += n_trials
        self._counts += counts
    
    def combine(self, other: 'MultinomialSuf') -> 'MultinomialSuf':
        """Combine with another MultinomialSuf."""
        if not isinstance(other, MultinomialSuf):
            raise TypeError("Can only combine with another MultinomialSuf")
        
        if self._k != other._k and self._k > 0 and other._k > 0:
            raise ValueError("Cannot combine sufficient statistics of different dimensions")
        
        result_k = max(self._k, other._k)
        result = MultinomialSuf(result_k)
        result._n = self._n + other._n
        
        if self._counts is not None and other._counts is not None:
            result._counts = self._counts + other._counts
        elif self._counts is not None:
            result._counts = self._counts.copy()
        elif other._counts is not None:
            result._counts = other._counts.copy()
        
        return result
    
    def n(self) -> int:
        """Get total number of trials."""
        return self._n
    
    def k(self) -> int:
        """Get number of categories."""
        return self._k
    
    def counts(self) -> Optional[np.ndarray]:
        """Get counts vector."""
        if self._counts is None:
            return None
        return self._counts.copy()
    
    def count(self, category: int) -> int:
        """Get count for specific category."""
        if self._counts is None:
            return 0
        if not 0 <= category < self._k:
            raise IndexError(f"Category {category} out of range [0, {self._k})")
        return int(self._counts[category])
    
    def proportions(self) -> Optional[np.ndarray]:
        """Get empirical proportions."""
        if self._counts is None or self._n == 0:
            return None
        return self._counts / self._n
    
    def clone(self) -> 'MultinomialSuf':
        """Create copy."""
        result = MultinomialSuf(self._k)
        result._n = self._n
        if self._counts is not None:
            result._counts = self._counts.copy()
        return result
    
    def __str__(self) -> str:
        """String representation."""
        props = self.proportions()
        props_str = str(props) if props is not None else "None"
        return f"MultinomialSuf(n={self._n}, k={self._k}, props={props_str})"


class MultinomialModel(ConjugateModel):
    """Multinomial model with unknown probability vector.
    
    This model assumes observations are multinomial random variables:
    X ~ Multinomial(n, p) with known n and unknown p.
    
    The model uses a conjugate Dirichlet prior:
    p ~ Dirichlet(alpha)
    
    This gives the Dirichlet-multinomial conjugate family.
    """
    
    def __init__(self, probs: Union[List[float], np.ndarray, Vector] = None, k: int = 2):
        """Initialize Multinomial model.
        
        Args:
            probs: Initial probability vector. If None, uses uniform probabilities.
            k: Number of categories (used if probs is None).
        """
        super().__init__()
        
        if probs is not None:
            if isinstance(probs, Vector):
                probs_array = probs.to_numpy()
            else:
                probs_array = np.array(probs, dtype=float)
            
            if np.any(probs_array < 0):
                raise ValueError("Probabilities must be non-negative")
            
            # Normalize to ensure they sum to 1
            probs_sum = np.sum(probs_array)
            if probs_sum <= 0:
                raise ValueError("At least one probability must be positive")
            probs_array /= probs_sum
            
            self._k = len(probs_array)
        else:
            if k < 2:
                raise ValueError("Number of categories must be at least 2")
            self._k = k
            probs_array = np.ones(k) / k  # Uniform probabilities
        
        # Create parameter object
        self._probs_param = VectorParams(Vector(probs_array))
        
        # Set up parameter management
        self.set_parameter('probs', self._probs_param)
        
        # Initialize sufficient statistics
        self._suf = MultinomialSuf(self._k)
        
        # Prior parameters (default to uniform: Dirichlet(1, 1, ..., 1))
        self._alpha = np.ones(self._k)
    
    # ============================================================================
    # Parameter Access
    # ============================================================================
    
    def probs(self) -> Vector:
        """Get probability vector."""
        return self._probs_param.get_value().copy()
    
    def set_probs(self, probs: Union[List[float], np.ndarray, Vector]):
        """Set probability vector."""
        if isinstance(probs, Vector):
            probs_array = probs.to_numpy()
        else:
            probs_array = np.array(probs, dtype=float)
        
        if len(probs_array) != self._k:
            raise ValueError(f"Probability vector length {len(probs_array)} must match k={self._k}")
        
        if np.any(probs_array < 0):
            raise ValueError("Probabilities must be non-negative")
        
        # Normalize
        probs_sum = np.sum(probs_array)
        if probs_sum <= 0:
            raise ValueError("At least one probability must be positive")
        probs_array /= probs_sum
        
        self._probs_param.set_value(Vector(probs_array))
        self._notify_observers()
    
    def prob(self, category: int) -> float:
        """Get probability for specific category."""
        if not 0 <= category < self._k:
            raise IndexError(f"Category {category} out of range [0, {self._k})")
        return self._probs_param.get_value()[category]
    
    def k(self) -> int:
        """Get number of categories."""
        return self._k
    
    # ============================================================================
    # Prior Management
    # ============================================================================
    
    def set_conjugate_prior(self, alpha: Union[List[float], np.ndarray, Vector]):
        """Set conjugate Dirichlet prior.
        
        Args:
            alpha: Concentration parameters of Dirichlet prior.
        """
        if isinstance(alpha, Vector):
            alpha_array = alpha.to_numpy()
        else:
            alpha_array = np.array(alpha, dtype=float)
        
        if len(alpha_array) != self._k:
            raise ValueError(f"Alpha vector length {len(alpha_array)} must match k={self._k}")
        
        if np.any(alpha_array <= 0):
            raise ValueError("Dirichlet prior parameters must be positive")
        
        self._alpha = alpha_array.copy()
    
    def log_prior(self) -> float:
        """Compute log prior probability under Dirichlet prior."""
        probs_vec = self.probs()
        
        # Dirichlet log pdf: log(Gamma(sum(alpha))) - sum(log(Gamma(alpha_i))) + sum((alpha_i-1)*log(p_i))
        log_gamma_sum_alpha = gammaln(np.sum(self._alpha))
        sum_log_gamma_alpha = np.sum(gammaln(self._alpha))
        sum_weighted_log_p = np.sum((self._alpha - 1) * np.log(probs_vec.to_numpy()))
        
        return log_gamma_sum_alpha - sum_log_gamma_alpha + sum_weighted_log_p
    
    # ============================================================================
    # Data Management
    # ============================================================================
    
    def add_data(self, data: Union[Data, tuple, dict, List]):
        """Add data and update sufficient statistics.
        
        Args:
            data: Can be:
                - MultinomialData object
                - tuple: (n_trials, counts_array)
                - dict: {'trials': n, 'counts': counts}
                - list of any of the above
        """
        if isinstance(data, list):
            for item in data:
                self.add_data(item)
            return
        
        if isinstance(data, tuple) and len(data) == 2:
            n_trials, counts = data
            data_obj = MultinomialData(n_trials, counts)
        elif isinstance(data, dict):
            n_trials = data.get('trials', 0)
            counts = data.get('counts', [])
            data_obj = MultinomialData(n_trials, counts)
        elif isinstance(data, MultinomialData):
            data_obj = data
        else:
            raise TypeError(f"Cannot add data of type {type(data)}")
        
        super().add_data(data_obj)
        self._suf.update(data_obj)
    
    def clear_data(self):
        """Clear all data and reset sufficient statistics."""
        super().clear_data()
        self._suf.clear()
    
    def suf(self) -> MultinomialSuf:
        """Get sufficient statistics."""
        return self._suf
    
    # ============================================================================
    # Model Interface Implementation
    # ============================================================================
    
    def log_likelihood(self, data: Optional[List[Data]] = None) -> float:
        """Compute log likelihood.
        
        Args:
            data: Data to compute likelihood for. If None, uses model's data.
            
        Returns:
            Log likelihood value.
        """
        if data is not None:
            # Compute likelihood for provided data
            log_lik = 0.0
            probs_vec = self.probs().to_numpy()
            
            for data_point in data:
                if isinstance(data_point, MultinomialData):
                    n_trials = data_point.trials()
                    counts = data_point.counts()
                    
                    # Multinomial log likelihood: log(n!) - sum(log(k_i!)) + sum(k_i * log(p_i))
                    # We omit the factorial terms as they don't depend on p
                    log_lik += np.sum(counts * np.log(probs_vec))
                else:
                    raise TypeError(f"Expected MultinomialData, got {type(data_point)}")
            
            return log_lik
        else:
            # Use sufficient statistics for efficiency
            if self._suf.n() == 0:
                return 0.0
            
            counts = self._suf.counts()
            probs_vec = self.probs().to_numpy()
            
            if counts is None:
                return 0.0
            
            # Log likelihood = sum(k_i * log(p_i))
            return np.sum(counts * np.log(probs_vec))
    
    def simulate_data(self, n: Optional[int] = None, trials_per_obs: Optional[int] = None) -> List[MultinomialData]:
        """Simulate data from the model.
        
        Args:
            n: Number of observations to simulate. If None, uses current sample size.
            trials_per_obs: Number of trials per observation. If None, uses 1.
            
        Returns:
            List of simulated MultinomialData objects.
        """
        if n is None:
            n = len(self._data) if self._data else 1
        
        if n <= 0:
            return []
        
        if trials_per_obs is None:
            trials_per_obs = 1
        
        probs_vec = self.probs().to_numpy()
        rng = RNG()
        
        simulated = []
        for _ in range(n):
            # Use numpy's multinomial
            counts = rng._rng.multinomial(trials_per_obs, probs_vec)
            simulated.append(MultinomialData(trials_per_obs, counts))
        
        return simulated
    
    # ============================================================================
    # Parameter Vectorization
    # ============================================================================
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Convert parameters to vector form using log-ratio transformation."""
        probs_vec = self.probs().to_numpy()
        
        if minimal:
            # Use log-ratio transformation: log(p_i/p_k) for i=0,...,k-2
            # The last probability p_k is determined by constraint sum(p_i) = 1
            if self._k == 1:
                return Vector([])  # No free parameters
            
            log_ratios = np.log(probs_vec[:-1] / probs_vec[-1])
            return Vector(log_ratios)
        else:
            # Use log transformation with constraint handling
            return Vector(np.log(probs_vec))
    
    def unvectorize_params(self, theta: Vector, minimal: bool = True):
        """Set parameters from vector form."""
        if minimal:
            if self._k == 1:
                # No free parameters
                return
            
            if len(theta) != self._k - 1:
                raise ValueError(f"Parameter vector must have exactly {self._k-1} elements")
            
            # Convert from log-ratios
            log_ratios = theta.to_numpy()
            
            # Compute probabilities: p_i = exp(log_ratio_i) / (1 + sum(exp(log_ratio_j)))
            exp_ratios = np.exp(log_ratios)
            normalizer = 1.0 + np.sum(exp_ratios)
            
            probs_array = np.zeros(self._k)
            probs_array[:-1] = exp_ratios / normalizer
            probs_array[-1] = 1.0 / normalizer
        else:
            if len(theta) != self._k:
                raise ValueError(f"Parameter vector must have exactly {self._k} elements")
            
            # Convert from log and normalize
            probs_array = np.exp(theta.to_numpy())
            probs_array /= np.sum(probs_array)
        
        self.set_probs(probs_array)
    
    # ============================================================================
    # Maximum Likelihood Estimation
    # ============================================================================
    
    def mle(self):
        """Compute maximum likelihood estimate."""
        if self._suf.n() == 0:
            return  # No data to estimate from
        
        # MLE for multinomial is simply the empirical proportions
        proportions = self._suf.proportions()
        if proportions is not None:
            self.set_probs(proportions)
    
    # ============================================================================
    # Conjugate Model Implementation
    # ============================================================================
    
    def posterior_mode(self) -> Vector:
        """Compute posterior mode under conjugate Dirichlet prior.
        
        Returns:
            Posterior mode of probability vector.
        """
        counts = self._suf.counts()
        if counts is None:
            counts = np.zeros(self._k)
        
        # Posterior parameters
        alpha_post = self._alpha + counts
        
        # Mode of Dirichlet(alpha) is (α_i - 1) / (sum(α_j) - k) for all α_i > 1
        # If any α_i <= 1, use mean instead
        if np.all(alpha_post > 1):
            normalizer = np.sum(alpha_post) - self._k
            p_mode = (alpha_post - 1) / normalizer
        else:
            # Use mean
            p_mode = alpha_post / np.sum(alpha_post)
        
        return Vector(p_mode)
    
    def posterior_mean(self) -> Vector:
        """Compute posterior mean under conjugate Dirichlet prior.
        
        Returns:
            Posterior mean of probability vector.
        """
        counts = self._suf.counts()
        if counts is None:
            counts = np.zeros(self._k)
        
        # Posterior parameters
        alpha_post = self._alpha + counts
        
        # Mean of Dirichlet(alpha) is α_i / sum(α_j)
        p_mean = alpha_post / np.sum(alpha_post)
        
        return Vector(p_mean)
    
    def sample_posterior(self, n: int = 1) -> List[Vector]:
        """Sample from posterior distribution.
        
        Args:
            n: Number of samples.
            
        Returns:
            List of probability vector samples.
        """
        counts = self._suf.counts()
        if counts is None:
            counts = np.zeros(self._k)
        
        # Posterior parameters
        alpha_post = self._alpha + counts
        
        dirichlet = Dirichlet(alpha_post)
        rng = RNG()
        
        samples = []
        for _ in range(n):
            p_sample = dirichlet.rvs(rng)
            samples.append(Vector(p_sample))
        
        return samples
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def clone(self) -> 'MultinomialModel':
        """Create a deep copy of the model."""
        cloned = MultinomialModel(self.probs(), self.k())
        cloned.set_conjugate_prior(self._alpha)
        
        # Copy data
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        """String representation."""
        probs_str = str(self.probs().to_numpy())
        return (f"MultinomialModel(k={self.k()}, probs={probs_str}, "
                f"data_points={len(self._data)})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)