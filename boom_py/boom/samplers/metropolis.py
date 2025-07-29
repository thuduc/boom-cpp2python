"""Metropolis-Hastings samplers."""
import numpy as np
from typing import Callable, Optional, Union
from .base import Sampler, AdaptiveSampler, MoveAccounting
from ..distributions.rng import GlobalRng


class MetropolisHastings(Sampler):
    """Metropolis-Hastings sampler."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float],
                 proposal_cov: Optional[np.ndarray] = None,
                 rng: Optional[GlobalRng] = None):
        """Initialize Metropolis-Hastings sampler.
        
        Args:
            target_log_density: Function that computes log target density
            proposal_cov: Proposal covariance matrix
            rng: Random number generator
        """
        super().__init__(target_log_density)
        self.proposal_cov = proposal_cov
        self.rng = rng or GlobalRng()
        self.accounting = MoveAccounting("MetropolisHastings")
    
    def set_proposal_covariance(self, cov: np.ndarray):
        """Set proposal covariance matrix."""
        cov = np.asarray(cov)
        if cov.ndim == 0:
            # Scalar - use as diagonal
            self.proposal_cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            # Vector - use as diagonal matrix
            self.proposal_cov = np.diag(cov)
        else:
            # Matrix
            self.proposal_cov = cov
    
    def draw(self) -> np.ndarray:
        """Draw one sample using Metropolis-Hastings."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Generate proposal
        if self.proposal_cov is None:
            # Default to identity matrix
            dim = len(self.current_value)
            proposal = self.current_value + self.rng.rnorm_vec(dim)
        else:
            # Use specified covariance
            L = np.linalg.cholesky(self.proposal_cov)
            z = self.rng.rnorm_vec(len(self.current_value))
            proposal = self.current_value + L @ z
        
        # Evaluate target density at proposal
        try:
            proposal_log_density = self.target_log_density(proposal)
        except:
            # If evaluation fails, reject
            proposal_log_density = -np.inf
        
        # Metropolis-Hastings acceptance step
        log_accept_prob = proposal_log_density - self.current_log_density
        
        accept = False
        if log_accept_prob >= 0:
            accept = True
        elif np.log(self.rng.runif()) < log_accept_prob:
            accept = True
        
        # Update state
        self.n_draws += 1
        self.accounting.record_attempt(accept, proposal_log_density)
        
        if accept:
            self.current_value = proposal
            self.current_log_density = proposal_log_density
            self.n_accepted += 1
        
        return self.current_value.copy()
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate."""
        return self.accounting.acceptance_rate


class RandomWalkMetropolis(MetropolisHastings):
    """Random walk Metropolis sampler with adaptive step size."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float],
                 step_size: float = 1.0,
                 target_acceptance_rate: float = 0.44,
                 rng: Optional[GlobalRng] = None):
        """Initialize random walk Metropolis sampler.
        
        Args:
            target_log_density: Function that computes log target density
            step_size: Initial step size
            target_acceptance_rate: Target acceptance rate for adaptation
            rng: Random number generator
        """
        super().__init__(target_log_density, rng=rng)
        self.step_size = step_size
        self.target_acceptance_rate = target_acceptance_rate
        self.accounting = MoveAccounting("RandomWalkMetropolis")
    
    def draw(self) -> np.ndarray:
        """Draw using random walk proposal."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Random walk proposal
        dim = len(self.current_value)
        proposal = self.current_value + self.step_size * self.rng.rnorm_vec(dim)
        
        # Evaluate target density
        try:
            proposal_log_density = self.target_log_density(proposal)
        except:
            proposal_log_density = -np.inf
        
        # Accept/reject
        log_accept_prob = proposal_log_density - self.current_log_density
        
        accept = False
        if log_accept_prob >= 0:
            accept = True
        elif np.log(self.rng.runif()) < log_accept_prob:
            accept = True
        
        # Update
        self.n_draws += 1
        self.accounting.record_attempt(accept, proposal_log_density)
        
        if accept:
            self.current_value = proposal
            self.current_log_density = proposal_log_density
            self.n_accepted += 1
        
        return self.current_value.copy()
    
    def adapt_step_size(self, factor: float = 1.01):
        """Adapt step size based on acceptance rate."""
        current_rate = self.acceptance_rate
        if current_rate > self.target_acceptance_rate:
            self.step_size *= factor
        else:
            self.step_size /= factor


class AdaptiveMetropolis(AdaptiveSampler):
    """Adaptive Metropolis sampler that learns proposal covariance."""
    
    def __init__(self, target_log_density: Callable[[np.ndarray], float],
                 initial_cov: Optional[np.ndarray] = None,
                 adaptation_frequency: int = 100,
                 rng: Optional[GlobalRng] = None):
        """Initialize adaptive Metropolis sampler.
        
        Args:
            target_log_density: Function that computes log target density
            initial_cov: Initial proposal covariance
            adaptation_frequency: How often to adapt
            rng: Random number generator
        """
        super().__init__(target_log_density, adaptation_frequency)
        self.rng = rng or GlobalRng()
        self.initial_cov = initial_cov
        self.proposal_cov = None
        self.sample_history = []
        self.accounting = MoveAccounting("AdaptiveMetropolis")
        
        # Adaptation parameters
        self.sd = 2.4  # Scaling factor (Gelman et al. 1996)
        self.epsilon = 1e-8  # Regularization
    
    def _draw_impl(self) -> np.ndarray:
        """Implementation of the draw."""
        if self.current_value is None:
            raise ValueError("Must set initial value before drawing")
        
        # Set up proposal covariance if not done
        if self.proposal_cov is None:
            dim = len(self.current_value)
            if self.initial_cov is not None:
                self.proposal_cov = self.initial_cov.copy()
            else:
                self.proposal_cov = np.eye(dim)
        
        # Generate proposal
        L = np.linalg.cholesky(self.proposal_cov + self.epsilon * np.eye(len(self.current_value)))
        z = self.rng.rnorm_vec(len(self.current_value))
        proposal = self.current_value + L @ z
        
        # Evaluate target
        try:
            proposal_log_density = self.target_log_density(proposal)
        except:
            proposal_log_density = -np.inf
        
        # Accept/reject
        log_accept_prob = proposal_log_density - self.current_log_density
        
        accept = False
        if log_accept_prob >= 0:
            accept = True
        elif np.log(self.rng.runif()) < log_accept_prob:
            accept = True
        
        # Update
        self.n_draws += 1
        self.accounting.record_attempt(accept, proposal_log_density)
        
        if accept:
            self.current_value = proposal
            self.current_log_density = proposal_log_density
            self.n_accepted += 1
        
        # Store sample for adaptation
        self.sample_history.append(self.current_value.copy())
        
        return self.current_value.copy()
    
    def adapt(self):
        """Adapt proposal covariance based on sample history."""
        if len(self.sample_history) < 2:
            return
        
        # Compute sample covariance
        samples = np.array(self.sample_history[-self.adaptation_frequency:])
        if len(samples) < 2:
            return
        
        sample_cov = np.cov(samples.T)
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])
        
        # Update proposal covariance
        d = len(self.current_value)
        self.proposal_cov = (self.sd ** 2 / d) * sample_cov
        
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(self.proposal_cov)
        eigvals = np.maximum(eigvals, self.epsilon)
        self.proposal_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate."""
        return self.accounting.acceptance_rate