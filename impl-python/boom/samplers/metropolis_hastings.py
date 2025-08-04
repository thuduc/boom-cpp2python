"""Metropolis-Hastings MCMC sampler implementation."""

from typing import Callable, Optional, Union, List
import numpy as np
from abc import ABC, abstractmethod
from ..linalg import Vector
from ..distributions import RNG


class ProposalDistribution(ABC):
    """Abstract base class for proposal distributions in Metropolis-Hastings sampling."""
    
    @abstractmethod
    def propose(self, current_state: Vector, rng: RNG) -> Vector:
        """Generate a proposal given the current state.
        
        Args:
            current_state: Current parameter values.
            rng: Random number generator.
            
        Returns:
            Proposed parameter values.
        """
        pass
    
    @abstractmethod
    def log_density(self, proposal: Vector, current_state: Vector) -> float:
        """Compute log density of proposal given current state.
        
        Args:
            proposal: Proposed parameter values.
            current_state: Current parameter values.
            
        Returns:
            Log density of the proposal.
        """
        pass
    
    def is_symmetric(self) -> bool:
        """Return True if the proposal distribution is symmetric."""
        return True


class RandomWalkProposal(ProposalDistribution):
    """Random walk proposal: new = current + Normal(0, covariance)."""
    
    def __init__(self, covariance: Union[float, Vector, np.ndarray]):
        """Initialize random walk proposal.
        
        Args:
            covariance: Proposal covariance. Can be:
                - float: scalar variance (assumes independence)
                - Vector: diagonal covariance
                - np.ndarray: full covariance matrix
        """
        if isinstance(covariance, (int, float)):
            self._is_scalar = True
            self._variance = float(covariance)
            self._std = np.sqrt(self._variance)
        elif isinstance(covariance, Vector):
            self._is_scalar = False
            self._is_diagonal = True
            self._variances = covariance.to_numpy()
            if np.any(self._variances <= 0):
                raise ValueError("All variances must be positive")
            self._std_devs = np.sqrt(self._variances)
        elif isinstance(covariance, np.ndarray):
            self._is_scalar = False
            self._is_diagonal = False
            if covariance.ndim == 1:
                # Diagonal covariance given as 1D array
                self._is_diagonal = True
                self._variances = covariance.copy()
                if np.any(self._variances <= 0):
                    raise ValueError("All variances must be positive")
                self._std_devs = np.sqrt(self._variances)
            else:
                # Full covariance matrix
                if covariance.shape[0] != covariance.shape[1]:
                    raise ValueError("Covariance matrix must be square")
                # Check positive definiteness
                try:
                    self._chol = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError:
                    raise ValueError("Covariance matrix must be positive definite")
                self._covariance = covariance.copy()
        else:
            raise TypeError("Covariance must be float, Vector, or numpy array")
    
    def propose(self, current_state: Vector, rng: RNG) -> Vector:
        """Generate random walk proposal."""
        current_array = current_state.to_numpy()
        
        if self._is_scalar:
            # Scalar variance
            noise = np.array([rng.normal(0.0, self._std) for _ in range(len(current_array))])
        elif self._is_diagonal:
            # Diagonal covariance
            if len(self._std_devs) != len(current_array):
                raise ValueError(f"Covariance dimension {len(self._std_devs)} doesn't match state dimension {len(current_array)}")
            noise = np.array([rng.normal(0.0, std) for std in self._std_devs])
        else:
            # Full covariance matrix
            if self._covariance.shape[0] != len(current_array):
                raise ValueError(f"Covariance dimension {self._covariance.shape[0]} doesn't match state dimension {len(current_array)}")
            # Generate multivariate normal using Cholesky decomposition
            z = np.array([rng.normal(0.0, 1.0) for _ in range(len(current_array))])
            noise = self._chol @ z
        
        proposal_array = current_array + noise
        return Vector(proposal_array)
    
    def log_density(self, proposal: Vector, current_state: Vector) -> float:
        """Compute log density (symmetric proposal, so returns 0)."""
        return 0.0  # Symmetric proposal
    
    def is_symmetric(self) -> bool:
        """Random walk is symmetric."""
        return True


class IndependenceProposal(ProposalDistribution):
    """Independence proposal: new ~ fixed_distribution (independent of current state)."""
    
    def __init__(self, mean: Vector, covariance: Union[float, Vector, np.ndarray]):
        """Initialize independence proposal.
        
        Args:
            mean: Mean of the proposal distribution.
            covariance: Covariance of the proposal distribution.
        """
        self._mean = mean.copy()
        
        if isinstance(covariance, (int, float)):
            self._is_scalar = True
            self._variance = float(covariance)
            self._std = np.sqrt(self._variance)
        elif isinstance(covariance, Vector):
            self._is_diagonal = True
            self._variances = covariance.to_numpy()
            if np.any(self._variances <= 0):
                raise ValueError("All variances must be positive")
            self._std_devs = np.sqrt(self._variances)
        elif isinstance(covariance, np.ndarray):
            if covariance.ndim == 1:
                self._is_diagonal = True
                self._variances = covariance.copy()
                if np.any(self._variances <= 0):
                    raise ValueError("All variances must be positive")
                self._std_devs = np.sqrt(self._variances)
            else:
                self._is_diagonal = False
                if covariance.shape[0] != covariance.shape[1]:
                    raise ValueError("Covariance matrix must be square")
                try:
                    self._chol = np.linalg.cholesky(covariance)
                    self._cov_inv = np.linalg.inv(covariance)
                    self._log_det_cov = np.log(np.linalg.det(covariance))
                except np.linalg.LinAlgError:
                    raise ValueError("Covariance matrix must be positive definite")
                self._covariance = covariance.copy()
        else:
            raise TypeError("Covariance must be float, Vector, or numpy array")
    
    def propose(self, current_state: Vector, rng: RNG) -> Vector:
        """Generate independence proposal (ignores current state)."""
        mean_array = self._mean.to_numpy()
        
        if hasattr(self, '_is_scalar') and self._is_scalar:
            # Scalar variance
            noise = np.array([rng.normal(0.0, self._std) for _ in range(len(mean_array))])
        elif hasattr(self, '_is_diagonal') and self._is_diagonal:
            # Diagonal covariance
            noise = np.array([rng.normal(0.0, std) for std in self._std_devs])
        else:
            # Full covariance matrix
            z = np.array([rng.normal(0.0, 1.0) for _ in range(len(mean_array))])
            noise = self._chol @ z
        
        proposal_array = mean_array + noise
        return Vector(proposal_array)
    
    def log_density(self, proposal: Vector, current_state: Vector) -> float:
        """Compute log density of multivariate normal proposal."""
        x = proposal.to_numpy() - self._mean.to_numpy()
        
        if hasattr(self, '_is_scalar') and self._is_scalar:
            # Scalar variance
            return -0.5 * (len(x) * np.log(2 * np.pi * self._variance) + np.sum(x * x) / self._variance)
        elif hasattr(self, '_is_diagonal') and self._is_diagonal:
            # Diagonal covariance
            log_det = np.sum(np.log(self._variances))
            quad_form = np.sum(x * x / self._variances)
            return -0.5 * (len(x) * np.log(2 * np.pi) + log_det + quad_form)
        else:
            # Full covariance matrix
            quad_form = x @ self._cov_inv @ x
            return -0.5 * (len(x) * np.log(2 * np.pi) + self._log_det_cov + quad_form)
    
    def is_symmetric(self) -> bool:
        """Independence proposal is not symmetric."""
        return False


class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""
    
    def __init__(self, log_target: Callable[[Vector], float], 
                 proposal: ProposalDistribution,
                 rng: Optional[RNG] = None):
        """Initialize Metropolis-Hastings sampler.
        
        Args:
            log_target: Function that computes log target density.
            proposal: Proposal distribution.
            rng: Random number generator. If None, creates a new one.
        """
        self._log_target = log_target
        self._proposal = proposal
        self._rng = rng if rng is not None else RNG()
        
        # Statistics
        self._n_proposals = 0
        self._n_accepted = 0
    
    def sample(self, initial_state: Vector, n_samples: int) -> tuple:
        """Run Metropolis-Hastings sampling.
        
        Args:
            initial_state: Starting parameter values.
            n_samples: Number of samples to generate.
            
        Returns:
            Tuple of (samples, acceptance_rate) where samples is a list of Vector objects.
        """
        samples = []
        current_state = initial_state.copy()
        current_log_target = self._log_target(current_state)
        
        # Reset statistics
        self._n_proposals = 0
        self._n_accepted = 0
        
        for i in range(n_samples):
            # Generate proposal
            proposal = self._proposal.propose(current_state, self._rng)
            self._n_proposals += 1
            
            # Compute acceptance probability
            try:
                proposal_log_target = self._log_target(proposal)
                
                if self._proposal.is_symmetric():
                    # Symmetric proposal: only need target densities
                    log_alpha = proposal_log_target - current_log_target
                else:
                    # Asymmetric proposal: need proposal densities too
                    log_q_proposal_to_current = self._proposal.log_density(current_state, proposal)
                    log_q_current_to_proposal = self._proposal.log_density(proposal, current_state)
                    log_alpha = (proposal_log_target - current_log_target + 
                               log_q_proposal_to_current - log_q_current_to_proposal)
                
                # Accept or reject
                if log_alpha >= 0 or np.log(self._rng()) < log_alpha:
                    # Accept
                    current_state = proposal
                    current_log_target = proposal_log_target
                    self._n_accepted += 1
                
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                # Reject proposals that cause numerical issues
                pass
            
            samples.append(current_state.copy())
        
        acceptance_rate = self._n_accepted / self._n_proposals if self._n_proposals > 0 else 0.0
        return samples, acceptance_rate
    
    def tune_proposal(self, initial_state: Vector, n_tune: int = 1000, 
                     target_acceptance: float = 0.44) -> 'MetropolisHastings':
        """Tune proposal distribution to achieve target acceptance rate.
        
        Args:
            initial_state: Starting parameter values for tuning.
            n_tune: Number of tuning iterations.
            target_acceptance: Target acceptance rate.
            
        Returns:
            New MetropolisHastings object with tuned proposal.
        """
        if not isinstance(self._proposal, RandomWalkProposal):
            raise NotImplementedError("Tuning only implemented for RandomWalkProposal")
        
        # Simple tuning: adjust scalar variance
        if hasattr(self._proposal, '_is_scalar') and self._proposal._is_scalar:
            current_var = self._proposal._variance
            
            # Try a few different variances
            variances = [current_var * factor for factor in [0.5, 0.8, 1.0, 1.25, 2.0]]
            best_var = current_var
            best_score = float('inf')
            
            for var in variances:
                tuned_proposal = RandomWalkProposal(var)
                tuned_sampler = MetropolisHastings(self._log_target, tuned_proposal, self._rng)
                
                _, acceptance_rate = tuned_sampler.sample(initial_state, n_tune)
                score = abs(acceptance_rate - target_acceptance)
                
                if score < best_score:
                    best_score = score
                    best_var = var
            
            # Create new sampler with best variance
            best_proposal = RandomWalkProposal(best_var)
            return MetropolisHastings(self._log_target, best_proposal, self._rng)
        
        else:
            # For more complex proposals, return unchanged
            return self
    
    def acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        if self._n_proposals == 0:
            return 0.0
        return self._n_accepted / self._n_proposals
    
    def n_proposals(self) -> int:
        """Get number of proposals made."""
        return self._n_proposals
    
    def n_accepted(self) -> int:
        """Get number of proposals accepted."""
        return self._n_accepted