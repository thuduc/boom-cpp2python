"""Base classes for Hidden Markov Models."""

import numpy as np
from typing import List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

from boom.models.base import Model
from boom.models.data import Data
from boom.linalg import Vector, Matrix
from boom.distributions import RNG


class HmmData(Data):
    """Data container for HMM observations."""
    
    def __init__(self, observations: Union[List[Union[float, int, Any]], Union[float, int, Any]], timestamp: Optional[int] = None):
        """Initialize HMM data.
        
        Args:
            observations: List of observations or single observation
            timestamp: Time index (optional)
        """
        super().__init__()
        if isinstance(observations, list):
            self._observations = observations
        else:
            # Single observation - for backward compatibility
            self._y = observations
            self._observations = [observations]
        self._timestamp = timestamp
    
    def y(self) -> Any:
        """Get observation (first observation for backward compatibility)."""
        return self._observations[0] if self._observations else None
    
    def get_observations(self) -> List[Any]:
        """Get all observations."""
        return self._observations.copy()
    
    def n_observations(self) -> int:
        """Get number of observations."""
        return len(self._observations)
    
    def timestamp(self) -> Optional[int]:
        """Get timestamp."""
        return self._timestamp
    
    def set_y(self, y: Any):
        """Set observation (updates first observation)."""
        if self._observations:
            self._observations[0] = y
        else:
            self._observations = [y]
        self._y = y  # For backward compatibility
    
    def set_timestamp(self, timestamp: Optional[int]):
        """Set timestamp."""
        self._timestamp = timestamp
    
    def clone(self) -> 'HmmData':
        """Create a copy of this data point."""
        return HmmData(self._observations.copy(), self._timestamp)
    
    def __repr__(self) -> str:
        return f"HmmData(observations={self._observations}, timestamp={self._timestamp})"


class HmmModel(Model, ABC):
    """Base class for Hidden Markov Models.
    
    An HMM consists of:
    - Initial state distribution π
    - Transition probability matrix A
    - Emission distributions for each state
    """
    
    def __init__(self, n_states: int):
        """Initialize HMM model.
        
        Args:
            n_states: Number of hidden states
        """
        super().__init__()
        self._n_states = n_states
        self._initial_probs = Vector(np.ones(n_states) / n_states)
        self._transition_matrix = Matrix(np.ones((n_states, n_states)) / n_states)
        self._data: List[HmmData] = []
    
    @property
    def n_states(self) -> int:
        """Get number of states."""
        return self._n_states
    
    def initial_probs(self) -> Vector:
        """Get initial state probabilities."""
        return self._initial_probs.copy()
    
    def set_initial_probs(self, probs: Union[List[float], np.ndarray, Vector]):
        """Set initial state probabilities."""
        if isinstance(probs, (list, np.ndarray)):
            probs_vec = Vector(probs)
        elif isinstance(probs, Vector):
            probs_vec = probs.copy()
        else:
            raise ValueError(f"probs must be list, ndarray, or Vector, got {type(probs)}")
        
        if len(probs_vec) != self._n_states:
            raise ValueError(f"probs length {len(probs_vec)} doesn't match n_states {self._n_states}")
        
        if not np.allclose(np.sum(probs_vec.to_numpy()), 1.0):
            raise ValueError("Initial probabilities must sum to 1")
        
        if np.any(probs_vec.to_numpy() < 0):
            raise ValueError("Initial probabilities must be non-negative")
        
        self._initial_probs = probs_vec
        self._notify_observers()
    
    def transition_matrix(self) -> Matrix:
        """Get transition probability matrix."""
        return self._transition_matrix.copy()
    
    def set_transition_matrix(self, A: Union[List[List[float]], np.ndarray, Matrix]):
        """Set transition probability matrix."""
        if isinstance(A, (list, np.ndarray)):
            A_mat = Matrix(A)
        elif isinstance(A, Matrix):
            A_mat = A.copy()
        else:
            raise ValueError(f"A must be list of lists, ndarray, or Matrix, got {type(A)}")
        
        if A_mat.shape() != (self._n_states, self._n_states):
            raise ValueError(f"Transition matrix shape {A_mat.shape()} doesn't match ({self._n_states}, {self._n_states})")
        
        # Check stochastic constraints
        A_array = A_mat.to_numpy()
        if not np.allclose(np.sum(A_array, axis=1), 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")
        
        if np.any(A_array < 0):
            raise ValueError("Transition probabilities must be non-negative")
        
        self._transition_matrix = A_mat
        self._notify_observers()
    
    def add_data(self, data: Union[HmmData, Any, List[Union[HmmData, Any]]]):
        """Add observation data."""
        if isinstance(data, HmmData):
            self._add_single_data(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, HmmData):
                    self._add_single_data(item)
                else:
                    hmm_data = HmmData(item)
                    self._add_single_data(hmm_data)
        else:
            hmm_data = HmmData(data)
            self._add_single_data(hmm_data)
    
    def _add_single_data(self, data: HmmData):
        """Add a single data point."""
        if data.timestamp() is None:
            data.set_timestamp(len(self._data))
        self._data.append(data)
    
    def clear_data(self):
        """Clear all data."""
        self._data.clear()
    
    def get_data(self, t: int) -> HmmData:
        """Get data at time t."""
        return self._data[t]
    
    def time_dimension(self) -> int:
        """Get number of time points."""
        return len(self._data)
    
    @abstractmethod
    def emission_log_prob(self, state: int, observation: Any) -> float:
        """Compute log probability of observation given state.
        
        Args:
            state: Hidden state index
            observation: Observation value
            
        Returns:
            Log probability of observation given state
        """
        pass
    
    @abstractmethod
    def sample_emission(self, state: int, rng: Optional[RNG] = None) -> Any:
        """Sample observation from emission distribution for given state.
        
        Args:
            state: Hidden state index
            rng: Random number generator
            
        Returns:
            Sampled observation
        """
        pass
    
    def forward_backward(self) -> Tuple[Matrix, Matrix, float]:
        """Run forward-backward algorithm.
        
        Returns:
            Tuple of (alpha, beta, log_likelihood) where:
            - alpha[t, i] = P(y_1:t, S_t = i)
            - beta[t, i] = P(y_{t+1}:T | S_t = i)  
            - log_likelihood = log P(y_1:T)
        """
        T = self.time_dimension()
        if T == 0:
            return Matrix((0, self._n_states)), Matrix((0, self._n_states)), 0.0
        
        alpha = Matrix((T, self._n_states))
        beta = Matrix((T, self._n_states))
        
        # Forward pass
        # Initialize: α_1(i) = π_i * b_i(y_1)
        for i in range(self._n_states):
            obs_logprob = self.emission_log_prob(i, self._data[0].y())
            alpha[0, i] = np.log(self._initial_probs[i]) + obs_logprob
        
        # Normalize to prevent underflow
        log_norm_0 = self._log_sum_exp(alpha.row(0).to_numpy())
        for i in range(self._n_states):
            alpha[0, i] -= log_norm_0
        
        log_likelihood = log_norm_0
        
        # Forward recursion: α_{t+1}(j) = [Σ_i α_t(i) * a_{ij}] * b_j(y_{t+1})
        for t in range(1, T):
            for j in range(self._n_states):
                log_sum = -np.inf
                for i in range(self._n_states):
                    log_trans = np.log(self._transition_matrix[i, j])
                    log_sum = self._log_sum_exp([log_sum, alpha[t-1, i] + log_trans])
                
                obs_logprob = self.emission_log_prob(j, self._data[t].y())
                alpha[t, j] = log_sum + obs_logprob
            
            # Normalize
            log_norm_t = self._log_sum_exp(alpha.row(t).to_numpy())
            for j in range(self._n_states):
                alpha[t, j] -= log_norm_t
            
            log_likelihood += log_norm_t
        
        # Backward pass
        # Initialize: β_T(i) = 1 (in log space: 0)
        for i in range(self._n_states):
            beta[T-1, i] = 0.0
        
        # Backward recursion: β_t(i) = Σ_j a_{ij} * b_j(y_{t+1}) * β_{t+1}(j)
        for t in range(T-2, -1, -1):
            for i in range(self._n_states):
                log_sum = -np.inf
                for j in range(self._n_states):
                    log_trans = np.log(self._transition_matrix[i, j])
                    obs_logprob = self.emission_log_prob(j, self._data[t+1].y())
                    log_sum = self._log_sum_exp([log_sum, log_trans + obs_logprob + beta[t+1, j]])
                
                beta[t, i] = log_sum
        
        return alpha, beta, log_likelihood
    
    def viterbi(self) -> Tuple[List[int], float]:
        """Find most likely state sequence using Viterbi algorithm.
        
        Returns:
            Tuple of (state_sequence, log_probability)
        """
        T = self.time_dimension()
        if T == 0:
            return [], 0.0
        
        delta = Matrix((T, self._n_states))
        psi = Matrix((T, self._n_states))
        
        # Initialize: δ_1(i) = π_i * b_i(y_1)
        for i in range(self._n_states):
            obs_logprob = self.emission_log_prob(i, self._data[0].y())
            delta[0, i] = np.log(self._initial_probs[i]) + obs_logprob
            psi[0, i] = 0
        
        # Recursion: δ_t(j) = max_i [δ_{t-1}(i) * a_{ij}] * b_j(y_t)
        for t in range(1, T):
            for j in range(self._n_states):
                max_val = -np.inf
                max_idx = 0
                
                for i in range(self._n_states):
                    log_trans = np.log(self._transition_matrix[i, j])
                    val = delta[t-1, i] + log_trans
                    if val > max_val:
                        max_val = val
                        max_idx = i
                
                obs_logprob = self.emission_log_prob(j, self._data[t].y())
                delta[t, j] = max_val + obs_logprob
                psi[t, j] = max_idx
        
        # Termination: find best final state
        max_prob = -np.inf
        best_state = 0
        for i in range(self._n_states):
            if delta[T-1, i] > max_prob:
                max_prob = delta[T-1, i]
                best_state = i
        
        # Backtrack to find path
        path = [0] * T
        path[T-1] = best_state
        
        for t in range(T-2, -1, -1):
            path[t] = int(psi[t+1, path[t+1]])
        
        return path, max_prob
    
    def posterior_state_probs(self) -> Matrix:
        """Compute posterior state probabilities P(S_t = i | y_1:T).
        
        Returns:
            Matrix where gamma[t, i] = P(S_t = i | y_1:T)
        """
        alpha, beta, log_likelihood = self.forward_backward()
        T = self.time_dimension()
        
        if T == 0:
            return Matrix((0, self._n_states))
        
        gamma = Matrix((T, self._n_states))
        
        for t in range(T):
            log_probs = []
            for i in range(self._n_states):
                log_probs.append(alpha[t, i] + beta[t, i])
            
            log_norm = self._log_sum_exp(log_probs)
            
            for i in range(self._n_states):
                gamma[t, i] = np.exp(log_probs[i] - log_norm)
        
        return gamma
    
    def log_likelihood(self) -> float:
        """Compute log likelihood of observed data."""
        _, _, log_lik = self.forward_backward()
        return log_lik
    
    def simulate_data(self, n: int, rng: Optional[RNG] = None) -> Tuple[List[HmmData], List[int]]:
        """Simulate data from the HMM.
        
        Args:
            n: Number of time points to simulate
            rng: Random number generator
            
        Returns:
            Tuple of (observations, hidden_states)
        """
        if rng is None:
            rng = RNG()
        
        observations = []
        states = []
        
        # Sample initial state
        state = rng.choice(self._n_states, p=self._initial_probs.to_numpy())
        
        for t in range(n):
            states.append(state)
            
            # Sample observation
            obs = self.sample_emission(state, rng)
            observations.append(HmmData(obs, timestamp=t))
            
            # Sample next state
            if t < n - 1:
                trans_probs = self._transition_matrix.row(state).to_numpy()
                state = rng.choice(self._n_states, p=trans_probs)
        
        return observations, states
    
    def _log_sum_exp(self, log_values: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        if len(log_values) == 0:
            return -np.inf
        
        max_val = np.max(log_values)
        if max_val == -np.inf:
            return -np.inf
        
        return max_val + np.log(np.sum(np.exp(log_values - max_val)))
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Vectorize parameters for optimization."""
        params = []
        
        # Initial probabilities (use log-ratio to ensure simplex constraint)
        if minimal and self._n_states > 1:
            init_probs = self._initial_probs.to_numpy()
            log_ratios = np.log(init_probs[:-1] / init_probs[-1])
            params.extend(log_ratios)
        else:
            params.extend(self._initial_probs.to_numpy())
        
        # Transition matrix (use log-ratio for each row)
        A = self._transition_matrix.to_numpy()
        if minimal and self._n_states > 1:
            for i in range(self._n_states):
                row = A[i, :]
                log_ratios = np.log(row[:-1] / row[-1])
                params.extend(log_ratios)
        else:
            params.extend(A.flatten())
        
        return Vector(params)
    
    def unvectorize_params(self, theta: Vector):
        """Set parameters from vector."""
        theta_array = theta.to_numpy()
        idx = 0
        
        # Initial probabilities
        if self._n_states > 1:
            log_ratios = theta_array[idx:idx+self._n_states-1]
            idx += self._n_states - 1
            
            # Convert from log-ratio to probabilities
            ratios = np.exp(log_ratios)
            probs = np.zeros(self._n_states)
            probs[:-1] = ratios
            probs[-1] = 1.0
            probs /= np.sum(probs)
            
            self.set_initial_probs(probs)
        
        # Transition matrix
        A = np.zeros((self._n_states, self._n_states))
        for i in range(self._n_states):
            if self._n_states > 1:
                log_ratios = theta_array[idx:idx+self._n_states-1]
                idx += self._n_states - 1
                
                # Convert from log-ratio to probabilities
                ratios = np.exp(log_ratios)
                row = np.zeros(self._n_states)
                row[:-1] = ratios
                row[-1] = 1.0
                row /= np.sum(row)
                
                A[i, :] = row
            else:
                A[i, 0] = 1.0
        
        self.set_transition_matrix(A)
    
    def clone(self) -> 'HmmModel':
        """Create a copy of this model."""
        cloned = self.__class__(self._n_states)
        cloned.set_initial_probs(self._initial_probs)
        cloned.set_transition_matrix(self._transition_matrix)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(n_states={self._n_states}, "
                f"time_points={self.time_dimension()})")
    
    def baum_welch(self, max_iterations: int = 100, tolerance: float = 1e-4) -> bool:
        """Run Baum-Welch algorithm for parameter estimation.
        
        Args:
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance for log likelihood.
            
        Returns:
            True if converged, False otherwise.
        """
        T = self.time_dimension()
        if T == 0:
            return True  # No data to train on
        
        prev_log_lik = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: compute forward-backward probabilities
            alpha, beta, log_lik = self.forward_backward()
            
            # Check convergence
            if abs(log_lik - prev_log_lik) < tolerance:
                return True
            prev_log_lik = log_lik
            
            # Compute gamma and xi
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(alpha, beta)
            
            # M-step: update parameters
            self._update_initial_probs(gamma)
            # Compute xi_sum for transition matrix update
            xi_sum = Matrix((self._n_states, self._n_states))
            for t in range(T-1):
                for i in range(self._n_states):
                    for j in range(self._n_states):
                        xi_sum[i, j] += xi[t, i, j]
            self._update_transition_matrix(xi_sum)
            self._update_emission_params(gamma)
        
        return False  # Did not converge
    
    def _compute_gamma(self, alpha: Matrix, beta: Matrix) -> Matrix:
        """Compute posterior state probabilities gamma."""
        T = self.time_dimension()
        gamma = Matrix((T, self._n_states))
        
        for t in range(T):
            log_probs = []
            for i in range(self._n_states):
                log_probs.append(alpha[t, i] + beta[t, i])
            
            log_norm = self._log_sum_exp(log_probs)
            
            for i in range(self._n_states):
                gamma[t, i] = np.exp(log_probs[i] - log_norm)
        
        return gamma
    
    def _compute_xi(self, alpha: Matrix, beta: Matrix) -> np.ndarray:
        """Compute joint posterior probabilities xi."""
        T = self.time_dimension()
        xi = np.zeros((T-1, self._n_states, self._n_states))
        
        for t in range(T-1):
            log_probs = []
            for i in range(self._n_states):
                for j in range(self._n_states):
                    log_trans = np.log(self._transition_matrix[i, j])
                    obs_logprob = self.emission_log_prob(j, self._data[t+1].y())
                    log_prob = alpha[t, i] + log_trans + obs_logprob + beta[t+1, j]
                    log_probs.append(log_prob)
            
            log_norm = self._log_sum_exp(log_probs)
            
            idx = 0
            for i in range(self._n_states):
                for j in range(self._n_states):
                    xi[t, i, j] = np.exp(log_probs[idx] - log_norm)
                    idx += 1
        
        return xi
    
    def _update_initial_probs(self, gamma: Matrix):
        """Update initial state probabilities."""
        new_probs = Vector(gamma.row(0).to_numpy())
        self.set_initial_probs(new_probs)
    
    @abstractmethod
    def _update_transition_matrix(self, xi_sum: Matrix):
        """Update transition matrix. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _update_emission_params(self, gamma: Matrix):
        """Update emission parameters. Must be implemented by subclasses."""
        pass
    
    def predict_state_probabilities(self, observation: Any) -> Vector:
        """Predict state probabilities for a new observation.
        
        Args:
            observation: New observation value.
            
        Returns:
            Vector of state probabilities.
        """
        # Use current parameters to predict state probabilities
        # This uses Bayes' rule: P(S|y) ∝ P(y|S) * P(S)
        
        # If we have data, use the last state distribution
        if self.time_dimension() > 0:
            # Get posterior state probs at last time point
            gamma = self.posterior_state_probs()
            last_state_probs = gamma.row(gamma.nrow() - 1).to_numpy()
            
            # Evolve one step forward using transition matrix
            prior_probs = self._transition_matrix.to_numpy().T @ last_state_probs
        else:
            # Use initial state probabilities
            prior_probs = self._initial_probs.to_numpy()
        
        # Compute likelihoods P(y|S=i)
        log_likelihoods = []
        for i in range(self._n_states):
            log_likelihoods.append(self.emission_log_prob(i, observation))
        
        # Compute posterior using Bayes' rule
        log_posteriors = []
        for i in range(self._n_states):
            log_posteriors.append(log_likelihoods[i] + np.log(prior_probs[i]))
        
        # Normalize
        log_norm = self._log_sum_exp(log_posteriors)
        posteriors = []
        for i in range(self._n_states):
            posteriors.append(np.exp(log_posteriors[i] - log_norm))
        
        return Vector(posteriors)