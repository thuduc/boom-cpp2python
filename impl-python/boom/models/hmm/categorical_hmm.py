"""Categorical Hidden Markov Model."""

import numpy as np
from typing import List, Optional, Union, Tuple, Any
import scipy.special

from .base import HmmModel, HmmData
from boom.linalg import Vector, Matrix
from boom.distributions import RNG


class CategoricalHmm(HmmModel):
    """Hidden Markov Model with categorical emissions.
    
    Each hidden state has a categorical emission distribution over K categories.
    """
    
    def __init__(self, n_states: int, n_categories: int,
                 emission_probs: Optional[Union[List[List[float]], np.ndarray, Matrix]] = None):
        """Initialize Categorical HMM.
        
        Args:
            n_states: Number of hidden states
            n_categories: Number of emission categories
            emission_probs: Emission probability matrix (n_states x n_categories)
                          emission_probs[i, k] = P(observation = k | state = i)
                          If None, initialized uniformly
        """
        super().__init__(n_states)
        self._n_categories = n_categories
        
        # Initialize emission probabilities
        if emission_probs is None:
            # Uniform emission probabilities
            self._emission_probs = Matrix((n_states, n_categories), fill_value=1.0/n_categories)
        else:
            if isinstance(emission_probs, (list, np.ndarray)):
                emit_mat = Matrix(emission_probs)
            elif isinstance(emission_probs, Matrix):
                emit_mat = emission_probs.copy()
            else:
                raise ValueError(f"emission_probs must be list, ndarray, or Matrix, got {type(emission_probs)}")
            
            if emit_mat.shape() != (n_states, n_categories):
                raise ValueError(f"emission_probs shape {emit_mat.shape()} doesn't match "
                               f"({n_states}, {n_categories})")
            
            # Check stochastic constraints
            emit_array = emit_mat.to_numpy()
            if not np.allclose(np.sum(emit_array, axis=1), 1.0):
                raise ValueError("Each row of emission_probs must sum to 1")
            
            if np.any(emit_array < 0):
                raise ValueError("Emission probabilities must be non-negative")
            
            self._emission_probs = emit_mat
    
    @property
    def n_categories(self) -> int:
        """Get number of emission categories."""
        return self._n_categories
    
    def emission_probs(self) -> Matrix:
        """Get emission probability matrix."""
        return self._emission_probs.copy()
    
    def set_emission_probs(self, probs: Union[List[List[float]], np.ndarray, Matrix]):
        """Set emission probability matrix."""
        if isinstance(probs, (list, np.ndarray)):
            probs_mat = Matrix(probs)
        elif isinstance(probs, Matrix):
            probs_mat = probs.copy()
        else:
            raise ValueError(f"probs must be list, ndarray, or Matrix, got {type(probs)}")
        
        if probs_mat.shape() != (self._n_states, self._n_categories):
            raise ValueError(f"probs shape {probs_mat.shape()} doesn't match "
                           f"({self._n_states}, {self._n_categories})")
        
        # Check stochastic constraints
        probs_array = probs_mat.to_numpy()
        if not np.allclose(np.sum(probs_array, axis=1), 1.0):
            raise ValueError("Each row of emission probabilities must sum to 1")
        
        if np.any(probs_array < 0):
            raise ValueError("Emission probabilities must be non-negative")
        
        self._emission_probs = probs_mat
        self._notify_observers()
    
    def emission_log_prob(self, state: int, observation: Any) -> float:
        """Compute log probability of observation given state."""
        if not isinstance(observation, int) or observation < 0 or observation >= self._n_categories:
            raise ValueError(f"Observation must be integer in [0, {self._n_categories-1}] for Categorical HMM")
        
        prob = self._emission_probs[state, observation]
        return np.log(max(prob, 1e-10))  # Avoid log(0)
    
    def sample_emission(self, state: int, rng: Optional[RNG] = None) -> int:
        """Sample observation from emission distribution for given state."""
        if rng is None:
            rng = RNG()
        
        emit_probs = self._emission_probs.row(state).to_numpy()
        return rng.choice(self._n_categories, p=emit_probs)
    
    def fit_em(self, max_iterations: int = 100, tolerance: float = 1e-6,
               verbose: bool = False) -> Tuple[List[float], bool]:
        """Fit model parameters using Expectation-Maximization algorithm.
        
        Args:
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance for log-likelihood
            verbose: Whether to print progress
            
        Returns:
            Tuple of (log_likelihood_history, converged)
        """
        if self.time_dimension() == 0:
            raise ValueError("No data to fit")
        
        log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: compute posterior state probabilities
            gamma = self.posterior_state_probs()
            
            # Compute transition expectations
            xi = self._compute_xi_expectations(gamma)
            
            # M-step: update parameters
            self._update_initial_probs(gamma)
            self._update_transition_matrix(xi)
            self._update_emission_params(gamma)
            
            # Compute log-likelihood
            current_log_likelihood = self.log_likelihood()
            log_likelihood_history.append(current_log_likelihood)
            
            if verbose:
                print(f"EM Iteration {iteration + 1}: Log-likelihood = {current_log_likelihood:.6f}")
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                if verbose:
                    print(f"EM converged after {iteration + 1} iterations")
                return log_likelihood_history, True
            
            prev_log_likelihood = current_log_likelihood
        
        if verbose:
            print(f"EM did not converge after {max_iterations} iterations")
        
        return log_likelihood_history, False
    
    def _compute_xi_expectations(self, gamma: Matrix) -> Matrix:
        """Compute transition expectations ξ_t(i,j) = P(S_t = i, S_{t+1} = j | y_1:T)."""
        T = self.time_dimension()
        xi_sum = Matrix((self._n_states, self._n_states))
        
        if T <= 1:
            return xi_sum
        
        alpha, beta, log_likelihood = self.forward_backward()
        
        for t in range(T - 1):
            xi_t = Matrix((self._n_states, self._n_states))
            
            for i in range(self._n_states):
                for j in range(self._n_states):
                    log_trans = np.log(self._transition_matrix[i, j])
                    obs_logprob = self.emission_log_prob(j, self._data[t + 1].y())
                    
                    log_xi = alpha[t, i] + log_trans + obs_logprob + beta[t + 1, j]
                    xi_t[i, j] = np.exp(log_xi - log_likelihood)
            
            xi_sum += xi_t
        
        return xi_sum
    
    def _update_initial_probs(self, gamma: Matrix):
        """Update initial state probabilities."""
        new_probs = gamma.row(0).to_numpy()
        self.set_initial_probs(new_probs)
    
    def _update_transition_matrix(self, xi: Matrix):
        """Update transition probability matrix."""
        A_new = xi.to_numpy()
        
        # Normalize each row
        for i in range(self._n_states):
            row_sum = np.sum(A_new[i, :])
            if row_sum > 0:
                A_new[i, :] /= row_sum
            else:
                # If no transitions from state i, use uniform distribution
                A_new[i, :] = 1.0 / self._n_states
        
        self.set_transition_matrix(A_new)
    
    def _update_emission_params(self, gamma: Matrix):
        """Update emission parameters."""
        T = self.time_dimension()
        
        new_emission_probs = np.zeros((self._n_states, self._n_categories))
        
        for i in range(self._n_states):
            for k in range(self._n_categories):
                # Count weighted observations for state i and category k
                weighted_count = 0.0
                total_weight = 0.0
                
                for t in range(T):
                    weight = gamma[t, i]
                    observation = int(self._data[t].y())
                    
                    total_weight += weight
                    if observation == k:
                        weighted_count += weight
                
                if total_weight > 1e-10:
                    new_emission_probs[i, k] = weighted_count / total_weight
                else:
                    # If no weight for state i, use uniform distribution
                    new_emission_probs[i, k] = 1.0 / self._n_categories
            
            # Ensure row sums to 1 (handle numerical precision issues)
            row_sum = np.sum(new_emission_probs[i, :])
            if row_sum > 0:
                new_emission_probs[i, :] /= row_sum
            else:
                new_emission_probs[i, :] = 1.0 / self._n_categories
        
        self.set_emission_probs(new_emission_probs)
    
    def predict_next(self, n_ahead: int = 1) -> List[Vector]:
        """Predict future observation probabilities.
        
        Args:
            n_ahead: Number of steps ahead to predict
            
        Returns:
            List of probability vectors for predicted observations
        """
        if self.time_dimension() == 0:
            raise ValueError("No data to predict from")
        
        # Get posterior state probabilities at final time
        gamma = self.posterior_state_probs()
        final_state_probs = gamma.row(gamma.nrow() - 1).to_numpy()
        
        predictions = []
        current_state_probs = final_state_probs.copy()
        
        for h in range(n_ahead):
            # Predict observation probabilities based on current state probabilities
            pred_probs = np.zeros(self._n_categories)
            
            for k in range(self._n_categories):
                for i in range(self._n_states):
                    pred_probs[k] += current_state_probs[i] * self._emission_probs[i, k]
            
            predictions.append(Vector(pred_probs))
            
            # Evolve state probabilities for next prediction
            if h < n_ahead - 1:
                A = self._transition_matrix.to_numpy()
                current_state_probs = current_state_probs @ A
        
        return predictions
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Vectorize parameters for optimization."""
        base_params = super().vectorize_params(minimal).to_numpy()
        
        # Add emission parameters (use log-ratio for each row)
        emission_params = []
        B = self._emission_probs.to_numpy()
        
        if minimal and self._n_categories > 1:
            for i in range(self._n_states):
                row = B[i, :]
                log_ratios = np.log(row[:-1] / row[-1])
                emission_params.extend(log_ratios)
        else:
            emission_params.extend(B.flatten())
        
        all_params = np.concatenate([base_params, emission_params])
        return Vector(all_params)
    
    def unvectorize_params(self, theta: Vector):
        """Set parameters from vector."""
        theta_array = theta.to_numpy()
        
        # Extract base parameters (initial probs and transition matrix)
        n_base_params = (self._n_states - 1) + self._n_states * (self._n_states - 1)
        base_params = Vector(theta_array[:n_base_params])
        super().unvectorize_params(base_params)
        
        # Extract emission parameters
        idx = n_base_params
        B = np.zeros((self._n_states, self._n_categories))
        
        if self._n_categories > 1:
            for i in range(self._n_states):
                log_ratios = theta_array[idx:idx + self._n_categories - 1]
                idx += self._n_categories - 1
                
                # Convert from log-ratio to probabilities
                ratios = np.exp(log_ratios)
                row = np.zeros(self._n_categories)
                row[:-1] = ratios
                row[-1] = 1.0
                row /= np.sum(row)
                
                B[i, :] = row
        else:
            B[:, 0] = 1.0
        
        self.set_emission_probs(B)
    
    def clone(self) -> 'CategoricalHmm':
        """Create a copy of this model."""
        cloned = CategoricalHmm(self._n_states, self._n_categories, self._emission_probs)
        cloned.set_initial_probs(self._initial_probs)
        cloned.set_transition_matrix(self._transition_matrix)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"CategoricalHmm(n_states={self._n_states}, "
                f"n_categories={self._n_categories}, "
                f"time_points={self.time_dimension()})")