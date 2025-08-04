"""Gaussian Hidden Markov Model."""

import numpy as np
from typing import List, Optional, Union, Tuple, Any
import scipy.stats

from .base import HmmModel, HmmData
from boom.linalg import Vector
from boom.distributions import RNG


class GaussianHmm(HmmModel):
    """Hidden Markov Model with Gaussian emissions.
    
    Each hidden state has a Gaussian emission distribution N(μ_i, σ²_i).
    """
    
    def __init__(self, n_states: int, means: Optional[Union[List[float], np.ndarray, Vector]] = None,
                 variances: Optional[Union[List[float], np.ndarray, Vector]] = None):
        """Initialize Gaussian HMM.
        
        Args:
            n_states: Number of hidden states
            means: Mean parameters for each state (default: zeros)
            variances: Variance parameters for each state (default: ones)
        """
        super().__init__(n_states)
        
        # Initialize emission parameters
        if means is None:
            self._means = Vector(np.zeros(n_states))
        else:
            if isinstance(means, (list, np.ndarray)):
                means_vec = Vector(means)
            elif isinstance(means, Vector):
                means_vec = means.copy()
            else:
                raise ValueError(f"means must be list, ndarray, or Vector, got {type(means)}")
            
            if len(means_vec) != n_states:
                raise ValueError(f"means length {len(means_vec)} doesn't match n_states {n_states}")
            
            self._means = means_vec
        
        if variances is None:
            self._variances = Vector(np.ones(n_states))
        else:
            if isinstance(variances, (list, np.ndarray)):
                var_vec = Vector(variances)
            elif isinstance(variances, Vector):
                var_vec = variances.copy()
            else:
                raise ValueError(f"variances must be list, ndarray, or Vector, got {type(variances)}")
            
            if len(var_vec) != n_states:
                raise ValueError(f"variances length {len(var_vec)} doesn't match n_states {n_states}")
            
            if np.any(var_vec.to_numpy() <= 0):
                raise ValueError("All variances must be positive")
            
            self._variances = var_vec
    
    @property
    def means(self) -> Vector:
        """Get emission means."""
        return self._means.copy()
    
    def set_means(self, means: Union[List[float], np.ndarray, Vector]):
        """Set emission means."""
        if isinstance(means, (list, np.ndarray)):
            means_vec = Vector(means)
        elif isinstance(means, Vector):
            means_vec = means.copy()
        else:
            raise ValueError(f"means must be list, ndarray, or Vector, got {type(means)}")
        
        if len(means_vec) != self._n_states:
            raise ValueError(f"means length {len(means_vec)} doesn't match n_states {self._n_states}")
        
        self._means = means_vec
        self._notify_observers()
    
    @property
    def variances(self) -> Vector:
        """Get emission variances."""
        return self._variances.copy()
    
    def set_variances(self, variances: Union[List[float], np.ndarray, Vector]):
        """Set emission variances."""
        if isinstance(variances, (list, np.ndarray)):
            var_vec = Vector(variances)
        elif isinstance(variances, Vector):
            var_vec = variances.copy()
        else:
            raise ValueError(f"variances must be list, ndarray, or Vector, got {type(variances)}")
        
        if len(var_vec) != self._n_states:
            raise ValueError(f"variances length {len(var_vec)} doesn't match n_states {self._n_states}")
        
        if np.any(var_vec.to_numpy() <= 0):
            raise ValueError("All variances must be positive")
        
        self._variances = var_vec
        self._notify_observers()
    
    def emission_log_prob(self, state: int, observation: Any) -> float:
        """Compute log probability of observation given state."""
        if not isinstance(observation, (int, float)):
            raise ValueError("Observation must be numeric for Gaussian HMM")
        
        y = float(observation)
        mu = self._means[state]
        sigma2 = self._variances[state]
        
        return scipy.stats.norm.logpdf(y, loc=mu, scale=np.sqrt(sigma2))
    
    def sample_emission(self, state: int, rng: Optional[RNG] = None) -> float:
        """Sample observation from emission distribution for given state."""
        if rng is None:
            rng = RNG()
        
        mu = self._means[state]
        sigma = np.sqrt(self._variances[state])
        
        return rng.randn() * sigma + mu
    
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
    
    def _compute_xi_expectations(self, gamma: 'Matrix') -> 'Matrix':
        """Compute transition expectations ξ_t(i,j) = P(S_t = i, S_{t+1} = j | y_1:T)."""
        from boom.linalg import Matrix
        
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
    
    def _update_initial_probs(self, gamma: 'Matrix'):
        """Update initial state probabilities."""
        new_probs = gamma.row(0).to_numpy()
        self.set_initial_probs(new_probs)
    
    def _update_transition_matrix(self, xi: 'Matrix'):
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
    
    def _update_emission_params(self, gamma: 'Matrix'):
        """Update emission parameters (means and variances)."""
        T = self.time_dimension()
        
        new_means = np.zeros(self._n_states)
        new_variances = np.zeros(self._n_states)
        
        for i in range(self._n_states):
            # Compute weighted mean
            gamma_sum = 0.0
            weighted_sum = 0.0
            
            for t in range(T):
                weight = gamma[t, i]
                observation = float(self._data[t].y())
                
                gamma_sum += weight
                weighted_sum += weight * observation
            
            if gamma_sum > 1e-10:
                new_means[i] = weighted_sum / gamma_sum
            else:
                new_means[i] = self._means[i]  # Keep current value
            
            # Compute weighted variance
            weighted_var_sum = 0.0
            
            for t in range(T):
                weight = gamma[t, i]
                observation = float(self._data[t].y())
                diff = observation - new_means[i]
                
                weighted_var_sum += weight * diff * diff
            
            if gamma_sum > 1e-10:
                new_variances[i] = max(weighted_var_sum / gamma_sum, 1e-6)  # Minimum variance
            else:
                new_variances[i] = self._variances[i]  # Keep current value
        
        self.set_means(new_means)
        self.set_variances(new_variances)
    
    def predict_next(self, n_ahead: int = 1) -> List[Tuple[float, float]]:
        """Predict future observations.
        
        Args:
            n_ahead: Number of steps ahead to predict
            
        Returns:
            List of (mean, variance) pairs for predicted observations
        """
        if self.time_dimension() == 0:
            raise ValueError("No data to predict from")
        
        # Get posterior state probabilities at final time
        gamma = self.posterior_state_probs()
        final_state_probs = gamma.row(gamma.nrow() - 1).to_numpy()
        
        predictions = []
        current_state_probs = final_state_probs.copy()
        
        for h in range(n_ahead):
            # Predict observation based on current state probabilities
            pred_mean = np.sum(current_state_probs * self._means.to_numpy())
            
            # Compute prediction variance: E[Var[Y|S]] + Var[E[Y|S]]
            pred_var_1 = np.sum(current_state_probs * self._variances.to_numpy())  # E[Var[Y|S]]
            
            means_array = self._means.to_numpy()
            pred_var_2 = np.sum(current_state_probs * means_array**2) - pred_mean**2  # Var[E[Y|S]]
            
            pred_variance = pred_var_1 + pred_var_2
            
            predictions.append((pred_mean, pred_variance))
            
            # Evolve state probabilities for next prediction
            if h < n_ahead - 1:
                A = self._transition_matrix.to_numpy()
                current_state_probs = current_state_probs @ A
        
        return predictions
    
    def vectorize_params(self, minimal: bool = True) -> Vector:
        """Vectorize parameters for optimization."""
        base_params = super().vectorize_params(minimal).to_numpy()
        
        # Add emission parameters
        emission_params = []
        emission_params.extend(self._means.to_numpy())
        emission_params.extend(np.log(self._variances.to_numpy()))  # Log variances for unconstrained optimization
        
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
        means = theta_array[idx:idx + self._n_states]
        idx += self._n_states
        log_variances = theta_array[idx:idx + self._n_states]
        variances = np.exp(log_variances)
        
        self.set_means(means)
        self.set_variances(variances)
    
    def clone(self) -> 'GaussianHmm':
        """Create a copy of this model."""
        cloned = GaussianHmm(self._n_states, self._means, self._variances)
        cloned.set_initial_probs(self._initial_probs)
        cloned.set_transition_matrix(self._transition_matrix)
        
        for data_point in self._data:
            cloned.add_data(data_point.clone())
        
        return cloned
    
    def __str__(self) -> str:
        return (f"GaussianHmm(n_states={self._n_states}, "
                f"means={self._means.to_numpy()}, "
                f"variances={self._variances.to_numpy()}, "
                f"time_points={self.time_dimension()})")