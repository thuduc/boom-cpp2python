"""Kalman filter and smoother for linear state space models."""

import numpy as np
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from boom.linalg import Vector, Matrix, SpdMatrix
from .base import StateSpaceModel


@dataclass
class FilterState:
    """State of the Kalman filter at a single time point."""
    # Filtered state
    state_mean: Vector  # E[alpha_t | y_1:t]
    state_variance: SpdMatrix  # Var[alpha_t | y_1:t]
    
    # Predicted state
    predicted_state_mean: Vector  # E[alpha_t | y_1:t-1] 
    predicted_state_variance: SpdMatrix  # Var[alpha_t | y_1:t-1]
    
    # Prediction for observation
    predicted_observation: float  # E[y_t | y_1:t-1]
    prediction_variance: float  # Var[y_t | y_1:t-1]
    
    # Innovation
    innovation: float  # y_t - E[y_t | y_1:t-1]
    
    # Log likelihood contribution
    log_likelihood: float


@dataclass 
class SmootherState:
    """State of the Kalman smoother at a single time point."""
    # Smoothed state
    state_mean: Vector  # E[alpha_t | y_1:T]
    state_variance: SpdMatrix  # Var[alpha_t | y_1:T]
    
    # Smoothed state disturbance
    state_disturbance_mean: Vector  # E[eta_t | y_1:T]
    state_disturbance_variance: SpdMatrix  # Var[eta_t | y_1:T]


class KalmanFilter:
    """Kalman filter for linear Gaussian state space models.
    
    Implements the standard Kalman filter and RTS smoother algorithms
    for computing filtered and smoothed estimates of the state vector.
    """
    
    def __init__(self, model: StateSpaceModel):
        """Initialize Kalman filter.
        
        Args:
            model: State space model to filter
        """
        self.model = model
        self._filter_states: List[FilterState] = []
        self._smoother_states: List[SmootherState] = []
        self._log_likelihood = 0.0
    
    def filter(self, compute_likelihood: bool = True) -> List[FilterState]:
        """Run forward Kalman filter.
        
        Args:
            compute_likelihood: Whether to compute log likelihood
            
        Returns:
            List of filter states for each time point
        """
        self._filter_states = []
        self._log_likelihood = 0.0
        
        if self.model.time_dimension == 0:
            return self._filter_states
        
        # Initialize with prior
        state_mean = self.model.initial_state_mean()
        state_variance = self.model.initial_state_variance()
        
        for t in range(self.model.time_dimension):
            data_t = self.model.get_data(t)
            
            # Prediction step
            if t > 0:
                T_t = self.model.transition_matrix(t-1)
                Q_t = self.model.state_error_variance(t-1)
                
                # Predict state: a_{t|t-1} = T_t * a_{t-1|t-1}
                predicted_state_mean = Vector(T_t.to_numpy() @ state_mean.to_numpy())
                
                # Predict state variance: P_{t|t-1} = T_t * P_{t-1|t-1} * T_t' + Q_t
                P_pred = T_t.to_numpy() @ state_variance.to_numpy() @ T_t.T.to_numpy() + Q_t.to_numpy()
                predicted_state_variance = SpdMatrix(P_pred)
            else:
                predicted_state_mean = state_mean.copy()
                predicted_state_variance = state_variance.copy()
            
            # Observation prediction
            Z_t = self.model.observation_matrix(t)
            H_t = self.model.observation_variance
            
            # Predicted observation: E[y_t | y_1:t-1] = Z_t * a_{t|t-1}
            predicted_obs = float((Z_t @ Vector(predicted_state_mean.to_numpy())).to_numpy()[0])
            
            # Prediction variance: F_t = Z_t * P_{t|t-1} * Z_t' + H_t
            F_t = float((Z_t @ predicted_state_variance @ Z_t.T).to_numpy()[0, 0] + H_t)
            
            # Innovation: v_t = y_t - E[y_t | y_1:t-1]
            if data_t.is_observed():
                innovation = data_t.y() - predicted_obs
            else:
                innovation = 0.0  # No innovation for missing observations
            
            # Update step (only if observation is observed)
            if data_t.is_observed() and F_t > 1e-10:
                # Kalman gain: K_t = P_{t|t-1} * Z_t' * F_t^{-1}
                K_t = (predicted_state_variance.to_numpy() @ Z_t.T.to_numpy()) / F_t
                
                # Update state: a_{t|t} = a_{t|t-1} + K_t * v_t
                updated_state_mean = Vector(predicted_state_mean.to_numpy() + K_t.flatten() * innovation)
                
                # Update state variance: P_{t|t} = P_{t|t-1} - K_t * Z_t * P_{t|t-1}
                P_upd = (predicted_state_variance.to_numpy() - 
                        K_t @ Z_t.to_numpy() @ predicted_state_variance.to_numpy())
                updated_state_variance = SpdMatrix(P_upd)
            else:
                # No update for missing observations
                updated_state_mean = predicted_state_mean.copy()
                updated_state_variance = predicted_state_variance.copy()
            
            # Log likelihood contribution
            if compute_likelihood and data_t.is_observed() and F_t > 1e-10:
                log_lik_t = -0.5 * (np.log(2 * np.pi) + np.log(F_t) + innovation**2 / F_t)
            else:
                log_lik_t = 0.0
            
            self._log_likelihood += log_lik_t
            
            # Store filter state
            filter_state = FilterState(
                state_mean=updated_state_mean,
                state_variance=updated_state_variance,
                predicted_state_mean=predicted_state_mean,
                predicted_state_variance=predicted_state_variance,
                predicted_observation=predicted_obs,
                prediction_variance=F_t,
                innovation=innovation,
                log_likelihood=log_lik_t
            )
            self._filter_states.append(filter_state)
            
            # Update for next iteration
            state_mean = updated_state_mean
            state_variance = updated_state_variance
        
        return self._filter_states
    
    def smooth(self) -> List[SmootherState]:
        """Run backward RTS smoother.
        
        Must be called after filter().
        
        Returns:
            List of smoother states for each time point
        """
        if not self._filter_states:
            raise ValueError("Must run filter() before smooth()")
        
        T = self.model.time_dimension
        self._smoother_states = [None] * T  # type: ignore
        
        if T == 0:
            return []
        
        # Initialize with filtered estimates at final time
        final_filter = self._filter_states[-1]
        final_smoother = SmootherState(
            state_mean=final_filter.state_mean.copy(),
            state_variance=final_filter.state_variance.copy(),
            state_disturbance_mean=Vector(np.zeros(self.model.state_dimension)),
            state_disturbance_variance=SpdMatrix(np.zeros((self.model.state_dimension, 
                                                          self.model.state_dimension)))
        )
        self._smoother_states[T-1] = final_smoother
        
        # Work backwards
        for t in range(T-2, -1, -1):
            filter_t = self._filter_states[t]
            filter_t_plus_1 = self._filter_states[t+1]
            smoother_t_plus_1 = self._smoother_states[t+1]
            
            # Transition matrix and state error variance
            T_t = self.model.transition_matrix(t)
            Q_t = self.model.state_error_variance(t)
            
            # Smoother gain: A_t = P_{t|t} * T_t' * P_{t+1|t}^{-1}
            P_t_given_t = filter_t.state_variance.to_numpy()
            P_t_plus_1_given_t = filter_t_plus_1.predicted_state_variance.to_numpy()
            T_t_array = T_t.to_numpy()
            
            try:
                P_inv = np.linalg.inv(P_t_plus_1_given_t)
                A_t = P_t_given_t @ T_t_array.T @ P_inv
            except np.linalg.LinAlgError:
                # Use pseudoinverse if singular
                P_inv = np.linalg.pinv(P_t_plus_1_given_t)
                A_t = P_t_given_t @ T_t_array.T @ P_inv
            
            # Smoothed state mean: a_{t|T} = a_{t|t} + A_t * (a_{t+1|T} - a_{t+1|t})
            state_diff = (smoother_t_plus_1.state_mean.to_numpy() - 
                         filter_t_plus_1.predicted_state_mean.to_numpy())
            smoothed_state_mean = Vector(filter_t.state_mean.to_numpy() + A_t @ state_diff)
            
            # Smoothed state variance: P_{t|T} = P_{t|t} + A_t * (P_{t+1|T} - P_{t+1|t}) * A_t'
            P_diff = (smoother_t_plus_1.state_variance.to_numpy() - 
                     filter_t_plus_1.predicted_state_variance.to_numpy())
            smoothed_state_variance = SpdMatrix(P_t_given_t + A_t @ P_diff @ A_t.T)
            
            # State disturbance (simplified - would need more complex calculation for full implementation)
            state_disturbance_mean = Vector(np.zeros(self.model.state_dimension))
            state_disturbance_variance = Q_t.copy()
            
            smoother_state = SmootherState(
                state_mean=smoothed_state_mean,
                state_variance=smoothed_state_variance,
                state_disturbance_mean=state_disturbance_mean,
                state_disturbance_variance=state_disturbance_variance
            )
            self._smoother_states[t] = smoother_state
        
        return self._smoother_states
    
    def log_likelihood(self) -> float:
        """Get log likelihood from last filter run."""
        return self._log_likelihood
    
    def filter_states(self) -> List[FilterState]:
        """Get filter states from last filter run."""
        return self._filter_states
    
    def smoother_states(self) -> List[SmootherState]:
        """Get smoother states from last smooth run."""
        return self._smoother_states
    
    def predict(self, n_ahead: int = 1) -> List[Tuple[float, float]]:
        """Predict future observations.
        
        Args:
            n_ahead: Number of steps ahead to predict
            
        Returns:
            List of (mean, variance) pairs for predicted observations
        """
        if not self._filter_states:
            raise ValueError("Must run filter() before predict()")
        
        T = self.model.time_dimension
        if T == 0:
            raise ValueError("No data to predict from")
        
        # Start from final filtered state
        state_mean = self._filter_states[-1].state_mean.copy()
        state_variance = self._filter_states[-1].state_variance.copy()
        
        predictions = []
        
        for h in range(1, n_ahead + 1):
            t = T + h - 1  # Time index for prediction
            
            # Evolve state
            T_t = self.model.transition_matrix(t-1)
            Q_t = self.model.state_error_variance(t-1)
            
            # Predict state
            predicted_state_mean = Vector(T_t.to_numpy() @ state_mean.to_numpy())
            predicted_state_variance = SpdMatrix(
                T_t.to_numpy() @ state_variance.to_numpy() @ T_t.T.to_numpy() + Q_t.to_numpy()
            )
            
            # Predict observation
            Z_t = self.model.observation_matrix(t)
            H_t = self.model.observation_variance
            
            predicted_obs_mean = float((Z_t @ Vector(predicted_state_mean.to_numpy())).to_numpy()[0])
            predicted_obs_variance = float(
                (Z_t @ predicted_state_variance @ Z_t.T).to_numpy()[0, 0] + H_t
            )
            
            predictions.append((predicted_obs_mean, predicted_obs_variance))
            
            # Update state for next prediction
            state_mean = predicted_state_mean
            state_variance = predicted_state_variance
        
        return predictions
    
    def residuals(self) -> Tuple[Vector, Vector]:
        """Compute standardized residuals.
        
        Returns:
            Tuple of (residuals, standardized_residuals)
        """
        if not self._filter_states:
            raise ValueError("Must run filter() before computing residuals()")
        
        residuals = []
        standardized_residuals = []
        
        for t, filter_state in enumerate(self._filter_states):
            data_t = self.model.get_data(t)
            
            if data_t.is_observed():
                residual = filter_state.innovation
                std_residual = residual / np.sqrt(filter_state.prediction_variance)
            else:
                residual = np.nan
                std_residual = np.nan
            
            residuals.append(residual)
            standardized_residuals.append(std_residual)
        
        return Vector(residuals), Vector(standardized_residuals)