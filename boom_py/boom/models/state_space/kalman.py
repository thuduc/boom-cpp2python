"""Kalman filter and smoother implementation."""
import numpy as np
from typing import Tuple, Optional, NamedTuple
from ...linalg import Vector, Matrix, SpdMatrix


class KalmanFilterResult(NamedTuple):
    """Results from Kalman filtering."""
    filtered_states: Matrix  # a_t|t
    filtered_variances: list  # P_t|t (list of SpdMatrix)
    predicted_states: Matrix  # a_t|t-1
    predicted_variances: list  # P_t|t-1 (list of SpdMatrix)
    prediction_errors: Vector  # v_t
    prediction_error_variances: Vector  # F_t
    log_likelihood: float


class KalmanSmootherResult(NamedTuple):
    """Results from Kalman smoothing."""
    smoothed_states: Matrix  # a_t|n
    smoothed_variances: list  # P_t|n (list of SpdMatrix)
    state_disturbances: Matrix  # u_t
    disturbance_variances: list  # D_t (list of SpdMatrix)


class KalmanFilter:
    """Kalman filter for state space models."""
    
    def __init__(self, model):
        """Initialize Kalman filter.
        
        Args:
            model: StateSpaceModel instance
        """
        self.model = model
    
    def filter(self, y: Vector) -> KalmanFilterResult:
        """Run Kalman filter on observations.
        
        Args:
            y: Vector of observations
            
        Returns:
            KalmanFilterResult with filtering results
        """
        n = len(y)
        state_dim = self.model.state_dimension()
        
        if state_dim == 0:
            # No state - just compute likelihood of observations
            log_lik = -0.5 * n * np.log(2 * np.pi * self.model.observation_variance)
            log_lik -= 0.5 * np.sum(y**2) / self.model.observation_variance
            
            return KalmanFilterResult(
                filtered_states=Matrix.zero(n, 0),
                filtered_variances=[],
                predicted_states=Matrix.zero(n, 0),
                predicted_variances=[],
                prediction_errors=y,
                prediction_error_variances=Vector([self.model.observation_variance] * n),
                log_likelihood=log_lik
            )
        
        # Initialize storage
        filtered_states = Matrix.zero(n, state_dim)
        predicted_states = Matrix.zero(n, state_dim)
        filtered_variances = []
        predicted_variances = []
        prediction_errors = Vector.zero(n)
        prediction_error_variances = Vector.zero(n)
        
        log_likelihood = 0.0
        
        # Initial conditions
        a = self.model.initial_state_mean()
        P = self.model.initial_state_variance()
        
        for t in range(n):
            # Get system matrices for time t
            T = self.model.transition_matrix(t)
            Z = self.model.observation_matrix(t)
            Q = self.model.state_variance(t)
            H = self.model.observation_variance  # scalar
            
            # Prediction step
            if t == 0:
                a_pred = Vector(a)
                P_pred = P
            else:
                a_pred = Vector(T @ a)
                P_pred = SpdMatrix(T @ P @ T.T + Q)
            
            predicted_states[t, :] = a_pred
            predicted_variances.append(P_pred)
            
            # Observation prediction
            if Z.ncol() > 0:
                y_pred = float((Z @ a_pred)[0])
                F = float((Z @ P_pred @ Z.T)[0, 0]) + H
            else:
                y_pred = 0.0
                F = H
            
            # Prediction error
            v = y[t] - y_pred
            prediction_errors[t] = v
            prediction_error_variances[t] = F
            
            # Update log likelihood
            if F > 1e-10:  # Avoid numerical issues
                log_likelihood -= 0.5 * (np.log(2 * np.pi * F) + v**2 / F)
            
            # Update step (if state dimension > 0 and F > 0)
            if state_dim > 0 and F > 1e-10:
                # Kalman gain
                if Z.ncol() > 0:
                    K_matrix = (P_pred @ Z.T) / F
                    K = Vector(K_matrix.flatten())  # Ensure it's a vector
                else:
                    K = Vector.zero(state_dim)
                
                # Update state and covariance
                a = Vector(a_pred + K * v)
                P = SpdMatrix(P_pred - np.outer(K, K) * F)
            else:
                a = Vector(a_pred)
                P = P_pred
            
            filtered_states[t, :] = a
            filtered_variances.append(P)
        
        return KalmanFilterResult(
            filtered_states=filtered_states,
            filtered_variances=filtered_variances,
            predicted_states=predicted_states,
            predicted_variances=predicted_variances,
            prediction_errors=prediction_errors,
            prediction_error_variances=prediction_error_variances,
            log_likelihood=log_likelihood
        )
    
    def log_likelihood(self, y: Vector) -> float:
        """Compute log likelihood of observations.
        
        Args:
            y: Vector of observations
            
        Returns:
            Log likelihood value
        """
        return self.filter(y).log_likelihood


class KalmanSmoother:
    """Kalman smoother for state space models."""
    
    def __init__(self, model):
        """Initialize Kalman smoother.
        
        Args:
            model: StateSpaceModel instance
        """
        self.model = model
        self.filter = KalmanFilter(model)
    
    def smooth(self, y: Vector) -> Tuple[KalmanFilterResult, KalmanSmootherResult]:
        """Run Kalman smoother on observations.
        
        Args:
            y: Vector of observations
            
        Returns:
            Tuple of (filter_result, smoother_result)
        """
        # First run the filter
        filter_result = self.filter.filter(y)
        
        n = len(y)
        state_dim = self.model.state_dimension()
        
        if state_dim == 0:
            # No state to smooth
            return filter_result, KalmanSmootherResult(
                smoothed_states=Matrix.zero(n, 0),
                smoothed_variances=[],
                state_disturbances=Matrix.zero(n, 0),
                disturbance_variances=[]
            )
        
        # Initialize backward pass
        smoothed_states = Matrix(filter_result.filtered_states)
        smoothed_variances = list(filter_result.filtered_variances)
        state_disturbances = Matrix.zero(n, state_dim)
        disturbance_variances = []
        
        # Backward pass
        r = Vector.zero(state_dim)  # Smoothing recursion
        N = Matrix.zero(state_dim, state_dim)  # Smoothing variance recursion
        
        for t in reversed(range(n)):
            # Get system matrices
            T = self.model.transition_matrix(t)
            Z = self.model.observation_matrix(t)
            Q = self.model.state_variance(t)
            H = self.model.observation_variance
            
            # Prediction error and variance from filter
            v = filter_result.prediction_errors[t]
            F = filter_result.prediction_error_variances[t]
            
            # Smoothing corrections
            if Z.ncol() > 0 and F > 1e-10:
                r = Z.T * (v / F) + T.T @ r
                N = Z.T @ Z / F + T.T @ N @ T
            else:
                r = T.T @ r
                N = T.T @ N @ T
            
            # Smoothed state
            if t > 0:
                P_pred = filter_result.predicted_variances[t]
                smoothed_states[t, :] = filter_result.predicted_states[t, :] + P_pred @ r
                smoothed_variances[t] = SpdMatrix(P_pred - P_pred @ N @ P_pred)
            
            # State disturbance
            QN = Q @ N
            state_disturbances[t, :] = Q @ r
            disturbance_variances.append(SpdMatrix(Q - QN @ Q))
        
        # Reverse disturbance variances to match time order
        disturbance_variances.reverse()
        
        return filter_result, KalmanSmootherResult(
            smoothed_states=smoothed_states,
            smoothed_variances=smoothed_variances,
            state_disturbances=state_disturbances,
            disturbance_variances=disturbance_variances
        )