"""
Tests for HMM (Hidden Markov Model) implementations.

This module tests the HMM models including Gaussian and Categorical HMMs.
"""

import pytest
import numpy as np
import sys
import os

# Add the impl-python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from boom.models.hmm import HmmModel, GaussianHmm, CategoricalHmm, HmmData
from boom.linalg import Vector, Matrix


class TestHmmData:
    """Test HMM data structures."""
    
    def test_hmm_data_creation(self):
        """Test creation of HMM data."""
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        data = HmmData(observations)
        
        assert data.n_observations() == 5
        assert len(data.get_observations()) == 5
        
    def test_hmm_data_vector_input(self):
        """Test HMM data with Vector input."""
        observations = Vector(np.array([1.0, 2.0, 3.0]))
        data = HmmData([observations])
        
        assert data.n_observations() == 1


class TestGaussianHmm:
    """Test Gaussian HMM implementation."""
    
    def test_initialization(self):
        """Test Gaussian HMM initialization."""
        hmm = GaussianHmm(n_states=2)
        
        assert hmm.n_states == 2
        assert len(hmm.means) == 2
        assert len(hmm.variances) == 2
        
    def test_parameter_setting(self):
        """Test setting HMM parameters."""
        hmm = GaussianHmm(n_states=2)
        
        # Set parameters
        means = np.array([0.0, 5.0])
        variances = np.array([1.0, 2.0])
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        initial_probs = np.array([0.5, 0.5])
        
        hmm.set_means(means)
        hmm.set_variances(variances)
        hmm.set_transition_matrix(Matrix(transition_matrix))
        hmm.set_initial_probs(Vector(initial_probs))
        
        np.testing.assert_array_almost_equal(hmm.means, means)
        np.testing.assert_array_almost_equal(hmm.variances, variances)
        
    def test_emission_probability(self):
        """Test emission probability computation."""
        hmm = GaussianHmm(n_states=2)
        hmm.set_means(np.array([0.0, 5.0]))
        hmm.set_variances(np.array([1.0, 1.0]))
        
        # Test emission probability
        log_prob_0 = hmm.emission_log_prob(0, 0.0)  # Mean of state 0
        log_prob_1 = hmm.emission_log_prob(1, 5.0)  # Mean of state 1
        
        # Should be higher probability at the mean
        assert log_prob_0 > hmm.emission_log_prob(0, 2.0)
        assert log_prob_1 > hmm.emission_log_prob(1, 2.0)
        
    def test_forward_backward_algorithm(self):
        """Test forward-backward algorithm."""
        hmm = GaussianHmm(n_states=2)
        
        # Set up simple model
        hmm.set_means(np.array([0.0, 3.0]))
        hmm.set_variances(np.array([1.0, 1.0]))
        hmm.set_transition_matrix(Matrix(np.array([[0.8, 0.2], [0.3, 0.7]])))
        hmm.set_initial_probs(Vector(np.array([0.6, 0.4])))
        
        # Create data
        observations = [0.1, 0.2, 2.8, 2.9, 3.1]
        for obs in observations:
            hmm.add_data(HmmData(obs))
        
        # Run forward-backward
        alpha, beta, log_likelihood = hmm.forward_backward()
        
        assert alpha.shape() == (5, 2)  # n_obs x n_states
        assert beta.shape() == (5, 2)
        assert isinstance(log_likelihood, float)
        assert not np.isnan(log_likelihood)
        
    def test_viterbi_algorithm(self):
        """Test Viterbi algorithm."""
        hmm = GaussianHmm(n_states=2)
        
        # Set up model
        hmm.set_means(np.array([0.0, 3.0]))
        hmm.set_variances(np.array([1.0, 1.0]))
        hmm.set_transition_matrix(Matrix(np.array([[0.8, 0.2], [0.3, 0.7]])))
        hmm.set_initial_probs(Vector(np.array([0.6, 0.4])))
        
        # Create data
        observations = [0.1, 0.2, 2.8, 2.9]
        for obs in observations:
            hmm.add_data(HmmData(obs))
        
        # Run Viterbi
        path, log_prob = hmm.viterbi()
        
        assert len(path) == 4
        assert all(0 <= state < 2 for state in path)
        assert isinstance(log_prob, float)
        
    def test_baum_welch_training(self):
        """Test Baum-Welch parameter estimation."""
        hmm = GaussianHmm(n_states=2)
        
        # Generate synthetic data from known model
        np.random.seed(42)
        true_means = np.array([0.0, 5.0])
        true_vars = np.array([1.0, 1.0])
        
        # Synthetic observations
        observations = []
        for _ in range(50):
            state = np.random.choice(2)
            obs = np.random.normal(true_means[state], np.sqrt(true_vars[state]))
            observations.append(obs)
        
        for obs in observations:
            hmm.add_data(HmmData(obs))
        
        # Initial parameters (slightly wrong)
        hmm.set_means(np.array([0.5, 4.5]))
        hmm.set_variances(np.array([1.5, 1.5]))
        
        initial_ll = hmm.log_likelihood()
        
        # Train with Baum-Welch
        hmm.baum_welch(max_iterations=10)
        
        final_ll = hmm.log_likelihood()
        
        # Likelihood should improve
        assert final_ll >= initial_ll
        
    def test_prediction(self):
        """Test state prediction."""
        hmm = GaussianHmm(n_states=2)
        
        # Set up model
        hmm.set_means(np.array([0.0, 3.0]))
        hmm.set_variances(np.array([1.0, 1.0]))
        hmm.set_transition_matrix(Matrix(np.array([[0.8, 0.2], [0.3, 0.7]])))
        hmm.set_initial_probs(Vector(np.array([0.6, 0.4])))
        
        # Test prediction
        state_probs = hmm.predict_state_probabilities(0.1)  # Close to mean of state 0
        
        assert len(state_probs.to_numpy()) == 2
        assert np.sum(state_probs.to_numpy()) == pytest.approx(1.0)
        assert state_probs.to_numpy()[0] > state_probs.to_numpy()[1]  # Should favor state 0


class TestCategoricalHmm:
    """Test Categorical HMM implementation."""
    
    def test_initialization(self):
        """Test Categorical HMM initialization."""
        hmm = CategoricalHmm(n_states=2, n_categories=3)
        
        assert hmm.n_states == 2
        assert hmm.n_categories == 3
        assert hmm.emission_probs().shape() == (2, 3)
        
    def test_emission_probability(self):
        """Test emission probability computation."""
        hmm = CategoricalHmm(n_states=2, n_categories=3)
        
        # Set emission probabilities
        emission_probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        hmm.set_emission_probs(Matrix(emission_probs))
        
        # Test emission probability
        log_prob = hmm.emission_log_prob(0, 0)  # State 0, category 0
        expected_log_prob = np.log(0.5)
        
        assert log_prob == pytest.approx(expected_log_prob)
        
    def test_parameter_estimation(self):
        """Test parameter estimation for categorical HMM."""
        hmm = CategoricalHmm(n_states=2, n_categories=3)
        
        # Create synthetic data
        observations = [0, 0, 1, 2, 2, 1, 0, 2]
        for obs in observations:
            hmm.add_data(HmmData(obs))
        
        # Set initial parameters
        hmm.set_transition_matrix(Matrix(np.array([[0.7, 0.3], [0.4, 0.6]])))
        hmm.set_initial_probs(Vector(np.array([0.5, 0.5])))
        
        initial_ll = hmm.log_likelihood()
        
        # Train model
        hmm.baum_welch(max_iterations=5)
        
        final_ll = hmm.log_likelihood()
        
        # Likelihood should not decrease
        assert final_ll >= initial_ll
        
    def test_invalid_observation(self):
        """Test handling of invalid observations."""
        hmm = CategoricalHmm(n_states=2, n_categories=3)
        
        # Test with invalid category
        with pytest.raises(ValueError):
            hmm.emission_log_prob(0, 3)  # Category 3 doesn't exist
            
        with pytest.raises(ValueError):
            hmm.emission_log_prob(0, -1)  # Negative category


class TestHmmIntegration:
    """Integration tests for HMM functionality."""
    
    def test_hmm_workflow(self):
        """Test complete HMM workflow."""
        # Create and train a Gaussian HMM
        hmm = GaussianHmm(n_states=2)
        
        # Generate synthetic data
        np.random.seed(123)
        observations = []
        
        # State sequence: 0, 0, 1, 1, 0
        true_sequence = [0, 0, 1, 1, 0]
        true_means = [1.0, 5.0]
        
        for state in true_sequence:
            obs = np.random.normal(true_means[state], 0.5)
            observations.append(obs)
        
        for obs in observations:
            hmm.add_data(HmmData(obs))
        
        # Train model
        hmm.baum_welch(max_iterations=20)
        
        # Decode most likely sequence
        decoded_path, _ = hmm.viterbi()
        
        # Check that we get a reasonable decoding
        assert len(decoded_path) == len(true_sequence)
        
    def test_parameter_vectorization(self):
        """Test parameter vectorization and unvectorization."""
        hmm = GaussianHmm(n_states=2)
        
        # Set some parameters
        hmm.set_means(np.array([1.0, 3.0]))
        hmm.set_variances(np.array([0.5, 1.5]))
        
        # Get parameter vector
        param_vector = hmm.vectorize_params()
        
        # Create new model and set parameters from vector
        hmm2 = GaussianHmm(n_states=2)
        hmm2.unvectorize_params(param_vector)
        
        # Check that parameters match
        np.testing.assert_array_almost_equal(hmm.means, hmm2.means)
        np.testing.assert_array_almost_equal(hmm.variances, hmm2.variances)


if __name__ == '__main__':
    pytest.main([__file__])