"""Tests for Metropolis-Hastings sampler."""

import pytest
import numpy as np
from boom.samplers.metropolis_hastings import (
    RandomWalkProposal, IndependenceProposal, MetropolisHastings
)
from boom.linalg import Vector
from boom.distributions import RNG


class TestRandomWalkProposal:
    """Test RandomWalkProposal class."""
    
    def test_scalar_covariance(self):
        """Test with scalar covariance."""
        proposal = RandomWalkProposal(1.0)
        rng = RNG(seed=42)
        
        current = Vector([1.0, 2.0])
        new_proposal = proposal.propose(current, rng)
        
        # Should be same dimension
        assert len(new_proposal) == len(current)
        
        # Should be symmetric
        assert proposal.is_symmetric()
        assert proposal.log_density(new_proposal, current) == 0.0
    
    def test_vector_covariance(self):
        """Test with vector covariance."""
        variances = Vector([1.0, 4.0, 0.25])
        proposal = RandomWalkProposal(variances)
        rng = RNG(seed=42)
        
        current = Vector([0.0, 0.0, 0.0])
        new_proposal = proposal.propose(current, rng)
        
        assert len(new_proposal) == 3
        assert proposal.is_symmetric()
    
    def test_matrix_covariance(self):
        """Test with matrix covariance."""
        cov_matrix = np.array([[1.0, 0.2], [0.2, 0.5]])
        proposal = RandomWalkProposal(cov_matrix)
        rng = RNG(seed=42)
        
        current = Vector([1.0, -1.0])
        new_proposal = proposal.propose(current, rng)
        
        assert len(new_proposal) == 2
        assert proposal.is_symmetric()
    
    def test_invalid_covariance(self):
        """Test with invalid covariance."""
        # Negative variance
        with pytest.raises(ValueError):
            RandomWalkProposal(Vector([-1.0, 1.0]))
        
        # Non-positive definite matrix
        bad_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite
        with pytest.raises(ValueError):
            RandomWalkProposal(bad_matrix)


class TestIndependenceProposal:
    """Test IndependenceProposal class."""
    
    def test_scalar_covariance(self):
        """Test with scalar covariance."""
        mean = Vector([0.0, 1.0])
        proposal = IndependenceProposal(mean, 2.0)
        rng = RNG(seed=42)
        
        current = Vector([5.0, -3.0])  # Should be ignored
        new_proposal = proposal.propose(current, rng)
        
        assert len(new_proposal) == 2
        assert not proposal.is_symmetric()
        
        # Log density should be reasonable
        log_dens = proposal.log_density(new_proposal, current)
        assert isinstance(log_dens, float)
        assert not np.isnan(log_dens)
    
    def test_vector_covariance(self):
        """Test with vector covariance."""
        mean = Vector([1.0, -1.0])
        variances = Vector([0.5, 2.0])
        proposal = IndependenceProposal(mean, variances)
        rng = RNG(seed=42)
        
        current = Vector([0.0, 0.0])
        new_proposal = proposal.propose(current, rng)
        
        assert len(new_proposal) == 2
        
        # Test log density computation
        log_dens = proposal.log_density(mean, current)  # Density at mean should be high
        assert isinstance(log_dens, float)


class TestMetropolisHastings:
    """Test MetropolisHastings sampler."""
    
    def test_standard_normal_sampling(self):
        """Test sampling from standard normal distribution."""
        # Target: standard normal log density
        def log_target(x):
            if len(x) != 1:
                return -np.inf
            return -0.5 * x[0]**2 - 0.5 * np.log(2 * np.pi)
        
        # Random walk proposal
        proposal = RandomWalkProposal(1.0)
        sampler = MetropolisHastings(log_target, proposal, RNG(seed=42))
        
        # Sample
        initial_state = Vector([0.0])
        samples, acceptance_rate = sampler.sample(initial_state, 1000)
        
        assert len(samples) == 1000
        assert 0.0 < acceptance_rate < 1.0
        
        # Extract values
        values = [s[0] for s in samples]
        
        # Check approximate properties
        sample_mean = np.mean(values)
        sample_var = np.var(values)
        
        # Should be approximately N(0, 1)
        assert abs(sample_mean) < 0.2  # Allow some Monte Carlo error
        assert 0.7 < sample_var < 1.3
    
    def test_bivariate_normal_sampling(self):
        """Test sampling from bivariate normal."""
        # Target: bivariate standard normal
        def log_target(x):
            if len(x) != 2:
                return -np.inf
            return -0.5 * (x[0]**2 + x[1]**2) - np.log(2 * np.pi)
        
        # Random walk proposal
        proposal = RandomWalkProposal(np.eye(2))
        sampler = MetropolisHastings(log_target, proposal, RNG(seed=123))
        
        # Sample
        initial_state = Vector([0.0, 0.0])
        samples, acceptance_rate = sampler.sample(initial_state, 500)
        
        assert len(samples) == 500
        assert acceptance_rate > 0.1  # Should have reasonable acceptance
        
        # Extract values
        x_values = [s[0] for s in samples]
        y_values = [s[1] for s in samples]
        
        # Check approximate properties
        assert abs(np.mean(x_values)) < 0.3
        assert abs(np.mean(y_values)) < 0.3
        assert 0.4 < np.var(x_values) < 1.6
        assert 0.4 < np.var(y_values) < 1.6
    
    def test_independence_proposal(self):
        """Test with independence proposal."""
        # Target: N(2, 1)
        def log_target(x):
            if len(x) != 1:
                return -np.inf
            return -0.5 * (x[0] - 2)**2 - 0.5 * np.log(2 * np.pi)
        
        # Independence proposal: N(1.5, 2)
        mean = Vector([1.5])
        proposal = IndependenceProposal(mean, 2.0)
        sampler = MetropolisHastings(log_target, proposal, RNG(seed=456))
        
        # Sample
        initial_state = Vector([0.0])
        samples, acceptance_rate = sampler.sample(initial_state, 1000)
        
        assert len(samples) == 1000
        assert acceptance_rate > 0.05  # Should accept some proposals
        
        # Check that samples are reasonable
        values = [s[0] for s in samples]
        sample_mean = np.mean(values)
        
        # Should be close to true mean of 2
        assert 1.5 < sample_mean < 2.5
    
    def test_tuning(self):
        """Test proposal tuning."""
        # Target: standard normal
        def log_target(x):
            return -0.5 * x[0]**2 - 0.5 * np.log(2 * np.pi)
        
        # Start with bad proposal variance
        proposal = RandomWalkProposal(10.0)  # Too large
        sampler = MetropolisHastings(log_target, proposal, RNG(seed=789))
        
        initial_state = Vector([0.0])
        
        # Check initial acceptance rate
        _, initial_acceptance = sampler.sample(initial_state, 100)
        
        # Tune sampler
        tuned_sampler = sampler.tune_proposal(initial_state, n_tune=500)
        
        # Check if tuning helped
        _, tuned_acceptance = tuned_sampler.sample(initial_state, 100)
        
        # Can't guarantee tuning always improves, but should be reasonable
        assert 0.1 < tuned_acceptance < 0.9
    
    def test_error_handling(self):
        """Test error handling."""
        # Target that sometimes returns invalid values
        def problematic_target(x):
            if x[0] < -5 or x[0] > 5:
                raise ValueError("Out of bounds")
            return -0.5 * x[0]**2
        
        proposal = RandomWalkProposal(1.0)
        sampler = MetropolisHastings(problematic_target, proposal, RNG(seed=101))
        
        # Should still work, rejecting problematic proposals
        initial_state = Vector([0.0])
        samples, acceptance_rate = sampler.sample(initial_state, 100)
        
        assert len(samples) == 100
        # All samples should be valid
        for sample in samples:
            assert -5 <= sample[0] <= 5
    
    def test_statistics(self):
        """Test sampler statistics."""
        def log_target(x):
            return -0.5 * x[0]**2
        
        proposal = RandomWalkProposal(1.0)
        sampler = MetropolisHastings(log_target, proposal, RNG(seed=202))
        
        initial_state = Vector([0.0])
        samples, acceptance_rate = sampler.sample(initial_state, 50)
        
        # Check statistics
        assert sampler.n_proposals() == 50
        assert sampler.n_accepted() == int(50 * acceptance_rate)
        assert abs(sampler.acceptance_rate() - acceptance_rate) < 1e-10