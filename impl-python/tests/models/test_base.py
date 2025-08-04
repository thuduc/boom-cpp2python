"""Tests for base model classes."""

import pytest
import numpy as np
from boom.models.base import Data, Model, LoglikeModel
from boom.linalg import Vector


class TestData:
    """Test Data base class."""
    
    def test_construction(self):
        """Test Data construction through a concrete subclass."""
        # We need a concrete implementation to test
        class ConcreteData(Data):
            def __init__(self, value):
                super().__init__()
                self.value = value
            
            def clone(self):
                result = ConcreteData(self.value)
                result.set_missing(self.is_missing())
                return result
        
        data = ConcreteData(42)
        assert not data.is_missing()
        assert data.value == 42
    
    def test_missing_status(self):
        """Test missing status management."""
        class ConcreteData(Data):
            def clone(self):
                return ConcreteData()
        
        data = ConcreteData()
        assert not data.is_missing()
        
        data.set_missing(True)
        assert data.is_missing()
        
        data.set_missing(False)
        assert not data.is_missing()
    
    def test_str_representation(self):
        """Test string representation."""
        class ConcreteData(Data):
            def clone(self):
                return ConcreteData()
        
        data = ConcreteData()
        s = str(data)
        assert "ConcreteData" in s
        assert "missing=False" in s
        
        data.set_missing(True)
        s = str(data)
        assert "missing=True" in s


class TestModel:
    """Test Model base class."""
    
    def test_construction(self):
        """Test Model construction through concrete subclass."""
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return 0.0
            
            def simulate_data(self, n=None):
                return []
        
        model = ConcreteModel()
        assert model.sample_size() == 0
        assert len(model.parameter_names()) == 0
    
    def test_parameter_management(self):
        """Test parameter management."""
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return 0.0
            
            def simulate_data(self, n=None):
                return []
        
        model = ConcreteModel()
        
        # Set parameters
        model.set_parameter('alpha', 1.5)
        model.set_parameter('beta', 2.0)
        
        # Get parameters
        assert model.get_parameter('alpha') == 1.5
        assert model.get_parameter('beta') == 2.0
        assert model.get_parameter('gamma') is None
        assert model.get_parameter('gamma', 3.0) == 3.0
        
        # Parameter names
        names = model.parameter_names()
        assert 'alpha' in names
        assert 'beta' in names
        assert len(names) == 2
        
        # All parameters
        params = model.parameters()
        assert params['alpha'] == 1.5
        assert params['beta'] == 2.0
    
    def test_data_management(self):
        """Test data management."""
        class ConcreteData(Data):
            def __init__(self, value):
                super().__init__()
                self.value = value
            
            def clone(self):
                return ConcreteData(self.value)
        
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return 0.0
            
            def simulate_data(self, n=None):
                return []
        
        model = ConcreteModel()
        
        # Add data
        data1 = ConcreteData(1)
        data2 = ConcreteData(2)
        
        model.add_data(data1)
        assert model.sample_size() == 1
        
        model.add_data_batch([data2])
        assert model.sample_size() == 2
        
        # Get data
        all_data = model.data()
        assert len(all_data) == 2
        assert all_data[0].value == 1
        assert all_data[1].value == 2
        
        # Clear data
        model.clear_data()
        assert model.sample_size() == 0
    
    def test_observer_pattern(self):
        """Test observer pattern for parameter changes."""
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return 0.0
            
            def simulate_data(self, n=None):
                return []
        
        class Observer:
            def __init__(self):
                self.update_count = 0
            
            def update(self, model):
                self.update_count += 1
        
        model = ConcreteModel()
        observer = Observer()
        
        model.add_observer(observer)
        assert observer.update_count == 0
        
        # Setting parameter should notify observer
        model.set_parameter('test', 1.0)
        assert observer.update_count == 1
        
        # Setting same value shouldn't notify
        model.set_parameter('test', 1.0)
        assert observer.update_count == 1
        
        # Setting different value should notify
        model.set_parameter('test', 2.0)
        assert observer.update_count == 2
        
        # Remove observer
        model.remove_observer(observer)
        model.set_parameter('test', 3.0)
        assert observer.update_count == 2  # No more updates
    
    def test_default_methods(self):
        """Test default method implementations."""
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return -5.0
            
            def simulate_data(self, n=None):
                return []
        
        model = ConcreteModel()
        
        # Default log_prior
        assert model.log_prior() == 0.0
        
        # Default log_posterior
        assert model.log_posterior() == -5.0  # log_likelihood + log_prior
        
        # Default mle does nothing
        model.mle()  # Should not raise error
        
        # Default gradient/hessian raise NotImplementedError
        with pytest.raises(NotImplementedError):
            model.gradient()
        
        with pytest.raises(NotImplementedError):
            model.hessian()
        
        # Default vectorization
        theta = model.vectorize_params()
        assert isinstance(theta, Vector)
        assert len(theta) == 0
        
        # Default unvectorization
        model.unvectorize_params(Vector([]))  # Should not raise error
        
        # Default clone raises NotImplementedError
        with pytest.raises(NotImplementedError):
            model.clone()
    
    def test_str_representation(self):
        """Test string representation."""
        class ConcreteModel(Model):
            def log_likelihood(self, data=None):
                return 0.0
            
            def simulate_data(self, n=None):
                return []
        
        model = ConcreteModel()
        model.set_parameter('alpha', 1.0)
        
        s = str(model)
        assert "ConcreteModel" in s
        assert "sample_size=0" in s
        assert "alpha" in s


class TestLoglikeModel:
    """Test LoglikeModel class."""
    
    def test_likelihood_computation(self):
        """Test likelihood computation."""
        class ConcreteLoglikeModel(LoglikeModel):
            def log_likelihood(self, data=None):
                return -2.0
            
            def simulate_data(self, n=None):
                return []
            
            def vectorize_params(self, minimal=True):
                return Vector([1.0, 2.0])  # 2 parameters
        
        model = ConcreteLoglikeModel()
        
        # Likelihood
        assert abs(model.likelihood() - np.exp(-2.0)) < 1e-10
        
        # AIC = 2k - 2*log_likelihood = 2*2 - 2*(-2) = 8
        assert abs(model.AIC() - 8.0) < 1e-10
        
        # Deviance = -2*log_likelihood = -2*(-2) = 4
        assert abs(model.deviance() - 4.0) < 1e-10
    
    def test_bic_computation(self):
        """Test BIC computation."""
        class ConcreteData(Data):
            def clone(self):
                return ConcreteData()
        
        class ConcreteLoglikeModel(LoglikeModel):
            def log_likelihood(self, data=None):
                if data is None:
                    data = self._data
                return -len(data)  # Simple likelihood based on sample size
            
            def simulate_data(self, n=None):
                return []
            
            def vectorize_params(self, minimal=True):
                return Vector([1.0])  # 1 parameter
        
        model = ConcreteLoglikeModel()
        
        # Add some data
        for _ in range(10):
            model.add_data(ConcreteData())
        
        # BIC = log(n)*k - 2*log_likelihood = log(10)*1 - 2*(-10) = log(10) + 20
        expected_bic = np.log(10) + 20
        assert abs(model.BIC() - expected_bic) < 1e-10