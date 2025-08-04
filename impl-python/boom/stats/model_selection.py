"""
Model selection and comparison utilities.

This module provides tools for comparing models and performing
cross-validation for model selection.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from ..models.base import Model


class ModelComparison:
    """
    Compare multiple models using various criteria.
    """
    
    def __init__(self, models: List[Model], names: Optional[List[str]] = None):
        """
        Initialize model comparison.
        
        Args:
            models: List of fitted models
            names: Optional names for models
        """
        self._models = models
        self._names = names if names is not None else [f"Model_{i+1}" for i in range(len(models))]
        
        if len(self._names) != len(self._models):
            raise ValueError("Number of names must match number of models")
    
    def compare_ic(self, criteria: List[str] = ['aic', 'bic']) -> Dict[str, Dict[str, float]]:
        """
        Compare models using information criteria.
        
        Args:
            criteria: List of criteria to compute
            
        Returns:
            Dictionary with results for each model
        """
        from .information_criteria import compute_ic
        
        results = {}
        
        for name, model in zip(self._names, self._models):
            try:
                ic_values = compute_ic(model, criteria)
                results[name] = ic_values
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def compare_likelihood(self) -> Dict[str, float]:
        """
        Compare models using log likelihood.
        
        Returns:
            Dictionary with log likelihood for each model
        """
        results = {}
        
        for name, model in zip(self._names, self._models):
            try:
                results[name] = model.log_likelihood()
            except Exception as e:
                results[name] = np.nan
        
        return results
    
    def likelihood_ratio_test(self, model1_idx: int, model2_idx: int) -> Dict[str, Any]:
        """
        Perform likelihood ratio test between two nested models.
        
        Args:
            model1_idx: Index of restricted (simpler) model
            model2_idx: Index of unrestricted (more complex) model
            
        Returns:
            Dictionary with test results
        """
        if model1_idx >= len(self._models) or model2_idx >= len(self._models):
            raise ValueError("Model index out of range")
        
        model1 = self._models[model1_idx]
        model2 = self._models[model2_idx]
        
        # Check if models are nested
        p1 = model1.n_parameters()
        p2 = model2.n_parameters()
        
        if p1 >= p2:
            raise ValueError("Model 1 must be simpler (fewer parameters) than Model 2")
        
        # Compute likelihood ratio statistic
        ll1 = model1.log_likelihood()
        ll2 = model2.log_likelihood()
        
        lr_statistic = 2 * (ll2 - ll1)
        df = p2 - p1
        
        # P-value from chi-square distribution
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(lr_statistic, df)
        
        return {
            'lr_statistic': lr_statistic,
            'degrees_freedom': df,
            'p_value': p_value,
            'model1_ll': ll1,
            'model2_ll': ll2,
            'model1_params': p1,
            'model2_params': p2,
            'significant': p_value < 0.05
        }
    
    def summary_table(self) -> str:
        """
        Generate summary comparison table.
        
        Returns:
            Formatted comparison table
        """
        # Get comparison results
        ic_results = self.compare_ic()
        ll_results = self.compare_likelihood()
        
        # Create table
        lines = []
        lines.append("Model Comparison Summary")
        lines.append("=" * 50)
        lines.append(f"{'Model':<15} {'LogLik':<12} {'AIC':<12} {'BIC':<12}")
        lines.append("-" * 50)
        
        for name in self._names:
            ll = ll_results.get(name, np.nan)
            ic = ic_results.get(name, {})
            aic = ic.get('AIC', np.nan)
            bic = ic.get('BIC', np.nan)
            
            lines.append(f"{name:<15} {ll:<12.3f} {aic:<12.3f} {bic:<12.3f}")
        
        return "\n".join(lines)


class CrossValidator:
    """
    Cross-validation for model selection and evaluation.
    """
    
    def __init__(self, n_folds: int = 5):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of cross-validation folds
        """
        self._n_folds = n_folds
    
    def k_fold_cv(self, model_factory: Callable[[], Model],
                  data: Any, 
                  scoring_func: Callable[[Model, Any], float]) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_factory: Function that creates a new model instance
            data: Dataset to split
            scoring_func: Function to score model on test data
            
        Returns:
            Dictionary with CV results
        """
        # Simple implementation - assumes data can be indexed
        if hasattr(data, '__len__'):
            n_samples = len(data)
        else:
            raise ValueError("Data must be indexable")
        
        fold_size = n_samples // self._n_folds
        scores = []
        
        for fold in range(self._n_folds):
            # Split data into train and test
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self._n_folds - 1 else n_samples
            
            # Create train and test indices
            test_indices = list(range(test_start, test_end))
            train_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
            
            # Split data (this is simplified - real implementation would depend on data type)
            train_data = self._subset_data(data, train_indices)
            test_data = self._subset_data(data, test_indices)
            
            # Train model on training data
            model = model_factory()
            model.set_data(train_data)
            model.fit()
            
            # Score on test data
            score = scoring_func(model, test_data)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_folds': self._n_folds
        }
    
    def _subset_data(self, data: Any, indices: List[int]) -> Any:
        """
        Extract subset of data using indices.
        
        This is a simplified implementation - real version would handle
        different data types appropriately.
        """
        if hasattr(data, '__getitem__'):
            if hasattr(data, 'subset'):
                return data.subset(indices)
            else:
                # Generic indexing
                return [data[i] for i in indices]
        else:
            raise ValueError("Cannot subset this data type")
    
    def leave_one_out_cv(self, model_factory: Callable[[], Model],
                        data: Any,
                        scoring_func: Callable[[Model, Any], float]) -> Dict[str, Any]:
        """
        Perform leave-one-out cross-validation.
        
        Args:
            model_factory: Function that creates a new model instance
            data: Dataset
            scoring_func: Function to score model
            
        Returns:
            Dictionary with LOOCV results
        """
        if hasattr(data, '__len__'):
            n_samples = len(data)
        else:
            raise ValueError("Data must be indexable")
        
        scores = []
        
        for i in range(n_samples):
            # Leave out sample i
            train_indices = list(range(n_samples))
            train_indices.remove(i)
            test_indices = [i]
            
            # Split data
            train_data = self._subset_data(data, train_indices)
            test_data = self._subset_data(data, test_indices)
            
            # Train and score
            model = model_factory()
            model.set_data(train_data)
            model.fit()
            
            score = scoring_func(model, test_data)
            scores.append(score)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_samples': n_samples
        }