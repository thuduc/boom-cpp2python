"""Data classes for BOOM models."""

from typing import Union, List, Optional, Any
import numpy as np
from ..linalg import Vector, Matrix
from .base import Data


class DoubleData(Data):
    """Data class for single floating point values."""
    
    def __init__(self, value: float):
        """Initialize with a float value.
        
        Args:
            value: The data value.
        """
        super().__init__()
        self._value = float(value)
    
    def value(self) -> float:
        """Get the data value."""
        return self._value
    
    def set_value(self, value: float):
        """Set the data value."""
        self._value = float(value)
    
    def clone(self) -> 'DoubleData':
        """Create a copy."""
        result = DoubleData(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __float__(self) -> float:
        """Convert to float."""
        return self._value
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"DoubleData({self._value}){missing_str}"


class VectorData(Data):
    """Data class for vector observations."""
    
    def __init__(self, value: Union[Vector, List[float], np.ndarray]):
        """Initialize with a vector value.
        
        Args:
            value: The vector data.
        """
        super().__init__()
        if isinstance(value, Vector):
            self._value = value.copy()
        else:
            self._value = Vector(value)
    
    def value(self) -> Vector:
        """Get the data value."""
        return self._value.copy()
    
    def set_value(self, value: Union[Vector, List[float], np.ndarray]):
        """Set the data value."""
        if isinstance(value, Vector):
            self._value = value.copy()
        else:
            self._value = Vector(value)
    
    def dim(self) -> int:
        """Get the dimension of the vector."""
        return len(self._value)
    
    def clone(self) -> 'VectorData':
        """Create a copy."""
        result = VectorData(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __len__(self) -> int:
        """Length of the vector."""
        return len(self._value)
    
    def __getitem__(self, index: int) -> float:
        """Get element."""
        return self._value[index]
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"VectorData(dim={len(self._value)}){missing_str}"


class MatrixData(Data):
    """Data class for matrix observations."""
    
    def __init__(self, value: Union[Matrix, List[List[float]], np.ndarray]):
        """Initialize with a matrix value.
        
        Args:
            value: The matrix data.
        """
        super().__init__()
        if isinstance(value, Matrix):
            self._value = value.copy()
        else:
            self._value = Matrix(value)
    
    def value(self) -> Matrix:
        """Get the data value."""
        return self._value.copy()
    
    def set_value(self, value: Union[Matrix, List[List[float]], np.ndarray]):
        """Set the data value."""
        if isinstance(value, Matrix):
            self._value = value.copy()
        else:
            self._value = Matrix(value)
    
    def nrow(self) -> int:
        """Get number of rows."""
        return self._value.nrow()
    
    def ncol(self) -> int:
        """Get number of columns."""
        return self._value.ncol()
    
    def shape(self) -> tuple:
        """Get matrix shape."""
        return self._value.shape()
    
    def clone(self) -> 'MatrixData':
        """Create a copy."""
        result = MatrixData(self._value)
        result.set_missing(self.is_missing())
        return result
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"MatrixData({self._value.shape()}){missing_str}"


class CategoricalData(Data):
    """Data class for categorical observations."""
    
    def __init__(self, value: Union[int, str], levels: Optional[List[str]] = None):
        """Initialize with categorical value.
        
        Args:
            value: Category value (integer index or string).
            levels: List of category names. If None and value is string,
                   creates single-level list.
        """
        super().__init__()
        
        if isinstance(value, str):
            if levels is None:
                levels = [value]
            if value in levels:
                self._value = levels.index(value)
            else:
                raise ValueError(f"Value '{value}' not in levels {levels}")
        else:
            self._value = int(value)
        
        self._levels = levels if levels is not None else []
    
    def value(self) -> int:
        """Get the category index."""
        return self._value
    
    def set_value(self, value: Union[int, str]):
        """Set the category value."""
        if isinstance(value, str):
            if value not in self._levels:
                raise ValueError(f"Value '{value}' not in levels {self._levels}")
            self._value = self._levels.index(value)
        else:
            self._value = int(value)
    
    def level(self) -> Optional[str]:
        """Get the category name."""
        if 0 <= self._value < len(self._levels):
            return self._levels[self._value]
        return None
    
    def levels(self) -> List[str]:
        """Get all category levels."""
        return self._levels.copy()
    
    def set_levels(self, levels: List[str]):
        """Set category levels."""
        if self._value >= len(levels):
            raise ValueError(f"Current value {self._value} invalid for levels {levels}")
        self._levels = levels.copy()
    
    def n_levels(self) -> int:
        """Get number of levels."""
        return len(self._levels)
    
    def clone(self) -> 'CategoricalData':
        """Create a copy."""
        result = CategoricalData(self._value, self._levels)
        result.set_missing(self.is_missing())
        return result
    
    def __str__(self) -> str:
        """String representation."""
        level_str = self.level() or str(self._value)
        missing_str = " (missing)" if self.is_missing() else ""
        return f"CategoricalData({level_str}){missing_str}"


class BinomialData(Data):
    """Data class for binomial observations (n trials, k successes)."""
    
    def __init__(self, n: int, k: int):
        """Initialize with binomial data.
        
        Args:
            n: Number of trials.
            k: Number of successes.
        """
        super().__init__()
        if k > n or k < 0 or n < 0:
            raise ValueError(f"Invalid binomial data: n={n}, k={k}")
        
        self._n = int(n)
        self._k = int(k)
    
    def trials(self) -> int:
        """Get number of trials."""
        return self._n
    
    def successes(self) -> int:
        """Get number of successes.""" 
        return self._k
    
    def failures(self) -> int:
        """Get number of failures."""
        return self._n - self._k
    
    def success_rate(self) -> float:
        """Get success rate."""
        if self._n == 0:
            return 0.0
        return self._k / self._n
    
    def set_trials(self, n: int):
        """Set number of trials."""
        if n < self._k:
            raise ValueError(f"Number of trials {n} cannot be less than successes {self._k}")
        self._n = int(n)
    
    def set_successes(self, k: int):
        """Set number of successes."""
        if k > self._n or k < 0:
            raise ValueError(f"Invalid number of successes {k} for {self._n} trials")
        self._k = int(k)
    
    def clone(self) -> 'BinomialData':
        """Create a copy."""
        result = BinomialData(self._n, self._k)
        result.set_missing(self.is_missing()) 
        return result
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"BinomialData(n={self._n}, k={self._k}){missing_str}"


class CompositeData(Data):
    """Data class that can hold multiple data objects."""
    
    def __init__(self, data_list: Optional[List[Data]] = None):
        """Initialize with list of data objects.
        
        Args:
            data_list: List of data objects to store.
        """
        super().__init__()
        self._data = data_list.copy() if data_list else []
    
    def add_data(self, data: Data):
        """Add a data object."""
        self._data.append(data)
    
    def get_data(self, index: int) -> Data:
        """Get data object at index."""
        return self._data[index]
    
    def data_list(self) -> List[Data]:
        """Get all data objects."""
        return self._data.copy()
    
    def size(self) -> int:
        """Get number of data objects.""" 
        return len(self._data)
    
    def clear(self):
        """Remove all data objects."""
        self._data.clear()
    
    def clone(self) -> 'CompositeData':
        """Create a copy."""
        cloned_data = [data.clone() for data in self._data]
        result = CompositeData(cloned_data)
        result.set_missing(self.is_missing())
        return result
    
    def __len__(self) -> int:
        """Number of data objects."""
        return len(self._data)
    
    def __getitem__(self, index: int) -> Data:
        """Get data object at index."""
        return self._data[index]
    
    def __iter__(self):
        """Iterate over data objects."""
        return iter(self._data)
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"CompositeData(size={len(self._data)}){missing_str}"


# Convenience functions for creating data objects

def create_double_data(value: float) -> DoubleData:
    """Create a DoubleData object."""
    return DoubleData(value)


def create_vector_data(value: Union[Vector, List[float], np.ndarray]) -> VectorData:
    """Create a VectorData object."""
    return VectorData(value)


def create_matrix_data(value: Union[Matrix, List[List[float]], np.ndarray]) -> MatrixData:
    """Create a MatrixData object."""
    return MatrixData(value)


def create_categorical_data(value: Union[int, str], 
                           levels: Optional[List[str]] = None) -> CategoricalData:
    """Create a CategoricalData object."""
    return CategoricalData(value, levels)


def create_binomial_data(n: int, k: int) -> BinomialData:
    """Create a BinomialData object."""
    return BinomialData(n, k)


class MultinomialData(Data):
    """Data class for multinomial observations (n trials, k categories)."""
    
    def __init__(self, n: int, counts: Union[List[int], np.ndarray, Vector]):
        """Initialize with multinomial data.
        
        Args:
            n: Total number of trials.
            counts: Count for each category.
        """
        super().__init__()
        
        if isinstance(counts, Vector):
            counts_array = np.array(counts.to_numpy(), dtype=int)
        elif isinstance(counts, np.ndarray):
            counts_array = counts.astype(int)
        else:
            counts_array = np.array(counts, dtype=int)
        
        if np.any(counts_array < 0):
            raise ValueError("Counts must be non-negative")
        
        if np.sum(counts_array) != n:
            raise ValueError(f"Sum of counts ({np.sum(counts_array)}) must equal n ({n})")
        
        self._n = int(n)
        self._counts = counts_array
        self._k = len(counts_array)
    
    def trials(self) -> int:
        """Get total number of trials."""
        return self._n
    
    def counts(self) -> np.ndarray:
        """Get count array."""
        return self._counts.copy()
    
    def count(self, category: int) -> int:
        """Get count for specific category."""
        if not 0 <= category < self._k:
            raise IndexError(f"Category {category} out of range [0, {self._k})")
        return int(self._counts[category])
    
    def n_categories(self) -> int:
        """Get number of categories."""
        return self._k
    
    def proportions(self) -> np.ndarray:
        """Get proportion for each category."""
        if self._n == 0:
            return np.zeros(self._k)
        return self._counts / self._n
    
    def set_counts(self, counts: Union[List[int], np.ndarray]):
        """Set counts array."""
        if isinstance(counts, np.ndarray):
            counts_array = counts.astype(int)
        else:
            counts_array = np.array(counts, dtype=int)
        
        if len(counts_array) != self._k:
            raise ValueError(f"Counts length {len(counts_array)} must match categories {self._k}")
        
        if np.any(counts_array < 0):
            raise ValueError("Counts must be non-negative")
        
        self._n = int(np.sum(counts_array))
        self._counts = counts_array
    
    def clone(self) -> 'MultinomialData':
        """Create a copy."""
        result = MultinomialData(self._n, self._counts)
        result.set_missing(self.is_missing())
        return result
    
    def __str__(self) -> str:
        """String representation."""
        missing_str = " (missing)" if self.is_missing() else ""
        return f"MultinomialData(n={self._n}, k={self._k}){missing_str}"


def create_composite_data(data_list: Optional[List[Data]] = None) -> CompositeData:
    """Create a CompositeData object."""
    return CompositeData(data_list)


def create_multinomial_data(n: int, counts: Union[List[int], np.ndarray, Vector]) -> MultinomialData:
    """Create a MultinomialData object."""
    return MultinomialData(n, counts)