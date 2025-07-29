"""Data types for BOOM models."""
import numpy as np
from typing import Union, Optional, List
from ..linalg import Vector, Matrix
from .base import Data


class DoubleData(Data):
    """Single numeric observation."""
    
    def __init__(self, value: float):
        """Initialize with a single value."""
        self.value = float(value)
    
    def __float__(self):
        return self.value
    
    def __repr__(self):
        return f"DoubleData({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, DoubleData):
            return self.value == other.value
        return self.value == other


class VectorData(Data):
    """Vector observation."""
    
    def __init__(self, values: Union[Vector, np.ndarray, List[float]]):
        """Initialize with vector values."""
        self.value = Vector(values)
    
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __repr__(self):
        return f"VectorData({self.value})"
    
    def dim(self) -> int:
        """Dimension of the vector."""
        return len(self.value)


class RegressionData(Data):
    """Data for regression models."""
    
    def __init__(self, y: float, x: Union[Vector, np.ndarray, List[float]]):
        """Initialize regression data.
        
        Args:
            y: Response variable
            x: Predictor variables
        """
        self.y = float(y)
        self.x = Vector(x)
    
    @property
    def xdim(self) -> int:
        """Dimension of predictors."""
        return len(self.x)
    
    def __repr__(self):
        return f"RegressionData(y={self.y}, x={self.x})"


class CategoricalData(Data):
    """Categorical data."""
    
    def __init__(self, value: Union[int, str], levels: Optional[List] = None):
        """Initialize categorical data.
        
        Args:
            value: The categorical value
            levels: Optional list of possible levels
        """
        self.value = value
        self.levels = levels
        if levels is not None:
            if value not in levels:
                raise ValueError(f"Value {value} not in levels {levels}")
            self._index = levels.index(value)
        else:
            self._index = None
    
    @property
    def index(self) -> Optional[int]:
        """Get numeric index of value (if levels are defined)."""
        return self._index
    
    def __repr__(self):
        return f"CategoricalData({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, CategoricalData):
            return self.value == other.value
        return self.value == other


class BinomialData(Data):
    """Binomial data (successes out of trials)."""
    
    def __init__(self, successes: int, trials: int):
        """Initialize binomial data.
        
        Args:
            successes: Number of successes
            trials: Number of trials
        """
        if successes < 0 or trials < 0:
            raise ValueError("Successes and trials must be non-negative")
        if successes > trials:
            raise ValueError("Successes cannot exceed trials")
        self.successes = int(successes)
        self.trials = int(trials)
    
    @property
    def failures(self) -> int:
        """Number of failures."""
        return self.trials - self.successes
    
    @property
    def proportion(self) -> float:
        """Success proportion."""
        if self.trials == 0:
            return 0.0
        return self.successes / self.trials
    
    def __repr__(self):
        return f"BinomialData(successes={self.successes}, trials={self.trials})"


class PoissonData(Data):
    """Poisson data with optional exposure."""
    
    def __init__(self, count: int, exposure: float = 1.0):
        """Initialize Poisson data.
        
        Args:
            count: Event count
            exposure: Exposure/offset (default 1.0)
        """
        if count < 0:
            raise ValueError("Count must be non-negative")
        if exposure <= 0:
            raise ValueError("Exposure must be positive")
        self.count = int(count)
        self.exposure = float(exposure)
    
    @property
    def rate(self) -> float:
        """Rate (count/exposure)."""
        return self.count / self.exposure
    
    def __repr__(self):
        if self.exposure == 1.0:
            return f"PoissonData(count={self.count})"
        return f"PoissonData(count={self.count}, exposure={self.exposure})"


class MultinomialData(Data):
    """Multinomial data."""
    
    def __init__(self, counts: Union[Vector, np.ndarray, List[int]]):
        """Initialize multinomial data.
        
        Args:
            counts: Counts for each category
        """
        self.counts = Vector(counts).astype(int)
        if np.any(self.counts < 0):
            raise ValueError("All counts must be non-negative")
    
    @property
    def total(self) -> int:
        """Total count."""
        return int(self.counts.sum())
    
    @property
    def proportions(self) -> Vector:
        """Category proportions."""
        total = self.total
        if total == 0:
            return Vector(np.zeros(len(self.counts)))
        return self.counts / total
    
    @property
    def ncategories(self) -> int:
        """Number of categories."""
        return len(self.counts)
    
    def __repr__(self):
        return f"MultinomialData({self.counts})"


class TimeSeriesData(Data):
    """Time series data point."""
    
    def __init__(self, value: float, time: float):
        """Initialize time series data.
        
        Args:
            value: Observation value
            time: Time point
        """
        self.value = float(value)
        self.time = float(time)
    
    def __repr__(self):
        return f"TimeSeriesData(value={self.value}, time={self.time})"


class WeightedData(Data):
    """Weighted data wrapper."""
    
    def __init__(self, data: Data, weight: float = 1.0):
        """Initialize weighted data.
        
        Args:
            data: The underlying data
            weight: Weight (default 1.0)
        """
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        self.data = data
        self.weight = float(weight)
    
    def __repr__(self):
        return f"WeightedData({self.data}, weight={self.weight})"
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying data."""
        return getattr(self.data, name)


class SpdData(Data):
    """Symmetric positive definite matrix data."""
    
    def __init__(self, matrix: Union[Matrix, np.ndarray]):
        """Initialize SPD matrix data.
        
        Args:
            matrix: Symmetric positive definite matrix
        """
        from ..linalg import SpdMatrix
        self.value = SpdMatrix(matrix)
    
    @property
    def dim(self) -> int:
        """Dimension of the matrix."""
        return self.value.nrow()
    
    def __repr__(self):
        return f"SpdData({self.value})"


class CompositeData(Data):
    """Composite data containing multiple data elements."""
    
    def __init__(self, *components):
        """Initialize composite data.
        
        Args:
            *components: Data components
        """
        self.components = components
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, idx):
        return self.components[idx]
    
    def __repr__(self):
        return f"CompositeData{self.components}"