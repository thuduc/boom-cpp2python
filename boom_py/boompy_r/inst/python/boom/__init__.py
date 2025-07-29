"""BOOM: Bayesian Object Oriented Modeling - Python Implementation."""

__version__ = "0.1.0"

# Import key components for easy access
from boom.linalg import Vector, Matrix, SpdMatrix
from boom.distributions import rng

__all__ = [
    "Vector",
    "Matrix", 
    "SpdMatrix",
    "rng",
]