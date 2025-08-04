"""Setup configuration for BOOM Python package."""

from setuptools import setup, find_packages

setup(
    name="boom",
    version="0.1.0",
    description="Bayesian Object Oriented Modeling - Python implementation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
            "mypy",
        ]
    },
    python_requires=">=3.8",
)