# BOOM Python Implementation

This is the Python implementation of BOOM (Bayesian Object Oriented Modeling), migrated from the original C++ library.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -r requirements-dev.txt
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=boom tests/
```

Run benchmarks:
```bash
pytest benchmarks/ -v
```

## Project Structure

- `boom/`: Main package
  - `linalg/`: Linear algebra components
  - `distributions/`: Probability distributions
  - `models/`: Statistical models
  - `samplers/`: MCMC samplers
  - `utils/`: Utility functions
- `tests/`: Test suite
- `benchmarks/`: Performance benchmarks