# BOOM C++ to Python Migration Plan - Simplified Version

## Executive Summary

This plan migrates the BOOM (Bayesian Object Oriented Modeling) library from C++ to Python. The codebase contains ~220,000 lines of C++ code across 1,819 files, including extensive linear algebra operations via Eigen library.

## Core Strategy

### Principles
1. **Simple is better**: Use NumPy/SciPy for all numerical operations
2. **No premature optimization**: Pure Python first, optimize only if needed
3. **Leverage existing work**: Build on existing Python bindings in Interfaces/python/
4. **Direct mapping**: C++ classes → Python classes with minimal redesign
5. **Sequential migration**: One component at a time, maintaining functionality

## Technology Stack (Simplified)

### Essential Dependencies Only
- **NumPy**: All linear algebra and array operations
- **SciPy**: Statistical distributions and special functions
- **pytest**: Testing framework
- **typing**: Type hints for clarity

### No Advanced Features
- No Cython unless absolutely necessary
- No parallel processing
- No GPU acceleration
- No JIT compilation

## Migration Phases (24 months)

### Phase 0: Foundation and Existing Code Review (Month 1)
**Goal**: Set up project and analyze existing Python bindings

1. **Project Setup**
   ```
   boom/
   ├── __init__.py
   ├── linalg/
   ├── distributions/
   ├── math/
   ├── models/
   ├── samplers/
   ├── stats/
   ├── utils/
   └── tests/
   ```

2. **Analyze Existing Python Code**
   - Review Interfaces/python/BayesBoom/
   - Review Interfaces/python/bsts/
   - Identify reusable components
   - Document existing API patterns

### Phase 1: Core Mathematics (Months 2-4)
**Goal**: Replace Eigen and establish mathematical foundation

1. **Eigen → NumPy Mapping**
   - LinAlg/Vector.hpp → boom/linalg/vector.py (NumPy array wrapper)
   - LinAlg/Matrix.hpp → boom/linalg/matrix.py (NumPy 2D array wrapper)
   - LinAlg/SpdMatrix.hpp → boom/linalg/spd_matrix.py
   - All Eigen operations → NumPy equivalents

2. **Basic Math Functions**
   - math/ → boom/math/ (26 files)
   - Bmath/ → boom/math/rmath.py (R math functions using SciPy)
   - Special functions using scipy.special

3. **Utilities**
   - cpputil/ → boom/utils/ (basic utilities only)
   - Focus on essential functions used by other components

### Phase 2: Distributions and Random Numbers (Months 5-6)
**Goal**: Implement probability distributions

1. **Core Components**
   - distributions/rng.hpp → boom/distributions/rng.py (NumPy RandomState)
   - Use scipy.stats for standard distributions
   - Custom distributions as simple Python classes

2. **Implementation Strategy**
   ```python
   # Example: Simple wrapper around scipy
   class GammaDistribution:
       def __init__(self, alpha, beta):
           self.alpha = alpha
           self.beta = beta
       
       def pdf(self, x):
           return scipy.stats.gamma.pdf(x, self.alpha, scale=1/self.beta)
   ```

### Phase 3: Base Model Framework (Months 7-9)
**Goal**: Establish model hierarchy

1. **Core Classes**
   - Models/Model.hpp → boom/models/base.py
   - Models/ParamTypes.hpp → boom/models/params.py
   - Models/DataTypes.hpp → boom/models/data.py
   - Models/Sufstat.hpp → boom/models/sufstat.py

2. **Simple Python Implementation**
   ```python
   class Model:
       def __init__(self):
           self.parameters = {}
           
       def log_likelihood(self, data):
           raise NotImplementedError
   ```

### Phase 4: Basic Statistical Models (Months 10-12)
**Goal**: Implement fundamental models

1. **Priority Models** (Simple implementations)
   - GaussianModel
   - BinomialModel  
   - PoissonModel
   - MultinomialModel
   - GammaModel
   - BetaModel

2. **Testing**
   - Unit tests comparing with C++ outputs
   - Use existing C++ test cases as reference

### Phase 5: Sampling Algorithms (Months 13-15)
**Goal**: Basic MCMC samplers

1. **Core Samplers**
   - Samplers/ → boom/samplers/
   - Simple Metropolis-Hastings
   - Basic slice sampler
   - Gibbs sampler components

2. **No Optimization**
   - Pure Python implementations
   - Focus on correctness over speed

### Phase 6: GLM Models (Months 16-18)
**Goal**: Generalized linear models

1. **Components**
   - Models/Glm/ → boom/models/glm/
   - Focus on most-used models
   - Simple posterior samplers

### Phase 7: State Space Models (Months 19-21)
**Goal**: Time series models

1. **Core Components**
   - Models/StateSpace/ → boom/models/state_space/
   - Basic Kalman filter
   - Essential state space models

### Phase 8: Remaining Components (Months 22-24)
**Goal**: Complete migration

1. **Additional Models**
   - HMM (essential subset)
   - Mixtures (basic models)
   - Time series (core models)

2. **Optimization and Stats**
   - numopt/ → boom/optimization/ (scipy.optimize wrappers)
   - stats/ → boom/stats/
   - TargetFun/ → boom/target_functions/

## Implementation Guidelines

### Direct C++ to Python Mapping

```python
# C++: Vector class
# Python: Simple NumPy wrapper
class Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
    
    def __add__(self, other):
        return Vector(self.data + other.data)
    
    def norm(self):
        return np.linalg.norm(self.data)
```

### No Complex Patterns
- No metaclasses
- No complex inheritance hierarchies  
- No advanced Python features
- Direct, readable code

## Testing Strategy

### Simple Test Structure
```python
def test_gaussian_model():
    # Create model
    model = GaussianModel(mean=0.0, variance=1.0)
    
    # Test against known values
    assert abs(model.log_likelihood([0.0]) - (-0.9189385)) < 1e-6
    
    # Compare with C++ output
    # (Run C++ version separately and compare results)
```

### Validation Approach
1. Port C++ test cases directly
2. Compare numerical outputs
3. Accept small numerical differences (< 1e-10)

## Migration Execution

### File-by-File Process
1. Read C++ header and implementation
2. Write Python equivalent
3. Port test cases
4. Validate against C++ output
5. Document any differences

### No Premature Optimization
- Write clean, simple Python first
- Profile only if performance is unacceptable
- Consider Cython only as last resort

## Risk Management

### Simplified Risks
1. **Performance**: Accept 5-10x slower initially
2. **Numerical differences**: Document all variances
3. **Missing features**: Implement only what's needed

### Mitigation
- Clear documentation of limitations
- Maintain C++ version during migration
- Focus on correctness over performance

## Timeline Summary

**Total Duration: 24 months**

| Phase | Duration | Component |
|-------|----------|-----------|
| 0 | 1 month | Setup & existing code review |
| 1 | 3 months | Core math & Eigen replacement |
| 2 | 2 months | Distributions |
| 3 | 3 months | Model framework |
| 4 | 3 months | Basic models |
| 5 | 3 months | Samplers |
| 6 | 3 months | GLM models |
| 7 | 3 months | State space models |
| 8 | 3 months | Remaining components |

## Success Criteria

1. **Correctness**: Numerical agreement with C++ (within tolerance)
2. **Completeness**: Core functionality available
3. **Simplicity**: Code is readable and maintainable
4. **Testing**: All components have test coverage

## Team Requirements

- 2-3 Python developers with numerical computing experience
- 1 Statistician familiar with Bayesian methods
- Access to C++ BOOM experts for questions

## Deliverables

Each phase produces:
1. Python module(s)
2. Test suite
3. Validation report comparing with C++
4. Simple documentation

This simplified plan focuses on getting a working Python version without complexity, accepting performance trade-offs for maintainability and ease of migration.