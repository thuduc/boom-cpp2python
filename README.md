## GenAI Proof of Concept: C++ to Python conversion using Claude Code
The purpose of this proof of concept is to find out if an LLM can take an existing complex  C++ codebase and convert it to Python. The project we will be using for this PoC is this BOOM C++ library for Bayesian modeling: https://github.com/steve-the-bayesian/BOOM

### LLM & AI Tool
* LLMs used: Both Claude Sonnet 4 and Opus 4 (best coding LLMs) - https://www.anthropic.com/news/claude-4
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

### Conversion Process: 
* Step 1 - use Claude Code (together with Opus 4 LLM) to analyze an existing code repository, then ask it to put together a comprehensive conversion plan for converting the entire codebase from C++ to Python. 
* Step 2 - ask Claude Code (together with both Claude 4 LLMs) to use this conversion plan (see [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md)) to implement all phases defined in the plan. Make sure the migration plan includes requirements for comprehensive test coverage of the converted code, via unit and integration tests.

### PoC Results
* The [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md) specifies a migration timeline of 24 months, to be done in 9 phases. The conversion took Claude Code over 5 hours to complete. 
* See [CLAUDE_CODE_SESSION.md](CLAUDE_CODE_SESSION.md) for all prompts issued to Claude Code. A summary response to each prompt by Claude Code is also included. Note the switching between Sonnet 4 for regular tasks and Opus 4 for advanced tasks.
* The conversion effort by Claude Code did not perform a line-by-line conversion of the original C++ code into Python. It analyzed the entire C++ codebase before coming up with a new modular design, then scaffolded the entire project, before proceeding with the C++ to Python conversion.
* The converted Python code resides under impl-python/ directory
* Successful passing of all unit and integration tests. See [impl-python/TEST_SUMMARY.md](impl-python/TEST_SUMMARY.md) for details.

### PoC Assessment
* See [POC_ASSESSMENT.md](POC_ASSESSMENT.md) for detailed assessment of the C++ to Python conversion plan ([PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md)) and the conversion implementation.
* We manually confirmed the core of these findings, especially from Claude Code (using Opus 4) since it does a much more comprehensive analysis of the code than Gemini
    * Functional Completeness: Excellent match of all critical functionality.

    * Core Math functionality: Very well-structured

    * Idiomatic Python usage: Well-structures, uses NumPy effectively.

    * Readability & safety: Simple API, safe defaults like symmetry enforcement (SPD Matrix).

    * Performance efficiency: Could be improved with scipy.linalg, parallelism (multiprocessing) and other open-source high-performance computing libraries.

### Running the Generated Code
See [impl-python/README.md](impl-python/README.md)
