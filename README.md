## GenAI Proof of Concept: C++ to Python conversion using Claude Code

The purpose of this proof of concept is to find out if an LLM can take an existing complex  C++ codebase and converted it to Python. The project we will be using for this PoC is this BOOM C++ library for Bayesian modeling: https://github.com/steve-the-bayesian/BOOM

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

#### Conversion Process: 
* Step 1 - use Claude Code (together with Opus 4 LLM) to analyze an existing code repository, then ask it to put together a comprehensive conversion plan for converting the entire codebase from C++ to Python. 
* Step 2 - ask Claude Code to use this conversion plan (see [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md)) to implement all phases defined in the plan. Make sure the migration plan includes requirements for comprehensive test coverage of the converted code, via unit and integration tests.

### PoC Results
* The [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md) specified a migration timeline of 18 months, to be done in 8 phases. The conversion took Claude Code over 4 hours to complete. 
* The conversion effort by Claude Code did not perform a line-by-line conversion of the original C++ code into Python. It analyzed the entire C++ codebase before coming up with a new modular design, then scaffold the entire project, before proceeding with the C++ to Python conversion.
* The converted Python code resides under boom_py/ directory
* Successful passing of all unit and integration tests. See [boom_py/TEST_SUMMARY.md](boom_py/TEST_SUMMARY.md) for details.


## Running the code
See [boom_py/README.md](boom_py/README.md)
