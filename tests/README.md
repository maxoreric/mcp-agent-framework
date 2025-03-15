# MCP Agent Framework Tests

This directory contains tests for the MCP Agent Framework.

## Test Structure

- `unit/`: Unit tests for individual components
- `integration/`: Tests for component interactions
- `e2e/`: End-to-end tests for complete workflows

## Running Tests

To run all tests:

```bash
pytest tests/
```

To run a specific test module:

```bash
pytest tests/unit/test_agent.py
```

To run a specific test case:

```bash
pytest tests/unit/test_agent.py::TestAgent::test_initialization
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=mcp_agent_framework tests/
```

## Writing Tests

When adding new features, please ensure:

1. Unit tests cover individual functions and methods
2. Integration tests verify component interactions
3. End-to-end tests validate complete workflows
4. All tests have clear docstrings explaining their purpose
5. Mock external dependencies to ensure tests are deterministic
