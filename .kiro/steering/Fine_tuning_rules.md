---
inclusion: always
---

# Project Conventions and Guidelines

## Python Environment

- Use `python -m venv .venv` to create virtual environments; activate before running any script or installing packages.
- Install all dependencies inside the active venv via `pip` — never install globally.
- Pin exact versions in `requirements.txt`; update it whenever dependencies change.

## Code Style

- Follow PEP 8: 4-space indentation, no tabs, 88-character line limit (Black-compatible).
- Add type hints to every function signature, including return types.
- Use f-strings for all string formatting — never `%` or `.format()`.
- Write Google-style docstrings for all public modules, classes, and functions.

## Architecture

- Separate code into three distinct layers: I/O, business logic, and data models — keep them decoupled.
- Prefer composition over inheritance.
- Keep functions small and single-purpose; minimize side effects.
- Inject dependencies explicitly — no global mutable state.

## Testing

- Write `pytest` tests for every new feature and bug fix under `tests/`, mirroring the source structure.
- Cover edge cases and failure paths, not just the happy path.
- Use `hypothesis` for property-based tests on data-transformation and parsing logic.

## Security

- Never commit secrets, API keys, or credentials — use environment variables or a secrets manager.
- Validate and sanitize all external inputs before use.
- Use parameterized queries for all database interactions.

## General

- Prefer explicit over implicit.
- Delete dead code — do not comment it out.
- Keep commits focused and atomic with clear, descriptive messages.
