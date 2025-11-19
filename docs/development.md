# Development Guide

## Running Tests

```bash
pytest -q                    # Run all tests
pytest testing/test_*.py -v  # Verbose output
pytest testing/test_pytable.py::TestJoins -v  # Run specific test class
```

## Project Structure

```
py-vector/
├── py_vector.py           # Core PyVector class and typed subclasses
├── py_table.py            # PyTable (joins, aggregate, window)
├── _errors.py             # Exception types
├── _typeutils.py          # Helper utilities
├── _alias_tracker.py      # Alias tracking implementation
├── __init__.py            # Public API exports
├── testing/               # Test suite (pytest)
│   ├── test_pyvector_*.py
│   ├── test_pytable.py
│   ├── test_joins.py
│   ├── test_aggregate_window.py
│   └── ...
└── docs/                  # Documentation
    ├── design-philosophy.md
    ├── performance.md
    └── ...
```

## Key Modules

### `py_vector.py`
- Core `PyVector` class with typed subclasses (`_PyInt`, `_PyFloat`, `_PyString`, `_PyDate`)
- Arithmetic operations and operator overloading
- Fingerprinting and alias tracking
- Elementwise operations

### `py_table.py`
- `PyTable` class (vector-of-vectors)
- Join operations (inner, left, full outer)
- Aggregate and window functions
- Column access and sanitization

### `_errors.py`
- Custom exception hierarchy
- `PyVectorError` base class
- Specific exceptions: `PyVectorKeyError`, `PyVectorValueError`, `PyVectorTypeError`, `PyVectorIndexError`

### `_alias_tracker.py`
- Identity-based alias tracking via weakref registry
- Copy-on-write triggering for shared data

### `_typeutils.py`
- Helper utilities (e.g., `slice_length`)

## Development Status

**Current State:** Pre-release. API may change.

**No versioning/changelog yet**—this is active development.

## Contributing

(Future: Add contribution guidelines, code style, PR process)

## Design Philosophy

See `docs/design-philosophy.md` for detailed rationale behind PyVector's design decisions.
