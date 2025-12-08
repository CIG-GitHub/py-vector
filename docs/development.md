# Development Guide

## Running Tests

```bash
pytest -q                    # Run all tests
pytest testing/test_*.py -v  # Verbose output
pytest testing/test_Table.py::TestJoins -v  # Run specific test class
```

## Project Structure

```
serif/
├── serif.py           # Core Vector class and typed subclasses
├── py_table.py            # Table (joins, aggregate, window)
├── _errors.py             # Exception types
├── _typeutils.py          # Helper utilities
├── _alias_tracker.py      # Alias tracking implementation
├── __init__.py            # Public API exports
├── testing/               # Test suite (pytest)
│   ├── test_Vector_*.py
│   ├── test_Table.py
│   ├── test_joins.py
│   ├── test_aggregate_window.py
│   └── ...
└── docs/                  # Documentation
    ├── design-philosophy.md
    ├── performance.md
    └── ...
```

## Key Modules

### `serif.py`
- Core `Vector` class with typed subclasses (`_Int`, `_Float`, `_String`, `_Date`)
- Arithmetic operations and operator overloading
- Fingerprinting and alias tracking
- Elementwise operations

### `table.py`
- `Table` class (vector-of-vectors)
- Join operations (inner, left, full outer)
- Aggregate and window functions
- Column access and sanitization

### `_errors.py`
- Custom exception hierarchy
- `VectorError` base class
- Specific exceptions: `VectorKeyError`, `VectorValueError`, `VectorTypeError`, `VectorIndexError`

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

See `docs/design-philosophy.md` for detailed rationale behind Vector's design decisions.

