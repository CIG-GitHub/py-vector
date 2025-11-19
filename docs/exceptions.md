# Exception Handling

PyVector raises specific exception types for clear error handling.

## Exception Types

### PyVectorKeyError
Subclass of `KeyError`. Raised when:
- Column not found in table
- Key missing in dictionary operations

```python
from py_vector import PyVectorKeyError

try:
    column = table['missing_column']
except PyVectorKeyError:
    print("Column not found")
```

### PyVectorValueError
Subclass of `ValueError`. Raised when:
- Invalid values provided
- Mismatched lengths in operations
- Invalid join key configurations

```python
from py_vector import PyVectorValueError

try:
    table.inner_join(other, left_on=['a', 'b'], right_on=['x'])
except PyVectorValueError:
    print("Mismatched join key lengths")
```

### PyVectorTypeError
Subclass of `TypeError`. Raised when:
- Invalid types provided to type-safe vectors
- Type mismatches in operations

```python
from py_vector import PyVectorTypeError

try:
    result = typesafe_int_vector + "string"
except PyVectorTypeError:
    print("Type mismatch")
```

### PyVectorIndexError
Subclass of `IndexError`. Raised when:
- Out-of-bounds indexing
- Invalid slice operations

```python
from py_vector import PyVectorIndexError

try:
    value = vector[1000]  # index out of range
except PyVectorIndexError:
    print("Index out of bounds")
```

## Broad Exception Catching

All custom exceptions inherit from `PyVectorError`:

```python
from py_vector import PyVectorError

try:
    # ... operations ...
except PyVectorError:
    # Catch all PyVector-specific errors
    pass
```

## Attribute Access

`table.missing_column` raises `AttributeError` (Pythonic behavior). 

Use `table['col']` for dictionary-style access or check existence with `'col' in table._underlying`.
