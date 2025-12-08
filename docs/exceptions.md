# Exception Handling

Vector raises specific exception types for clear error handling.

## Exception Types

### JibKeyError
Subclass of `KeyError`. Raised when:
- Column not found in table
- Key missing in dictionary operations

```python
from jib import JibKeyError

try:
    column = table['missing_column']
except JibKeyError:
    print("Column not found")
```

### JibValueError
Subclass of `ValueError`. Raised when:
- Invalid values provided
- Mismatched lengths in operations
- Invalid join key configurations

```python
from jib import JibValueError

try:
    table.inner_join(other, left_on=['a', 'b'], right_on=['x'])
except JibValueError:
    print("Mismatched join key lengths")
```

### JibTypeError
Subclass of `TypeError`. Raised when:
- Invalid types provided to type-safe vectors
- Type mismatches in operations

```python
from jib import JibTypeError

try:
    result = typesafe_int_vector + "string"
except JibTypeError:
    print("Type mismatch")
```

### JibIndexError
Subclass of `IndexError`. Raised when:
- Out-of-bounds indexing
- Invalid slice operations

```python
from jib import JibIndexError

try:
    value = vector[1000]  # index out of range
except JibIndexError:
    print("Index out of bounds")
```

## Broad Exception Catching

All custom exceptions inherit from `JibError`:

```python
from jib import JibError

try:
    # ... operations ...
except JibError:
    # Catch all Vector-specific errors
    pass
```

## Attribute Access

`table.missing_column` raises `AttributeError` (Pythonic behavior). 

Use `table['col']` for dictionary-style access or check existence with `'col' in table._underlying`.


