class PyVectorError(Exception):
    """Base exception for py-vector library."""
    pass


class PyVectorKeyError(PyVectorError, KeyError):
    """Raised when a column/key is missing."""
    pass


class PyVectorTypeError(PyVectorError, TypeError):
    """Raised for invalid types in API calls."""
    pass


class PyVectorValueError(PyVectorError, ValueError):
    """Raised for invalid values or mismatched lengths."""
    pass


class PyVectorIndexError(PyVectorError, IndexError):
    """Raised for invalid indexing operations."""
    pass
