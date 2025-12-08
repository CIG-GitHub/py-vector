class JibError(Exception):
    """Base exception for jib library."""
    pass


class JibKeyError(JibError, KeyError):
    """Raised when a column/key is missing."""
    pass


class JibTypeError(JibError, TypeError):
    """Raised for invalid types in API calls."""
    pass


class JibValueError(JibError, ValueError):
    """Raised for invalid values or mismatched lengths."""
    pass


class JibIndexError(JibError, IndexError):
    """Raised for invalid indexing operations."""
    pass
