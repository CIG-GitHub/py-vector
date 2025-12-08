class SerifError(Exception):
    """Base exception for serif library."""
    pass


class SerifKeyError(SerifError, KeyError):
    """Raised when a column/key is missing."""
    pass


class SerifTypeError(SerifError, TypeError):
    """Raised for invalid types in API calls."""
    pass


class SerifValueError(SerifError, ValueError):
    """Raised for invalid values or mismatched lengths."""
    pass


class SerifIndexError(SerifError, IndexError):
    """Raised for invalid indexing operations."""
    pass
