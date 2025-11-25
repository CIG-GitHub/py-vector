"""
py-vector: A Pythonic, zero-dependency vector and table library

Main classes:
    - PyVector: 1D vector with optional type safety
    - PyTable: 2D table (multiple columns of equal length)
    
Type-specific subclasses (auto-created):
    - _PyFloat: Vector of floats with float method proxying
    - _PyInt: Vector of integers with int method proxying
    - _PyString: Vector of strings with string method proxying
    - _PyDate: Vector of dates with date method proxying

Zero external dependencies - pure Python stdlib only.
"""

from .alias_tracker import AliasError
from .backends.tuple import TupleBackend
from .vector import PyVector
from .table import PyTable
from .errors import PyVectorError
from .errors import PyVectorKeyError
from .errors import PyVectorValueError
from .errors import PyVectorTypeError
from .errors import PyVectorIndexError
from .csv import read_csv
from .typing import DataType

__version__ = "0.1.0"
__all__ = [
    "PyVector", 
    "PyTable",
    "read_csv",
    "DataType",
    "AliasError",
    "PyVectorError",
    "PyVectorKeyError",
    "PyVectorValueError",
    "PyVectorTypeError",
    "PyVectorIndexError"
]