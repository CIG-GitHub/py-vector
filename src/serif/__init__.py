"""
serif: A Pythonic, zero-dependency vector and table library

Designed for Python users who need to work with datasets beyond Excel's limits
(>1000 rows) but want the ease-of-use and intuitive feel of Excel or SQL.

Main classes:
    - Vector: 1D vector with optional type safety
    - Table: 2D table (multiple columns of equal length)
    
Type-specific subclasses (auto-created):
    - _Float: Vector of floats with float method proxying
    - _Int: Vector of integers with int method proxying
    - _String: Vector of strings with string method proxying
    - _Date: Vector of dates with date method proxying

Zero external dependencies - pure Python stdlib only.
"""

from .alias_tracker import _ALIAS_TRACKER, AliasError
from .vector import Vector, _Float, _Int, _String, _Date
from .table import Table
from .errors import SerifError, SerifKeyError, SerifValueError, SerifTypeError, SerifIndexError
from .csv import read_csv
from .typing import DataType

__version__ = "0.1.0"
__all__ = [
	"Vector", 
	"Table",
	"read_csv",
	"DataType",
	"AliasError",
	"SerifError",
	"SerifKeyError",
	"SerifValueError",
	"SerifTypeError",
	"SerifIndexError"
]
