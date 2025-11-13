"""
py-vector: A Pythonic, zero-dependency vector and table library

Designed for Python users who need to work with datasets beyond Excel's limits
(>1000 rows) but want the ease-of-use and intuitive feel of Excel or SQL.

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

from py_vector import PyVector, PyTable, _PyFloat, _PyInt, _PyString, _PyDate

__version__ = "0.1.0"
__all__ = ["PyVector", "PyTable", "_PyFloat", "_PyInt", "_PyString", "_PyDate"]
