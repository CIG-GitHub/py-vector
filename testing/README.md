"""
PyVector Test Suite
===================

Run all tests with: pytest testing/

Individual test files:
- test_pyvector_basic.py      : Basic PyVector operations (creation, copy, iteration)
- test_pyvector_indexing.py   : Indexing, slicing, boolean masks
- test_pyvector_math.py        : Math operations, comparisons, aggregations
- test_pytable.py              : PyTable 2D operations, column/row access
- test_pyfloat.py              : _PyFloat proxy methods
- test_pyint.py                : _PyInt proxy methods  
- test_pystring.py             : _PyString proxy methods
- test_pydate.py               : _PyDate proxy methods
- test_fingerprint.py          : Fingerprint change detection (old format)

Total test coverage:
- ~150+ parametrized test cases
- All core functionality covered
- Type-specific method proxying tested
- Change detection validated

Run specific test file:
    pytest testing/test_pyvector_basic.py -v

Run specific test class:
    pytest testing/test_pyvector_basic.py::TestCreation -v

Run with coverage:
    pytest testing/ --cov=py_vector --cov-report=html
"""
