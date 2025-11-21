import pytest
from py_vector import PyVector
from py_vector import PyTable
from py_vector.errors import PyVectorKeyError, PyVectorValueError


def test_missing_column_raises_pyvector_keyerror():
    t = PyTable({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(PyVectorKeyError):
        _ = t['missing']


def test_join_mismatched_lengths_raises_pyvector_valueerror():
    left = PyTable({'id': [1, 2], 'date': ['a', 'b']})
    right = PyTable({'id': [2, 3]})
    with pytest.raises(PyVectorValueError):
        left.inner_join(right, left_on=['id', 'date'], right_on=['id'])
