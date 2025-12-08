import pytest
from jib import Vector
from jib import Table
from jib.errors import JibKeyError, JibValueError


def test_missing_column_raises_Vector_keyerror():
    t = Table({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(JibKeyError):
        _ = t['missing']


def test_join_mismatched_lengths_raises_Vector_valueerror():
    left = Table({'id': [1, 2], 'date': ['a', 'b']})
    right = Table({'id': [2, 3]})
    with pytest.raises(JibValueError):
        left.inner_join(right, left_on=['id', 'date'], right_on=['id'])



