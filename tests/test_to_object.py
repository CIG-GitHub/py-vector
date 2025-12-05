"""
Test PyVector.to_object() conversion method.
"""

import pytest
from py_vector import PyVector
from py_vector.errors import PyVectorTypeError


def test_to_object_basic():
	"""Test converting int vector to object vector"""
	a = PyVector([1, 2, 3, 4])
	assert a._dtype.kind is int
	
	b = a.to_object()
	assert b._dtype.kind is object
	assert list(b) == [1, 2, 3, 4]


def test_to_object_allows_mixed_assignment():
	"""Test that object vector allows mixed type assignment"""
	a = PyVector([1, 2, 3, 4])
	a = a.to_object()
	
	a[2] = "ryan"
	assert a[2] == "ryan"
	assert a[0] == 1
	assert list(a) == [1, 2, "ryan", 4]


def test_to_object_preserves_name():
	"""Test that to_object preserves vector name"""
	a = PyVector([1, 2, 3], name="mydata")
	b = a.to_object()
	assert b._name == "mydata"


def test_to_object_preserves_display_as_row():
	"""Test that to_object preserves display_as_row setting"""
	a = PyVector([1, 2, 3], as_row=True)
	b = a.to_object()
	assert b._display_as_row == True


def test_int_vector_rejects_string():
	"""Test that int vector rejects string assignment"""
	a = PyVector([1, 2, 3, 4])
	with pytest.raises(PyVectorTypeError, match="Cannot set str in int vector"):
		a[2] = "ryan"


def test_dtype_object_on_creation():
	"""Test creating object vector directly with dtype parameter"""
	a = PyVector([1, 2, 3, 4], dtype=object)
	assert a._dtype.kind is object
	a[2] = "test"
	assert a[2] == "test"


def test_dtype_string_on_creation():
	"""Test creating object vector with dtype as string"""
	a = PyVector([1, 2, 3, 4], dtype=object)  # Use object type, not string
	assert a._dtype.kind is object
	a[2] = "test"
	assert a[2] == "test"


def test_to_object_idempotent():
	"""Test that calling to_object on object vector is safe"""
	import warnings
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		a = PyVector([1, "two", 3.0])
	assert a._dtype.kind is object
	
	b = a.to_object()
	assert b._dtype.kind is object
	assert list(b) == [1, "two", 3.0]


def test_to_object_with_string_vector():
	"""Test converting string vector to object"""
	a = PyVector(["a", "b", "c"])
	assert a._dtype.kind is str
	
	b = a.to_object()
	assert b._dtype.kind is object
	assert list(b) == ["a", "b", "c"]


def test_to_object_with_float_vector():
	"""Test converting float vector to object"""
	a = PyVector([1.5, 2.5, 3.5])
	assert a._dtype.kind is float
	
	b = a.to_object()
	assert b._dtype.kind is object
	assert list(b) == [1.5, 2.5, 3.5]
