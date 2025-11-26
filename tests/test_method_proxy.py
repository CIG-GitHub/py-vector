"""
Tests for MethodProxy and __getattr__ property vs method distinction.

This tests the refactored __getattr__ that automatically distinguishes between:
- Properties (like date.year) → evaluated immediately
- Methods (like str.replace) → return MethodProxy that waits for ()
"""

from py_vector import PyVector
from datetime import date


def test_date_properties():
	"""date.year, date.month, date.day are properties, not methods."""
	dates = PyVector([date(2024, 1, 15), date(2024, 12, 25)])
	
	# Properties should work without parentheses
	years = dates.year
	assert (years == PyVector([2024, 2024])).all()
	
	months = dates.month
	assert (months == PyVector([1, 12])).all()
	
	days = dates.day
	assert (days == PyVector([15, 25])).all()


def test_date_methods():
	"""date.replace, date.strftime are methods, not properties."""
	dates = PyVector([date(2024, 1, 15), date(2024, 12, 25)])
	
	# Methods should require parentheses
	replaced = dates.replace(year=2025)
	assert (replaced == PyVector([date(2025, 1, 15), date(2025, 12, 25)])).all()
	
	formatted = dates.strftime("%Y-%m-%d")
	assert (formatted == PyVector(["2024-01-15", "2024-12-25"])).all()


def test_str_methods():
	"""str.replace, str.upper are methods."""
	words = PyVector(["hello", "world"])
	
	# Methods require parentheses
	replaced = words.replace("o", "0")
	assert (replaced == PyVector(["hell0", "w0rld"])).all()
	
	uppercased = words.upper()
	assert (uppercased == PyVector(["HELLO", "WORLD"])).all()


def test_int_properties():
	"""int.real, int.imag are properties."""
	nums = PyVector([1, 2, 3])
	
	# These are properties (not methods)
	reals = nums.real
	assert (reals == PyVector([1, 2, 3])).all()
	
	imags = nums.imag
	assert (imags == PyVector([0, 0, 0])).all()


def test_int_methods():
	"""int.bit_length, int.bit_count are methods."""
	nums = PyVector([1, 2, 3, 4])
	
	# Methods require parentheses
	lengths = nums.bit_length()
	assert (lengths == PyVector([1, 2, 2, 3])).all()
	
	counts = nums.bit_count()
	assert (counts == PyVector([1, 1, 2, 1])).all()


def test_float_properties():
	"""float.real, float.imag are properties."""
	floats = PyVector([1.5, 2.5, 3.5])
	
	# Properties work without parentheses
	reals = floats.real
	assert (reals == PyVector([1.5, 2.5, 3.5])).all()
	
	imags = floats.imag
	assert (imags == PyVector([0.0, 0.0, 0.0])).all()


def test_float_methods():
	"""float.is_integer, float.hex are methods."""
	floats = PyVector([1.0, 2.5, 3.0])
	
	# Methods require parentheses
	is_ints = floats.is_integer()
	assert (is_ints == PyVector([True, False, True])).all()
	
	hexes = floats.hex()
	assert len(hexes) == 3
	assert all(isinstance(h, str) for h in hexes)


def test_object_dtype_raises():
	"""PyVector[object] should reject __getattr__ attempts."""
	v = PyVector([object(), object()])
	
	try:
		_ = v.some_attr
		assert False, "Should have raised AttributeError"
	except AttributeError as e:
		assert "PyVector[object]" in str(e)


def test_nonexistent_attribute_raises():
	"""Accessing non-existent attributes should raise AttributeError."""
	dates = PyVector([date(2024, 1, 1)])
	
	try:
		_ = dates.nonexistent_method
		assert False, "Should have raised AttributeError"
	except AttributeError as e:
		assert "date" in str(e)
		assert "nonexistent_method" in str(e)

def test_methodproxy_string_with_none():
    v = PyVector(["a", None, "b"])
    out = v.upper()
    assert list(out) == ["A", None, "B"]


def test_methodproxy_date_property_with_none():
    v = PyVector([date(2024, 1, 5), None, date(2020, 7, 1)])
    out = v.year
    assert list(out) == [2024, None, 2020]


def test_methodproxy_date_method_with_none():
    v = PyVector([date(2024, 1, 5), None])
    out = v.isoformat()
    assert list(out) == ["2024-01-05", None]


def test_methodproxy_string_property_none():
    v = PyVector(["abc", None, "DEF"])
    out = v.islower()
    assert list(out) == [True, None, False]
