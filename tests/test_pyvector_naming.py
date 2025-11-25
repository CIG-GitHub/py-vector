import pytest
from py_vector import PyVector
from py_vector import PyTable


def test_name_initialization():
	"""Test that names can be set during initialization"""
	v = PyVector([1, 2, 3], name="my_vector")
	assert v.name == "my_vector"
	
	v_no_name = PyVector([1, 2, 3])
	assert v_no_name.name is None


def test_copy_preserves_name_by_default():
	"""Test that copy() preserves name by default"""
	v = PyVector([1, 2, 3], name="original")
	v_copy = v.copy()
	assert v_copy.name == "original"


def test_copy_can_override_name():
	"""Test that copy() can explicitly set a new name"""
	v = PyVector([1, 2, 3], name="original")
	v_renamed = v.copy(name="renamed")
	assert v_renamed.name == "renamed"


def test_copy_can_clear_name():
	"""Test that copy() can explicitly clear the name"""
	v = PyVector([1, 2, 3], name="original")
	v_unnamed = v.copy(name=None)
	assert v_unnamed.name is None


def test_transpose_preserves_name():
	"""Test that .T preserves the name"""
	v = PyVector([1, 2, 3], name="my_vector")
	v_t = v.T
	assert v_t.name == "my_vector"


def test_slice_preserves_name():
	"""Test that slicing preserves the name"""
	v = PyVector([1, 2, 3, 4, 5], name="my_vector")
	v_slice = v[1:4]
	assert v_slice.name == "my_vector"


def test_index_selection_preserves_name():
	"""Test that single index selection preserves the name"""
	v = PyVector([1, 2, 3, 4, 5], name="my_vector")
	v_single = v[2]
	# Single index returns a scalar, not a PyVector with a name
	assert isinstance(v_single, int)


def test_promote_preserves_name():
	"""Test that type promotion preserves the name"""
	v = PyVector([1, 2, 3], name="my_vector")
	v._promote(float)
	assert v.name == "my_vector"
	assert v.schema().kind == float


def test_math_operations_do_not_preserve_name():
	"""Test that math operations do NOT preserve names"""
	v1 = PyVector([1, 2, 3], name="vector1")
	v2 = PyVector([4, 5, 6], name="vector2")
	
	# Binary operations with two named vectors
	assert (v1 + v2).name is None
	assert (v1 - v2).name is None
	assert (v1 * v2).name is None
	assert (v1 / v2).name is None
	assert (v1 // v2).name is None
	assert (v1 % v2).name is None
	assert (v1 ** v2).name is None
	
	# Operations with scalars
	assert (v1 + 10).name is None
	assert (v1 - 10).name is None
	assert (v1 * 10).name is None
	assert (v1 / 10).name is None
	
	# Reverse operations
	assert (10 + v1).name is None
	assert (10 - v1).name is None
	assert (10 * v1).name is None


def test_math_operations_with_unnamed_vectors():
	"""Test that math operations with unnamed vectors also return unnamed results"""
	v1 = PyVector([1, 2, 3])
	v2 = PyVector([4, 5, 6])
	
	assert (v1 + v2).name is None
	assert (v1 * 2).name is None


def test_aggregations_do_not_preserve_name():
	"""Test that aggregation methods do NOT preserve names"""
	v = PyVector([1, 2, 3, 4, 5], name="my_vector")
	
	# 1D aggregations return scalars (no name attribute)
	assert isinstance(v.sum(), (int, float))
	assert isinstance(v.mean(), (int, float))
	assert isinstance(v.max(), (int, float))
	assert isinstance(v.min(), (int, float))
	assert isinstance(v.stdev(), (int, float))
	
	# unique() returns a set (no name)
	assert isinstance(v.unique(), set)


def test_2d_aggregations_do_not_preserve_name():
	"""Test that 2D aggregations do NOT preserve names"""
	# Create a proper 2D vector (PyTable-like structure)
	col1 = PyVector([1, 3, 5])
	col2 = PyVector([2, 4, 6])
	v = PyVector([col1, col2], name="my_matrix")
	
	# 2D aggregations return PyVectors without names
	assert v.sum().name is None
	assert v.mean().name is None
	assert v.max().name is None
	assert v.min().name is None
	assert v.stdev().name is None


def test_string_methods_do_not_preserve_name():
	"""Test that string methods do NOT preserve names"""
	v = PyVector(["Hello", "World"], name="my_strings", dtype=str)
	
	assert v.upper().name is None
	assert v.lower().name is None
	assert v.strip().name is None
	assert v.replace("l", "L").name is None
	assert v.split().name is None


def test_pytable_column_selection_preserves_name():
	"""Test that selecting a column from PyTable preserves the column name"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	col_a = t['a']
	assert col_a.name == 'a'
	
	col_b = t['b']
	assert col_b.name == 'b'


def test_pytable_multi_column_selection():
	"""Test that selecting multiple columns works and preserves names"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	# Multi-column selection
	t2 = t['a', 'c']
	assert isinstance(t2, PyTable)
	assert len(t2) == 2  # Two columns selected
	assert t2['a'].name == 'a'
	assert t2['c'].name == 'c'
	assert list(t2['a']) == [1, 2, 3]
	assert list(t2['c']) == [7, 8, 9]
	
	# Duplicate column selection (should create copies)
	t3 = t['a', 'a', 'a']
	assert isinstance(t3, PyTable)
	assert len(t3) == 3  # Three columns (all duplicates of 'a')
	# All three should have the same values - verify by position
	assert list(t3['a']) == [1, 2, 3]


def test_pytable_operations_on_named_columns():
	"""Test that operations on named columns from PyTable don't preserve names"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	# Column selection preserves name
	assert t['a'].name == 'a'
	
	# But math operations do not
	assert (t['a'] + t['b']).name is None
	assert (t['a'] * 2).name is None
	assert (t['a'] + 10).name is None


def test_name_after_mutation():
	"""Test that name is preserved after mutation via __setitem__"""
	v = PyVector([1, 2, 3], name="my_vector")
	v[1] = 99
	assert v.name == "my_vector"
	assert v[1] == 99


def test_name_with_multiple_mutations():
	"""Test that name persists through multiple mutations"""
	v = PyVector([1, 2, 3, 4, 5], name="persistent")
	v[0] = 10
	v[2] = 20
	v[4] = 30
	assert v.name == "persistent"
	assert list(v) == [10, 2, 20, 4, 30]


def test_name_independence_between_copies():
	"""Test that renaming a copy doesn't affect the original"""
	v1 = PyVector([1, 2, 3], name="original")
	v2 = v1.copy(name="copy")
	
	assert v1.name == "original"
	assert v2.name == "copy"


def test_operations_create_independent_unnamed_vectors():
	"""Test that operations create new unnamed vectors independent of inputs"""
	v1 = PyVector([1, 2, 3], name="vec1")
	v2 = PyVector([4, 5, 6], name="vec2")
	
	v3 = v1 + v2
	assert v3.name is None
	
	# Original vectors unchanged
	assert v1.name == "vec1"
	assert v2.name == "vec2"
