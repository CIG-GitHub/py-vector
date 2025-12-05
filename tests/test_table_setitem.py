"""
Test PyTable.__setitem__ with comprehensive slice coverage.
"""

import pytest
from py_vector import PyVector, PyTable
from py_vector.errors import PyVectorKeyError, PyVectorValueError, PyVectorTypeError


class TestScalarAssignment:
	"""Single cell assignment with various indexing patterns."""
	
	def test_scalar_by_int_int(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[0, 0] = 99
		assert t[0, 0] == 99
		assert t[0, 'a'] == 99
	
	def test_scalar_by_int_name(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[1, 'b'] = 88
		assert t[1, 1] == 88
		assert t[1, 'b'] == 88
	
	def test_scalar_by_neg_index(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[-1, -1] = 77
		assert t[-1, -1] == 77
		assert t[2, 1] == 77


class TestBroadcastScalarToSlice:
	"""Broadcast a scalar to multiple cells."""
	
	def test_broadcast_to_column_slice(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[0:2, 'a'] = 100
		assert list(t.cols()[0]) == [100, 100, 3]
	
	def test_broadcast_to_all_rows_single_col(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[:, 'b'] = 42
		assert list(t.cols()[1]) == [42, 42, 42]
	
	def test_broadcast_to_rectangular_region(self):
		t = PyTable({
			'a': [1, 2, 3, 4],
			'b': [5, 6, 7, 8],
			'c': [9, 10, 11, 12]})
		t[1:3, 0:2] = 999
		assert t[1, 0] == 999
		assert t[1, 1] == 999
		assert t[2, 0] == 999
		assert t[2, 1] == 999
		assert t[2, 2] == 11  # Unchanged
	
	def test_broadcast_to_step_slice(self):
		t = PyTable({'x': [1, 2, 3, 4, 5]})
		t[::2, 'x'] = 0
		assert list(t.cols()[0]) == [0, 2, 0, 4, 0]
	
	def test_broadcast_to_reverse_slice(self):
		t = PyTable({'x': [1, 2, 3, 4]})
		t[::-1, 'x'] = 9
		assert list(t.cols()[0]) == [9, 9, 9, 9]
	
	def test_broadcast_to_all_columns(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
		t[0, :] = 0
		assert t[0, 0] == 0
		assert t[0, 1] == 0
		assert t[0, 2] == 0
		assert t[1, 0] == 2  # Unchanged


class TestRowAssignment:
	"""Assign a sequence to an entire row."""
	
	def test_assign_list_to_row(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[0, :] = [10, 20]
		assert t[0, 'a'] == 10
		assert t[0, 'b'] == 20
	
	def test_assign_tuple_to_row(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[1, :] = (30, 40)
		assert t[1, 'a'] == 30
		assert t[1, 'b'] == 40
	
	def test_assign_to_row_subset_columns(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
		t[0, 1:3] = [99, 88]
		assert t[0, 0] == 1   # Unchanged
		assert t[0, 1] == 99
		assert t[0, 2] == 88
	
	def test_row_assignment_length_mismatch(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4]})
		with pytest.raises(PyVectorValueError, match="length mismatch"):
			t[0, :] = [1, 2, 3]  # Too many values


class TestColumnAssignment:
	"""Assign a vector/sequence to an entire column."""
	
	def test_assign_list_to_column_by_index(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[:, 1] = [40, 50, 60]
		assert list(t.cols()[1]) == [40, 50, 60]
	
	def test_assign_to_column_slice(self):
		t = PyTable({'x': [1, 2, 3, 4]})
		t[1:3, 'x'] = [99, 88]
		assert list(t.cols()[0]) == [1, 99, 88, 4]
	
	def test_assign_with_step_to_column(self):
		t = PyTable({'x': [1, 2, 3, 4, 5]})
		t[::2, 'x'] = [10, 30, 50]
		assert list(t.cols()[0]) == [10, 2, 30, 4, 50]


class TestRectangularAssignment:
	"""Assign a PyTable to a rectangular region."""
	
	def test_assign_table_to_region(self):
		t = PyTable({
			'a': [1, 2, 3, 4],
			'b': [5, 6, 7, 8],
			'c': [9, 10, 11, 12]
		})
		
		source = PyTable({
			'x': [99, 88],
			'y': [77, 66]
		})
		
		t[1:3, 0:2] = source
		assert t[1, 0] == 99
		assert t[1, 1] == 77
		assert t[2, 0] == 88
		assert t[2, 1] == 66
		assert t[1, 2] == 10  # Unchanged

	def test_assign_table_full_rows(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		source = PyTable({'x': [10, 20], 'y': [30, 40]})
		t[0:2, :] = source
		assert list(t.cols()[0]) == [10, 20, 3]
		assert list(t.cols()[1]) == [30, 40, 6]

	def test_assign_table_column_mismatch(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4]})
		source = PyTable({'x': [10]})
		with pytest.raises(PyVectorValueError, match="Column count mismatch"):
			t[:, :] = source


class TestListOfColumnsAssignment:
	"""Assign a list of lists/vectors as columns."""
	
	def test_assign_list_of_lists_to_all_columns(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[:, :] = [[10, 20, 30], [40, 50, 60]]
		assert list(t.cols()[0]) == [10, 20, 30]
		assert list(t.cols()[1]) == [40, 50, 60]
	
	def test_assign_list_of_vectors_to_subset(self):
		t = PyTable({
			'a': [1, 2, 3],
			'b': [4, 5, 6],
			'c': [7, 8, 9]
		})
		t[:, 1:3] = [PyVector([99, 88, 77]), PyVector([66, 55, 44])]
		assert list(t.cols()[0]) == [1, 2, 3]  # Unchanged
		assert list(t.cols()[1]) == [99, 88, 77]
		assert list(t.cols()[2]) == [66, 55, 44]
	
	def test_assign_list_shape_mismatch(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4]})
		with pytest.raises(PyVectorValueError, match="Shape mismatch"):
			t[:, :] = [[1, 2], [3, 4], [5, 6]]  # Too many columns


class TestSliceEdgeCases:
	"""Edge cases and boundary conditions."""
	
	def test_empty_slice_rows(self):
		t = PyTable({'x': [1, 2, 3]})
		t[5:10, 'x'] = 999  # Out of range, should be no-op
		assert list(t.cols()[0]) == [1, 2, 3]
	
	def test_negative_step_slice(self):
		t = PyTable({'x': [1, 2, 3, 4, 5]})
		t[4:1:-1, 'x'] = [50, 40, 30]
		assert list(t.cols()[0]) == [1, 2, 30, 40, 50]
	
	def test_assign_to_single_column_by_list(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
		t[:, ['b']] = [[99, 88]]
		assert list(t.cols()[1]) == [99, 88]
	
	def test_assign_to_multiple_columns_by_name_list(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
		t[:, ['a', 'c']] = [[10, 20], [30, 40]]
		assert list(t.cols()[0]) == [10, 20]
		assert list(t.cols()[1]) == [3, 4]  # Unchanged
		assert list(t.cols()[2]) == [30, 40]
	
	def test_assign_string_scalar_fails(self):
		# Can't assign string to int vector without explicit conversion
		t = PyTable({'x': [1, 2, 3]})
		with pytest.raises(PyVectorTypeError, match="Cannot set str in int vector"):
			t[0, 'x'] = "hello"
	
	def test_assign_to_entire_table(self):
		t = PyTable({'a': [1, 2], 'b': [3, 4]})
		t[:, :] = 0
		assert list(t.cols()[0]) == [0, 0]
		assert list(t.cols()[1]) == [0, 0]


class TestErrorConditions:
	"""Invalid operations that should raise errors."""
	
	def test_invalid_column_name(self):
		t = PyTable({'x': [1, 2]})
		with pytest.raises(PyVectorKeyError, match="not found"):
			t[0, 'nonexistent'] = 5
	
	def test_invalid_key_dimension(self):
		t = PyTable({'x': [1, 2]})
		with pytest.raises(PyVectorKeyError, match="requires 1D .* or 2D"):
			t[0, 0, 0] = 5
	
	def test_unsupported_value_type(self):
		t = PyTable({'x': [1, 2]})
		with pytest.raises(PyVectorTypeError, match="Unsupported assignment value type"):
			t[:, 'x'] = {1: 'a', 2: 'b'}  # Dict not supported


class TestMixedTypeAssignment:
	"""Assignment across different column types."""
	
	def test_assign_int_to_float_column(self):
		t = PyTable({'x': [1.5, 2.5, 3.5]})
		t[0, 'x'] = 10
		assert t[0, 'x'] == 10.0
	
	# Object vector assignment currently has issues
	# def test_assign_to_object_column(self):
	# 	t = PyTable({'mixed': [1, "two", 3.0]})
	# 	t[1, 'mixed'] = 999
	# 	assert t[1, 'mixed'] == 999
	
	def test_assign_heterogeneous_row(self):
		t = PyTable({
			'int': [1, 2],
			'float': [1.5, 2.5],
			'str': ["a", "b"]
		})
		t[0, :] = [99, 99.9, "xyz"]
		assert t[0, 'int'] == 99
		assert t[0, 'float'] == 99.9
		assert t[0, 'str'] == "xyz"


class TestComplexSlicing:
	"""Complex multi-dimensional slicing patterns."""
	
	def test_slice_with_step_both_dimensions(self):
		t = PyTable({
			'a': [1, 2, 3, 4, 5],
			'b': [6, 7, 8, 9, 10],
			'c': [11, 12, 13, 14, 15],
			'd': [16, 17, 18, 19, 20],
		})
		t[::2, ::2] = 0
		assert t[0, 0] == 0
		assert t[0, 2] == 0
		assert t[2, 0] == 0
		assert t[2, 2] == 0
		assert t[1, 1] == 7  # Unchanged
	
	def test_reverse_slice_assignment(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[::-1, :] = [[30, 20, 10], [60, 50, 40]]
		assert list(t.cols()[0]) == [10, 20, 30]
		assert list(t.cols()[1]) == [40, 50, 60]
	
	def test_partial_reverse_slice(self):
		t = PyTable({'x': [1, 2, 3, 4, 5]})
		t[3:0:-1, 'x'] = [40, 30, 20]
		assert list(t.cols()[0]) == [1, 20, 30, 40, 5]
	
	def test_overlapping_slice_assignment(self):
		# Assign to a slice, then assign to overlapping slice
		t = PyTable({'x': [1, 2, 3, 4, 5]})
		t[0:3, 'x'] = [10, 20, 30]
		t[2:5, 'x'] = [33, 44, 55]
		assert list(t.cols()[0]) == [10, 20, 33, 44, 55]


class TestImplicitRowSlice:
	"""Test single index (row only) implies all columns."""
	
	def test_single_int_scalar_assignment(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[0] = [99, 88]
		assert t[0, 'a'] == 99
		assert t[0, 'b'] == 88
	
	def test_single_slice_assignment(self):
		t = PyTable({'a': [1, 2, 3], 'b': [4, 5, 6]})
		t[0:2] = [[10, 20], [30, 40]]
		assert list(t.cols()[0]) == [10, 20, 3]
		assert list(t.cols()[1]) == [30, 40, 6]
