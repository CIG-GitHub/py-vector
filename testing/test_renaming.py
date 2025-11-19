import pytest
from py_vector import PyVector
from py_table import PyTable


def test_pyvector_rename():
	"""Test that PyVector.rename() changes the name"""
	v = PyVector([1, 2, 3], name="old_name")
	assert v._name == "old_name"
	
	result = v.rename("new_name")
	assert v._name == "new_name"
	assert result is v  # Returns self for chaining


def test_pyvector_rename_chaining():
	"""Test that rename returns self for method chaining"""
	v = PyVector([1, 2, 3], name="original")
	v2 = v.rename("renamed").copy()
	
	assert v._name == "renamed"
	assert v2._name == "renamed"


def test_pyvector_rename_to_none():
	"""Test that we can clear a name with rename(None)"""
	v = PyVector([1, 2, 3], name="has_name")
	v.rename(None)
	assert v._name is None


def test_pytable_rename_column():
	"""Test renaming a single column in PyTable"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	assert t['a']._name == 'a'
	
	result = t.rename_column('a', 'alpha')
	
	assert t['alpha']._name == 'alpha'
	assert result is t  # Returns self for chaining
	
	# Old name should not work
	with pytest.raises(KeyError):
		t['a']


def test_pytable_rename_column_not_found():
	"""Test that renaming a non-existent column raises KeyError"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	with pytest.raises(KeyError, match="Column 'z' not found"):
		t.rename_column('z', 'zeta')


def test_pytable_rename_columns_dict():
	"""Test renaming multiple columns at once"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	result = t.rename_columns(['a', 'b', 'c'], ['alpha', 'beta', 'gamma'])
	
	assert t['alpha']._name == 'alpha'
	assert t['beta']._name == 'beta'
	assert t['gamma']._name == 'gamma'
	assert result is t  # Returns self for chaining


def test_pytable_rename_columns_partial():
	"""Test renaming only some columns"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	t.rename_columns(['a', 'c'], ['alpha', 'gamma'])
	
	assert t['alpha']._name == 'alpha'
	assert t['b']._name == 'b'  # Unchanged
	assert t['gamma']._name == 'gamma'


def test_pytable_rename_columns_not_found():
	"""Test that renaming a non-existent column raises KeyError"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	with pytest.raises(KeyError, match="Column 'z' not found"):
		t.rename_columns(['a', 'z'], ['alpha', 'zeta'])


def test_pytable_rename_columns_atomic():
	"""Test that rename_columns is atomic - either all succeed or none"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6],
		'c': [7, 8, 9]
	})
	
	# Try to rename with one invalid column
	with pytest.raises(KeyError, match="Column 'invalid' not found"):
		t.rename_columns(['a', 'invalid', 'b'], ['alpha', 'oops', 'beta'])
	
	# No changes should have been made - 'a' should NOT be renamed to 'alpha'
	assert t._underlying[0]._name == 'a'
	assert t._underlying[1]._name == 'b'
	assert t._underlying[2]._name == 'c'
	
	# Should still be accessible by original names
	assert list(t['a']) == [1, 2, 3]
	assert list(t['b']) == [4, 5, 6]


def test_pytable_rename_columns_chaining():
	"""Test that rename_columns returns self for chaining"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	# Chain multiple operations
	t.rename_columns(['a'], ['alpha']).rename_column('b', 'beta')
	
	assert t['alpha']._name == 'alpha'
	assert t['beta']._name == 'beta'


def test_pytable_duplicate_column_names():
	"""Test the horrible real-world condition: duplicate column names"""
	# Create table with duplicate column names
	col1 = PyVector([1, 2, 3], name='a')
	col2 = PyVector([4, 5, 6], name='a')
	col3 = PyVector([7, 8, 9], name='b')
	t = PyTable([col1, col2, col3])
	
	# Should have two columns named 'a' and one named 'b'
	assert t._underlying[0]._name == 'a'
	assert t._underlying[1]._name == 'a'
	assert t._underlying[2]._name == 'b'
	
	# Accessing t['a'] should return the first match
	assert list(t['a']) == [1, 2, 3]
	
	# Renaming one 'a' should only rename the first match
	t.rename_column('a', 'alpha')
	assert t._underlying[0]._name == 'alpha'
	assert t._underlying[1]._name == 'a'  # Second 'a' unchanged
	assert t._underlying[2]._name == 'b'
	
	# Now we have alpha, a, b
	assert list(t['alpha']) == [1, 2, 3]
	assert list(t['a']) == [4, 5, 6]  # Gets the remaining 'a'


def test_pytable_rename_all_duplicates():
	"""Test renaming ALL columns with duplicate names"""
	# Create table with duplicate column names
	col1 = PyVector([1, 2, 3], name='a')
	col2 = PyVector([4, 5, 6], name='a')
	col3 = PyVector([7, 8, 9], name='a')
	t = PyTable([col1, col2, col3])
	
	# All three columns named 'a'
	assert all(c._name == 'a' for c in t._underlying)
	
	# rename_columns with parallel lists renames each match in order
	t.rename_columns(['a', 'a', 'a'], ['x', 'y', 'z'])
	
	# Each 'a' renamed in order
	assert t._underlying[0]._name == 'x'
	assert t._underlying[1]._name == 'y'
	assert t._underlying[2]._name == 'z'


def test_pytable_rename_duplicate_columns_separately():
	"""Test renaming duplicate columns to different names using parallel sequences"""
	# Create table with duplicate column names
	col1 = PyVector([1, 2, 3], name='data')
	col2 = PyVector([4, 5, 6], name='data')
	col3 = PyVector([7, 8, 9], name='label')
	t = PyTable([col1, col2, col3])
	
	# We want to rename both 'data' columns to different names
	# This is where parallel lists shine
	t.rename_columns(['data', 'data'], ['measurement', 'control'])
	
	assert t._underlying[0]._name == 'measurement'
	assert t._underlying[1]._name == 'control'
	assert t._underlying[2]._name == 'label'
	
	# Verify data preserved
	assert list(t['measurement']) == [1, 2, 3]
	assert list(t['control']) == [4, 5, 6]
	assert list(t['label']) == [7, 8, 9]


def test_pytable_getattr_after_rename():
	"""Test that __getattr__ works after renaming"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	# Should work before rename
	assert list(t.a) == [1, 2, 3]
	
	t.rename_column('a', 'alpha')
	
	# Old attribute should return None (column name doesn't match anymore)
	assert t.a is None
	
	# New attribute should work
	assert list(t.alpha) == [1, 2, 3]


def test_rename_preserves_data():
	"""Test that renaming doesn't affect the data"""
	v = PyVector([1, 2, 3], name="old")
	original_data = list(v)
	
	v.rename("new")
	
	assert list(v) == original_data
	assert v._name == "new"


def test_pytable_rename_preserves_data():
	"""Test that renaming columns doesn't affect the data"""
	t = PyTable({
		'a': [1, 2, 3],
		'b': [4, 5, 6]
	})
	
	original_a = list(t['a'])
	original_b = list(t['b'])
	
	t.rename_columns(['a', 'b'], ['x', 'y'])
	
	assert list(t['x']) == original_a
	assert list(t['y']) == original_b
