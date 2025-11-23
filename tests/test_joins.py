import pytest
from py_vector import PyVector
from py_vector import PyTable
from py_vector.errors import PyVectorTypeError


def test_inner_join_basic():
	"""Test basic inner join with single key"""
	left = PyTable({
		'id': [1, 2, 3],
		'name': ['Alice', 'Bob', 'Charlie']
	})
	right = PyTable({
		'id': [2, 3, 4],
		'age': [25, 30, 35]
	})
	
	# Single key join using string syntax
	result = left.inner_join(right, left_on='id', right_on='id')
	
	assert len(result) == 2
	assert list(result['name']) == ['Bob', 'Charlie']
	assert list(result['age']) == [25, 30]


def test_inner_join_multi_key():
	"""Test inner join with composite keys"""
	left = PyTable({
		'customer_id': [1, 1, 2],
		'date': ['2023-01-01', '2023-01-02', '2023-01-01'],
		'amount': [100, 200, 300]
	})
	right = PyTable({
		'customer_id': [1, 2, 1],
		'date': ['2023-01-01', '2023-01-01', '2023-01-03'],
		'status': ['active', 'active', 'inactive']
	})
	
	result = left.inner_join(
		right, 
		left_on=['customer_id', 'date'],
		right_on=['customer_id', 'date']
	)
	
	assert len(result) == 2
	assert list(result['amount']) == [100, 300]
	assert list(result['status']) == ['active', 'active']


def test_join_left():
	"""Test left join preserves all left rows"""
	left = PyTable({
		'id': [1, 2, 3],
		'name': ['Alice', 'Bob', 'Charlie']
	})
	right = PyTable({
		'id': [2, 4],
		'age': [25, 35]
	})
	
	result = left.join(right, left_on='id', right_on='id')
	
	assert len(result) == 3
	assert list(result['name']) == ['Alice', 'Bob', 'Charlie']
	assert list(result['age']) == [None, 25, None]


def test_full_join():
	"""Test full outer join includes all rows from both tables"""
	left = PyTable({
		'id': [1, 2],
		'left_val': ['A', 'B']
	})
	right = PyTable({
		'id': [2, 3],
		'right_val': ['X', 'Y']
	})
	
	result = left.full_join(right, left_on='id', right_on='id')
	
	assert len(result) == 3
	# Check that we have rows for id 1, 2, and 3
	assert None in list(result['left_val'])  # id=3 has no left match
	assert None in list(result['right_val'])  # id=1 has no right match


def test_many_to_one_cardinality():
	"""Test that many_to_one default catches duplicate right keys"""
	left = PyTable({
		'order_id': [1, 2, 3],
		'customer_id': [100, 100, 200]
	})
	right = PyTable({
		'customer_id': [100, 100, 200],  # Duplicate 100!
		'name': ['Alice', 'Alice2', 'Bob']
	})
	
	with pytest.raises(ValueError, match="many_to_one.*violated.*Right side has duplicate keys"):
		left.inner_join(right, left_on='customer_id', right_on='customer_id')


def test_one_to_one_cardinality():
	"""Test that one_to_one catches duplicates on both sides"""
	left = PyTable({
		'id': [1, 1, 2],  # Duplicate 1!
		'left_val': ['A', 'A2', 'B']
	})
	right = PyTable({
		'id': [1, 2],
		'right_val': ['X', 'Y']
	})
	
	with pytest.raises(ValueError, match="one_to_one.*violated.*Left side has duplicate key"):
		left.inner_join(right, left_on='id', right_on='id', expect='one_to_one')


def test_many_to_many_allows_duplicates():
	"""Test that many_to_many allows duplicates on both sides"""
	left = PyTable({
		'key': [1, 1, 2],
		'left_val': ['A', 'B', 'C']
	})
	right = PyTable({
		'key': [1, 1, 2],
		'right_val': ['X', 'Y', 'Z']
	})
	
	result = left.inner_join(right, left_on='key', right_on='key', expect='many_to_many')
	
	# 1 matches 1 twice = 4 combinations (A-X, A-Y, B-X, B-Y)
	# 2 matches 2 once = 1 combination (C-Z)
	assert len(result) == 5


def test_one_to_many_cardinality():
	"""Test that one_to_many allows multiple right matches per left"""
	left = PyTable({
		'id': [1, 2],
		'name': ['Alice', 'Bob']
	})
	right = PyTable({
		'id': [1, 1, 2],
		'order': ['Order1', 'Order2', 'Order3']
	})
	
	result = left.inner_join(right, left_on='id', right_on='id', expect='one_to_many')
	
	assert len(result) == 3
	assert list(result['name']) == ['Alice', 'Alice', 'Bob']
	assert list(result['order']) == ['Order1', 'Order2', 'Order3']


def test_join_empty_result():
	"""Test that join with no matches returns empty table"""
	left = PyTable({
		'id': [1, 2],
		'val': ['A', 'B']
	})
	right = PyTable({
		'id': [3, 4],
		'val': ['X', 'Y']
	})
	
	result = left.inner_join(right, left_on='id', right_on='id')
	
	assert len(result) == 0


def test_join_preserves_column_names():
	"""Test that column names are preserved after join"""
	left = PyTable({
		'customer_id': [1, 2],
		'order_total': [100, 200]
	})
	right = PyTable({
		'id': [1, 2],
		'customer_name': ['Alice', 'Bob']
	})
	
	result = left.inner_join(right, left_on='customer_id', right_on='id')
	
	# Check column names are accessible
	assert list(result.customer_id) == [1, 2]
	assert list(result.order_total) == [100, 200]
	assert list(result.customer_name) == ['Alice', 'Bob']


def test_left_join_with_multiple_right_matches():
	"""Test left join when right side has multiple matches (many_to_many)"""
	left = PyTable({
		'id': [1],
		'name': ['Alice']
	})
	right = PyTable({
		'id': [1, 1],
		'order': ['Order1', 'Order2']
	})
	
	result = left.join(right, left_on='id', right_on='id', expect='one_to_many')
	
	assert len(result) == 2
	assert list(result['name']) == ['Alice', 'Alice']
	assert list(result['order']) == ['Order1', 'Order2']


def test_full_join_no_matches():
	"""Test full join when no keys match"""
	left = PyTable({
		'id': [1, 2],
		'left_val': ['A', 'B']
	})
	right = PyTable({
		'id': [3, 4],
		'right_val': ['X', 'Y']
	})
	
	result = left.full_join(right, left_on='id', right_on='id')
	
	assert len(result) == 4
	# All left_val should have 2 values and 2 Nones
	assert list(result['left_val']).count(None) == 2
	# All right_val should have 2 values and 2 Nones
	assert list(result['right_val']).count(None) == 2


def test_join_different_column_names():
	"""Test join where left and right use different column names"""
	left = PyTable({
		'user_id': [1, 2, 3],
		'name': ['Alice', 'Bob', 'Charlie']
	})
	right = PyTable({
		'customer_id': [2, 3, 4],
		'purchases': [5, 10, 15]
	})
	
	result = left.inner_join(right, left_on='user_id', right_on='customer_id')
	
	assert len(result) == 2
	assert list(result['name']) == ['Bob', 'Charlie']
	assert list(result['purchases']) == [5, 10]


def test_join_with_pyvector_columns():
	"""Test join using PyVector columns directly instead of strings"""
	left = PyTable({'id': [1, 2]})
	right = PyTable({'id': [2, 3]})
	
	result = left.inner_join(right, left_on=left.id, right_on=right.id)
	
	assert len(result) == 1
	assert list(result['id']) == [2]


def test_join_multi_key_with_pyvectors():
	"""Test multi-key join using PyVector columns"""
	left = PyTable({
		'a': [1, 1, 2],
		'b': [10, 20, 10],
		'val': ['x', 'y', 'z']
	})
	right = PyTable({
		'a': [1, 2, 1],
		'b': [10, 10, 30],
		'data': ['p', 'q', 'r']
	})
	
	result = left.inner_join(right, left_on=[left.a, left.b], right_on=[right.a, right.b])
	
	assert len(result) == 2
	assert list(result['val']) == ['x', 'z']
	assert list(result['data']) == ['p', 'q']


def test_join_key_validation_length_mismatch():
	"""Test that mismatched left_on/right_on lengths raise error"""
	left = PyTable({'id': [1, 2], 'date': ['a', 'b']})
	right = PyTable({'id': [2, 3]})
	
	with pytest.raises(ValueError, match="same length"):
		left.inner_join(right, left_on=['id', 'date'], right_on=['id'])


def test_join_key_validation_missing_column():
	"""Test that non-existent column names raise error"""
	left = PyTable({'id': [1, 2]})
	right = PyTable({'id': [2, 3]})
    
	from py_vector.errors import PyVectorKeyError
	with pytest.raises(PyVectorKeyError):
		left.inner_join(right, left_on='missing_col', right_on='id')


def test_join_with_computed_keys():
	"""Test that float columns are rejected as join keys"""
	left = PyTable({
		'price': [100.0, 200.0, 300.0],
		'name': ['A', 'B', 'C']
	})
	right = PyTable({
		'price_with_tax': [108.0, 216.0, 324.0],
		'quantity': [1, 2, 3]
	})
	
	# Float keys should be rejected due to precision issues
	with pytest.raises(PyVectorTypeError, match="Invalid join key dtype 'float'"):
		left.inner_join(right, left_on=left['price'] * 1.08, right_on=right['price_with_tax'])


def test_join_with_constant_vector():
	"""Test joining with a constant PyVector (broadcast join pattern)"""
	left = PyTable({
		'id': [1, 2],
		'val': ['x', 'y']
	})
	right = PyTable({
		'flag': [1, 1, 1],
		'data': ['a', 'b', 'c']
	})
	
	# Create constant vector matching left table length
	constant_key = PyVector([1, 1])
	
	# This creates a cartesian-like product where every left row matches every right row
	result = left.inner_join(right, left_on=constant_key, right_on=right['flag'], expect='many_to_many')
	
	assert len(result) == 6  # 2 left rows * 3 right rows


def test_join_sanitized_column_name_lookup():
	"""Test that string lookup works with both exact and sanitized names"""
	left = PyTable({
		'Customer ID': [1, 2, 3],
		'Name': ['Alice', 'Bob', 'Charlie']
	})
	right = PyTable({
		'CUSTOMER_ID': [2, 3, 4],
		'Age': [25, 30, 35]
	})
	
	# Should find column using sanitized name (case-insensitive)
	result = left.inner_join(right, left_on='customer_id', right_on='customer_id')
	
	assert len(result) == 2
	assert list(result['Name']) == ['Bob', 'Charlie']


def test_join_key_wrong_length_left():
	"""Test that left_on PyVector with wrong length raises error"""
	left = PyTable({'id': [1, 2, 3]})
	right = PyTable({'id': [2, 3]})
	
	# Create PyVector with wrong length
	wrong_length_key = PyVector([1, 2])  # Length 2, but left table has 3 rows
	
	with pytest.raises(ValueError, match="Left join key.*has length 2.*left table has 3 rows"):
		left.inner_join(right, left_on=wrong_length_key, right_on=right.id)


def test_join_key_wrong_length_right():
	"""Test that right_on PyVector with wrong length raises error"""
	left = PyTable({'id': [1, 2]})
	right = PyTable({'id': [2, 3, 4]})
	
	# Create PyVector with wrong length
	wrong_length_key = PyVector([1, 2])  # Length 2, but right table has 3 rows
	
	with pytest.raises(ValueError, match="Right join key.*has length 2.*right table has 3 rows"):
		left.inner_join(right, left_on=left.id, right_on=wrong_length_key)


def test_join_multi_key_computed():
	"""Test multi-key join with mix of columns and computed values"""
	left = PyTable({
		'year': [2023, 2023, 2024],
		'month': [1, 2, 1],
		'amount': [100, 200, 300]
	})
	right = PyTable({
		'period_code': [202301, 202302, 202401],
		'budget': [150, 250, 350]
	})
	
	# Compute YYYYMM code from year and month
	left_period = left.year * 100 + left.month
	
	result = left.inner_join(right, left_on=left_period, right_on=right.period_code)
	
	assert len(result) == 3
	assert list(result['amount']) == [100, 200, 300]
	assert list(result['budget']) == [150, 250, 350]

