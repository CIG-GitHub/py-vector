import pytest
from py_vector import _sanitize_name, PyVector, PyTable


def test_sanitize_simple_name():
	"""Test that simple valid names are lowercased"""
	assert _sanitize_name("column") == "column"
	assert _sanitize_name("col_1") == "col_1"
	assert _sanitize_name("MyColumn") == "mycolumn"


def test_sanitize_spaces():
	"""Test that spaces are replaced with underscores"""
	assert _sanitize_name("first name") == "first_name"
	assert _sanitize_name("a b c") == "a_b_c"


def test_sanitize_special_chars():
	"""Test that special characters are replaced with underscores"""
	assert _sanitize_name("name-with-dashes") == "name_with_dashes"
	assert _sanitize_name("price$") == "price"
	assert _sanitize_name("percent%") == "percent"
	assert _sanitize_name("name@domain") == "name_domain"
	assert _sanitize_name("col.with.dots") == "col_with_dots"


def test_sanitize_collapse_underscores():
	"""Test that user underscores are preserved (not collapsed)"""
	assert _sanitize_name("a__b") == "a__b"
	assert _sanitize_name("a___b") == "a___b"
	assert _sanitize_name("multiple____underscores") == "multiple____underscores"


def test_sanitize_strip_underscores():
	"""Test that leading/trailing underscores are stripped"""
	assert _sanitize_name("_leading") == "leading"
	assert _sanitize_name("trailing_") == "trailing"
	assert _sanitize_name("_both_") == "both"
	assert _sanitize_name("__multiple__") == "multiple"


def test_sanitize_starts_with_digit():
	"""Test that names starting with digits get prefixed with underscore"""
	assert _sanitize_name("123") == "_123"
	assert _sanitize_name("2nd_column") == "_2nd_column"
	assert _sanitize_name("99problems") == "_99problems"


def test_sanitize_empty_result():
	"""Test that names that become empty return 'col'"""
	assert _sanitize_name("") == "col"
	assert _sanitize_name("___") == "col"
	assert _sanitize_name("$$$") == "col"
	assert _sanitize_name("@#$%") == "col"


def test_sanitize_unicode():
	"""Test that unicode characters are handled by isalnum()"""
	assert _sanitize_name("naïve") == "naïve"  # ï is alphanumeric in unicode
	assert _sanitize_name("café") == "café"  # é is alphanumeric in unicode
	assert _sanitize_name("αβγ") == "αβγ"  # Greek letters are alphanumeric


def test_sanitize_mixed_complexity():
	"""Test complex real-world scenarios"""
	assert _sanitize_name("User ID") == "user_id"
	assert _sanitize_name("2023-Revenue ($)") == "_2023_revenue"
	assert _sanitize_name("__private__var__") == "private__var"
	assert _sanitize_name("column.1.data") == "column_1_data"


def test_sanitize_non_string_input():
	"""Test that non-string inputs are converted to strings first"""
	assert _sanitize_name(123) == "_123"
	assert _sanitize_name(45.67) == "_45_67"
	assert _sanitize_name(None) == "none"


def test_sanitize_preserves_valid_python_identifiers():
	"""Test that identifiers are lowercased and sanitized"""
	assert _sanitize_name("valid_identifier") == "valid_identifier"
	assert _sanitize_name("_private") == "private"  # Leading _ stripped
	assert _sanitize_name("CamelCase") == "camelcase"
	assert _sanitize_name("snake_case_name") == "snake_case_name"


def test_sanitize_csv_headers():
	"""Test sanitization of typical messy CSV column names"""
	assert _sanitize_name("First Name") == "first_name"
	assert _sanitize_name("Email Address (Primary)") == "email_address_primary"
	assert _sanitize_name("Price ($USD)") == "price_usd"
	assert _sanitize_name("Q1 2023 Revenue") == "q1_2023_revenue"



def test_table_getattr_with_spaces():
	"""Test that column names with spaces work via sanitized attribute access"""
	t = PyTable({
		'first name': [1, 2, 3],
		'last name': [4, 5, 6]
	})
	
	# Original names work with brackets
	assert list(t['first name']) == [1, 2, 3]
	assert list(t['last name']) == [4, 5, 6]
	
	# Sanitized names work with attribute access
	assert list(t.first_name) == [1, 2, 3]
	assert list(t.last_name) == [4, 5, 6]


def test_table_getattr_with_special_chars():
	"""Test that column names with special characters work via sanitized attributes"""
	t = PyTable({
		'price ($)': [10, 20, 30],
		'count@time': [1, 2, 3],
		'col.with.dots': [4, 5, 6]
	})
	
	# Original names work with brackets
	assert list(t['price ($)']) == [10, 20, 30]
	
	# Sanitized names work with attributes
	assert list(t.price) == [10, 20, 30]
	assert list(t.count_time) == [1, 2, 3]
	assert list(t.col_with_dots) == [4, 5, 6]


def test_table_getitem_sanitized():
	"""Test that __getitem__ accepts both original and sanitized names"""
	t = PyTable({
		'first name': [1, 2, 3],
		'price ($)': [10, 20, 30]
	})
	
	# Original names
	assert list(t['first name']) == [1, 2, 3]
	assert list(t['price ($)']) == [10, 20, 30]
	
	# Sanitized names also work
	assert list(t['first_name']) == [1, 2, 3]
	assert list(t['price']) == [10, 20, 30]


def test_table_dir_sanitized():
	"""Test that __dir__ returns sanitized names for tab completion"""
	t = PyTable({
		'first name': [1, 2, 3],
		'price ($)': [10, 20, 30],
		'count@time': [4, 5, 6]
	})
	
	dir_names = t.__dir__()
	
	# Sanitized names should appear
	assert 'first_name' in dir_names
	assert 'price' in dir_names
	assert 'count_time' in dir_names
	
	# Original unsanitized names should NOT appear
	assert 'first name' not in dir_names
	assert 'price ($)' not in dir_names


def test_table_unnamed_columns():
	"""Test that unnamed columns get col_1, col_2, etc."""
	col1 = PyVector([1, 2, 3])
	col2 = PyVector([4, 5, 6])
	col3 = PyVector([7, 8, 9])
	t = PyTable([col1, col2, col3])
	
	# Attribute access with col_N
	assert list(t.col_1) == [1, 2, 3]
	assert list(t.col_2) == [4, 5, 6]
	assert list(t.col_3) == [7, 8, 9]
	
	# __dir__ should show col_1, col_2, col_3
	dir_names = t.__dir__()
	assert 'col_1' in dir_names
	assert 'col_2' in dir_names
	assert 'col_3' in dir_names


def test_table_mixed_named_unnamed():
	"""Test mix of named and unnamed columns"""
	col1 = PyVector([1, 2, 3], name='alpha')
	col2 = PyVector([4, 5, 6])  # No name
	col3 = PyVector([7, 8, 9], name='gamma')
	t = PyTable([col1, col2, col3])
	
	# Named columns work
	assert list(t.alpha) == [1, 2, 3]
	assert list(t.gamma) == [7, 8, 9]
	
	# Unnamed column accessible as col_2
	assert list(t.col_2) == [4, 5, 6]


def test_table_getattr_starts_with_digit():
	"""Test that column names starting with digits get prefixed with _"""
	t = PyTable({
		'2023 Revenue': [100, 200, 300],
		'1st Place': [1, 2, 3]
	})
	
	# Sanitized names with leading underscore (lowercase)
	assert list(t._2023_revenue) == [100, 200, 300]
	assert list(t._1st_place) == [1, 2, 3]


def test_table_getitem_priority():
	"""Test that exact match takes priority over sanitized match"""
	# Create a table where sanitized name might conflict
	col1 = PyVector([1, 2, 3], name='first_name')
	col2 = PyVector([4, 5, 6], name='first name')  # Sanitizes to same thing
	t = PyTable([col1, col2])
	
	# Exact matches should work
	assert list(t['first_name']) == [1, 2, 3]
	assert list(t['first name']) == [4, 5, 6]
	
	# Attribute access gets first match (first_name is exact)
	assert list(t.first_name) == [1, 2, 3]


def test_table_empty_column_name():
	"""Test that empty/special-only column names still accessible"""
	t = PyTable({
		'': [1, 2, 3],
		'   ': [4, 5, 6],  # All spaces
		'$$$': [7, 8, 9]   # All special chars
	})
	
	# Original names work with exact match in __getitem__
	assert list(t['']) == [1, 2, 3]
	assert list(t['   ']) == [4, 5, 6]
	assert list(t['$$$']) == [7, 8, 9]
	
	# All three sanitize to 'col', so attribute access gets first match
	assert list(t.col) == [1, 2, 3]


def test_sanitization_preserves_camelcase():
	"""Test that CamelCase and other valid identifiers work"""
	t = PyTable({
		'CamelCase': [1, 2, 3],
		'snake_case': [4, 5, 6],
		'UPPERCASE': [7, 8, 9]
	})
	
	assert list(t.camelcase) == [1, 2, 3]  # case-insensitive
	assert list(t.snake_case) == [4, 5, 6]
	assert list(t.uppercase) == [7, 8, 9]  # case-insensitive
