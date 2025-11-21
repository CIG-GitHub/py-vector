import warnings
from .vector import PyVector
from .naming import _sanitize_user_name, _uniquify
from .errors import PyVectorKeyError, PyVectorValueError


def _missing_col_error(name, context="PyTable"):
	return PyVectorKeyError(f"Column '{name}' not found in {context}")


class _RowView:
	"""Lightweight row view for iterating over table rows with attribute access."""
	__slots__ = ('_cols', '_column_map', '_index')
	
	def __init__(self, table, index):
		# Cache direct handles to underlying data (bypasses PyVector method dispatch)
		self._cols = [col._underlying for col in table._underlying]
		self._column_map = table._column_map
		self._index = index
	
	def set_index(self, index):
		"""Reuse this row view for a different index (avoids allocation during iteration)."""
		self._index = index
		return self
	
	def __getattr__(self, attr):
		"""Access column values by sanitized attribute name."""
		col_idx = self._column_map.get(attr.lower())
		if col_idx is None:
			raise AttributeError(f"Row has no attribute '{attr}'")
		return self._cols[col_idx][self._index]
	
	def __getitem__(self, key):
		"""Access column values by index or name (optimized for int hot path)."""
		# Fast path: integer indexing (no isinstance check overhead)
		try:
			return self._cols[key][self._index]
		except TypeError:
			# Fallback: string indexing
			if isinstance(key, str):
				return getattr(self, key)
			raise TypeError(f"Row indices must be int or str, not {type(key).__name__}")
	
	def __iter__(self):
		"""Iterate over column values in this row."""
		idx = self._index
		for col in self._cols:
			yield col[idx]
	
	def __len__(self):
		"""Return number of columns."""
		return len(self._cols)
	
	def __repr__(self):
		"""Return a simple representation of the row."""
		idx = self._index
		values = [repr(col[idx]) for col in self._cols]
		return f"Row({idx}: {', '.join(values)})"


class PyTable(PyVector):
	""" Multiple columns of the same length """
	_length = None
	
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		# Handle dict initialization {name: values, ...}
		if isinstance(initial, dict):
			# Create PyVectors with names from dict keys
			initial = [PyVector(values, name=col_name) for col_name, values in initial.items()]
		
		self._length = len(initial[0]) if initial else 0
		
		# Deep copy columns to enforce value semantics
		# Tables receive snapshots of vectors, preventing aliasing
		# Save original names BEFORE copying
		original_names = [vec._name for vec in initial] if initial else []
		
		# Make copies of the vectors
		if initial:
			initial = tuple(vec.copy() for vec in initial)
		else:
			initial = ()
		
		# Call parent constructor
		super().__init__(
			initial,
			default_element=default_element,
			dtype=dtype,
			typesafe=typesafe,
			name=name)
		
		# CRITICAL: Restore column names after parent init
		# The parent PyVector.__init__ may have modified self._underlying
		if original_names:
			for i, col_name in enumerate(original_names):
				if i < len(self._underlying):
					self._underlying[i]._name = col_name
		
		# Build column map once for fast row iteration
		self._column_map = self._build_column_map()

	def __len__(self):
		if not self:
			return 0
		if isinstance(self._underlying[0], PyTable):
			return len(self._underlying)
		return self._length

	def size(self):
		return (len(self),) + self[0].size()

	def _build_column_map(self):
		"""Build mapping from sanitized column names to column indices.
		
		This is computed once during table initialization and used by
		_RowView for O(1) attribute lookups during iteration.
		"""
		column_map = {}
		seen = set()
		for idx, col in enumerate(self._underlying):
			if col._name is not None:
				base = _sanitize_user_name(col._name)
				if base is None:
					# Empty after sanitization, use system name
					sanitized = f'col{idx}_'
				else:
					sanitized = _uniquify(base, seen)
					seen.add(sanitized)
			else:
				# Unnamed column, use system name
				sanitized = f'col{idx}_'
			column_map[sanitized] = idx
		return column_map
	
	def __dir__(self):
		"""Return list of available attributes including sanitized column names."""
		# Use object.__dir__ to get instance attributes, then add column names
		base_attrs = object.__dir__(self)
		return sorted(set(base_attrs + list(self._column_map.keys())))

	def __getattr__(self, attr):
		"""Access columns by sanitized attribute name using pre-computed column map."""
		col_idx = self._column_map.get(attr.lower())
		if col_idx is not None:
			return self._underlying[col_idx]
		
		# Attribute not found - raise AttributeError for Pythonic behavior
		raise AttributeError(f"{self.__class__.__name__!s} object has no attribute '{attr}'")

	def rename_column(self, old_name, new_name):
		"""Rename a column (modifies in place, returns self for chaining)"""
		for col in self._underlying:
			if col._name == old_name:
				col._name = new_name
				# Rebuild column map to reflect new name
				self._column_map = self._build_column_map()
				return self
		raise _missing_col_error(old_name)
	
	def rename_columns(self, old_names, new_names):
		"""
		Atomically rename multiple columns using parallel old_names and new_names lists.

		Rules:
		- old_names and new_names must be same length
		- each list-element renames EXACTLY ONE matching occurrence
		(left-to-right positional matching)
		- if renaming fails (old name not found), no columns are renamed and KeyError is raised
		"""

		if len(old_names) != len(new_names):
			raise PyVectorValueError("old_names and new_names must have the same length")

		# Simulate renames using a temporary list (avoid mid-state partial renames)
		simulated = [col._name for col in self._underlying]

		for old, new in zip(old_names, new_names):
			try:
				idx = simulated.index(old)
			except ValueError:
				raise _missing_col_error(old)
			simulated[idx] = new  # simulate rename

		# Apply renames for real
		for old, new in zip(old_names, new_names):
			# rename the FIRST matching column in the real table
			for col in self._underlying:
				if col._name == old:
					col._name = new
					break
		
		# Rebuild column map to reflect new names
		self._column_map = self._build_column_map()
		return self

	@property
	def T(self):
		if len(self.size())==2:
			# Transpose 2D table: columns become rows
			num_rows = self._length
			num_cols = len(self._underlying)
			rows = []
			for row_idx in range(num_rows):
				row = PyVector([col[row_idx] for col in self._underlying])
				rows.append(row)
			return PyTable(rows)
		return self.copy((tuple(x.T for x in self))) # higher dimensions

	def __getitem__(self, key):
		key = self._check_duplicate(key)
		
		# Handle string indexing for column names
		if isinstance(key, str):
			# Try exact match first
			for col in self._underlying:
				if col._name == key:
					return col
			
			# Try sanitized match (case-insensitive)
			key_lower = key.lower()
			seen = set()
			for idx, col in enumerate(self._underlying):
				if col._name is not None:
					base = _sanitize_user_name(col._name)
					# If sanitization returns None, match system name
					if base is None:
						if f'col{idx}_' == key_lower:
							return col
					else:
						unique_name = _uniquify(base, seen)
						seen.add(unique_name)
						if unique_name == key_lower:
							return col
				else:
					# Unnamed columns: match col{idx}_ pattern
					if f'col{idx}_' == key_lower:
						return col
			
			raise _missing_col_error(key)
		
		# Handle tuple of strings for multi-column selection
		if isinstance(key, tuple) and all(isinstance(k, str) for k in key):
			# Multiple column selection by names
			selected_cols = []
			for col_name in key:
				found = False
				# Try exact match first
				for col in self._underlying:
					if col._name == col_name:
						selected_cols.append(col.copy())  # Copy to preserve original
						found = True
						break
				
				# Try sanitized match (case-insensitive)
				if not found:
					col_name_lower = col_name.lower()
					seen = set()
					for idx, col in enumerate(self._underlying):
						if col._name is not None:
							base = _sanitize_user_name(col._name)
							if base is None:
								if f'col{idx}_' == col_name_lower:
									selected_cols.append(col.copy())
									found = True
									break
							else:
								unique_name = _uniquify(base, seen)
								seen.add(unique_name)
								if unique_name == col_name_lower:
									selected_cols.append(col.copy())
									found = True
									break
						else:
							if f'col{idx}_' == col_name_lower:
								selected_cols.append(col.copy())
								found = True
								break

								if not found:
									raise _missing_col_error(col_name)
			return PyTable(selected_cols)
		
		if isinstance(key, tuple):
			if len(key) != len(self.size()):
				raise PyVectorKeyError(f"Matrix indexing must provide an index in each dimension: {self.size()}")
			# for now.
			if len(key) > 2:
				return self[key[0]][key[1:]]

			if isinstance(key[0], slice):
				if isinstance(key[1], int):
					return self.cols(key[1])[key[0]]
				return self.copy([col[key[0]] for col in self.cols()[key[1]]])
			return self[key[0]]._underlying[key[1]]

		if isinstance(key, int):
			# Effectively a different input type (single not a list). Returning a value, not a vector.
			if isinstance(self._underlying[0], PyTable):
				return self._underlying[key]
			return PyVector(tuple(col[key] for col in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			).T

		if isinstance(key, PyVector) and key._dtype == bool and key._typesafe:
			assert (len(self) == len(key))
			return PyVector(tuple(x[key] for x in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			)
		if isinstance(key, list) and {type(e) for e in key} == {bool}:
			assert (len(self) == len(key))
			return PyVector(tuple(x[key] for x in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			)
		if isinstance(key, slice):
			return PyVector(tuple(x[key] for x in self._underlying), 
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe,
				name=self._name
			)

		# NOT RECOMMENDED
		if isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors; prefer slices or boolean masks')
			return PyVector(tuple(x[key] for x in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			)

	def __iter__(self):
		"""Iterate over rows using a reusable _RowView for memory efficiency."""
		row_view = _RowView(self, 0)
		for i in range(len(self)):
			row_view.set_index(i)
			yield row_view

	def __repr__(self):
		from .display import _printr
		return _printr(self)

	def __matmul__(self, other):
		if isinstance(other, PyVector):
			# we want the sum to operate 'in place' on integers
			if other.ndims() == 1:
				return PyVector(tuple(sum(u*v for u, v in zip(s._underlying, other._underlying)) for s in self))
			return self.copy(tuple(self.copy(tuple(x@y for x in other.cols())) for y in self)).T
		return super().__matmul__(other)

	def _elementwise_compare(self, other, op):
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector(tuple(op(x, y) for x, y in zip(self.cols(), other.cols(), strict=True)), False, bool, True)
		if hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector(tuple(op(x, y) for x, y in zip(self, other, strict=True)), False, bool, True).T
		return PyVector(tuple(op(x, other) for x in self.cols()), False, bool, True)

	def __rshift__(self, other):
		""" The >> operator behavior has been overridden to add the column(s) of other to self
		"""
		if self._dtype in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden. Use .bitshift() to shift bits.")

		if isinstance(other, PyTable):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector(self.cols() + other.cols(),
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector(self.cols() + (other,),
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		elif not self:
			return PyVector((other,),
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)
		raise PyVectorTypeError("Cannot add a column of constant values. Try using PyVector.new(element, length).")

	def __lshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if isinstance(other, PyTable):
			return PyVector([x << y for x, y in zip(self._underlying, other._underlying, strict=True)])
		return PyVector([x << y for x, y in zip(self._underlying, other, strict=True)])

	def _validate_join_keys(self, other, left_on, right_on):
		"""
		Validate and normalize join key specification.
		
		Args:
			other: Right table to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
		
		Accepted forms:
		- Single column:  left_on="id", right_on="customer_id"
		- Multiple:       left_on=["id","date"], right_on=["cust_id","trans_date"]
		- PyVector:       left_on=table1.id, right_on=table2.customer_id
		
		Returns:
			List of (left_col, right_col) tuples (PyVector objects)
		
		Raises:
			ValueError: If parameters are invalid or mismatched
		"""
		# Helper to get column by name or return PyVector as-is
		def get_column(table, col_spec, side_name):
			if isinstance(col_spec, str):
				try:
					return table[col_spec]
				except (KeyError, ValueError):
					raise _missing_col_error(col_spec, context=f"{side_name} table")
			elif isinstance(col_spec, PyVector):
				return col_spec
			else:
				raise PyVectorValueError(
					f"Column specification must be string or PyVector, got {type(col_spec).__name__}"
				)
		
		# Normalize to lists
		if isinstance(left_on, (str, PyVector)):
			left_on = [left_on]
		if isinstance(right_on, (str, PyVector)):
			right_on = [right_on]
		
		if not isinstance(left_on, list) or not isinstance(right_on, list):
			raise PyVectorValueError("left_on and right_on must be strings, PyVectors, or lists")
		
		if len(left_on) == 0 or len(right_on) == 0:
			raise PyVectorValueError("Must specify at least 1 join key")
		
		if len(left_on) != len(right_on):
			raise PyVectorValueError(
				f"left_on and right_on must have same length: "
				f"got {len(left_on)} and {len(right_on)}"
			)
		
		# Build list of (left_col, right_col) tuples
		normalized = []
		for i, (left_spec, right_spec) in enumerate(zip(left_on, right_on)):
			left_col = get_column(self, left_spec, "Left")
			right_col = get_column(other, right_spec, "Right")
			
			# Validate PyVector lengths match their respective tables
			if len(left_col) != len(self):
				raise ValueError(
					f"Left join key at index {i} has length {len(left_col)}, "
					f"but left table has {len(self)} rows"
				)
			if len(right_col) != len(other):
				raise ValueError(
					f"Right join key at index {i} has length {len(right_col)}, "
					f"but right table has {len(other)} rows"
				)
			
			normalized.append((left_col, right_col))
		
		return normalized

	def inner_join(self, other, left_on, right_on, expect='many_to_one'):
		"""
		Inner join two PyTables on specified key columns.
		Only returns rows where keys match in both tables.		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate and normalize join keys
		on = self._validate_join_keys(other, left_on, right_on)
		
		# Extract join key columns from left (self) and right (other)
		left_keys = [left_col for left_col, _ in on]
		right_keys = [right_col for _, right_col in on]
		
		# Build hash map: right_key_tuple -> list of right row indices
		right_index = {}
		for row_idx in range(len(other)):
			key = tuple(right_keys[i][row_idx] for i in range(len(right_keys)))
			if key not in right_index:
				right_index[key] = []
			right_index[key].append(row_idx)
		
		# Check cardinality expectations on right side
		if expect in ('one_to_one', 'many_to_one'):
			# Right side must have unique keys
			duplicates = {k: indices for k, indices in right_index.items() if len(indices) > 1}
			if duplicates:
				raise PyVectorValueError(
					f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
					f"Found {len(duplicates)} duplicate key(s), e.g., {list(duplicates.keys())[0]} "
					f"appears {len(list(duplicates.values())[0])} times."
				)
		
		# Track left side key uniqueness for one_to_many and one_to_one
		if expect in ('one_to_one', 'one_to_many'):
			left_keys_seen = set()
		
		# Perform join
		result_rows = []
		for left_idx in range(len(self)):
			key = tuple(left_keys[i][left_idx] for i in range(len(left_keys)))
			
			# Check one_to_many / one_to_one constraint
			if expect in ('one_to_one', 'one_to_many'):
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			# Find matching right rows
			if key in right_index:
				for right_idx in right_index[key]:
					# Combine left and right rows
					left_row = [col[left_idx] for col in self._underlying]
					right_row = [col[right_idx] for col in other._underlying]
					result_rows.append(left_row + right_row)
		
		if not result_rows:
			# Empty join result
			return PyTable([])
		
		# Transpose result_rows to get columns
		num_cols = len(self._underlying) + len(other._underlying)
		result_cols = []
		for col_idx in range(num_cols):
			col_data = [row[col_idx] for row in result_rows]
			
			# Preserve column names
			if col_idx < len(self._underlying):
				# Left table column
				orig_col = self._underlying[col_idx]
				result_cols.append(PyVector(col_data, name=orig_col._name))
			else:
				# Right table column
				orig_col = other._underlying[col_idx - len(self._underlying)]
				result_cols.append(PyVector(col_data, name=orig_col._name))
		
		return PyTable(result_cols)

	def join(self, other, left_on, right_on, expect='many_to_one'):
		"""
		Left join two PyTables on specified key columns.
		Returns all rows from left table, with matching rows from right (or None for no match).
		
		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate and normalize join keys
		on = self._validate_join_keys(other, left_on, right_on)
		
		# Extract join key columns from left (self) and right (other)
		left_keys = [left_col for left_col, _ in on]
		right_keys = [right_col for _, right_col in on]
		
		# Build hash map: right_key_tuple -> list of right row indices
		right_index = {}
		for row_idx in range(len(other)):
			key = tuple(right_keys[i][row_idx] for i in range(len(right_keys)))
			if key not in right_index:
				right_index[key] = []
			right_index[key].append(row_idx)
		
		# Check cardinality expectations on right side
		if expect in ('one_to_one', 'many_to_one'):
			# Right side must have unique keys
			duplicates = {k: indices for k, indices in right_index.items() if len(indices) > 1}
			if duplicates:
				raise PyVectorValueError(
					f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
					f"Found {len(duplicates)} duplicate key(s), e.g., {list(duplicates.keys())[0]} "
					f"appears {len(list(duplicates.values())[0])} times."
				)
		
		# Track left side key uniqueness for one_to_many and one_to_one
		if expect in ('one_to_one', 'one_to_many'):
			left_keys_seen = set()
		
		# Perform left join
		result_rows = []
		for left_idx in range(len(self)):
			key = tuple(left_keys[i][left_idx] for i in range(len(left_keys)))
			
			# Check one_to_many / one_to_one constraint
			if expect in ('one_to_one', 'one_to_many'):
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			# Find matching right rows
			if key in right_index:
				for right_idx in right_index[key]:
					# Combine left and right rows
					left_row = [col[left_idx] for col in self._underlying]
					right_row = [col[right_idx] for col in other._underlying]
					result_rows.append(left_row + right_row)
			else:
				# No match: left row with None for right columns
				left_row = [col[left_idx] for col in self._underlying]
				right_row = [None] * len(other._underlying)
				result_rows.append(left_row + right_row)
		
		if not result_rows:
			# Empty result
			return PyTable([])
		
		# Transpose result_rows to get columns
		num_cols = len(self._underlying) + len(other._underlying)
		result_cols = []
		for col_idx in range(num_cols):
			col_data = [row[col_idx] for row in result_rows]
			
			# Preserve column names
			if col_idx < len(self._underlying):
				# Left table column
				orig_col = self._underlying[col_idx]
				result_cols.append(PyVector(col_data, name=orig_col._name))
			else:
				# Right table column
				orig_col = other._underlying[col_idx - len(self._underlying)]
				result_cols.append(PyVector(col_data, name=orig_col._name))
		
		return PyTable(result_cols)

	def full_join(self, other, left_on, right_on, expect='many_to_many'):
		"""
		Full outer join two PyTables on specified key columns.
		Returns all rows from both tables, with None where no match exists.
		
		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate and normalize join keys
		on = self._validate_join_keys(other, left_on, right_on)
		
		# Extract join key columns from left (self) and right (other)
		left_keys = [left_col for left_col, _ in on]
		right_keys = [right_col for _, right_col in on]
		
		# Build hash maps for both sides
		right_index = {}
		for row_idx in range(len(other)):
			key = tuple(right_keys[i][row_idx] for i in range(len(right_keys)))
			if key not in right_index:
				right_index[key] = []
			right_index[key].append(row_idx)
		
		# Check cardinality expectations on right side
		if expect in ('one_to_one', 'many_to_one'):
			duplicates = {k: indices for k, indices in right_index.items() if len(indices) > 1}
			if duplicates:
				raise PyVectorValueError(
					f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
					f"Found {len(duplicates)} duplicate key(s), e.g., {list(duplicates.keys())[0]} "
					f"appears {len(list(duplicates.values())[0])} times."
				)
		
		# Track left side key uniqueness
		if expect in ('one_to_one', 'one_to_many'):
			left_keys_seen = set()
		
		# Track which right rows have been matched
		matched_right_rows = set()
		
		# Perform left side of full join
		result_rows = []
		for left_idx in range(len(self)):
			key = tuple(left_keys[i][left_idx] for i in range(len(left_keys)))
			
			# Check one_to_many / one_to_one constraint
			if expect in ('one_to_one', 'one_to_many'):
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			# Find matching right rows
			if key in right_index:
				for right_idx in right_index[key]:
					matched_right_rows.add(right_idx)
					# Combine left and right rows
					left_row = [col[left_idx] for col in self._underlying]
					right_row = [col[right_idx] for col in other._underlying]
					result_rows.append(left_row + right_row)
			else:
				# No match: left row with None for right columns
				left_row = [col[left_idx] for col in self._underlying]
				right_row = [None] * len(other._underlying)
				result_rows.append(left_row + right_row)
		
		# Add unmatched right rows
		for right_idx in range(len(other)):
			if right_idx not in matched_right_rows:
				left_row = [None] * len(self._underlying)
				right_row = [col[right_idx] for col in other._underlying]
				result_rows.append(left_row + right_row)
		
		if not result_rows:
			# Empty result
			return PyTable([])
		
		# Transpose result_rows to get columns
		num_cols = len(self._underlying) + len(other._underlying)
		result_cols = []
		for col_idx in range(num_cols):
			col_data = [row[col_idx] for row in result_rows]
			
			# Preserve column names
			if col_idx < len(self._underlying):
				orig_col = self._underlying[col_idx]
				result_cols.append(PyVector(col_data, name=orig_col._name))
			else:
				orig_col = other._underlying[col_idx - len(self._underlying)]
				result_cols.append(PyVector(col_data, name=orig_col._name))
		
		return PyTable(result_cols)
	
	def aggregate(
		self,
		# --- Partition keys ---
		over,
		
		# --- Built-in aggregations ---
		sum_over=None,
		mean_over=None,
		min_over=None,
		max_over=None,
		stdev_over=None,
		count_over=None,
		
		# --- Escape hatch ---
		apply=None,
	):
		"""
		Group rows by partition keys and compute aggregations.
		
		Args:
			over: PyVector(s) to partition/group by
			sum_over: PyVector(s) to sum within each group
			mean_over: PyVector(s) to average within each group
			min_over: PyVector(s) to find minimum within each group
			max_over: PyVector(s) to find maximum within each group
			stdev_over: PyVector(s) to compute standard deviation within each group
			count_over: PyVector(s) to count non-None values within each group
			apply: Dict of {name: (column, function)} for custom aggregations
		
		Returns:
			PyTable with one row per unique partition key combination
		
		Examples:
			# Group by customer_id, sum orders
			table.aggregate(over=table.customer_id, sum_over=table.order_total)
			
			# Multiple partition keys and aggregations
			table.aggregate(
				over=[table.year, table.month],
				sum_over=table.revenue,
				mean_over=table.score,
				count_over=table.transaction_id
			)
		"""
		# Normalize partition keys to list
		if isinstance(over, PyVector):
			over = [over]
		
		# Normalize aggregation columns to lists
		if sum_over is not None and isinstance(sum_over, PyVector):
			sum_over = [sum_over]
		if mean_over is not None and isinstance(mean_over, PyVector):
			mean_over = [mean_over]
		if min_over is not None and isinstance(min_over, PyVector):
			min_over = [min_over]
		if max_over is not None and isinstance(max_over, PyVector):
			max_over = [max_over]
		if stdev_over is not None and isinstance(stdev_over, PyVector):
			stdev_over = [stdev_over]
		if count_over is not None and isinstance(count_over, PyVector):
			count_over = [count_over]
		
		# Validate all columns have correct length
		for i, partition_col in enumerate(over):
			if len(partition_col) != len(self):
				raise PyVectorValueError(
					f"Partition key at index {i} has length {len(partition_col)}, "
					f"but table has {len(self)} rows"
				)
		
		# Build partition index: key_tuple -> list of row indices
		partition_index = {}
		for row_idx in range(len(self)):
			key = tuple(over[i][row_idx] for i in range(len(over)))
			if key not in partition_index:
				partition_index[key] = []
			partition_index[key].append(row_idx)
		
		# Helper to generate unique column name with suffix
		def make_agg_name(col, suffix):
			base_name = col._name if col._name else "col"
			base_sanitized = _sanitize_user_name(base_name)
			if base_sanitized is None:
				base_sanitized = "col"
			return f"{base_sanitized}_{suffix}"
		
		# Collect all result columns
		result_cols = []
		used_names = set()
		
		# Helper to ensure unique names
		def uniquify_name(name):
			if name not in used_names:
				used_names.add(name)
				return name
			counter = 2
			while f"{name}{counter}" in used_names:
				counter += 1
			unique_name = f"{name}{counter}"
			used_names.add(unique_name)
			return unique_name
		
		# Add partition key columns (unique values)
		# Use index-based enumeration to avoid any equality/operator overloads on PyVector
		for idx, partition_col in enumerate(over):
			unique_values = [key[idx] for key in partition_index.keys()]
			col_name = partition_col._name if partition_col._name else "key"
			result_cols.append(PyVector(unique_values, name=uniquify_name(col_name)))
		
		# Process aggregations
		def compute_aggregation(agg_cols, agg_func, suffix):
			if agg_cols is None:
				return
			for col in agg_cols:
				if len(col) != len(self):
					raise PyVectorValueError(f"Aggregation column has wrong length")
				
				agg_values = []
				for key in partition_index.keys():
					row_indices = partition_index[key]
					group_values = [col[idx] for idx in row_indices]
					agg_values.append(agg_func(group_values))
				
				col_name = make_agg_name(col, suffix)
				result_cols.append(PyVector(agg_values, name=uniquify_name(col_name)))
		
		# Apply built-in aggregations
		compute_aggregation(sum_over, lambda vals: sum(v for v in vals if v is not None), "sum")
		compute_aggregation(mean_over, lambda vals: sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None, "mean")
		compute_aggregation(min_over, lambda vals: min(v for v in vals if v is not None) if any(v is not None for v in vals) else None, "min")
		compute_aggregation(max_over, lambda vals: max(v for v in vals if v is not None) if any(v is not None for v in vals) else None, "max")
		compute_aggregation(count_over, lambda vals: len([v for v in vals if v is not None]), "count")
		
		# Standard deviation
		if stdev_over is not None:
			for col in stdev_over:
				if len(col) != len(self):
					raise PyVectorValueError(f"Aggregation column has wrong length")
				
				agg_values = []
				for key in partition_index.keys():
					row_indices = partition_index[key]
					group_values = [col[idx] for idx in row_indices if col[idx] is not None]
					if len(group_values) > 1:
						mean_val = sum(group_values) / len(group_values)
						variance = sum((v - mean_val) ** 2 for v in group_values) / (len(group_values) - 1)
						agg_values.append(variance ** 0.5)
					else:
						agg_values.append(None)
				
				col_name = make_agg_name(col, "stdev")
				result_cols.append(PyVector(agg_values, name=uniquify_name(col_name)))
		
		# Custom aggregations via apply
		if apply is not None:
			for agg_name, (col, func) in apply.items():
				if len(col) != len(self):
					raise PyVectorValueError(f"Custom aggregation column '{agg_name}' has wrong length")
				
				agg_values = []
				for key in partition_index.keys():
					row_indices = partition_index[key]
					group_values = [col[idx] for idx in row_indices]
					agg_values.append(func(group_values))
				
				result_cols.append(PyVector(agg_values, name=uniquify_name(agg_name)))
	
		return PyTable(result_cols)

	def window(
		self,
		# --- Partition keys ---
		over,
		
		# --- Built-in aggregations ---
		sum_over=None,
		mean_over=None,
		min_over=None,
		max_over=None,
		stdev_over=None,
		count_over=None,
		
		# --- Escape hatch ---
		apply=None,
	):
		"""
		Compute window functions over partitions, returning the same number of rows.
		
		Similar to aggregate(), but repeats the aggregated value for each row in the group.
		
		Args:
			over: PyVector(s) to partition/group by
			sum_over: PyVector(s) to sum within each group
			mean_over: PyVector(s) to average within each group
			min_over: PyVector(s) to find minimum within each group
			max_over: PyVector(s) to find maximum within each group
			stdev_over: PyVector(s) to compute standard deviation within each group
			count_over: PyVector(s) to count non-None values within each group
			apply: Dict of {name: (column, function)} for custom aggregations
		
		Returns:
			PyTable with same number of rows as input, with aggregated values repeated
		
		Examples:
			# Add running total per customer
			table.window(over=table.customer_id, sum_over=table.order_total)
			
			# Multiple window functions
			table.window(
				over=[table.year, table.month],
				sum_over=table.revenue,
				count_over=table.transaction_id
			)
		"""
		# Normalize partition keys to list
		if isinstance(over, PyVector):
			over = [over]
		
		# Normalize aggregation columns to lists
		if sum_over is not None and isinstance(sum_over, PyVector):
			sum_over = [sum_over]
		if mean_over is not None and isinstance(mean_over, PyVector):
			mean_over = [mean_over]
		if min_over is not None and isinstance(min_over, PyVector):
			min_over = [min_over]
		if max_over is not None and isinstance(max_over, PyVector):
			max_over = [max_over]
		if stdev_over is not None and isinstance(stdev_over, PyVector):
			stdev_over = [stdev_over]
		if count_over is not None and isinstance(count_over, PyVector):
			count_over = [count_over]
		
		# Validate all columns have correct length
		for i, partition_col in enumerate(over):
			if len(partition_col) != len(self):
				raise ValueError(
					f"Partition key at index {i} has length {len(partition_col)}, "
					f"but table has {len(self)} rows"
				)
		
		# Build partition index: key_tuple -> list of row indices
		partition_index = {}
		for row_idx in range(len(self)):
			key = tuple(over[i][row_idx] for i in range(len(over)))
			if key not in partition_index:
				partition_index[key] = []
			partition_index[key].append(row_idx)
		
		# Helper to generate unique column name with suffix
		def make_agg_name(col, suffix):
			base_name = col._name if col._name else "col"
			base_sanitized = _sanitize_user_name(base_name)
			if base_sanitized is None:
				base_sanitized = "col"
			return f"{base_sanitized}_{suffix}"
		
		# Collect all result columns
		result_cols = []
		used_names = set()
		
		# Helper to ensure unique names
		def uniquify_name(name):
			if name not in used_names:
				used_names.add(name)
				return name
			counter = 2
			while f"{name}{counter}" in used_names:
				counter += 1
			unique_name = f"{name}{counter}"
			used_names.add(unique_name)
			return unique_name
		
		# Add partition key columns (preserve original row order)
		for partition_col in over:
			# Just use the original column since we're keeping all rows
			col_name = partition_col._name if partition_col._name else "key"
			result_cols.append(PyVector(list(partition_col), name=uniquify_name(col_name)))
		
		# Process window aggregations - compute once per group, then expand
		def compute_window_aggregation(agg_cols, agg_func, suffix):
			if agg_cols is None:
				return
			for col in agg_cols:
				if len(col) != len(self):
					raise ValueError(f"Aggregation column has wrong length")
				
				# Compute aggregation for each group
				group_agg_values = {}
				for key, row_indices in partition_index.items():
					group_values = [col[idx] for idx in row_indices]
					group_agg_values[key] = agg_func(group_values)
				
				# Expand: assign aggregated value to each row in original order
				window_values = []
				for row_idx in range(len(self)):
					key = tuple(over[i][row_idx] for i in range(len(over)))
					window_values.append(group_agg_values[key])
				
				col_name = make_agg_name(col, suffix)
				result_cols.append(PyVector(window_values, name=uniquify_name(col_name)))
		
		# Apply built-in aggregations
		compute_window_aggregation(sum_over, lambda vals: sum(v for v in vals if v is not None), "sum")
		compute_window_aggregation(mean_over, lambda vals: sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None, "mean")
		compute_window_aggregation(min_over, lambda vals: min(v for v in vals if v is not None) if any(v is not None for v in vals) else None, "min")
		compute_window_aggregation(max_over, lambda vals: max(v for v in vals if v is not None) if any(v is not None for v in vals) else None, "max")
		compute_window_aggregation(count_over, lambda vals: len([v for v in vals if v is not None]), "count")
		
		# Standard deviation
		if stdev_over is not None:
			for col in stdev_over:
				if len(col) != len(self):
					raise ValueError(f"Aggregation column has wrong length")
				
				# Compute aggregation for each group
				group_agg_values = {}
				for key, row_indices in partition_index.items():
					group_values = [col[idx] for idx in row_indices if col[idx] is not None]
					if len(group_values) > 1:
						mean_val = sum(group_values) / len(group_values)
						variance = sum((v - mean_val) ** 2 for v in group_values) / (len(group_values) - 1)
						group_agg_values[key] = variance ** 0.5
					else:
						group_agg_values[key] = None
				
				# Expand to all rows
				window_values = []
				for row_idx in range(len(self)):
					key = tuple(over[i][row_idx] for i in range(len(over)))
					window_values.append(group_agg_values[key])
				
				col_name = make_agg_name(col, "stdev")
				result_cols.append(PyVector(window_values, name=uniquify_name(col_name)))
		
		# Custom aggregations via apply
		if apply is not None:
			for agg_name, (col, func) in apply.items():
				if len(col) != len(self):
					raise ValueError(f"Custom aggregation column '{agg_name}' has wrong length")
				
				# Compute aggregation for each group
				group_agg_values = {}
				for key, row_indices in partition_index.items():
					group_values = [col[idx] for idx in row_indices]
					group_agg_values[key] = func(group_values)
				
				# Expand to all rows
				window_values = []
				for row_idx in range(len(self)):
					key = tuple(over[i][row_idx] for i in range(len(over)))
					window_values.append(group_agg_values[key])
				
				result_cols.append(PyVector(window_values, name=uniquify_name(agg_name)))
		
		return PyTable(result_cols)
