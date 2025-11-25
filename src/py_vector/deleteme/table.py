import warnings
from .vector import PyVector
from .naming import _sanitize_user_name, _uniquify
from .errors import PyVectorKeyError, PyVectorValueError, PyVectorTypeError


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
	
	def __new__(cls, initial=(), dtype=None, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), dtype=None, name=None, as_row=False):
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
		
		# Set _dtype to None explicitly since PyTable bypasses PyVector.__new__
		self._dtype = None
		
		# Call parent constructor
		super().__init__(initial, dtype=dtype, name=name)
		
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
				dtype = self._dtype
			).T

		if isinstance(key, PyVector) and key.schema().kind == bool and not key.schema().nullable:
			assert (len(self) == len(key))
			return PyVector(tuple(x[key] for x in self._underlying),
				dtype = self._dtype
			)
		if isinstance(key, list) and {type(e) for e in key} == {bool}:
			assert (len(self) == len(key))
			return PyVector(tuple(x[key] for x in self._underlying),
				dtype = self._dtype
			)
		if isinstance(key, slice):
			return PyVector(tuple(x[key] for x in self._underlying), 
				dtype = self._dtype,
				name=self._name
			)

		# NOT RECOMMENDED
		if isinstance(key, PyVector) and key.schema().kind == int and not key.schema().nullable:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors; prefer slices or boolean masks')
			return PyVector(tuple(x[key] for x in self._underlying),
				dtype = self._dtype
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
		if self._dtype is not None and self._dtype.kind in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden. Use .bitshift() to shift bits.")

		if isinstance(other, PyTable):
			if self._dtype is not None and not self._dtype.nullable and other.schema() is not None and not other.schema().nullable and self._dtype.kind != other.schema().kind:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector(self.cols() + other.cols(),
				dtype=self._dtype)
		if isinstance(other, PyVector):
			# Adding a column to a table - tables can have mixed-type columns
			return PyVector(self.cols() + (other,),
				dtype=self._dtype)
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			# Convert iterable to PyVector and add as column (let PyVector infer dtype)
			return PyVector(self.cols() + (PyVector(other),),
				dtype=self._dtype)
		elif not self:
			return PyVector((other,),
				dtype=self._dtype)
		raise PyVectorTypeError("Cannot add a column of constant values. Try using PyVector.new(element, length).")

	def __lshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if isinstance(other, PyTable):
			return PyVector([x << y for x, y in zip(self._underlying, other._underlying, strict=True)])
		return PyVector([x << y for x, y in zip(self._underlying, other, strict=True)])

	@staticmethod
	def _validate_key_tuple_hashable(key_tuple, key_cols, row_idx):
		"""
		Validate that a join key tuple is hashable (for object dtype columns).
		
		Args:
			key_tuple: The tuple of key values to validate
			key_cols: List of key column PyVectors
			row_idx: Row index for error messages
		
		Raises:
			PyVectorTypeError: If any key component is not hashable
		"""
		try:
			hash(key_tuple)
		except TypeError as e:
			# Find which component failed
			for i, (component, col) in enumerate(zip(key_tuple, key_cols)):
				try:
					hash(component)
				except TypeError:
					col_name = col._name or f"key_{i}"
					raise PyVectorTypeError(
						f"Join key value in '{col_name}' at row {row_idx} is not hashable: "
						f"{type(component).__name__}. Join keys must be hashable."
					) from e
			# If we can't find the specific component, raise generic error
			raise PyVectorTypeError(
				f"Join key at row {row_idx} is not hashable."
			) from e

	def _validate_join_keys(self, other, left_on, right_on):
		"""
		Validate and normalize join key specification.
		
		Args:
			other: Right table to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
		
		Returns:
			List of (left_col, right_col) tuples (PyVector objects)
		
		Raises:
			PyVectorValueError: For malformed specs or validation failures
			PyVectorTypeError: For invalid dtypes or unhashable values
		"""
		from datetime import date, datetime
		
		# Helper: Resolve column from name or PyVector, validate ownership
		def get_column(table, col_spec, side_name):
			# CASE 1 — string column name
			if isinstance(col_spec, str):
				try:
					col = table[col_spec]
				except (KeyError, ValueError):
					raise _missing_col_error(
						col_spec,
						context=f"{side_name} table"
					)
			
			# CASE 2 — direct PyVector
			elif isinstance(col_spec, PyVector):
				col = col_spec
				# Note: Column ownership validation could be added here if PyVector
				# gains a _parent_table attribute in the future
			
			else:
				raise PyVectorValueError(
					f"Column specification must be string or PyVector, got "
					f"{type(col_spec).__name__}"
				)
			
			return col
		
		# Helper: Validate column dtype for join keys (static type check)
		def validate_key_dtype(col, side_name, idx):
			schema = col.schema()
			if schema is None:
				# Empty/untyped vectors - validate at runtime below
				return
			
			kind = schema.kind
			
			# Floats are NOT allowed — non-deterministic equality
			if kind is float:
				raise PyVectorTypeError(
					f"Invalid join key dtype 'float' at position {idx} on {side_name} side. "
					"Floating-point columns cannot be used as join keys due to precision issues."
				)
			
			# Allowed types: hashable and have stable equality
			# complex is excluded (not typically used for joins, can be added if needed)
			allowed_types = (int, str, bool, date, datetime, object)
			if kind not in allowed_types:
				raise PyVectorTypeError(
					f"Invalid join key dtype '{kind.__name__}' at position {idx} on {side_name} side. "
					"Join keys must support stable equality and hashing."
				)
		
		# Normalize to lists
		if isinstance(left_on, (str, PyVector)):
			left_on = [left_on]
		if isinstance(right_on, (str, PyVector)):
			right_on = [right_on]
		
		if not (isinstance(left_on, list) and isinstance(right_on, list)):
			raise PyVectorValueError("left_on and right_on must be strings, PyVectors, or lists")
		
		if not left_on or not right_on:
			raise PyVectorValueError("Must specify at least 1 join key")
		
		if len(left_on) != len(right_on):
			raise PyVectorValueError(
				f"left_on and right_on must have same length: "
				f"got {len(left_on)} and {len(right_on)}"
			)
		
		# Build final list of join key pairs
		normalized = []
		for i, (left_spec, right_spec) in enumerate(zip(left_on, right_on)):
			left_col = get_column(self, left_spec, "left")
			right_col = get_column(other, right_spec, "right")
			
			# Length validation
			if len(left_col) != len(self):
				raise PyVectorValueError(
					f"Left join key at index {i} has length {len(left_col)}, "
					f"but left table has {len(self)} rows"
				)
			if len(right_col) != len(other):
				raise PyVectorValueError(
					f"Right join key at index {i} has length {len(right_col)}, "
					f"but right table has {len(other)} rows"
				)
			
			# Dtype validation
			validate_key_dtype(left_col, "left", i)
			validate_key_dtype(right_col, "right", i)
			
			# Matching dtype validation (both must have schemas and same kind)
			left_schema = left_col.schema()
			right_schema = right_col.schema()
			if left_schema is not None and right_schema is not None:
				if left_schema.kind is not right_schema.kind:
					raise PyVectorTypeError(
						f"Join key at index {i} has mismatched dtypes: "
						f"{left_schema.kind.__name__} (left) vs {right_schema.kind.__name__} (right)"
					)
			
			normalized.append((left_col, right_col))
		
		return normalized

	def inner_join(self, other, left_on, right_on, expect='many_to_one'):
		"""
		Inner join two PyTables on specified key columns.
		Only returns rows where keys match in both tables.
		
		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate cardinality flag early
		if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
			raise PyVectorValueError(
				f"Invalid expect='{expect}'. "
				"Must be one of 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'."
			)
		
		# ------------------------------------------------------------------
		# 1. Validate and extract join keys
		# ------------------------------------------------------------------
		pairs = self._validate_join_keys(other, left_on, right_on)
		left_keys = [lk for lk, _ in pairs]
		right_keys = [rk for _, rk in pairs]
		
		# Determine if we need to validate hashability (only for object dtype columns)
		validate_hashable = any(
			(col.schema() is None or col.schema().kind is object)
			for col in (left_keys + right_keys)
		)
		
		# Pre-bind lengths and columns
		left_nrows = len(self)
		right_nrows = len(other)
		left_cols = self._underlying
		right_cols = other._underlying
		n_left_cols = len(left_cols)
		n_right_cols = len(right_cols)
		
		# ------------------------------------------------------------------
		# 2. Build hash map on right side
		# ------------------------------------------------------------------
		right_index = {}
		right_index_get = right_index.get
		
		check_right_unique = expect in ('one_to_one', 'many_to_one')
		if check_right_unique:
			duplicates = {}
		
		# Build key → [indices]
		for row_idx in range(right_nrows):
			key = tuple(col[row_idx] for col in right_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				PyTable._validate_key_tuple_hashable(key, right_keys, row_idx)
			
			bucket = right_index_get(key)
			if bucket is None:
				right_index[key] = [row_idx]
			else:
				bucket.append(row_idx)
				if check_right_unique:
					duplicates[key] = bucket
		
		# Cardinality check on right (one-to-one, many-to-one)
		if check_right_unique and duplicates:
			example_key, example_indices = next(iter(duplicates.items()))
			raise PyVectorValueError(
				f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
				f"Example: {example_key} appears {len(example_indices)} times."
			)
		
		# ------------------------------------------------------------------
		# 3. Left-side uniqueness enforcement
		# ------------------------------------------------------------------
		check_left_unique = expect in ('one_to_one', 'one_to_many')
		if check_left_unique:
			left_keys_seen = set()
		
		# ------------------------------------------------------------------
		# 4. Build RESULT in column-major order
		# ------------------------------------------------------------------
		result_data = [[] for _ in range(n_left_cols + n_right_cols)]
		append_cols = [col.append for col in result_data]
		
		# Perform join
		for left_idx in range(left_nrows):
			key = tuple(col[left_idx] for col in left_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				PyTable._validate_key_tuple_hashable(key, left_keys, left_idx)
			
			# Enforce left-side cardinality (if needed)
			if check_left_unique:
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			matches = right_index_get(key)
			if not matches:
				continue  # INNER JOIN → skip non-matches
			
			# Emit each match
			for right_idx in matches:
				# Left columns
				for c_idx, col in enumerate(left_cols):
					append_cols[c_idx](col[left_idx])
				
				# Right columns
				base = n_left_cols
				for offset, col in enumerate(right_cols):
					append_cols[base + offset](col[right_idx])
		
		# Handle empty result
		if all(len(col) == 0 for col in result_data):
			return PyTable([])
		
		# ------------------------------------------------------------------
		# 5. Wrap result_data in PyVectors
		# ------------------------------------------------------------------
		result_cols = []
		
		# Left columns (preserve name)
		for col_idx, orig_col in enumerate(left_cols):
			result_cols.append(PyVector(result_data[col_idx], name=orig_col._name))
		
		# Right columns (preserve name)
		base = n_left_cols
		for offset, orig_col in enumerate(right_cols):
			result_cols.append(PyVector(result_data[base + offset], name=orig_col._name))
		
		return PyTable(result_cols)

	def join(self, other, left_on, right_on, expect='many_to_one'):
		"""
		Left join two PyTables on specified key columns.
		Returns all rows from left table, with matching rows from right (or None for no match).
		
		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one',
			        'one_to_many', or 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate expectation value early
		if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
			raise PyVectorValueError(
				f"Invalid expect value '{expect}'. "
				"Must be one of 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'."
			)
		
		# Validate and normalize join keys
		pairs = self._validate_join_keys(other, left_on, right_on)
		
		# Extract join key columns (PyVectors)
		left_keys = [lk for lk, _ in pairs]
		right_keys = [rk for _, rk in pairs]
		
		# Check if any key columns have object dtype (need runtime validation)
		validate_hashable = any(
			(col.schema() is None or col.schema().kind is object)
			for col in (left_keys + right_keys)
		)
		
		left_nrows = len(self)
		right_nrows = len(other)
		left_cols = self._underlying
		right_cols = other._underlying
		n_left_cols = len(left_cols)
		n_right_cols = len(right_cols)
		
		# Build hash map on right: key_tuple -> list of row indices
		right_index = {}
		check_right_unique = expect in ('one_to_one', 'many_to_one')
		if check_right_unique:
			duplicates = {}
		
		for row_idx in range(right_nrows):
			key = tuple(col[row_idx] for col in right_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				self._validate_key_tuple_hashable(key, right_keys, row_idx)
			
			bucket = right_index.get(key)
			if bucket is None:
				right_index[key] = [row_idx]
			else:
				bucket.append(row_idx)
				if check_right_unique and key not in duplicates:
					duplicates[key] = bucket
		
		# Enforce right-side uniqueness if required
		if check_right_unique and duplicates:
			example_key, example_indices = next(iter(duplicates.items()))
			raise PyVectorValueError(
				f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
				f"Found at least {len(duplicates)} duplicate key(s), e.g., {example_key} "
				f"appears {len(example_indices)} times."
			)
		
		# Prepare left-side uniqueness tracking if needed
		check_left_unique = expect in ('one_to_one', 'one_to_many')
		if check_left_unique:
			left_keys_seen = set()
		
		# Perform left join, building result in COLUMN-MAJOR form
		total_cols = n_left_cols + n_right_cols
		result_data = [[] for _ in range(total_cols)]
		
		# Local binds for speed
		result_append_cols = [col.append for col in result_data]
		right_index_get = right_index.get
		
		for left_idx in range(left_nrows):
			key = tuple(col[left_idx] for col in left_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				self._validate_key_tuple_hashable(key, left_keys, left_idx)
			
			# Enforce left-side uniqueness if needed
			if check_left_unique:
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			matches = right_index_get(key)
			
			if matches:
				# For each matching right row, append combined row
				for right_idx in matches:
					# Append left columns
					for c_idx, col in enumerate(left_cols):
						result_append_cols[c_idx](col[left_idx])
					
					# Append right columns
					base = n_left_cols
					for offset, col in enumerate(right_cols):
						result_append_cols[base + offset](col[right_idx])
			else:
				# No match: left row with None for all right columns
				for c_idx, col in enumerate(left_cols):
					result_append_cols[c_idx](col[left_idx])
				
				base = n_left_cols
				for offset in range(n_right_cols):
					result_append_cols[base + offset](None)
		
		# Handle completely empty result
		if left_nrows == 0:
			return PyTable([])
		
		# Wrap result_data into PyVectors, preserving column names
		result_cols = []
		
		# Left table columns
		for col_idx, orig_col in enumerate(left_cols):
			col_data = result_data[col_idx]
			result_cols.append(PyVector(col_data, name=orig_col._name))
		
		# Right table columns
		for j, orig_col in enumerate(right_cols):
			col_data = result_data[n_left_cols + j]
			result_cols.append(PyVector(col_data, name=orig_col._name))
		
		return PyTable(result_cols)

	def full_join(self, other, left_on, right_on, expect='many_to_many'):
		"""
		Full outer join of two PyTables. Includes:
			- All rows from left table
			- All rows from right table
			- Matching rows combined
			- None where no match exists
		
		Args:
			other: PyTable to join with
			left_on: Column name(s) or PyVector(s) from left table
			right_on: Column name(s) or PyVector(s) from right table
			expect: Cardinality expectation - 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
		
		Returns:
			PyTable with joined results
		"""
		# Validate expectation string
		if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
			raise PyVectorValueError(
				f"Invalid expect='{expect}'. "
				"Must be 'one_to_one', 'many_to_one', 'one_to_many', or 'many_to_many'."
			)
		
		# ------------------------------------------------------------------
		# 1. Validate join keys and extract columns
		# ------------------------------------------------------------------
		pairs = self._validate_join_keys(other, left_on, right_on)
		left_keys = [lk for lk, _ in pairs]
		right_keys = [rk for _, rk in pairs]
		
		# Determine if we need to validate hashability (only for object dtype columns)
		validate_hashable = any(
			(col.schema() is None or col.schema().kind is object)
			for col in (left_keys + right_keys)
		)
		
		left_nrows = len(self)
		right_nrows = len(other)
		
		left_cols = self._underlying
		right_cols = other._underlying
		n_left_cols = len(left_cols)
		n_right_cols = len(right_cols)
		
		# ------------------------------------------------------------------
		# 2. Build hash index for right table
		# ------------------------------------------------------------------
		right_index = {}
		right_index_get = right_index.get
		
		check_right_unique = expect in ('one_to_one', 'many_to_one')
		if check_right_unique:
			duplicates = {}
		
		for right_idx in range(right_nrows):
			key = tuple(col[right_idx] for col in right_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				PyTable._validate_key_tuple_hashable(key, right_keys, right_idx)
			
			bucket = right_index_get(key)
			if bucket is None:
				right_index[key] = [right_idx]
			else:
				bucket.append(right_idx)
				if check_right_unique:
					duplicates[key] = bucket
		
		# Enforce right-side cardinality if necessary
		if check_right_unique and duplicates:
			example_key, example_inds = next(iter(duplicates.items()))
			raise PyVectorValueError(
				f"Join expectation '{expect}' violated: Right side has duplicate keys.\n"
				f"Example: {example_key} appears {len(example_inds)} times."
			)
		
		# ------------------------------------------------------------------
		# 3. Prepare left-side cardinality tracking
		# ------------------------------------------------------------------
		check_left_unique = expect in ('one_to_one', 'one_to_many')
		if check_left_unique:
			left_keys_seen = set()
		
		# Track which right rows are matched
		matched_right_rows = set()
		matched_right_add = matched_right_rows.add
		
		# ------------------------------------------------------------------
		# 4. Build RESULT in column-major form
		# ------------------------------------------------------------------
		total_cols = n_left_cols + n_right_cols
		result_data = [[] for _ in range(total_cols)]
		append_cols = [col.append for col in result_data]
		
		# ------------------------------------------------------------------
		# 5. Process LEFT table (major phase of full join)
		# ------------------------------------------------------------------
		for left_idx in range(left_nrows):
			key = tuple(col[left_idx] for col in left_keys)
			
			# Validate hashability for object dtype columns
			if validate_hashable:
				PyTable._validate_key_tuple_hashable(key, left_keys, left_idx)
			
			# Enforce left-side cardinality
			if check_left_unique:
				if key in left_keys_seen:
					raise PyVectorValueError(
						f"Join expectation '{expect}' violated: Left side has duplicate key {key}"
					)
				left_keys_seen.add(key)
			
			matches = right_index_get(key)
			if matches:
				# Emit matched combinations
				for right_idx in matches:
					matched_right_add(right_idx)
					
					# Left
					for c_idx, col in enumerate(left_cols):
						append_cols[c_idx](col[left_idx])
					
					# Right
					base = n_left_cols
					for offset, col in enumerate(right_cols):
						append_cols[base + offset](col[right_idx])
			else:
				# No match → left row + None right
				for c_idx, col in enumerate(left_cols):
					append_cols[c_idx](col[left_idx])
				
				base = n_left_cols
				for offset in range(n_right_cols):
					append_cols[base + offset](None)
		
		# ------------------------------------------------------------------
		# 6. Add unmatched RIGHT rows
		# ------------------------------------------------------------------
		for right_idx in range(right_nrows):
			if right_idx not in matched_right_rows:
				# Left side: all None
				for c_idx in range(n_left_cols):
					append_cols[c_idx](None)
				
				# Right side: real values
				base = n_left_cols
				for offset, col in enumerate(right_cols):
					append_cols[base + offset](col[right_idx])
		
		# ------------------------------------------------------------------
		# 7. If empty, return empty table
		# ------------------------------------------------------------------
		if left_nrows == 0 and right_nrows == 0:
			return PyTable([])
		
		# ------------------------------------------------------------------
		# 8. Wrap into PyVectors with names preserved
		# ------------------------------------------------------------------
		result_cols = []
		
		# Left columns
		for col_idx, orig_col in enumerate(left_cols):
			result_cols.append(PyVector(result_data[col_idx], name=orig_col._name))
		
		# Right columns
		base = n_left_cols
		for offset, orig_col in enumerate(right_cols):
			result_cols.append(PyVector(result_data[base + offset], name=orig_col._name))
		
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
		# ------------------------------------------------------------------
		# 1. Normalize inputs
		# ------------------------------------------------------------------
		if isinstance(over, PyVector):
			over = [over]
		
		# Normalize all agg lists
		def normalize(v):
			if v is None:
				return None
			return v if isinstance(v, list) else [v]
		
		sum_over   = normalize(sum_over)
		mean_over  = normalize(mean_over)
		min_over   = normalize(min_over)
		max_over   = normalize(max_over)
		stdev_over = normalize(stdev_over)
		count_over = normalize(count_over)
		
		# ------------------------------------------------------------------
		# 2. Validate partition key lengths
		# ------------------------------------------------------------------
		nrows = len(self)
		for i, col in enumerate(over):
			if len(col) != nrows:
				raise PyVectorValueError(
					f"Partition key at index {i} has length {len(col)}, "
					f"but table has {nrows} rows."
				)
		
		# ------------------------------------------------------------------
		# 3. Build partition index: key_tuple -> list of row indices
		# ------------------------------------------------------------------
		partition_index = {}
		pk_len = len(over)
		
		# Prebind key-cols for speed
		over_data = [c._underlying for c in over]
		
		for row_idx in range(nrows):
			key = tuple(over_data[i][row_idx] for i in range(pk_len))
			# Small fast-path
			bucket = partition_index.get(key)
			if bucket is None:
				partition_index[key] = [row_idx]
			else:
				bucket.append(row_idx)
		
		# Prebind items iteration
		group_items = list(partition_index.items())
		
		# ------------------------------------------------------------------
		# 4. Name sanitization helpers
		# ------------------------------------------------------------------
		def make_agg_name(col, suffix):
			base = col._name or "col"
			s = _sanitize_user_name(base)
			if s is None:
				s = "col"
			return f"{s}_{suffix}"
		
		used_names = set()
		
		def uniquify(name):
			if name not in used_names:
				used_names.add(name)
				return name
			i = 2
			while f"{name}{i}" in used_names:
				i += 1
			new = f"{name}{i}"
			used_names.add(new)
			return new
		
		# ------------------------------------------------------------------
		# 5. Build result columns array
		# ------------------------------------------------------------------
		result_cols = []
		
		# --- Partition key columns (one per unique group) ---
		# Pre-bind: this is fast because group_items holds (key, rows)
		for idx, col in enumerate(over):
			values = [key[idx] for key, _ in group_items]
			result_cols.append(PyVector(values, name=uniquify(col._name or "key")))
		
		# ------------------------------------------------------------------
		# 6. Column-major helper: aggregate one column for all groups
		# ------------------------------------------------------------------
		def aggregate_col(col, func, suffix):
			data = col._underlying
			out = []
			
			for key, row_indices in group_items:
				# column-major group extraction
				vals = [data[i] for i in row_indices]
				
				res = func(vals)
				out.append(res)
			
			name = uniquify(make_agg_name(col, suffix))
			result_cols.append(PyVector(out, name=name))
		
		# ------------------------------------------------------------------
		# 7. Built-in aggregations (fast, no repeated scans)
		# ------------------------------------------------------------------
		
		# SUM
		if sum_over:
			for col in sum_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				aggregate_col(
					col,
					lambda vals, d=d: sum(v for v in vals if v is not None),
					"sum"
				)
		
		# MEAN
		if mean_over:
			for col in mean_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				def mean_func(vals, d=d):
					clean = [v for v in vals if v is not None]
					return sum(clean) / len(clean) if clean else None
				
				aggregate_col(col, mean_func, "mean")
		
		# MIN
		if min_over:
			for col in min_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				def min_func(vals, d=d):
					clean = [v for v in vals if v is not None]
					return min(clean) if clean else None
				
				aggregate_col(col, min_func, "min")
		
		# MAX
		if max_over:
			for col in max_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				def max_func(vals, d=d):
					clean = [v for v in vals if v is not None]
					return max(clean) if clean else None
				
				aggregate_col(col, max_func, "max")
		
		# COUNT
		if count_over:
			for col in count_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				aggregate_col(
					col,
					lambda vals, d=d: sum(1 for v in vals if v is not None),
					"count"
				)
		
		# STDEV (sample standard deviation)
		if stdev_over:
			for col in stdev_over:
				if len(col) != nrows:
					raise PyVectorValueError(f"Aggregation column has wrong length")
				d = col._underlying
				
				def stdev_func(vals, d=d):
					clean = [v for v in vals if v is not None]
					n = len(clean)
					if n <= 1:
						return None
					mean_val = sum(clean) / n
					variance = sum((v - mean_val) ** 2 for v in clean) / (n - 1)
					return variance ** 0.5
				
				aggregate_col(col, stdev_func, "stdev")
		
		# ------------------------------------------------------------------
		# 8. Custom apply aggregations
		# ------------------------------------------------------------------
		if apply is not None:
			for agg_name, (col, func) in apply.items():
				if len(col) != nrows:
					raise PyVectorValueError(f"Custom aggregation column '{agg_name}' has wrong length")
				d = col._underlying
				out = []
				for key, row_indices in group_items:
					vals = [d[i] for i in row_indices]
					out.append(func(vals))
				
				result_cols.append(PyVector(out, name=uniquify(agg_name)))
		
		# ------------------------------------------------------------------
		# 9. Final table
		# ------------------------------------------------------------------
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
		
		# --- Custom aggregations ---
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
		# ----------------------------------------------------------------------
		# 1. Normalize inputs
		# ----------------------------------------------------------------------
		if isinstance(over, PyVector):
			over = [over]
		
		# Normalize aggregation lists
		def norm(v):
			return None if v is None else (v if isinstance(v, list) else [v])
		
		sum_over   = norm(sum_over)
		mean_over  = norm(mean_over)
		min_over   = norm(min_over)
		max_over   = norm(max_over)
		stdev_over = norm(stdev_over)
		count_over = norm(count_over)
		
		# ----------------------------------------------------------------------
		# 2. Validate column lengths
		# ----------------------------------------------------------------------
		nrows = len(self)
		for i, col in enumerate(over):
			if len(col) != nrows:
				raise ValueError(
					f"Partition key at index {i} has length {len(col)}, "
					f"but table has {nrows} rows."
				)
		
		# ----------------------------------------------------------------------
		# 3. Build partition index: key -> list[row indices]
		# ----------------------------------------------------------------------
		partition_index = {}
		pk_len = len(over)
		over_data = [c._underlying for c in over]
		
		# Build once; reused everywhere
		row_keys = [None] * nrows
		
		for i in range(nrows):
			key = tuple(over_data[k][i] for k in range(pk_len))
			row_keys[i] = key
			bucket = partition_index.get(key)
			if bucket is None:
				partition_index[key] = [i]
			else:
				bucket.append(i)
		
		group_items = list(partition_index.items())
		
		# ----------------------------------------------------------------------
		# 4. Name logic
		# ----------------------------------------------------------------------
		used = set()
		
		def sanitize(col, suffix):
			base = col._name or "col"
			s = _sanitize_user_name(base) or "col"
			return f"{s}_{suffix}"
		
		def uniquify(name):
			if name not in used:
				used.add(name)
				return name
			i = 2
			while f"{name}{i}" in used:
				i += 1
			final = f"{name}{i}"
			used.add(final)
			return final
		
		# ----------------------------------------------------------------------
		# 5. Start with partition key columns (copy directly)
		# ----------------------------------------------------------------------
		result_cols = []
		for col in over:
			result_cols.append(
				PyVector(list(col), name=uniquify(col._name or "key"))
			)
		
		# ----------------------------------------------------------------------
		# 6. Helper: compute group-level aggregation for one column
		# ----------------------------------------------------------------------
		def compute_group_values(col, fn):
			data = col._underlying
			out = {}
			for key, rows in group_items:
				vals = [data[i] for i in rows]
				out[key] = fn(vals)
			return out
		
		# ----------------------------------------------------------------------
		# 7. Helper: expand group-level values back to all rows
		# ----------------------------------------------------------------------
		def expand_to_rows(group_map):
			return [group_map[row_keys[i]] for i in range(nrows)]
		
		# ----------------------------------------------------------------------
		# 8. Built-in aggregations
		# ----------------------------------------------------------------------
		# Each aggregator: compute group-level -> expand -> append column
		
		# SUM
		if sum_over:
			for col in sum_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					return sum(v for v in vals if v is not None)
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "sum")))
				)
		
		# MEAN
		if mean_over:
			for col in mean_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					clean = [v for v in vals if v is not None]
					return sum(clean) / len(clean) if clean else None
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "mean")))
				)
		
		# MIN
		if min_over:
			for col in min_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					clean = [v for v in vals if v is not None]
					return min(clean) if clean else None
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "min")))
				)
		
		# MAX
		if max_over:
			for col in max_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					clean = [v for v in vals if v is not None]
					return max(clean) if clean else None
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "max")))
				)
		
		# COUNT
		if count_over:
			for col in count_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					return sum(1 for v in vals if v is not None)
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "count")))
				)
		
		# STDEV
		if stdev_over:
			for col in stdev_over:
				if len(col) != nrows:
					raise ValueError(f"Aggregation column has wrong length")
				def fn(vals):
					clean = [v for v in vals if v is not None]
					n = len(clean)
					if n <= 1:
						return None
					mean_val = sum(clean) / n
					return (sum((v - mean_val)**2 for v in clean) / (n - 1)) ** 0.5
				
				gm = compute_group_values(col, fn)
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(sanitize(col, "stdev")))
				)
		
		# ----------------------------------------------------------------------
		# 9. Custom aggregation(s)
		# ----------------------------------------------------------------------
		if apply:
			for name, (col, fn) in apply.items():
				if len(col) != nrows:
					raise ValueError(f"Custom aggregation column '{name}' has wrong length")
				data = col._underlying
				gm = {
					key: fn([data[i] for i in rows])
					for key, rows in group_items
				}
				result_cols.append(
					PyVector(expand_to_rows(gm), name=uniquify(name))
				)
		
		# ----------------------------------------------------------------------
		# 10. Final table
		# ----------------------------------------------------------------------
		return PyTable(result_cols)
