import operator
import warnings
import re

from _alias_tracker import _ALIAS_TRACKER, AliasError
from copy import deepcopy
from datetime import date
from datetime import datetime


def _sanitize_name(name: str) -> str:
	"""
	Convert a user-facing column name into a safe Python attribute name.

	Rules:
	- Keep [A-Za-z0-9_] as-is.
	- Convert each contiguous run of other characters → single '_'.
	- Lowercase the result.
	- Strip leading/trailing underscores (these convey no semantic meaning).
	- If empty after stripping → 'col'.
	- If resulting name starts with a digit → prefix '_' (Python requirement).
	"""

	# Input normalization
	if not isinstance(name, str):
		name = str(name)

	out = []
	idx = 0
	length = len(name)

	while idx < length:
		char = name[idx]

		if char.isalnum() or char == "_":
			out.append(char)
			idx += 1
		else:
			# Non-allowed run → emit a single underscore
			out.append("_")
			while idx < length and not (name[idx].isalnum() or name[idx] == "_"):
				idx += 1

	sanitized = "".join(out).lower()

	# Strip sanitation underscores on ends
	sanitized = sanitized.strip("_")

	# Empty fallback
	if sanitized == "":
		sanitized = "col"

	# Must not start with a digit
	if sanitized[0].isdigit():
		sanitized = "_" + sanitized

	return sanitized


def _uniquify(base: str, existing: set) -> str:
	"""
	Ensure `base` is unique inside `existing`.
	If not, append '__1', '__2', ... until it becomes unique.
	"""

	if base not in existing:
		return base

	counter = 1
	while True:
		candidate = f"{base}__{counter}"
		if candidate not in existing:
			return candidate
		counter += 1


def slice_length(s: slice, sequence_length: int) -> int:
	"""
	Calculates the length of a slice given the slice object and the sequence length.

	Args:
		s: The slice object.
		sequence_length: The length of the sequence being sliced.

	Returns:
		The length of the slice.
	"""
	start, stop, step = s.indices(sequence_length)
	return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)

def _str_to_repr(head, tail, joiner):
	pass


def _format_col(col, num_rows=5):
	"""Format a column as a string vector with truncation and alignment."""
	# Handle truncation
	if len(col) <= num_rows * 2:
		und = col._underlying
	else:
		und = col._underlying[:num_rows] + ('...',) + col._underlying[-num_rows:]
	
	# Format values based on dtype
	if col._dtype == float:
		str_vec = PyVector([f"{val:g}" if val != '...' else val for val in und])
	elif col._dtype == int:
		str_vec = PyVector([str(val) if val != '...' else val for val in und])
	elif col._dtype == date:
		str_vec = PyVector([val.isoformat() if val != '...' else val for val in und])
	elif col._dtype == str:
		str_vec = PyVector([repr(val) if val != '...' else val for val in und])
	else:
		str_vec = PyVector([str(val) for val in und])
	
	# Right-align numeric columns, left-align others
	max_len = max(len(val) for val in str_vec)
	if col._dtype in (int, float):
		return str_vec.rjust(max_len)
	else:
		return str_vec.ljust(max_len)

def _recursive_colnames(pv):
	if len(pv.size()) > 2:
		all_colnames = {_recursive_colnames(c) for c in pv}
		if len(all_colnames) == 1:
			return all_colnames.pop()
		return tuple(None for _ in all_colnames.pop())
	if len(pv.size()) == 1:
		return ('',)
	names = tuple(c._name or '' for c in pv.cols())
	return names


def format_footer(pv, col_dtypes=None, truncated=False, num_shown=5):
	"""Generate clean footer line."""
	shape = '×'.join(str(s) for s in pv.size())
	
	if len(pv.size()) == 1:
		# Vector
		dtype_name = pv._dtype.__name__ if pv._dtype else 'object'
		return f"# {len(pv)} element vector <{dtype_name}>"
	elif len(pv.size()) == 2:
		# Table
		if col_dtypes:
			if truncated:
				# Show first num_shown, ..., last num_shown dtypes
				dtype_str = ', '.join(col_dtypes[:num_shown]) + ', ..., ' + ', '.join(col_dtypes[-num_shown:])
			else:
				dtype_str = ', '.join(col_dtypes)
		else:
			dtype_str = pv._dtype.__name__ if pv._dtype else 'object'
		return f"# {shape} table <{dtype_str}>"
	else:
		# Tensor
		dtype_str = pv._dtype.__name__ if pv._dtype else 'object'
		return f"# {shape} tensor <{dtype_str}>"

def _printr(pv):
	"""Clean display-focused repr."""
	if len(pv.size()) == 1:
		# 1D vector
		formatted = _format_col(pv)
		lines = []
		# Add name as first line if present
		if pv._name:
			lines.append(pv._name)
		lines.extend(formatted._underlying)
		lines.append('')
		lines.append(format_footer(pv))
		return '\n'.join(lines)
	
	elif len(pv.size()) == 2:
		# 2D table
		cols = pv.cols()
		if len(cols) == 0:
			return '# 0×0 table'
		
		# Determine which columns to display (truncate if too many)
		num_cols_to_show = 5  # Show first/last 5 columns
		if len(cols) <= num_cols_to_show * 2:
			cols_to_display = cols
			col_indices = list(range(len(cols)))
			truncated = False
		else:
			cols_to_display = list(cols[:num_cols_to_show]) + list(cols[-num_cols_to_show:])
			col_indices = list(range(num_cols_to_show)) + list(range(len(cols) - num_cols_to_show, len(cols)))
			truncated = True
		
		# Get column names and dtypes
		col_names = []
		col_dtypes = []
		formatted_cols = []
		
		for idx, col in zip(col_indices, cols_to_display):
			name = col._name or ''  # Use empty string for unnamed columns
			col_names.append(name)
			dtype_name = col._dtype.__name__ if col._dtype else 'object'
			col_dtypes.append(dtype_name)
			formatted_cols.append(_format_col(col))
		
		# Insert '...' separator for truncated columns
		if truncated:
			col_names.insert(num_cols_to_show, '...')
			formatted_cols.insert(num_cols_to_show, PyVector(['...' for _ in range(len(formatted_cols[0]))]))
		
		# Check if any columns have names
		has_any_names = any(name for name in col_names if name != '...')
		
		# Adjust column widths to fit headers (if we're showing them)
		aligned_cols = []
		aligned_names = []
		for name, col, fmt_col in zip(col_names, cols_to_display if not truncated else cols_to_display[:num_cols_to_show] + cols_to_display[num_cols_to_show:], formatted_cols):
			if has_any_names:
				width = max(len(name), max(len(cell) for cell in fmt_col._underlying))
			else:
				width = max(len(cell) for cell in fmt_col._underlying)
			
			# Re-align to new width
			if not truncated and col._dtype in (int, float):
				aligned_cols.append(fmt_col.rjust(width))
				if has_any_names:
					aligned_names.append(name.rjust(width) if name else ' ' * width)
			else:
				aligned_cols.append(fmt_col.ljust(width))
				if has_any_names:
					aligned_names.append(name.ljust(width))
		
		# Build output
		lines = []
		
		# Only add header row if at least one column has a name
		if has_any_names:
			lines.append('  '.join(aligned_names))
		
		# Zip columns into rows
		for row_idx in range(len(aligned_cols[0])):
			row = '  '.join(col._underlying[row_idx] for col in aligned_cols)
			lines.append(row)
		
		lines.append('')
		lines.append(format_footer(pv, col_dtypes, truncated, num_cols_to_show))
		return '\n'.join(lines)
	
	else:
		# Tensor (not implemented)
		return format_footer(pv) + ' (repr not yet implemented)'


class PyVector():
	""" Iterable vector with optional type safety """
	_dtype = None
	_typesafe = None
	_default = None
	_underlying = None
	_name = None
	_display_as_row = False
	
	# Fingerprint constants for O(1) change detection
	_FP_P = (1 << 61) - 1  # Large prime (~2^61)
	_FP_B = 1315423911     # Base for rolling hash


	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		""" Decide what type of PyVector this is """
		if initial and all(isinstance(x, PyVector) for x in initial):
			if len({len(x) for x in initial}) == 1:
				return PyTable(initial=initial,
					default_element=default_element,
					dtype=dtype,
					typesafe=typesafe,
					name=name, 
					as_row=as_row)
			warnings.warn('Passing vectors of different length will not produce a PyTable.')
		if initial and all(isinstance(x, str) for x in initial):
			return _PyString(initial=initial,
				default_element=default_element,
				dtype=dtype,
				typesafe=typesafe,
				name=name,
				as_row=as_row)
		if initial and all(isinstance(x, int) for x in initial) and all(not isinstance(x, bool) for x in initial):
			return _PyInt(initial=initial,
				default_element=default_element,
				dtype=dtype,
				typesafe=typesafe,
				name=name,
				as_row=as_row)
		if initial and all(isinstance(x, (int, float)) for x in initial) and any(isinstance(x, float) for x in initial) and all(not isinstance(x, bool) for x in initial):
			# Mixed int/float or all float - promote to float
			return _PyFloat(initial=initial,
				default_element=default_element,
				dtype=dtype,
				typesafe=typesafe,
				name=name,
				as_row=as_row)
		if initial and all(isinstance(x, date) for x in initial):
			return _PyDate(initial=initial,
				default_element=default_element,
				dtype=dtype,
				typesafe=typesafe,
				name=name,
				as_row=as_row)
		return super(PyVector, cls).__new__(cls)



	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		""" Create a new PyVector from an initial list """
		self._typesafe = typesafe
		self._dtype = dtype
		self._name = None
		if name is not None:
			self.rename(name)
		self._default = default_element
		self._display_as_row = as_row

		if self._typesafe and default_element is not None and not isinstance(default_element, dtype):
			raise TypeError(f"Default element cannot be of type {type(default_element)} for typesafe PyVector(<{dtype}>)")

		promote = False
		if initial: # Know it's a list with at least one item.
			try:
				init_dtypes = {x._dtype for x in initial}
			except:
				init_dtypes = {type(x) for x in initial}
			if dtype:
				init_dtypes.add(dtype)

			if len(init_dtypes) == 1:
				self._dtype = init_dtypes.pop()
			elif init_dtypes == {float, int} and self._dtype is not int:
				# if we specify int, cannot convert to float. Otherwise, convert ints to float
				self._dtype = float
				promote = True
			else:
				assert not typesafe
				self._dtype = None

		if promote:
			self._underlying = tuple(x + 0.0 if self._dtype == float else x for x in initial)
		else:
			self._underlying = tuple(initial)
		
		# Initialize fingerprint for O(1) change detection
		self._fp = self._compute_fp(self._underlying)
		self._fp_powers = [1]
		for _ in range(len(self._underlying)):
			self._fp_powers.append((self._fp_powers[-1] * self._FP_B) % self._FP_P)
		
		# Register with alias tracker after full initialization
		_ALIAS_TRACKER.register(self, id(self._underlying))

	@staticmethod
	def _compute_fp(data):
		"""Compute fingerprint from data"""
		P = PyVector._FP_P
		B = PyVector._FP_B
		fp = 0
		for x in data:
			# Recursively use fingerprint for nested PyVectors
			if hasattr(x, '_fp'):
				h = x._fp
			else:
				try:
					h = hash(x)
				except TypeError:
					# Unhashable type (like list) - convert to tuple
					h = hash(tuple(x)) if hasattr(x, '__iter__') else hash(str(x))
			fp = (fp * B + h) % P
		return fp


	def size(self):
		if not self:
			return tuple()
		# if isinstance(self._underlying[0], PyVector):
		# 	return (len(self._underlying),) + self._underlying[0].size()
		return (len(self),)

	def fingerprint(self):
		"""
		Get current fingerprint for change detection.
		For nested structures, recursively uses child fingerprints.
		Returns an integer hash that changes when the vector is modified.
		"""
		if self.ndims() == 1:
			# Simple vector: return stored fingerprint (O(1))
			return self._fp
		
		# Table/nested: hash from column fingerprints
		return hash(tuple(col.fingerprint() for col in self.cols()))

	def _update_fp_single(self, idx, old, new):
		"""O(1) update fingerprint when replacing element at index i"""
		# Get hashes
		if hasattr(old, '_fp'):
			old_hash = old._fp
		else:
			try:
				old_hash = hash(old)
			except TypeError:
				old_hash = hash(tuple(old)) if hasattr(old, '__iter__') else hash(str(old))
		
		if hasattr(new, '_fp'):
			new_hash = new._fp
		else:
			try:
				new_hash = hash(new)
			except TypeError:
				new_hash = hash(tuple(new)) if hasattr(new, '__iter__') else hash(str(new))
		
		# Update rolling hash
		diff = (new_hash - old_hash) * self._fp_powers[len(self) - idx - 1]
		self._fp = (self._fp + diff) % self._FP_P

	def _fp_update_multi(self, updates):
		"""O(k) fingerprint update for k changes. updates = list of (index, old_value, new_value)"""
		fp = self._fp
		for idx, old_v, new_v in updates:
			old_h = hash(old_v) if not hasattr(old_v, '_fp') else old_v._fp
			new_h = hash(new_v) if not hasattr(new_v, '_fp') else new_v._fp
			position_weight = self._fp_powers[len(self) - 1 - idx]
			diff = (new_h - old_h) * position_weight
			fp = (fp + diff) % self._FP_P
		self._fp = fp

	def rename(self, new_name):
		if new_name.startswith('_'):
			raise NameError("Vector names cannot begin with '_'")
		self._name = new_name

	@classmethod
	def new(cls, default_element, length, typesafe=False):
		""" create a new, initialized vector of length * default_element"""
		if length:
			assert isinstance(length, int)
			return cls([default_element for _ in range(length)], dtype=type(default_element), typesafe=typesafe)
		return cls(default_element=default_element, dtype=type(default_element), typesafe=typesafe)


	def copy(self, new_values = None, name=...):
		# Preserve name if not explicitly overridden
		# Use sentinel value (...) to distinguish between name=None (clear) and not passing name (preserve)
		use_name = self._name if name is ... else name
		return PyVector(list(new_values or self._underlying),
			default_element = self._default,
			dtype = self._dtype,
			typesafe = self._typesafe,
			name = use_name,
			as_row = self._display_as_row)

	def rename(self, new_name):
		"""Rename this vector (returns self for chaining)"""
		self._name = new_name
		return self

	def __repr__(self):
		return(_printr(self))

	@property
	def _(self):
		""" streamlined display """
		return ''


	def __iter__(self):
		""" iterate over the underlying tuple """
		return iter(self._underlying)

	def __len__(self):
		""" length of the underlying tuple """
		return len(self._underlying)

	@property
	def T(self):
		inverted = self.copy(name = self._name)
		inverted._display_as_row = not self._display_as_row
		return inverted


	def __getitem__(self, key):
		""" Get item(s) from self. Behavior varies by input type:
		The following return a PyVector:
			# PyVector of bool: Logical indexing (masking). Get all items where the boolean is True
			# List where every element is a bool. See PyVector of bool
			# Slice: return the array elements of the slice.

		Special: Indexing a single index returns a value
			# Int: 
		"""
		if isinstance(key, int):
			# Effectively a different input type (single not a list). Returning a value, not a vector.
			return self._underlying[key]

		if isinstance(key, tuple):
			if len(key) != len(self.size()):
				raise KeyError(f'Matrix indexing must provide an index in each dimension: {self.size()}')
			if len(key) == 1:
				return self[key[0]]
			return self._underlying[key[-1]][key[:-1]]

		key = self._check_duplicate(key)
		if isinstance(key, PyVector) and key._dtype == bool and key._typesafe:
			assert (len(self) == len(key))
			return self.copy((x for x, y in zip(self, key, strict=True) if y), name=self._name)
		if isinstance(key, list) and {type(e) for e in key} == {bool}:
			assert (len(self) == len(key))
			return self.copy((x for x, y in zip(self, key, strict=True) if y), name=self._name)
		if isinstance(key, slice):
			return self.copy(self._underlying[key], name=self._name)

		# NOT RECOMMENDED
		if isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return self.copy((self[x] for x in key), name=self._name)

		# NOT RECOMMENDED
		if isinstance(key, list) and {type(e) for e in key} == {int}:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return self.copy((self[x] for x in key), name=self._name)
		raise TypeError(f'Vector indices must be boolean vectors, integer vectors or integers, not {str(type(key))}')


	def __setitem__(self, key, value):
		"""
		Set the item at the specified index (key) with the provided value.
		Mutates self in place with alias tracking.
		Supports boolean indexing, slicing, and standard indexing.
		"""
		# Check for aliasing before any mutation
		_ALIAS_TRACKER.check_writable(self, id(self._underlying))
		
		key = self._check_duplicate(key)
		value = self._check_duplicate(value)
		
		# Collect updates as (index, new_value) pairs
		updates = []
		
		# Handle boolean vector or list as key
		if (isinstance(key, PyVector) and key._dtype == bool and key._typesafe) \
			or (isinstance(key, list) and {type(e) for e in key} == {bool}):
			
			assert len(self) == len(key), "Boolean mask length must match vector length."

			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				assert sum(key) == len(value), "Iterable length must match the number of True elements in the mask."
				iter_v = iter(value)
				for idx, mask_value in enumerate(key):
					if mask_value:
						updates.append((idx, next(iter_v)))
			else:
				for idx, mask_value in enumerate(key):
					if mask_value:
						updates.append((idx, value))
		
		# Handle slice assignment
		elif isinstance(key, slice):
			slice_len = slice_length(key, len(self))
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if slice_len != len(value):
					raise ValueError("Slice length and value length must be the same.")
				values_to_assign = value
			else:
				values_to_assign = [value for _ in range(slice_len)]

			start, stop, step = key.indices(len(self))
			indices = range(start, stop, step)
			
			for idx, new_val in zip(indices, values_to_assign):
				updates.append((idx, new_val))
		
		# Single integer index assignment
		elif isinstance(key, int):
			if not (-len(self) <= key < len(self)):
				raise IndexError(f"Index {key} is out of bounds for vector of length {len(self)}")
			updates.append((key, value))
		
		# Subscript indexing with PyVector of integers
		elif isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise ValueError("Number of indices must match the length of values.")
				for idx, val in zip(key, value):
					updates.append((idx, val))
			else:
				for idx in key:
					updates.append((idx, value))
		
		# List or tuple of integers
		elif isinstance(key, (list, tuple)) and {type(e) for e in key} == {int}:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise ValueError("Number of indices must match the length of values.")
				for idx, val in zip(key, value):
					updates.append((idx, val))
			else:
				for idx in key:
					updates.append((idx, value))
		else:
			raise TypeError(f"Invalid key type: {type(key)}. Must be boolean vector, integer vector, slice, or single index.")
		
		# Copy-on-write: convert to list, apply mutations, convert back to tuple
		data = list(self._underlying)
		
		# Apply mutations and track for fingerprint
		fp_updates = []
		for idx, new_val in updates:
			old_val = data[idx]
			data[idx] = new_val
			fp_updates.append((idx, old_val, new_val))
		
		# Convert back to tuple
		new_tuple = tuple(data)
		
		# Unregister old tuple identity before registering new one
		old_tuple_id = id(self._underlying)

		# Update and re-register with alias tracker
		_ALIAS_TRACKER.unregister(self, old_tuple_id)
		self._underlying = new_tuple
		_ALIAS_TRACKER.register(self, id(new_tuple))

		
		# Update fingerprint
		self._fp_update_multi(fp_updates)


	""" Comparison Operators - equality and hashing
		# __eq__ ==
		# __ge__ >=
		# __gt__ >
		# __lt__ <
		# __le__ <=
		# __ne__ !=
	"""
	def _elementwise_compare(self, other, op):
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), False, bool, True)
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), False, bool, True)
		return PyVector(tuple(bool(op(x, other)) for x in self), False, bool, True)

	# Now, we can redefine the comparison methods using the helper function

	def __eq__(self, other):
		return self._elementwise_compare(other, operator.eq)

	def __ge__(self, other):
		return self._elementwise_compare(other, operator.ge)

	def __gt__(self, other):
		return self._elementwise_compare(other, operator.gt)

	def __le__(self, other):
		return self._elementwise_compare(other, operator.le)

	def __lt__(self, other):
		return self._elementwise_compare(other, operator.lt)

	def __ne__(self, other):
		return self._elementwise_compare(other, operator.ne)

	def __and__(self, other):
		return self._elementwise_compare(other, operator.and_)

	def __or__(self, other):
		return self._elementwise_compare(other, operator.or_)

	def __xor__(self, other):
		return self._elementwise_compare(other, operator.xor)

	def __rand__(self, other):
		return self._elementwise_compare(other, operator.and_)

	def __ror__(self, other):
		return self._elementwise_compare(other, operator.or_)

	def __rxor__(self, other):
		return self._elementwise_compare(other, operator.xor)


	""" Math operations """

	def _elementwise_operation(self, other, op_func, op_name: str, op_symbol: str):
		"""Helper function to handle element-wise operations with broadcasting."""
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			assert len(self) == len(other)
			if self._typesafe:
				other._promote(self._dtype)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self.cols(), other.cols(), strict=True)),
							default_element=(op_func(self._default, other._default) if self._default is not None and other._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							name=None,
							typesafe=self._typesafe,
							as_row=self._display_as_row)

		if isinstance(other, self._dtype or object) or self._check_native_typesafe(other):
			return PyVector(tuple(op_func(x, other) for x in self._underlying),
							default_element=(op_func(self._default,  other) if self._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							typesafe=self._typesafe,
							name=None,
							as_row=self._display_as_row)

		if hasattr(other, '__iter__'):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)),
				self._default,
				self._dtype,
				self._typesafe,
				None,
				self._display_as_row
				)

		raise TypeError(f"Unsupported operand type(s) for '{op_symbol}': '{self._dtype.__name__}' and '{type(other).__name__}'.")

	def __add__(self, other):
		return self._elementwise_operation(other, operator.add, '__add__', '+')

	def __mul__(self, other):
		return self._elementwise_operation(other, operator.mul, '__mul__', '*')

	def __sub__(self, other):
		return self._elementwise_operation(other, operator.sub, '__sub__', '-')

	def __truediv__(self, other):
		return self._elementwise_operation(other, operator.truediv, '__truediv__', '/')

	def __floordiv__(self, other):
		return self._elementwise_operation(other, operator.floordiv, '__floordiv__', '//')

	def __mod__(self, other):
		return self._elementwise_operation(other, operator.mod, '__mod__', '%')

	def __pow__(self, other):
		return self._elementwise_operation(other, operator.pow, '__pow__', '**')

	def __radd__(self, other):
		""" Change behavior for strings """
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			assert len(self) == len(other)
			if self._typesafe:
				other._promote(self._dtype)
			return PyVector(tuple(y + x for x, y in zip(self, other, strict=True)),
							default_element=(other._default + self._default if self._default is not None and other._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							name=None,
							typesafe=self._typesafe,
							as_row=self._display_as_row)

		if isinstance(other, self._dtype or object) or self._check_native_typesafe(other):
			return PyVector(tuple(other + x for x in self.cols()),
							default_element=(other + self._default if self._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							name=None,
							typesafe=self._typesafe,
							as_row=self._display_as_row)

		if hasattr(other, '__iter__'):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)),
				self._default,
				self._dtype,
				self._typesafe,
				None,
				self._display_as_row)
		return self + other

	def __rmul__(self, other):
		return self.__mul__(other)

	def __rsub__(self, other):
		return self._elementwise_operation(other, lambda y, x: x - y, '__rsub__', '-')

	def __rtruediv__(self, other):
		return self._elementwise_operation(other, lambda y, x: x / y, '__rtruediv__', '/')

	def __rfloordiv__(self, other):
		return self._elementwise_operation(other, lambda y, x: x // y, '__rfloordiv__', '//')

	def __rmod__(self, other):
		return self._elementwise_operation(other, lambda y, x: x % y, '__rmod__', '%')

	def __rpow__(self, other):
		return self._elementwise_operation(other, lambda y, x: x ** y, '__rpow__', '**')




	def _promote(self, new_dtype):
		""" Check if a vector can change data type (int -> float, float -> complex) """
		if self._dtype == new_dtype:
			return
		if self._typesafe:
			raise TypeError(f'Cannot convert typesafe PyVector from {self._dtype.__name__} to {new_dtype.__name__}.')
		if new_dtype == float and self._dtype == int:
			self._underlying = tuple(float(x) for x in self._underlying)
			self._dtype = float
		if new_dtype == complex and self._dtype in (int, float):
			self._underlying = tuple(complex(x) for x in self._underlying)
			self._dtype = complex
		return

	def ndims(self):
		return len(self.size())

	def cols(self, key=None):
		if isinstance(key, int):
			return self._underlying[key]
		if isinstance(key, slice):
			print(key)
		return self._underlying

	"""
	Recursive Vector Operations
	"""
	def max(self):
		if self.ndims() == 2:
			return self.copy((c.max() for c in self.cols()), name=None).T
		return max(self)

	def min(self):
		if self.ndims() == 2:
			return self.copy((c.min() for c in self.cols()), name=None).T
		return min(self)

	def sum(self):
		if self.ndims() == 2:
			return self.copy((c.sum() for c in self.cols()), name=None).T
		return sum(self)


	def mean(self):
		if self.ndims() == 2:
			return self.copy((c.mean() for c in self.cols()), name=None).T
		return sum(self._underlying) / len(self._underlying)

	def stdev(self, population=False):
		if self.ndims() == 2:
			return self.copy((c.stdev(population) for c in self.cols()), name=None).T
		m = self.mean()

		# use in-place sum over generator for fastness. I AM SPEED!
		# This is still 10x slower than numpy.
		num = sum((x-m)*(x-m) for x in self._underlying)
		return (num/(len(self) - 1 + population))**0.5

	def unique(self):
		return {x for x in self}

	def argsort(self):
		return [i for i, _ in sorted(enumerate(self._underlying), key=lambda x: x[1])]


	def _check_duplicate(self, other):
		if id(self) == id(other):
			# If the object references match, we need to copy other
			# return PyVector((x for x in other), other._default, other._dtype, other._typesafe)
			return deepcopy(other)
		return other


	def _check_native_typesafe(self, other):
		""" Ensure native type conversions (python) will not affect underlying type """
		if not self._dtype:
			return True
		if self._dtype == type(other):
			return True
		if self._dtype == PyVector:
			return True
		if not (self._typesafe or hasattr(other, '__iter__')):
			return True
		if self._dtype == float and type(other) == int: # includes bool since isinstance(True, int) returns True
			return True
		if self._dtype == complex and type(other) in (int, float): # ditto
			return True
		return False


	def __matmul__(self, other):
		""" Recursive matrix multiplication - I think this applies to all tensor contraction, but could be wrong """
		other = self._check_duplicate(other)
		if isinstance(other, PyTable):
			return PyVector(tuple(self @ z for z in other.cols()))
		return sum(x*y for x, y in zip(self._underlying, other._underlying, strict=True))
		raise TypeError(f"Unsupported operand type(s) for '*': '{self._dtype.__name__}' and '{type(other).__name__}'.")
		# PyVector(tuple(PyVector(tuple(sum((u*v for u, v in zip(x._underlying, y._underlying))) for y in q.cols())) for x in p))


	def __rmatmul__(self, other):
		other = self._check_duplicate(other)
		if len(self.size()) > 1:
			return PyVector(tuple(x @ other for x in self.cols()))
		return sum(x*y for x, y in zip(self._underlying, other, strict=True))
		raise TypeError(f"Unsupported operand type(s) for '*': '{self._dtype.__name__}' and '{type(other).__name__}'.")


	def __bool__(self):
		""" We expect the behavior to mimic that of an empty list or string
		namely, if the underlying list (tuple) is empty, we return False, otherwise return True.

		The rationale here is that even a typed empty list is empty. Even a typed empty list with
		a default value is empty.
		"""
		if self._dtype == bool and self._typesafe:
			warnings.warn(f"For element-by-element operations use (&, |, ~) instead of 'and', 'or', 'not' keywords.")
		if self._underlying:
			return True
		return False


	def __lshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if self._dtype in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden. Use .bitshift() to shift bits.")

		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise TypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector(self._underlying + other._underlying,
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			return PyVector(self._underlying + list(other),
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)
		return PyVector(self._underlying + [other],
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)


	def __rshift__(self, other):
		""" The >> operator behavior has been overridden to add the column(s) of other to self
		"""
		if self._dtype in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden. Use .bitshift() to shift bits.")

		if isinstance(other, PyTable):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise TypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector((self,) + other.cols(),
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise TypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector((self,) + (other,),
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			return PyVector([self, PyVector(tuple(x for x in other))],
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)
		elif not self:
			return PyVector((other,),
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)
		raise TypeError("Cannot add a column of constant values. Try using PyVector.new(element, length).")

class PyTable(PyVector):
	""" Multiple columns of the same length """
	_length = None
	
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		# Handle dict initialization {name: values, ...}
		if isinstance(initial, dict):
			initial = [PyVector(values, name=col_name) for col_name, values in initial.items()]
		
		self._length = len(initial[0]) if initial else 0
		
		# Deep copy columns to enforce value semantics
		# Tables receive snapshots of vectors, preventing aliasing
		# Preserve names through the copy (name=... means preserve)
		initial = [vec.copy(name=vec._name) for vec in initial] if initial else ()
		
		return super().__init__(
			initial,
			default_element=default_element,
			dtype=dtype,
			typesafe=typesafe,
			name=name)

	def __len__(self):
		if not self:
			return 0
		if isinstance(self._underlying[0], PyTable):
			return len(self._underlying)
		return self._length

	def size(self):
		# if isinstance(self._underlying[0], PyVector):
		# 	return 
		return (len(self),) + self[0].size()

	def __dir__(self):
		sanitized_names = []
		seen = set()
		for i, col in enumerate(self._underlying):
			if col._name is not None:
				base = _sanitize_name(col._name)
				unique_name = _uniquify(base, seen)
				seen.add(unique_name)
				sanitized_names.append(unique_name)
			else:
				# Unnamed columns get col_N
				sanitized_names.append(f'col_{i+1}')
		return dir(PyVector) + sanitized_names

	def __getattr__(self, attr):
		# Build sanitized name mapping with uniquification
		attr_lower = attr.lower()
		seen = set()
		for col in self._underlying:
			if col._name is not None:
				base = _sanitize_name(col._name)
				unique_name = _uniquify(base, seen)
				seen.add(unique_name)
				if unique_name == attr_lower:
					return col
			
		# Try positional access for col_N pattern
		if attr.startswith('col_'):
			try:
				idx = int(attr[4:]) - 1  # col_1 -> index 0
				if 0 <= idx < len(self._underlying):
					return self._underlying[idx]
			except ValueError:
				pass
		
		return None

	def rename_column(self, old_name, new_name):
		"""Rename a column (modifies in place, returns self for chaining)"""
		for col in self._underlying:
			if col._name == old_name:
				col._name = new_name
				return self
		raise KeyError(f"Column '{old_name}' not found")
	
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
			raise ValueError("old_names and new_names must have the same length")

		# Simulate renames using a temporary list (avoid mid-state partial renames)
		simulated = [col._name for col in self._underlying]

		for old, new in zip(old_names, new_names):
			try:
				idx = simulated.index(old)
			except ValueError:
				raise KeyError(f"Column '{old}' not found")
			simulated[idx] = new  # simulate rename

		# Apply renames for real
		rename_idx = 0
		for old, new in zip(old_names, new_names):
			# rename the FIRST matching column in the real table
			for col in self._underlying:
				if col._name == old:
					col._name = new
					break

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
			for col in self._underlying:
				if col._name is not None and _sanitize_name(col._name) == key_lower:
					return col
			
			# Try uniquified sanitized names
			seen = set()
			for col in self._underlying:
				if col._name is not None:
					base = _sanitize_name(col._name)
					unique_name = _uniquify(base, seen)
					seen.add(unique_name)
					if unique_name == key_lower:
						return col
			
			raise KeyError(f"Column '{key}' not found")
		
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
					for col in self._underlying:
						if col._name is not None and _sanitize_name(col._name) == col_name_lower:
							selected_cols.append(col.copy())
							found = True
							break
				
				if not found:
					raise KeyError(f"Column '{col_name}' not found")
			return PyTable(selected_cols)
		
		if isinstance(key, tuple):
			if len(key) != len(self.size()):
				raise KeyError(f'Matrix indexing must provide an index in each dimension: {self.size()}')
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
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return PyVector(tuple(x[key] for x in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			)

	def __iter__(self):
		current = 0
		while current < len(self):
			yield self[current]
			current += 1


	def __repr__(self):
		return _printr(self)

	def __matmul__(self, other):
		if isinstance(other, PyVector): #reverse this soon
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
				raise TypeError("Cannot concatenate two typesafe PyVectors of different types")
			# complicated typesafety rules here - what if a whole bunch of things.
			return PyVector(self.cols() + other.cols(),
				self._default, # self does not inherit other's default element
				self._dtype or other._dtype,
				self._typesafe or other._typesafe)
		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise TypeError("Cannot concatenate two typesafe PyVectors of different types")
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
		raise TypeError("Cannot add a column of constant values. Try using PyVector.new(element, length).")

	def __lshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if isinstance(other, PyTable):
			return PyVector([x << y for x, y in zip(self._underlying, other._underlying, strict=True)])
		return PyVector([x << y for x, y in zip(self._underlying, other, strict=True)])

	def __rshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if isinstance(other, PyTable):
			return PyVector(self.cols() + other.cols())
		return PyVector(self.cols() + (other,))

class _PyFloat(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(
			initial,
			default_element=default_element,
			dtype=float,
			typesafe=typesafe,
			name=name,
			as_row=as_row)

	def __getattr__(self, name):
		""" get float attributes first (numerator, denominator, real, etc.) """
		if hasattr(float, name):
			return PyVector(tuple(s.__getattr__(name) for s in self._underlying))
		return super().__getattr__(name)

	def __setattr__(self, name, value):
		""" attempt to set float attributes before PyVector attributes
		this ording is mainly to prevent attribute name collisions with float attributes"""
		if hasattr(float, name):
			return PyVector(tuple(s.__setattr__(name, value) for s in self._underlying))
		return super().__setattr__(name, value)

	def __delattr__(self, name):
		""" attempt to remove float attributes before PyVector attributes """
		if hasattr(float, name):
			return PyVector(tuple(s.__delattr__(name) for s in self._underlying))
		return super().__delattr__(name)

	def as_integer_ratio(self, *args, **kwargs):
		""" Call the as_integer_ratio method on float """
		return PyVector([s.as_integer_ratio() for s in self._underlying])

	def conjugate(self, *args, **kwargs):
		""" Call the conjugate method on float """
		return PyVector([s.conjugate() for s in self._underlying])

	def fromhex(self, *args, **kwargs):
		""" Call the fromhex method on float """
		return PyVector([s.fromhex() for s in self._underlying])

	def hex(self, *args, **kwargs):
		""" Call the hex method on float """
		return PyVector([s.hex() for s in self._underlying])

	def is_integer(self, *args, **kwargs):
		""" Call the is_integer method on float """
		return PyVector([s.is_integer() for s in self._underlying])



class _PyInt(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(
			initial,
			default_element=default_element,
			dtype=int,
			typesafe=typesafe,
			name=name,
			as_row=as_row)

	def __getattr__(self, name):
		""" get integer attributes first (numerator, denominator, real, etc.) """
		if hasattr(int, name):
			return PyVector(tuple(s.__getattr__(name) for s in self._underlying))
		return super().__getattr__(name)

	def __setattr__(self, name, value):
		""" attempt to set integer attributes before PyVector attributes
		this ording is mainly to prevent attribute name collisions with integer attributes"""
		if hasattr(int, name):
			return PyVector(tuple(s.__setattr__(name, value) for s in self._underlying))
		return super().__setattr__(name, value)

	def __delattr__(self, name):
		""" attempt to remove integer attributes before PyVector attributes """
		if hasattr(int, name):
			return PyVector(tuple(s.__delattr__(name) for s in self._underlying))
		return super().__delattr__(name)

	def as_integer_ratio(self, *args, **kwargs):
		""" Call the as_integer_ratio method on int """
		return PyVector([s.as_integer_ratio() for s in self._underlying])

	def conjugate(self, *args, **kwargs):
		""" Call the conjugate method on int """
		return PyVector([s.conjugate() for s in self._underlying])

	def bit_count(self, *args, **kwargs):
		""" Call the bit_count method on int """
		return PyVector([s.bit_count() for s in self._underlying])

	def bit_length(self, *args, **kwargs):
		""" Call the bit_length method on int """
		return PyVector([s.bit_length() for s in self._underlying])

	def fromhex(self, *args, **kwargs):
		""" Call the fromhex method on int """
		return PyVector([s.fromhex() for s in self._underlying])

	def hex(self, *args, **kwargs):
		""" Call the hex method on int """
		return PyVector([s.hex() for s in self._underlying])

	def is_integer(self, *args, **kwargs):
		""" Call the is_integer method on int """
		return PyVector([s.is_integer() for s in self._underlying])

	def to_bytes(self, *args, **kwargs):
		""" Call the to_bytes method on int """
		return PyVector([s.to_bytes(*args, **kwargs) for s in self._underlying])


class _PyString(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(
			initial,
			default_element=default_element,
			dtype=str,
			typesafe=typesafe,
			name=name,
			as_row=as_row)

	def capitalize(self):
		""" Call the internal capitalize method on string """
		return PyVector([s.capitalize() for s in self._underlying])

	def casefold(self):
		""" Call the internal casefold method on string """
		return PyVector([s.casefold() for s in self._underlying])

	def center(self, *args, **kwargs):
		""" Call the internal center method on string """
		return PyVector([s.center(*args, **kwargs) for s in self._underlying])

	def count(self, *args, **kwargs):
		""" Call the internal count method on string """
		return PyVector([s.count(*args, **kwargs) for s in self._underlying])

	def encode(self, *args, **kwargs):
		""" Call the internal encode method on string """
		return PyVector([s.encode(*args, **kwargs) for s in self._underlying])

	def endswith(self, *args, **kwargs):
		""" Call the internal endswith method on string """
		return PyVector([s.endswith(*args, **kwargs) for s in self._underlying])

	def expandtabs(self, *args, **kwargs):
		""" Call the internal expandtabs method on string """
		return PyVector([s.expandtabs(*args, **kwargs) for s in self._underlying])

	def find(self, *args, **kwargs):
		""" Call the internal find method on string """
		return PyVector([s.find(*args, **kwargs) for s in self._underlying])

	def format(self, *args, **kwargs):
		""" Call the internal format method on string """
		return PyVector([s.format(*args, **kwargs) for s in self._underlying])

	def format_map(self, *args, **kwargs):
		""" Call the internal format_map method on string """
		return PyVector([s.format_map(*args, **kwargs) for s in self._underlying])

	def index(self, *args, **kwargs):
		""" Call the internal index method on string """
		return PyVector([s.index(*args, **kwargs) for s in self._underlying])

	def isalnum(self, *args, **kwargs):
		""" Call the internal isalnum method on string """
		return PyVector([s.isalnum(*args, **kwargs) for s in self._underlying])

	def isalpha(self, *args, **kwargs):
		""" Call the internal isalpha method on string """
		return PyVector([s.isalpha(*args, **kwargs) for s in self._underlying])

	def isascii(self, *args, **kwargs):
		""" Call the internal isascii method on string """
		return PyVector([s.isascii(*args, **kwargs) for s in self._underlying])

	def isdecimal(self, *args, **kwargs):
		""" Call the internal isdecimal method on string """
		return PyVector([s.isdecimal(*args, **kwargs) for s in self._underlying])

	def isdigit(self, *args, **kwargs):
		""" Call the internal isdigit method on string """
		return PyVector([s.isdigit(*args, **kwargs) for s in self._underlying])

	def isidentifier(self, *args, **kwargs):
		""" Call the internal isidentifier method on string """
		return PyVector([s.isidentifier(*args, **kwargs) for s in self._underlying])

	def islower(self, *args, **kwargs):
		""" Call the internal islower method on string """
		return PyVector([s.islower(*args, **kwargs) for s in self._underlying])

	def isnumeric(self, *args, **kwargs):
		""" Call the internal isnumeric method on string """
		return PyVector([s.isnumeric(*args, **kwargs) for s in self._underlying])

	def isprintable(self, *args, **kwargs):
		""" Call the internal isprintable method on string """
		return PyVector([s.isprintable(*args, **kwargs) for s in self._underlying])

	def isspace(self, *args, **kwargs):
		""" Call the internal isspace method on string """
		return PyVector([s.isspace(*args, **kwargs) for s in self._underlying])

	def istitle(self, *args, **kwargs):
		""" Call the internal istitle method on string """
		return PyVector([s.istitle(*args, **kwargs) for s in self._underlying])

	def isupper(self, *args, **kwargs):
		""" Call the internal isupper method on string """
		return PyVector([s.isupper(*args, **kwargs) for s in self._underlying])

	def join(self, *args, **kwargs):
		""" Call the internal join method on string """
		return PyVector([s.join(*args, **kwargs) for s in self._underlying])

	def ljust(self, *args, **kwargs):
		""" Call the internal ljust method on string """
		return PyVector([s.ljust(*args, **kwargs) for s in self._underlying])

	def lower(self, *args, **kwargs):
		""" Call the internal lower method on string """
		return PyVector([s.lower(*args, **kwargs) for s in self._underlying])

	def lstrip(self, *args, **kwargs):
		""" Call the internal lstrip method on string """
		return PyVector([s.lstrip(*args, **kwargs) for s in self._underlying])

	def maketrans(self, *args, **kwargs):
		""" Call the internal maketrans method on string """
		return PyVector([s.maketrans(*args, **kwargs) for s in self._underlying])

	def partition(self, *args, **kwargs):
		""" Call the internal partition method on string """
		return PyVector([s.partition(*args, **kwargs) for s in self._underlying])

	def removeprefix(self, *args, **kwargs):
		""" Call the internal removeprefix method on string """
		return PyVector([s.removeprefix(*args, **kwargs) for s in self._underlying])

	def removesuffix(self, *args, **kwargs):
		""" Call the internal removesuffix method on string """
		return PyVector([s.removesuffix(*args, **kwargs) for s in self._underlying])

	def replace(self, *args, **kwargs):
		""" Call the internal replace method on string """
		return PyVector([s.replace(*args, **kwargs) for s in self._underlying])

	def rfind(self, *args, **kwargs):
		""" Call the internal rfind method on string """
		return PyVector([s.rfind(*args, **kwargs) for s in self._underlying])

	def rindex(self, *args, **kwargs):
		""" Call the internal rindex method on string """
		return PyVector([s.rindex(*args, **kwargs) for s in self._underlying])

	def rjust(self, *args, **kwargs):
		""" Call the internal rjust method on string """
		return PyVector([s.rjust(*args, **kwargs) for s in self._underlying])

	def rpartition(self, *args, **kwargs):
		""" Call the internal rpartition method on string """
		return PyVector([s.rpartition(*args, **kwargs) for s in self._underlying])

	def rsplit(self, *args, **kwargs):
		""" Call the internal rsplit method on string """
		return PyVector([s.rsplit(*args, **kwargs) for s in self._underlying])

	def rstrip(self, *args, **kwargs):
		""" Call the internal rstrip method on string """
		return PyVector([s.rstrip(*args, **kwargs) for s in self._underlying])

	def split(self, *args, **kwargs):
		""" Call the internal split method on string """
		return PyVector([s.split(*args, **kwargs) for s in self._underlying])

	def splitlines(self, *args, **kwargs):
		""" Call the internal splitlines method on string """
		return PyVector([s.splitlines(*args, **kwargs) for s in self._underlying])

	def startswith(self, *args, **kwargs):
		""" Call the internal startswith method on string """
		return PyVector([s.startswith(*args, **kwargs) for s in self._underlying])

	def strip(self, *args, **kwargs):
		""" Call the internal strip method on string """
		return PyVector([s.strip(*args, **kwargs) for s in self._underlying])

	def swapcase(self, *args, **kwargs):
		""" Call the internal swapcase method on string """
		return PyVector([s.swapcase(*args, **kwargs) for s in self._underlying])

	def title(self, *args, **kwargs):
		""" Call the internal title method on string """
		return PyVector([s.title(*args, **kwargs) for s in self._underlying])

	def translate(self, *args, **kwargs):
		""" Call the internal translate method on string """
		return PyVector([s.translate(*args, **kwargs) for s in self._underlying])

	def upper(self, *args, **kwargs):
		""" Call the internal upper method on string """
		return PyVector([s.upper(*args, **kwargs) for s in self._underlying])

	def zfill(self, *args, **kwargs):
		""" Call the internal zfill method on string """
		return PyVector([s.zfill(*args, **kwargs) for s in self._underlying])


class _PyDate(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=None):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(
			initial,
			default_element=default_element,
			dtype=date,
			typesafe=typesafe,
			name=name,
			as_row=as_row)

	def _elementwise_compare(self, other, op):
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			# Raise mismatched lengths
			assert len(self) == len(other)
			if other._dtype == str:
				return PyVector(tuple(bool(op(x, date.fromisoformat(y))) for x, y in zip(self, other, strict=True)), False, bool, True)
			if other._dtype == datetime:
				return PyVector(tuple(bool(op(datetime.combine(x, datetime.time(0, 0)), y)) for x, y in zip(self, other, strict=True)), False, bool, True)
		elif hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			# Raise mismatched lengths
			assert len(self) == len(other)
			# If it's not a PyVector or Constant, don't apply date compare logic
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), False, bool, True)
		elif isinstance(other, str):
			return PyVector(tuple(bool(op(x, date.fromisoformat(other))) for x in self), False, bool, True)
		elif isinstance(other, datetime):
			return PyVector(tuple(bool(op(datetime.combine(x, datetime.time(0, 0)), other)) for x in self), False, bool, True)
		# finally, 
		return super()._elementwise_compare(other, op)


	def ctime(self, *args, **kwargs):
		return PyVector([s.ctime(*args, **kwargs) for s in self._underlying])

	def day(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector([s.day for s in self._underlying])

	def fromisocalendar(self, *args, **kwargs):
		return PyVector([s.fromisocalendar(*args, **kwargs) for s in self._underlying])

	def fromisoformat(self, *args, **kwargs):
		return PyVector([s.fromisoformat(*args, **kwargs) for s in self._underlying])

	def fromordinal(self, *args, **kwargs):
		return PyVector([s.fromordinal(*args, **kwargs) for s in self._underlying])

	def fromtimestamp(self, *args, **kwargs):
		return PyVector([s.fromtimestamp(*args, **kwargs) for s in self._underlying])

	def isocalendar(self, *args, **kwargs):
		return PyVector([s.isocalendar(*args, **kwargs) for s in self._underlying])

	def isoformat(self, *args, **kwargs):
		return PyVector([s.isoformat(*args, **kwargs) for s in self._underlying])

	def isoweekday(self, *args, **kwargs):
		return PyVector([s.isoweekday(*args, **kwargs) for s in self._underlying])

	def max(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector([s.max for s in self._underlying])

	def min(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector([s.min for s in self._underlying])

	def month(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector([s.month for s in self._underlying])

	def replace(self, *args, **kwargs):
		return PyVector([s.replace(*args, **kwargs) for s in self._underlying])

	def resolution(self):
		return PyVector([s.resolution for s in self._underlying])

	def strftime(self, *args, **kwargs):
		return PyVector([s.strftime(*args, **kwargs) for s in self._underlying])

	def timetuple(self, *args, **kwargs):
		return PyVector([s.timetuple(*args, **kwargs) for s in self._underlying])

	def today(self, *args, **kwargs):
		return PyVector([s.today(*args, **kwargs) for s in self._underlying])

	def toordinal(self, *args, **kwargs):
		return PyVector([s.toordinal(*args, **kwargs) for s in self._underlying])

	def weekday(self, *args, **kwargs):
		return PyVector([s.weekday(*args, **kwargs) for s in self._underlying])

	def year(self):
		return PyVector([s.year for s in self._underlying])

	def __add__(self, other):
		""" adding integers is adding days """
		if isinstance(other, PyVector) and other._dtype == int:
			return PyVector([date.fromordinal(s.toordinal() + y) for s, y in zip(self._underlying, other, strict=True)])
		if isinstance(other, int):
			return PyVector([date.fromordinal(s.toordinal() + other) for s in self._underlying])
		return super().add(other)

	def eomonth(self):
		return self
