import operator
import warnings
import re

from .alias_tracker import _ALIAS_TRACKER, AliasError
from .errors import PyVectorTypeError, PyVectorIndexError, PyVectorValueError, PyVectorKeyError
from .display import _printr
from .naming import _sanitize_user_name, _uniquify
from .typing import (
	DataType,
	infer_dtype,
	validate_scalar,
)
from copy import deepcopy
from datetime import date
from datetime import datetime
from .typeutils import slice_length


class MethodProxy:
	"""Proxy that defers method calls to each element in a PyVector."""
	def __init__(self, vector, method_name):
		self._vector = vector
		self._method_name = method_name
	
	def __call__(self, *args, **kwargs):
		"""Execute the method on each element and return a new PyVector."""
		return PyVector(tuple(getattr(elem, self._method_name)(*args, **kwargs) 
		                for elem in self._vector._underlying))


class PyVector():
	""" Iterable vector with optional type safety """
	dtype = None  # DType instance
	_underlying = None
	_name = None
	_display_as_row = False
	
	# Fingerprint constants for O(1) change detection
	_FP_P = (1 << 61) - 1  # Large prime (~2^61)
	_FP_B = 1315423911     # Base for rolling hash


	def __new__(cls, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		"""
		Decide what type of PyVector to create based on contents.
		
		Parameters
		----------
		initial : iterable
			Initial values
		dtype : DType, type, or None
			Type specification. Can be DType instance or Python type (int, float, str)
		name : str, optional
			Name for the vector
		as_row : bool, default False
			Display as row instead of column
		**kwargs : dict
			Backwards compatibility - accepts but ignores 'typesafe' and 'default_element'
		"""
		# Backwards compatibility: ignore old parameters
		if 'typesafe' in kwargs or 'default_element' in kwargs:
			pass  # Just ignore them
		
		# Handle backwards compatibility: dtype can be Python type or DataType
		if dtype is not None and not isinstance(dtype, DataType):
			# Old API: dtype=int, typesafe=True → new API: dtype=DataType(int)
			dtype = DataType(dtype)
		
		# Check if we're creating a PyTable (all elements are vectors of same length)
		if initial and all(isinstance(x, PyVector) for x in initial):
			if len({len(x) for x in initial}) == 1:
				from .table import PyTable
				return PyTable(initial=initial, dtype=dtype, name=name, as_row=as_row)
			warnings.warn('Passing vectors of different length will not produce a PyTable.')
		
		# Infer dtype if not provided
		if dtype is None and initial:
			dtype = infer_dtype(initial)
		# Empty vector keeps dtype=None for backwards compatibility
		
		# Dispatch to typed subclasses based on inferred dtype
		if dtype is not None and dtype.kind is str:
			return _PyString(initial=initial, dtype=dtype, name=name, as_row=as_row)
		if dtype is not None and dtype.kind is int:
			return _PyInt(initial=initial, dtype=dtype, name=name, as_row=as_row)
		if dtype is not None and dtype.kind is float:
			return _PyFloat(initial=initial, dtype=dtype, name=name, as_row=as_row)
		if dtype is not None and dtype.kind is date:
			return _PyDate(initial=initial, dtype=dtype, name=name, as_row=as_row)
		
		# Default: base PyVector
		return super(PyVector, cls).__new__(cls)



	def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		"""
		Create a new PyVector from an initial sequence.
		
		Parameters
		----------
		initial : iterable
			Initial values
		dtype : DType, type, or None
			Type specification
		name : str, optional
			Name for the vector
		as_row : bool, default False
			Display as row instead of column
		**kwargs : dict
			Backwards compatibility - accepts but ignores 'typesafe' and 'default_element'
		"""
		# Backwards compatibility: ignore old parameters
		if 'typesafe' in kwargs or 'default_element' in kwargs:
			pass  # Just ignore them
		
		# Handle backwards compatibility
		if dtype is not None and not isinstance(dtype, DataType):
			dtype = DataType(dtype)
		
		# Infer dtype if not provided
		if dtype is None and initial:
			dtype = infer_dtype(initial)
		# Keep dtype=None for empty vectors (backwards compatibility)
		
		self.dtype = dtype
		self._name = None
		if name is not None:
			self.rename(name)
		self._display_as_row = as_row

		# Convert initial to tuple
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

	@classmethod
	def new(cls, default_element, length, typesafe=False):
		""" create a new, initialized vector of length * default_element"""
		if length:
			assert isinstance(length, int)
			dtype = infer_dtype([default_element])
			if typesafe:
				dtype = dtype.with_nullable(False)
			return cls([default_element for _ in range(length)], dtype=dtype)
		dtype = infer_dtype([default_element]) if default_element is not None else DataType(object)
		if typesafe:
			dtype = dtype.with_nullable(False).with_default(default_element)
		return cls(dtype=dtype)


	def copy(self, new_values = None, name=...):
		# Preserve name if not explicitly overridden
		# Use sentinel value (...) to distinguish between name=None (clear) and not passing name (preserve)
		use_name = self._name if name is ... else name
		return PyVector(list(new_values or self._underlying),
			dtype = self.dtype,
			name = use_name,
			as_row = self._display_as_row)

	def rename(self, new_name):
		"""Rename this vector (returns self for chaining)"""
		self._name = new_name
		return self

	# Backwards compatibility properties
	@property
	def _dtype(self):
		"""Backwards compat: return Python type from DataType.kind"""
		if self.dtype is None:
			return None
		# DataType.kind is already a Python type!
		return self.dtype.kind
	
	@property
	def _typesafe(self):
		"""Backwards compat: non-nullable dtypes are 'typesafe'"""
		return self.dtype is not None and not self.dtype.nullable
	
	@property
	def _default(self):
		"""Backwards compat: get default value"""
		return self.dtype.default if self.dtype else None

	def __repr__(self):
		return(_printr(self))

	def cast(self, target_type):
		"""
		Convert each element to target_type.
		
		Parameters
		----------
		target_type : type
			The type to convert to (e.g., int, float, str)
		
		Returns
		-------
		PyVector
			New vector with converted elements
		
		Raises
		------
		ValueError
			If any element fails to convert (includes index of failure)
		
		Examples
		--------
		>>> v = PyVector(['10', '20', '30'])
		>>> v.cast(float)
		PyVector([10.0, 20.0, 30.0])
		"""
		# The "I hate Python Dates" interceptor
		if target_type is date:
			# Swap the constructor for the parser transparently
			target_type = date.fromisoformat
		elif target_type is datetime:
			# Might as well fix this one too while you're at it
			target_type = datetime.fromisoformat
		
		# Hot path: try conversion on all elements
		try:
			return PyVector(tuple(target_type(elem) for elem in self._underlying))
		
		# Error path: find exact failure point for helpful error message
		except (ValueError, TypeError) as e:
			for i, elem in enumerate(self._underlying):
				try:
					target_type(elem)
				except Exception:
					raise ValueError(
						f"Cast failed at index {i}: '{elem}' cannot be converted to {target_type.__name__}"
					) from e

	def fillna(self, value):
		"""
		Replace None values with a fill value.
		
		Parameters
		----------
		value : object
			Value to replace Nones with
		
		Returns
		-------
		PyVector
			New vector with Nones replaced
		
		Examples
		--------
		>>> v = PyVector([1, None, 3])
		>>> v.fillna(0)
		PyVector([1, 0, 3])
		"""
		return PyVector(tuple(value if elem is None else elem for elem in self._underlying), dtype=self.dtype.with_nullable(False))

	def dropna(self):
		"""
		Remove None values from the vector.
		
		Returns
		-------
		PyVector
			New vector with Nones removed
		
		Examples
		--------
		>>> v = PyVector([1, None, 3, None, 5])
		>>> v.dropna()
		PyVector([1, 3, 5])
		"""
		return PyVector(tuple(elem for elem in self._underlying if elem is not None), dtype=self.dtype.with_nullable(False))

	def isna(self):
		"""
		Return boolean mask of None values.
		
		Returns
		-------
		PyVector
			Boolean vector, True where value is None
		
		Examples
		--------
		>>> v = PyVector([1, None, 3])
		>>> v.isna()
		PyVector([False, True, False])
		"""
		return PyVector(tuple(elem is None for elem in self._underlying), dtype=DataType(bool))

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
				raise PyVectorKeyError(f"Matrix indexing must provide an index in each dimension: {self.size()}")
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
				warnings.warn('Subscript indexing is sub-optimal for large vectors; prefer slices or boolean masks')
			return self.copy((self[x] for x in key), name=self._name)

		# NOT RECOMMENDED
		if isinstance(key, list) and {type(e) for e in key} == {int}:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return self.copy((self[x] for x in key), name=self._name)
		raise PyVectorTypeError(f'Vector indices must be boolean vectors, integer vectors or integers, not {str(type(key))}')


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
					raise PyVectorValueError("Slice length and value length must be the same.")
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
				raise PyVectorIndexError(f"Index {key} is out of range for PyVector of length {len(self)}")
			updates.append((key, value))
		
		# Subscript indexing with PyVector of integers
		elif isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise PyVectorValueError("Number of indices must match the length of values.")
				for idx, val in zip(key, value):
					updates.append((idx, val))
			else:
				for idx in key:
					updates.append((idx, value))
		
		# List or tuple of integers
		elif isinstance(key, (list, tuple)) and {type(e) for e in key} == {int}:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise PyVectorValueError("Number of indices must match the length of values.")
				for idx, val in zip(key, value):
					updates.append((idx, val))
			else:
				for idx in key:
					updates.append((idx, value))
		else:
			raise PyVectorTypeError(f"Invalid key type: {type(key)}. Must be boolean vector, integer vector, slice, or single index.")
		
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
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), dtype=DataType(bool))
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), dtype=DataType(bool))
		return PyVector(tuple(bool(op(x, other)) for x in self), dtype=DataType(bool))

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
							dtype=self.dtype,
							name=None,
							as_row=self._display_as_row)

		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)),
				dtype=self.dtype,
				name=None,
				as_row=self._display_as_row
				)

		# Scalar operation - let Python handle type compatibility
		try:
			return PyVector(tuple(op_func(x, other) for x in self._underlying),
							dtype=self.dtype,
							name=None,
							as_row=self._display_as_row)
		except TypeError as e:
			raise PyVectorTypeError(f"Unsupported operand type(s) for '{op_symbol}': '{self._dtype.__name__}' and '{type(other).__name__}'.") from e

	def _unary_operation(self, op_func, op_name: str):
		"""Helper function to handle unary operations on each element."""
		return PyVector(
			tuple(op_func(x) for x in self),
			dtype=self.dtype,
			name=self._name,
			as_row=self._display_as_row
		)

	def __add__(self, other):
		return self._elementwise_operation(other, operator.add, '__add__', '+')

	def __mul__(self, other):
		return self._elementwise_operation(other, operator.mul, '__mul__', '*')

	def __sub__(self, other):
		return self._elementwise_operation(other, operator.sub, '__sub__', '-')

	def __neg__(self):
		return self._unary_operation(operator.neg, '__neg__')

	def __pos__(self):
		return self._unary_operation(operator.pos, '__pos__')

	def __abs__(self):
		return self._unary_operation(operator.abs, '__abs__')

	def __invert__(self):
		return self._unary_operation(operator.invert, '__invert__')

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
							dtype=self.dtype,
							name=None,
							as_row=self._display_as_row)

		if isinstance(other, self._dtype or object) or self._check_native_typesafe(other):
			return PyVector(tuple(other + x for x in self.cols()),
							dtype=self.dtype,
							name=None,
							as_row=self._display_as_row)

		if hasattr(other, '__iter__'):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)),
				dtype=self.dtype,
				name=None,
				as_row=self._display_as_row)
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
		# Handle both Python types and DataType instances
		if isinstance(new_dtype, DataType):
			target_kind = new_dtype.kind
		elif isinstance(new_dtype, type):
			# Python type like int, float
			target_kind = new_dtype
		else:
			# Old string-based API
			type_map = {
				"int": int, "float": float, "complex": complex,
				"string": str, "bool": bool,
				"date": date, "datetime": datetime,
			}
			target_kind = type_map.get(new_dtype, object)
		
		# Already the target type
		if self.dtype.kind is target_kind:
			return
		
		# Allow numeric promotions: int -> float, float -> complex
		if target_kind is float and self.dtype.kind is int:
			old_tuple_id = id(self._underlying)
			new_tuple = tuple(float(x) if x is not None else None for x in self._underlying)
			_ALIAS_TRACKER.unregister(self, old_tuple_id)
			self._underlying = new_tuple
			_ALIAS_TRACKER.register(self, id(new_tuple))
			self.dtype = DataType(float, nullable=self.dtype.nullable)
		elif target_kind is complex and self.dtype.kind in (int, float):
			old_tuple_id = id(self._underlying)
			new_tuple = tuple(complex(x) if x is not None else None for x in self._underlying)
			_ALIAS_TRACKER.unregister(self, old_tuple_id)
			self._underlying = new_tuple
			_ALIAS_TRACKER.register(self, id(new_tuple))
			self.dtype = DataType(complex, nullable=self.dtype.nullable)
		else:
			# For backwards compat, raise error if trying invalid promotion
			raise PyVectorTypeError(f'Cannot convert PyVector from {self.dtype.kind.__name__} to {target_kind.__name__}.')
		return

	def ndims(self):
		return len(self.size())

	def cols(self, key=None):
		if isinstance(key, int):
			return self._underlying[key]
		if isinstance(key, slice):
			return self._underlying[key]
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
		# Exclude None values from sum
		return sum(v for v in self._underlying if v is not None)


	def mean(self):
		if self.ndims() == 2:
			return self.copy((c.mean() for c in self.cols()), name=None).T
		# Exclude None values from mean
		non_none = [v for v in self._underlying if v is not None]
		return sum(non_none) / len(non_none) if non_none else None

	def stdev(self, population=False):
		if self.ndims() == 2:
			return self.copy((c.stdev(population) for c in self.cols()), name=None).T
		# Exclude None values from stdev
		non_none = [v for v in self._underlying if v is not None]
		if len(non_none) < 2:
			return None
		m = sum(non_none) / len(non_none)
		# use in-place sum over generator for fastness. I AM SPEED!
		# This is still 10x slower than numpy.
		num = sum((x-m)*(x-m) for x in non_none)
		return (num/(len(non_none) - 1 + population))**0.5

	def unique(self):
		return {x for x in self}

	def argsort(self):
		return [i for i, _ in sorted(enumerate(self._underlying), key=lambda x: x[1])]

	def pluck(self, key, default=None):
		"""Extract a key/index from each element, returning default if not found.
		
		Works with dicts, lists, tuples, strings, or any subscriptable object.
		"""
		results = []
		for item in self._underlying:
			# If item is None, can't subscript it
			if item is None:
				results.append(default)
				continue
			
			try:
				results.append(item[key])
			except (KeyError, IndexError, TypeError):
				# KeyError: dict key missing
				# IndexError: list/tuple index out of range
				# TypeError: item not subscriptable
				results.append(default)
		
		return PyVector(results)


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
		if type(other).__name__ == 'PyTable':
			return PyVector(tuple(self @ z for z in other.cols()))
		return sum(x*y for x, y in zip(self._underlying, other._underlying, strict=True))
		raise PyVectorTypeError(f"Unsupported operand type(s) for '*': '{self._dtype.__name__}' and '{type(other).__name__}'.")
		# PyVector(tuple(PyVector(tuple(sum((u*v for u, v in zip(x._underlying, y._underlying))) for y in q.cols())) for x in p))


	def __rmatmul__(self, other):
		other = self._check_duplicate(other)
		if len(self.size()) > 1:
			return PyVector(tuple(x @ other for x in self.cols()))
		return sum(x*y for x, y in zip(self._underlying, other, strict=True))
		raise PyVectorTypeError(f"Unsupported operand type(s) for '*': '{self._dtype.__name__}' and '{type(other).__name__}'.")


	def __bool__(self):
		""" We expect the behavior to mimic that of an empty list or string
		namely, if the underlying list (tuple) is empty, we return False, otherwise return True.

		The rationale here is that even a typed empty list is empty. Even a typed empty list with
		a default value is empty.
		
		Note: We intentionally allow __bool__ on boolean vectors (returns True if non-empty).
		The warning about using & instead of 'and' is handled elsewhere.
		"""
		if self._underlying:
			return True
		return False


	def __lshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append) the new array to the end of the first
		"""
		if self._dtype in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden for concatenation. Use .bitshift() to shift bits.")

		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			return PyVector(self._underlying + other._underlying,
				dtype=self.dtype)
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			return PyVector(self._underlying + tuple(other),
				dtype=self.dtype)
		return PyVector(self._underlying + (other,),
				dtype=self.dtype)


	def __rshift__(self, other):
		""" The >> operator behavior has been overridden to add the column(s) of other to self
		"""
		if self._dtype in (bool, int) and isinstance(other, int):
			warnings.warn(f"The behavior of >> and << have been overridden for concatenation. Use .bitshift() to shift bits.")

		if type(other).__name__ == 'PyTable':
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			return PyVector((self,) + other.cols(),
				dtype=self.dtype)
		if isinstance(other, PyVector):
			if self._typesafe and other._typesafe and self._dtype != other._dtype:
				raise PyVectorTypeError("Cannot concatenate two typesafe PyVectors of different types")
			return PyVector((self,) + (other,),
				dtype=self.dtype)
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
		raise PyVectorTypeError("Cannot add a column of constant values. Try using PyVector.new(element, length).")

	def __rlshift__(self, other):
		""" The << operator behavior has been overridden to attempt to concatenate (append)
		Handles: other << self (where other is not a PyVector)
		"""
		# Convert other to PyVector and concatenate with self
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			return PyVector(tuple(other) + self._underlying,
				None,  # other doesn't have a default element
				None,
				False)
		# Scalar case: [other] + self
		return PyVector((other,) + self._underlying,
			None,
			None,
			False)

	def __rrshift__(self, other):
		""" The >> operator behavior has been overridden to add columns
		Handles: other >> self (where other is not a PyVector)
		Creates a table with other as first column(s) and self as additional column(s)
		"""
		# Convert other to PyVector and combine column-wise
		if hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			return PyVector((PyVector(tuple(other)), self),
				None,
				None,
				False)
		# Scalar case: create a single-element vector for other
		return PyVector((PyVector((other,)), self),
			None,
			None,
			False)

	def __getattr__(self, name):
		"""Proxy attribute access to underlying dtype.
		
		Distinguishes between properties (like date.year) and methods (like str.replace):
		- Properties are evaluated immediately and return a PyVector
		- Methods return a MethodProxy that waits for () to be called
		"""
		# 1. If we are untyped (object), don't guess. Explicit > Implicit.
		if self._dtype is object:
			raise AttributeError(f"PyVector[object] has no attribute '{name}'")
		
		# 2. Inspect the class definition of the type we are holding
		# getattr(cls, name) returns the actual class member (method, property, slot)
		cls_attr = getattr(self._dtype, name, None)
		
		if cls_attr is None:
			# If the class doesn't have it, we definitely don't have it
			raise AttributeError(f"'{self._dtype.__name__}' object has no attribute '{name}'")
		
		# 3. Check if it's callable at the class level
		# If it's callable, it's a method. If not, it's a property/descriptor.
		if callable(cls_attr):
			# It's a method -> Return the proxy to wait for ()
			return MethodProxy(self, name)
		else:
			# It's a property/descriptor (like .year) -> Compute immediately
			return PyVector([getattr(x, name) for x in self._underlying])

class _PyFloat(PyVector):
	def __new__(cls, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		if dtype is None:
			dtype = DataType(float)
		return super().__init__(initial, dtype=dtype, name=name, as_row=as_row)


class _PyInt(PyVector):
	def __new__(cls, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		if dtype is None:
			dtype = DataType(int)
		return super().__init__(initial, dtype=dtype, name=name, as_row=as_row)


class _PyString(PyVector):
	def __new__(cls, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		if dtype is None:
			dtype = DataType(str)
		return super().__init__(initial, dtype=dtype, name=name, as_row=as_row)

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

	def before(self, sep):
		"""Return the part of each string before the first occurrence of sep."""
		return PyVector([s.partition(sep)[0] for s in self._underlying])

	def after(self, sep):
		"""Return the part of each string after the first occurrence of sep."""
		return PyVector([s.partition(sep)[2] for s in self._underlying])

	def before_last(self, sep):
		"""Return the part of each string before the last occurrence of sep."""
		return PyVector([s.rpartition(sep)[0] for s in self._underlying])

	def after_last(self, sep):
		"""Return the part of each string after the last occurrence of sep."""
		return PyVector([s.rpartition(sep)[2] for s in self._underlying])


class _PyDate(PyVector):
	def __new__(cls, initial=(), dtype=None, name=None, as_row=None, **kwargs):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
		if dtype is None:
			dtype = DataType(date)
		return super().__init__(initial, dtype=dtype, name=name, as_row=as_row)

	def _elementwise_compare(self, other, op):
		other = self._check_duplicate(other)
		if isinstance(other, PyVector):
			# Raise mismatched lengths
			assert len(self) == len(other)
			if other._dtype == str:
				return PyVector(tuple(bool(op(x, date.fromisoformat(y))) for x, y in zip(self, other, strict=True)), dtype=DataType(bool))
			if other._dtype == datetime:
				return PyVector(tuple(bool(op(datetime.combine(x, datetime.time(0, 0)), y)) for x, y in zip(self, other, strict=True)), dtype=DataType(bool))
		elif hasattr(other, '__iter__') and not isinstance(other, (str, bytes, bytearray)):
			# Raise mismatched lengths
			assert len(self) == len(other)
			# If it's not a PyVector or Constant, don't apply date compare logic
			return PyVector(tuple(bool(op(x, y)) for x, y in zip(self, other, strict=True)), dtype=DataType(bool))
		elif isinstance(other, str):
			return PyVector(tuple(bool(op(x, date.fromisoformat(other))) for x in self), dtype=DataType(bool))
		elif isinstance(other, datetime):
			return PyVector(tuple(bool(op(datetime.combine(x, datetime.time(0, 0)), other)) for x in self), dtype=DataType(bool))
		# finally, 
		return super()._elementwise_compare(other, op)


	def ctime(self, *args, **kwargs):
		return PyVector([s.ctime(*args, **kwargs) for s in self._underlying])

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

	def replace(self, *args, **kwargs):
		return PyVector([s.replace(*args, **kwargs) for s in self._underlying])

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

	def __add__(self, other):
		""" adding integers is adding days """
		if isinstance(other, PyVector) and other._dtype == int:
			return PyVector([date.fromordinal(s.toordinal() + y) for s, y in zip(self._underlying, other, strict=True)])
		if isinstance(other, int):
			return PyVector([date.fromordinal(s.toordinal() + other) for s in self._underlying])
		return super().add(other)

	def eomonth(self):
		return self
