import operator
import warnings

from copy import deepcopy
from datetime import date
from datetime import datetime


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


def _format_col(col, num_rows = 5):
	""" return a PyVector of formatted values """
	und = col._underlying if len(col)<=num_rows*2 else col._underlying[:num_rows] + col._underlying[-num_rows-1:]
	if col._dtype == float:
		x = PyVector([f"{v: g}" for v in und])
	elif col._dtype == int:
		x = PyVector([f"{v: d}" for v in und])
	elif col._dtype == str:
		x = PyVector([f"'{v}'" for v in und])
	else:
		x = PyVector([f"{v}" for v in und])
	max_len = {len(v) for v in x}
	return x.rjust(max(max_len))

def _str_to_repr(head, tail, joiner):
	pass

class PyVector():
	""" Iterable vector with optional type safety """
	_dtype = None
	_typesafe = None
	_default = None
	_underlying = None
	_name = None
	_display_as_row = False


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
		if initial and all(isinstance(x, float) for x in initial):
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
		if name:
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
		elif isinstance(initial, tuple):
			self._underlying = initial
		else:
			self._underlying = tuple(x for x in initial)


	def size(self):
		if not self:
			return tuple()
		# if isinstance(self._underlying[0], PyVector):
		# 	return (len(self._underlying),) + self._underlying[0].size()
		return (len(self),)

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


	def copy(self, new_values = None):
		return PyVector(tuple(x for x in (new_values or self._underlying)),
			default_element = self._default,
			dtype = self._dtype,
			typesafe = self._typesafe,
			as_row = self._display_as_row)

	def __repr__(self):
		joiner = ', ' if self._display_as_row else ',\n  '
		if len(self) == 0:
			return 'PyVector([])'
		elif len(self) <= 10:
			return 'PyVector([\n  ' + joiner.join([str(x) for x in self._underlying]) + '\n]'
		return '[\n  ' + joiner.join([str(x) for x in self._underlying[:5]]) + \
		    joiner + '...' + joiner[1:] + joiner.join([str(x) for x in self._underlying[-5:]]) + \
		    '\n]' + f' # {len(self)} {self._dtype.__name__}'


	def __iter__(self):
		""" iterate over the underlying tuple """
		return iter(self._underlying)

	def __len__(self):
		""" length of the underlying tuple """
		return len(self._underlying)

	def T(self):
		inverted = self.copy()
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
			return self.copy((x for x, y in zip(self, key, strict=True) if y))
		if isinstance(key, list) and {type(e) for e in key} == {bool}:
			assert (len(self) == len(key))
			return self.copy((x for x, y in zip(self, key, strict=True) if y))
		if isinstance(key, slice):
			return self.copy(self._underlying[key])

		# NOT RECOMMENDED
		if isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return self.copy((self[x] for x in key))

		# NOT RECOMMENDED
		if isinstance(key, list, ) and {type(e) for e in key} == {int}:
			if len(self) > 1000:
				warnings.warn('Subscript indexing is sub-optimal for large vectors')
			return self.copy((self[x] for x in key))
		raise TypeError(f'Vector indices must be boolean vectors, integer vectors or integers, not {str(type(key))}')


	def __setitem__(self, key, value):
		"""
		Set the item at the specified index (key) with the provided value.
		Supports boolean indexing, slicing, and standard indexing.
		
		- If key is a boolean mask, assigns values where True.
		- If key is a slice, assigns values in the slice.
		- Handles type safety through PyVector, and ensures lengths match when necessary.
		"""
		# Perform duplicate checks on key and value, assuming self._check_duplicate handles it.
		key = self._check_duplicate(key)
		value = self._check_duplicate(value)
		
		# Handle boolean vector or list as key
		if (isinstance(key, PyVector) and key._dtype == bool and key._typesafe) \
			or (isinstance(key, list) and {type(e) for e in key} == {bool}):
			
			# Ensure the key (mask) is the same length as the current vector
			assert len(self) == len(key), "Boolean mask length must match vector length."

			# If value is an iterable, ensure its length matches the number of True elements in key
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				assert sum(key) == len(value), "Iterable length must match the number of True elements in the mask."
				
				# Replace based on mask (True/False positions)
				iter_v = iter(value)
				new_vector = PyVector(
					tuple(next(iter_v) if t else x for x, t in zip(self._underlying, key)),
					default_element=self._default,
					dtype=self._dtype,
					typesafe=self._typesafe
				)
			else:
				# Replace all True positions with the same value
				new_vector = PyVector(
					tuple(value if t else x for x, t in zip(self._underlying, key)),
					default_element=self._default,
					dtype=self._dtype,
					typesafe=self._typesafe
				)
			
			# Update the underlying data
			self._underlying = new_vector._underlying
			self._dtype = new_vector._dtype
			return

		# Handle slice assignment
		if isinstance(key, slice):
			# Ensure that if value is an iterable, its length matches the slice length
			slice_len = slice_length(key, len(self))
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if slice_len != len(value):
					raise ValueError("Slice length and value length must be the same.")
			else:
				# need an iterable of the same length as the slice
				value = [value for _ in range(slice_len)]

			# Assign values in the slice
			tuple_as_list = list(self._underlying)
			tuple_as_list[key] = value
			
			# Create a new PyVector to handle type safety and consistency
			new_vector = self.copy(tuple(tuple_as_list))
			
			# Update the underlying data
			self._underlying = new_vector._underlying
			self._dtype = new_vector._dtype
			return

		# Single integer index assignment
		if isinstance(key, int):
			# Ensure the index is valid
			if not (0 <= key < len(self)):
				raise IndexError(f"Index {key} is out of bounds for vector of length {len(self)}")
			tuple_as_list = list(self._underlying)
			tuple_as_list[key] = value

			 # Create a new PyVector to handle type safety and consistency
			new_vector = self.copy(tuple(tuple_as_list))
			
			# Update the underlying data
			self._underlying = new_vector._underlying
			self._dtype = new_vector._dtype
			return

		# Subscript indexing with PyVector of integers
		if isinstance(key, PyVector) and key._dtype == int and key._typesafe:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise ValueError("Number of indices must match the length of values.")
				tuple_as_list = list(self._underlying)
				for idx, val in zip(key, value):
					tuple_as_list[idx] = val
			else:
				tuple_as_list = list(self._underlying)
				for idx in key:
					tuple_as_list[idx] = value
			new_vector = self.copy(tuple(tuple_as_list))
			
			# Update the underlying data
			self._underlying = new_vector._underlying
			self._dtype = new_vector._dtype
			return

		# List or tuple of integers
		if isinstance(key, (list, tuple)) and {type(e) for e in key} == {int}:
			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
				if len(key) != len(value):
					raise ValueError("Number of indices must match the length of values.")
				tuple_as_list = list(self._underlying)
				for idx, val in zip(key, value):
					tuple_as_list[idx] = val
			else:
				tuple_as_list = list(self._underlying)
				for idx in key:
					tuple_as_list[idx] = value
			new_vector = self.copy(tuple(tuple_as_list))
			
			# Update the underlying data
			self._underlying = new_vector._underlying
			self._dtype = new_vector._dtype
			return

		# If none of the cases match, raise a TypeError
		raise TypeError(f"Invalid key type: {type(key)}. Must be boolean vector, integer vector, slice, or single index.")


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
							typesafe=self._typesafe)

		if isinstance(other, self._dtype or object) or self._check_native_typesafe(other):
			return PyVector(tuple(op_func(x, other) for x in self._underlying),
							default_element=(op_func(self._default,  other) if self._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							typesafe=self._typesafe)

		if hasattr(other, '__iter__'):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)), self._default, self._dtype, self._typesafe)

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
							typesafe=self._typesafe)

		if isinstance(other, self._dtype or object) or self._check_native_typesafe(other):
			return PyVector(tuple(other + x for x in self.cols()),
							default_element=(other + self._default if self._default is not None else None),
							dtype=self._dtype if self._typesafe else None,
							typesafe=self._typesafe)

		if hasattr(other, '__iter__'):
			assert len(self) == len(other)
			return PyVector(tuple(op_func(x, y) for x, y in zip(self, other, strict=True)), self._default, self._dtype, self._typesafe)
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
		if self.ndims() > 1:
			return self.copy((c.max() for c in self.cols()))
		return max(self)

	def min(self):
		if self.ndims() > 1:
			return self.copy((c.min() for c in self.cols()))
		return min(self)

	def sum(self):
		if self.ndims() > 1:
			return self.copy((c.sum() for c in self.cols()))
		return sum(self)


	def mean(self):
		if self.ndims() > 1:
			return self.copy((c.mean() for c in self.cols()))
		return sum(self._underlying) / len(self._underlying)

	def stdev(self, population=False):
		if self.ndims() > 1:
			return self.copy((c.stdev(population) for c in self.cols()))
		m = self.mean()

		# use in-place sum over generator for fastness. I AM SPEED!
		# This is still 10x slower than numpy.
		num = sum((x-m)*(x-m) for x in self._underlying)
		return (num/(len(self) - 1 + population))**0.5

	def unique(self):
		return {x for x in self}

	def argsort(self):
		return [i for i, _ in sorted(enumerate(self._underlying), key=lambda x: x[1])]


	def __hash__(self):
		return hash((
			self._dtype,
			self._typesafe,
			self._default,
			self._underlying,
			self._name))


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
			return PyVector(self._underlying + tuple(x for x in other),
				self._default, # self does not inherit other's default element
				self._dtype,
				self._typesafe)
		return PyVector(self._underlying + (other,),
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
		self._length = len(initial[0]) if initial else 0
		return super().__init__(initial, default_element=default_element, dtype=dtype, typesafe=typesafe, name=name)

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
		return dir(PyVector) + [c._name for c in self._underlying]

	def __getattr__(self, attr):
		if attr in [c._name for c in self._underlying]:
			idx = [c._name for c in self._underlying].index(attr)
			return self._underlying[idx]

	def T(self):
		if len(self.size())==2:
			return self.copy((tuple(x.T() for x in self))) # rows
		return self.copy((tuple(x.T() for x in self))) # rows

	def __getitem__(self, key):
		key = self._check_duplicate(key)
		if isinstance(key, tuple):
			if len(key) != len(self.size()):
				raise KeyError(f'Matrix indexing must provide an index in each dimension: {self.size()}')
			# for now.
			if len(key) > 2:
				return self[key[0]][key[1:]]
			if isinstance(key[1], slice):
				if isinstance(key[0], slice):
					return self.copy(self[key[0]]._underlying[key[1]])
				return self[key[0]][key[1]]
			return self[key[0]]._underlying[key[1]]



		if isinstance(key, int):
			# Effectively a different input type (single not a list). Returning a value, not a vector.
			if isinstance(self._underlying[0], PyTable):
				return self._underlying[key]
			return PyVector(tuple(col[key] for col in self._underlying),
				default_element = self._default,
				dtype = self._dtype,
				typesafe = self._typesafe
			).T()

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
				typesafe = self._typesafe
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
		out = '['
		if len(self) == 0:
			return '[[]]'
		elif len(self._underlying) <= 10:
			for x in self._underlying[:-1]:
				out += _format_col(x) + ', '
			out += _format_col(self._underlying[-1]) + ']'
		else:
			cols = self._underlying
			for x in cols[:4]:
				out += _format_col(x) + ', '
			out += _format_col(cols[4])
			out += ' ... '
			for x in cols[-5:-1]:
				out += _format_col(x) + ', '
			out += _format_col(cols[-1]) + ']'

		return repr(out)

	def __matmul__(self, other):
		if isinstance(other, PyVector): #reverse this soon
			# we want the sum to operate 'in place' on integers
			if other.ndims() == 1:
				return PyVector(tuple(sum(u*v for u, v in zip(s._underlying, other._underlying)) for s in self))
			# return PyVector(tuple(self@c for c in other.cols()))
			# return PyVector(tuple(PyVector(tuple(sum((u*v for u, v in zip(x._underlying, y._underlying))) for x in other.cols())) for y in self)).T()
			return self.copy(tuple(self.copy(tuple(x@y for x in other.cols())) for y in self)).T()
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
			return PyVector(tuple(op(x, y) for x, y in zip(self, other, strict=True)), False, bool, True).T()
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
		return super().__init__(initial, default_element=default_element, dtype=float, typesafe=typesafe, name=name)

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
		return PyVector(tuple(s.as_integer_ratio() for s in self._underlying))

	def conjugate(self, *args, **kwargs):
		""" Call the conjugate method on float """
		return PyVector(tuple(s.conjugate() for s in self._underlying))

	def fromhex(self, *args, **kwargs):
		""" Call the fromhex method on float """
		return PyVector(tuple(s.fromhex() for s in self._underlying))

	def hex(self, *args, **kwargs):
		""" Call the hex method on float """
		return PyVector(tuple(s.hex() for s in self._underlying))

	def is_integer(self, *args, **kwargs):
		""" Call the is_integer method on float """
		return PyVector(tuple(s.is_integer() for s in self._underlying))



class _PyInt(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(initial, default_element=default_element, dtype=int, typesafe=typesafe, name=name, as_row=as_row)

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
		return PyVector(tuple(s.as_integer_ratio() for s in self._underlying))

	def conjugate(self, *args, **kwargs):
		""" Call the conjugate method on int """
		return PyVector(tuple(s.conjugate() for s in self._underlying))

	def bit_count(self, *args, **kwargs):
		""" Call the bit_count method on int """
		return PyVector(tuple(s.bit_count() for s in self._underlying))

	def bit_length(self, *args, **kwargs):
		""" Call the bit_length method on int """
		return PyVector(tuple(s.bit_length() for s in self._underlying))

	def fromhex(self, *args, **kwargs):
		""" Call the fromhex method on int """
		return PyVector(tuple(s.fromhex() for s in self._underlying))

	def hex(self, *args, **kwargs):
		""" Call the hex method on int """
		return PyVector(tuple(s.hex() for s in self._underlying))

	def is_integer(self, *args, **kwargs):
		""" Call the is_integer method on int """
		return PyVector(tuple(s.is_integer() for s in self._underlying))

	def to_bytes(self, *args, **kwargs):
		""" Call the to_bytes method on int """
		return PyVector(tuple(s.to_bytes() for s in self._underlying))


class _PyString(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=False):
		return super().__init__(initial, default_element=default_element, dtype=str, typesafe=typesafe, name=name, as_row=as_row)

	def capitalize(self):
		""" Call the internal capitalize method on string """
		return PyVector(tuple(s.capitalize() for s in self._underlying))

	def casefold(self):
		""" Call the internal casefold method on string """
		return PyVector(tuple(s.casefold() for s in self._underlying))

	def center(self, *args, **kwargs):
		""" Call the internal center method on string """
		return PyVector(tuple(s.center(*args, **kwargs) for s in self._underlying))

	def count(self, *args, **kwargs):
		""" Call the internal count method on string """
		return PyVector(tuple(s.count(*args, **kwargs) for s in self._underlying))

	def encode(self, *args, **kwargs):
		""" Call the internal encode method on string """
		return PyVector(tuple(s.encode(*args, **kwargs) for s in self._underlying))

	def endswith(self, *args, **kwargs):
		""" Call the internal endswith method on string """
		return PyVector(tuple(s.endswith(*args, **kwargs) for s in self._underlying))

	def expandtabs(self, *args, **kwargs):
		""" Call the internal expandtabs method on string """
		return PyVector(tuple(s.expandtabs(*args, **kwargs) for s in self._underlying))

	def find(self, *args, **kwargs):
		""" Call the internal find method on string """
		return PyVector(tuple(s.find(*args, **kwargs) for s in self._underlying))

	def format(self, *args, **kwargs):
		""" Call the internal format method on string """
		return PyVector(tuple(s.format(*args, **kwargs) for s in self._underlying))

	def format_map(self, *args, **kwargs):
		""" Call the internal format_map method on string """
		return PyVector(tuple(s.format_map(*args, **kwargs) for s in self._underlying))

	def index(self, *args, **kwargs):
		""" Call the internal index method on string """
		return PyVector(tuple(s.index(*args, **kwargs) for s in self._underlying))

	def isalnum(self, *args, **kwargs):
		""" Call the internal isalnum method on string """
		return PyVector(tuple(s.isalnum(*args, **kwargs) for s in self._underlying))

	def isalpha(self, *args, **kwargs):
		""" Call the internal isalpha method on string """
		return PyVector(tuple(s.isalpha(*args, **kwargs) for s in self._underlying))

	def isascii(self, *args, **kwargs):
		""" Call the internal isascii method on string """
		return PyVector(tuple(s.isascii(*args, **kwargs) for s in self._underlying))

	def isdecimal(self, *args, **kwargs):
		""" Call the internal isdecimal method on string """
		return PyVector(tuple(s.isdecimal(*args, **kwargs) for s in self._underlying))

	def isdigit(self, *args, **kwargs):
		""" Call the internal isdigit method on string """
		return PyVector(tuple(s.isdigit(*args, **kwargs) for s in self._underlying))

	def isidentifier(self, *args, **kwargs):
		""" Call the internal isidentifier method on string """
		return PyVector(tuple(s.isidentifier(*args, **kwargs) for s in self._underlying))

	def islower(self, *args, **kwargs):
		""" Call the internal islower method on string """
		return PyVector(tuple(s.islower(*args, **kwargs) for s in self._underlying))

	def isnumeric(self, *args, **kwargs):
		""" Call the internal isnumeric method on string """
		return PyVector(tuple(s.isnumeric(*args, **kwargs) for s in self._underlying))

	def isprintable(self, *args, **kwargs):
		""" Call the internal isprintable method on string """
		return PyVector(tuple(s.isprintable(*args, **kwargs) for s in self._underlying))

	def isspace(self, *args, **kwargs):
		""" Call the internal isspace method on string """
		return PyVector(tuple(s.isspace(*args, **kwargs) for s in self._underlying))

	def istitle(self, *args, **kwargs):
		""" Call the internal istitle method on string """
		return PyVector(tuple(s.istitle(*args, **kwargs) for s in self._underlying))

	def isupper(self, *args, **kwargs):
		""" Call the internal isupper method on string """
		return PyVector(tuple(s.isupper(*args, **kwargs) for s in self._underlying))

	def join(self, *args, **kwargs):
		""" Call the internal join method on string """
		return PyVector(tuple(s.join(*args, **kwargs) for s in self._underlying))

	def ljust(self, *args, **kwargs):
		""" Call the internal ljust method on string """
		return PyVector(tuple(s.ljust(*args, **kwargs) for s in self._underlying))

	def lower(self, *args, **kwargs):
		""" Call the internal lower method on string """
		return PyVector(tuple(s.lower(*args, **kwargs) for s in self._underlying))

	def lstrip(self, *args, **kwargs):
		""" Call the internal lstrip method on string """
		return PyVector(tuple(s.lstrip(*args, **kwargs) for s in self._underlying))

	def maketrans(self, *args, **kwargs):
		""" Call the internal maketrans method on string """
		return PyVector(tuple(s.maketrans(*args, **kwargs) for s in self._underlying))

	def partition(self, *args, **kwargs):
		""" Call the internal partition method on string """
		return PyVector(tuple(s.partition(*args, **kwargs) for s in self._underlying))

	def removeprefix(self, *args, **kwargs):
		""" Call the internal removeprefix method on string """
		return PyVector(tuple(s.removeprefix(*args, **kwargs) for s in self._underlying))

	def removesuffix(self, *args, **kwargs):
		""" Call the internal removesuffix method on string """
		return PyVector(tuple(s.removesuffix(*args, **kwargs) for s in self._underlying))

	def replace(self, *args, **kwargs):
		""" Call the internal replace method on string """
		return PyVector(tuple(s.replace(*args, **kwargs) for s in self._underlying))

	def rfind(self, *args, **kwargs):
		""" Call the internal rfind method on string """
		return PyVector(tuple(s.rfind(*args, **kwargs) for s in self._underlying))

	def rindex(self, *args, **kwargs):
		""" Call the internal rindex method on string """
		return PyVector(tuple(s.rindex(*args, **kwargs) for s in self._underlying))

	def rjust(self, *args, **kwargs):
		""" Call the internal rjust method on string """
		return PyVector(tuple(s.rjust(*args, **kwargs) for s in self._underlying))

	def rpartition(self, *args, **kwargs):
		""" Call the internal rpartition method on string """
		return PyVector(tuple(s.rpartition(*args, **kwargs) for s in self._underlying))

	def rsplit(self, *args, **kwargs):
		""" Call the internal rsplit method on string """
		return PyVector(tuple(s.rsplit(*args, **kwargs) for s in self._underlying))

	def rstrip(self, *args, **kwargs):
		""" Call the internal rstrip method on string """
		return PyVector(tuple(s.rstrip(*args, **kwargs) for s in self._underlying))

	def split(self, *args, **kwargs):
		""" Call the internal split method on string """
		return PyVector(tuple(s.split(*args, **kwargs) for s in self._underlying))

	def splitlines(self, *args, **kwargs):
		""" Call the internal splitlines method on string """
		return PyVector(tuple(s.splitlines(*args, **kwargs) for s in self._underlying))

	def startswith(self, *args, **kwargs):
		""" Call the internal startswith method on string """
		return PyVector(tuple(s.startswith(*args, **kwargs) for s in self._underlying))

	def strip(self, *args, **kwargs):
		""" Call the internal strip method on string """
		return PyVector(tuple(s.strip(*args, **kwargs) for s in self._underlying))

	def swapcase(self, *args, **kwargs):
		""" Call the internal swapcase method on string """
		return PyVector(tuple(s.swapcase(*args, **kwargs) for s in self._underlying))

	def title(self, *args, **kwargs):
		""" Call the internal title method on string """
		return PyVector(tuple(s.title(*args, **kwargs) for s in self._underlying))

	def translate(self, *args, **kwargs):
		""" Call the internal translate method on string """
		return PyVector(tuple(s.translate(*args, **kwargs) for s in self._underlying))

	def upper(self, *args, **kwargs):
		""" Call the internal upper method on string """
		return PyVector(tuple(s.upper(*args, **kwargs) for s in self._underlying))

	def zfill(self, *args, **kwargs):
		""" Call the internal zfill method on string """
		return PyVector(tuple(s.zfill(*args, **kwargs) for s in self._underlying))


class _PyDate(PyVector):
	def __new__(cls, initial=(), default_element=None, dtype=None, typesafe=False, name=None, as_row=None):
		return super(PyVector, cls).__new__(cls)

	def __init__(self, initial=(), default_element=None, dtype=None, typesafe=False, name=None):
		return super().__init__(initial, default_element=default_element, dtype=date, typesafe=typesafe, name=name, as_row=as_row)

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
		return PyVector(tuple(s.ctime(*args, **kwargs) for s in self._underlying))

	def day(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector(tuple(s.day for s in self._underlying))

	def fromisocalendar(self, *args, **kwargs):
		return PyVector(tuple(s.fromisocalendar(*args, **kwargs) for s in self._underlying))

	def fromisoformat(self, *args, **kwargs):
		return PyVector(tuple(s.fromisoformat(*args, **kwargs) for s in self._underlying))

	def fromordinal(self, *args, **kwargs):
		return PyVector(tuple(s.fromordinal(*args, **kwargs) for s in self._underlying))

	def fromtimestamp(self, *args, **kwargs):
		return PyVector(tuple(s.fromtimestamp(*args, **kwargs) for s in self._underlying))

	def isocalendar(self, *args, **kwargs):
		return PyVector(tuple(s.isocalendar(*args, **kwargs) for s in self._underlying))

	def isoformat(self, *args, **kwargs):
		return PyVector(tuple(s.isoformat(*args, **kwargs) for s in self._underlying))

	def isoweekday(self, *args, **kwargs):
		return PyVector(tuple(s.isoweekday(*args, **kwargs) for s in self._underlying))

	def max(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector(tuple(s.max for s in self._underlying))

	def min(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector(tuple(s.min for s in self._underlying))

	def month(self):
		# attributes must be converted to functions to avoid name collisions.
		return PyVector(tuple(s.month for s in self._underlying))

	def replace(self, *args, **kwargs):
		return PyVector(tuple(s.replace(*args, **kwargs) for s in self._underlying))

	def resolution(self):
		return PyVector(tuple(s.resolution for s in self._underlying))

	def strftime(self, *args, **kwargs):
		return PyVector(tuple(s.strftime(*args, **kwargs) for s in self._underlying))

	def timetuple(self, *args, **kwargs):
		return PyVector(tuple(s.timetuple(*args, **kwargs) for s in self._underlying))

	def today(self, *args, **kwargs):
		return PyVector(tuple(s.today(*args, **kwargs) for s in self._underlying))

	def toordinal(self, *args, **kwargs):
		return PyVector(tuple(s.toordinal(*args, **kwargs) for s in self._underlying))

	def weekday(self, *args, **kwargs):
		return PyVector(tuple(s.weekday(*args, **kwargs) for s in self._underlying))

	def year(self):
		return PyVector(tuple(s.year for s in self._underlying))

	def __add__(self, other):
		""" adding integers is adding days """
		if isinstance(other, PyVector) and other._dtype == int:
			return PyVector(tuple(date.fromordinal(s.toordinal() + y) for s, y in zip(self._underlying, other, strict=True)))
		if isinstance(other, int):
			return PyVector(tuple(date.fromordinal(s.toordinal() + 1) for s in self._underlying))
		return super().add(other)

	def eomonth(self):
		return self
