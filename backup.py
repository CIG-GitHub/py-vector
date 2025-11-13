class Scott():
	_underlying = None
	index = 0
	stuff = 'stuff'

	def __init__(self, values=None, names=None):
		self._underlying = values
		self._colnames = names
		self.index = 0


	def __dir__(self):
		return dir(Scott) + [c for c in self._colnames]

	def __getattr__(self, attr):
		if attr in self._colnames:
			idx = self._colnames.index(attr)
			return self._underlying[idx]

	def __repr__(self):
		return 'Scott(' + repr(self._underlying) + ',' + repr(self._colnames) + ')'



class Ryan():
	_underlying = None
	def __init__(self, value=0):
		self._underlying = value

	def increment(self):
		self._underlying += 1


	def __repr__(self):
		return 'Ryan(' + repr(self._underlying) + ')'

	def __add__(self, value):
		return Ryan(self._underlying + value)


from py_vector import PyVector
def _format_col(col, name_label=None, num_rows = 5):
	""" return a PyVector of formatted values """
	und = col._underlying if len(col)<=num_rows*2 else col._underlying[:num_rows] + col._underlying[-num_rows-1:]
	if col._dtype == float:
		x = PyVector([f"{v: g}" for v in und])
	elif col._dtype == int:
		x = PyVector([f"{v: d}" for v in und])
	elif col._dtype == str:
		x = PyVector([f"'{v}'" for v in und])
	else:
		x = PyVector([f"{repr(v)}" for v in und])
	max_len = max({len(v) for v in x})
	if name_label is not None:
		#print(f'started here [{name_label}]')
		name_label = f"'{name_label}'"
		if len(name_label) < max_len:
			#print(f'adjusting [{name_label}] to')
			name_label = name_label.ljust((max_len + len(name_label))//2).rjust(max_len)
			#print(f'this thing: [{name_label}] .')
		max_len = max(len(name_label), max_len)
		x = PyVector([name_label]) << x
	return x.rjust(max_len)

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


def format_footer(pv):
	base = f"\n) # {'x'.join([str(s) for s in pv.size()])}"
	identifier = ' table' if len(pv.size()) > 1 else ' element array'
	dataclass = f" <{pv._dtype.__name__}>" if pv._dtype else ''
	return base + identifier + dataclass

def printr(pv, outer=True, opener='', closer='', joiner='', warning=''):
	if outer:
		header = f'PyVector(name={repr(pv._name)},\n' if pv._name else 'PyVector(\n'
		opener = 'rows= ['
		col_headers = _recursive_colnames(pv)
		closer = ']'
		footer = format_footer(pv)

	# Still a lot of work to do for multi-dimensional arrays
	if len(pv.size()) == 2:
		out = '['
		if len(pv) == 0:
			return '[[]]'
		elif len(pv.cols()) <= 10:
			for x, y in zip(pv.cols()[:-1], col_headers[:-1], strict=True):
				# the name and element width are handled together.
				out += _format_col(x, name_label = y) + ', '
			out += _format_col(pv.cols()[-1], name_label=col_headers[-1]) + '],'
			out[-1] = out[-1][:-1] # remove the final comma
		else:
			cols = pv.cols()
			for x, y in zip(cols[:4], col_headers[:4], strict=True):
				out += _format_col(x) + ', '
			out += _format_col(cols[4], col_headers[4], strict=True)
			out += ' ... '
			for x, y in zip(cols[-5:-1], col_headers[-5:-1]):
				out += _format_col(x, y) + ', '
			out += _format_col(cols[-1], col_headers[-1]) + '],'
			out[-1] = out[-1][:-1] # remove the final comma
	if len(pv.size()) == 1:
		out = _format_col(pv, name_label = '') + ','
		out[-1] = out[-1][:-1] # remove the final comma

	if any(col_headers):
		return header  + 'cols=  (' + out[0][1:-2] + '),\n' + opener + '\n       '.join(out[1:]) + closer + footer

	return header + opener + '\n       '.join(out[1:]) + closer + footer
"""
	if len(pv.size()) > 5:
		opener = '['
		closer = ']'
	elif len(pv.size()) == 5:
		warning = '\n # repr will only show the 4 innermost dimensions.'
		content = printr(pv[0]) + warning
	elif len(pv.size()) == 4:
		joiner = f"\n,"
		content = 
	elif len(pv.size()) == 3:
		joiner = f"\n,"
	elif len(pv.size()) == 2:
		content = 
"""


	"""


			if isinstance(key, list) and {type(e) for e in key} == {bool}:
				assert (len(self) == len(key))
				return PyVector([x for x, y in zip(self, key, strict=True) if y],
					default_element = self._default,
					dtype = self._dtype,
					typesafe = self._typesafe
				)
			if isinstance(key, slice):
				return PyVector(self._underlying[key], 
					default_element = self._default,
					dtype = self._dtype,
					typesafe = self._typesafe
				)
			if isinstance(key, int):
				# Effectively a different input type (single not a list). Returning a value, not a vector.
				return self._underlying[key]

			# NOT RECOMMENDED
			if isinstance(key, PyVector) and key._dtype == int and key._typesafe:
				if len(self) > 1000:
					warnings.warn('Subscript indexing is sub-optimal for large vectors')
				return PyVector([self[x] for x in key],
					default_element = self._default,
					dtype = self._dtype,
					typesafe = self._typesafe
				)

			# NOT RECOMMENDED
			if isinstance(key, (list, tuple)) and {type(e) for e in key} == {int}:
				if len(self) > 1000:
					warnings.warn('Subscript indexing is sub-optimal for large vectors')
				return PyVector([self[x] for x in key],
					default_element = self._default,
					dtype = self._dtype,
					typesafe = self._typesafe
				)
			raise TypeError(f'Vector indices must be boolean vectors, integer vectors or integers, not {str(type(key))}')
	"""
"""



	def __eq__(self, other):
		other = self._check_duplicate(other)
		if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
			return PyVector([x == other for x in self], False, bool, True)
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x == y for x, y in zip(self, other, strict=True)], False, bool, True)
		raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")


	def __ge__(self, other):
		other = self._check_duplicate(other)
		if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
			return PyVector([x >= other for x in self], False, bool, True)
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x >= y for x, y in zip(self, other, strict=True)], False, bool, True)
		raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")


	def __gt__(self, other):
		other = self._check_duplicate(other)
		if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
			return PyVector([x > other for x in self], False, bool, True)
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x > y for x, y in zip(self, other, strict=True)], False, bool, True)
		raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")


	def __le__(self, other):
		other = self._check_duplicate(other)
		#if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x <= y for x, y in zip(self, other, strict=True)], False, bool, True)

		return PyVector([x <= other for x in self], False, bool, True)
		#raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")


	def __lt__(self, other):
		other = self._check_duplicate(other)
		if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
			return PyVector([x < other for x in self], False, bool, True)
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x < y for x, y in zip(self, other, strict=True)], False, bool, True)
		raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")


	def __ne__(self, other):
		other = self._check_duplicate(other)
		if isinstance(other, self._dtype) or (isinstance(other, int) and self._dtype == float):
			return PyVector([x != other for x in self], False, bool, True)
		if isinstance(other, PyVector) or hasattr(other, '__iter__'):
			# Raise mismatched lengths
			assert len(self) == len(other)
			return PyVector([x != y for x, y in zip(self, other, strict=True)], False, bool, True)
		raise TypeError(f"Unable to compare {type(self)} with {type(other)}.")




	#def __next__(self):
	#	self._index = self._index or 0
	#	if self._index >= len(self._underlying):
	#		raise StopIteration
	#	value = self._underlying[self._index]
	#	self._index += 1
	#	return value
"""


#	def __setitem__(self, key, value):
#		""" geez """
#		key = self._check_duplicate(key)
#		value = self._check_duplicate(value)
#		if (isinstance(key, PyVector) and key._dtype == bool and key._typesafe) \
#			or (isinstance(key, list) and {type(e) for e in key} == {bool}):
#			assert (len(self) == len(key))
#			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytesarray)):
#				# We're in absurd territory, but if you assign based on logical index and pass
#				# a vector of values in, the vector has to be the same length as the number of 
#				# "True"s in the mask
#				assert sum(key) == len(value)
#
#				iter_v = iter(value)
#				# allow PyVector to take care of typesafety mismatches and up-scaling.
#				new_vector = PyVector(
#					tuple(next(iter_v) if t else x for x, t in zip(self._underlying, key)),
#					default_element = self._default,
#					dtype = self._dtype,
#					typesafe = self._typesafe
#				)
#			else:
#				new_vector = PyVector(
#					tuple(value if t else x for x, t in zip(self._underlying, key)),
#					default_element = self._default,
#					dtype = self._dtype,
#					typesafe = self._typesafe
#				)
#			self._underlying = new_vector._underlying
#			self._dtype = new_vector._dtype
#			return
#		if isinstance(key, slice):
#			if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytesarray)):
#				# Slice assignment will be forced to either of exactly the correct number of indices
#				# or individual values. Regular slice assigment rules are stupid.
#				if  slice_length(key, len(self)) != len(value):
#					raise ValueError(f'Input slice and assignment values must be of the same length.')
#
#			tuple_as_list = list(self._underlying)
#			tuple_as_list[key] = value
#			# allow PyVector to take care of typesafety mismatches and up-scaling.
#			new_vector = PyVector(tuple_as_list,
#				default_element = self._default,
#				dtype = self._dtype,
#				typesafe = self._typesafe
#			)
#			self._underlying = new_vector._underlying
#			self._dtype = new_vector._dtype
#			return
#		raise TypeError(f'Vector indices must be boolean vectors, integer vectors or integers, not {str(type(key))}')




#	def __add__(self, other):
#		other = self._check_duplicate(other)
#		#print(f"{self._name} {type(other)} {other}")
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x + y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			#print(f'we have entered here {type(other)}')
#			return PyVector([x + other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__add__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '+': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __radd__(self, other):
#		return self.__add__(other)
#
#	def __mod__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x % other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x % y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__mod__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '%': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __floordiv__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x // other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x // y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__floordiv__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '//': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#
#	def __rfloordiv__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([other // x for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([y // x for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__rfloordiv__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '//': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __truediv__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x / other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x / y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__truediv__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '/': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __rtruediv__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([other / x for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([y / x for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__rtruediv__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '/': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __mul__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x * other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x * y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__mul__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '*': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __rmul__(self, other):
#		return self.__mul__(other)
#
#
#	def __pow__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x ** other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype, 
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x ** y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__pow__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '**': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#
#	def __sub__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([x - other for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype, 
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([x - y for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__sub__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '-': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#
#	def __rsub__(self, other):
#		other = self._check_duplicate(other)
#		if isinstance(other, self._dtype) or self._check_native_typesafe(other):
#			return PyVector([other - x for x in self],
#				self._default + other if self._default is not None else None,
#				self._dtype,  
#				self._typesafe)
#		if isinstance(other, PyVector):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			if self._typesafe:
#				other._promote(self._dtype)
#			return PyVector([y - x for x, y in zip(self, other, strict=True)],
#				default_element = self._default + other._default if self._default is not None and other._default is not None else None,
#				dtype = self._dtype if self._typesafe else None,
#				typesafe = self._typesafe)
#		if hasattr(other, '__iter__'):
#			# Raise mismatched lengths
#			assert len(self) == len(other)
#			return self.__rsub__(PyVector([x for x in other], self._default, self._dtype, self._typesafe))
#
#		raise TypeError(f"Unsupported operand type(s) for '-': '{self._dtype.__name__}' and '{type(other).__name__}'.")
#