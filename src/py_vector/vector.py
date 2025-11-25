from __future__ import annotations
from typing import Any
from typing import Iterable
from datetime import date
from datetime import datetime

from . import config
from .alias_tracker import _ALIAS_TRACKER
from .alias_tracker import AliasError
from .backends.tuple import TupleBackend
from .backends.table import TableBackend
from .errors import PyVectorIndexError
from .typing import DataType
from .typing import infer_dtype

class PyVector:
    """
    The Public Shell. 
    Delegates storage and math to self._impl (the Backend).
    """
    # Default backend if none provided
    _backend_class = config.get_backend()

    def __new__(cls, data=None, dtype=None, name=None, _impl=None):
        """
        Factory method: Auto-promotes to typed subclasses (_PyInt, _PyDate)
        unless we are explicitly instantiating a subclass.
        """
        # If we are already inside a subclass (e.g. _PyInt(...)), just create it
        if cls is not PyVector:
            return super().__new__(cls)

        # If wrapping an existing implementation, infer from its dtype
        if _impl is not None:
            inferred_dtype = _impl._dtype
        else:
            # Temporary inference to decide which class to instantiate
            # (The backend will do full inference again, but we need the class now)
            # Optimization: could peek at first element?
            if dtype is None and data:
                inferred_dtype = infer_dtype(data)
            elif isinstance(dtype, DataType):
                inferred_dtype = dtype
            else:
                inferred_dtype = DataType(dtype) if dtype else None

        # Dispatch to subclass
        if inferred_dtype:
            if inferred_dtype.kind == int:
                return super().__new__(_PyInt)
            if inferred_dtype.kind == float:
                return super().__new__(_PyFloat)
            if inferred_dtype.kind == bool:
                return super().__new__(_PyBool)
            if inferred_dtype.kind == str:
                return super().__new__(_PyString)
            if inferred_dtype.kind == date:
                return super().__new__(_PyDate)
            if inferred_dtype.kind == datetime:
                return super().__new__(_PyDateTime)
        
        return super().__new__(PyVector)

    def __init__(self, data=None, dtype=None, name=None, _impl=None):
        if _impl is not None:
            # wrapping an existing backend implementation
            self._impl = _impl
        else:
            backend_cls = config.get_backend()
            self._impl = backend_cls(data, dtype=dtype, name=name)

        _ALIAS_TRACKER.register(self, id(self._impl))

    # --------------------------------------------------------
    # Properties & Delegations
    # --------------------------------------------------------

    @property
    def name(self):
        """Public name of the vector."""
        return self._impl._name

    @name.setter
    def name(self, value):
        """Rename in-place (updates the backend)."""
        self._impl._name = value

    def __len__(self):
        return len(self._impl)

    def __iter__(self):
        return iter(self._impl)

    def __repr__(self):
        from .display import _printr  # or whatever your formatter is named
        return _printr(self)

    def schema(self):
        return self._impl._dtype

    def fingerprint(self):
        return self._impl.fingerprint()

    def copy(self, name=None):
        new = self.__class__(None, _impl=self._impl)
        new._impl._name = name  # always respect explicit None
        return new

    # --------------------------------------------------------
    # Indexing
    # --------------------------------------------------------

    def __getitem__(self, key):
            """
            Public indexing entrypoint.

            Normalizes PyVector keys → raw Python types before delegating to the backend.
            Wraps backend slice/mask results back into a PyVector shell.
            """

            # 1. Normalize PyVector keys (boolean mask / integer fancy index)
            #    v[mask_vec]  or v[idx_vec]
            if isinstance(key, PyVector):
                key = list(key)  # becomes list[bool] or list[int]

            # 2. Delegate to backend
            result = self._impl[key]

            # 3. If backend returns another backend (slice/mask/gather), wrap it
            #    TupleBackend is the only slicing backend for vectors.
            if isinstance(result, (TupleBackend, TableBackend)):
                return PyVector(None, _impl=result)

            # 4. Scalar → return as-is
            return result

    def __setitem__(self, key, value):
        from .alias_tracker import _ALIAS_TRACKER, AliasError
        from .errors import PyVectorValueError, PyVectorTypeError

        # 1. Enforce Copy-On-Write
        try:
            _ALIAS_TRACKER.check_writable(self, id(self._impl))
        except AliasError:
            self._impl = self.copy()._impl
            _ALIAS_TRACKER.register(self, id(self._impl))

        # 2. Resolve Indices
        n = len(self)
        indices = []
        
        if isinstance(key, int):
            indices = [key + n if key < 0 else key]

        elif isinstance(key, slice):
            indices = list(range(*key.indices(n)))

        elif isinstance(key, (list, tuple, PyVector)):
            if isinstance(key, PyVector):
                k_list = list(key)
            else:
                k_list = key
            
            if not k_list:
                return  # Nothing to set
                
            # Boolean Mask
            if isinstance(k_list[0], bool):
                if len(k_list) != n:
                    raise PyVectorValueError(f"Mask length {len(k_list)} != vector length {n}")
                indices = [i for i, x in enumerate(k_list) if x]

            # Integer Fancy Indexing
            elif isinstance(k_list[0], int):
                indices = [i + n if i < 0 else i for i in k_list]

            else:
                raise PyVectorTypeError(f"Invalid index type: {type(k_list[0])}")

        else:
            raise PyVectorTypeError(f"Unsupported index type: {type(key)}")

        if not indices:
            return

        # 3. Resolve Values (Broadcast or Zip)
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            vals = list(value)
            if len(vals) != len(indices):
                raise PyVectorValueError(f"Mismatch: assigning {len(vals)} values to {len(indices)} indices")
            updates = list(zip(indices, vals))
        else:
            updates = [(i, value) for i in indices]

        # 4. Apply
        self._impl.set_values(updates)


    # --------------------------------------------------------
    # Arithmetic (The "One Function Call" Overhead)
    # --------------------------------------------------------

    def __add__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.add(other_impl))

    def __sub__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.sub(other_impl))

    def __mul__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.mul(other_impl))

    def __truediv__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.truediv(other_impl))
    
    def __floordiv__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.floordiv(other_impl))
    
    def __mod__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.mod(other_impl))

    def __pow__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(None, _impl=self._impl.pow(other_impl))

    # Reverse
    def __radd__(self, other):
        return PyVector(None, _impl=self._impl.radd(other))
    def __rsub__(self, other):
        return PyVector(None, _impl=self._impl.rsub(other))
    def __rmul__(self, other):
        return PyVector(None, _impl=self._impl.rmul(other))
    def __rtruediv__(self, other):
        return PyVector(None, _impl=self._impl.rtruediv(other))
    def __rfloordiv__(self, other):
        return PyVector(None, _impl=self._impl.rfloordiv(other))
    def __rmod__(self, other):
        return PyVector(None, _impl=self._impl.rmod(other))
    def __rpow__(self, other):
        return PyVector(None, _impl=self._impl.rpow(other))

    # Comparison (Returns PyVector of bools)
    def __eq__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        # Backend returns list[bool], we wrap in PyVector
        return PyVector(self._impl.eq(other_impl))

    def __ne__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(self._impl.ne(other_impl))

    def __lt__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(self._impl.lt(other_impl))

    def __le__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(self._impl.le(other_impl))

    def __gt__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(self._impl.gt(other_impl))

    def __ge__(self, other):
        other_impl = other._impl if isinstance(other, PyVector) else other
        return PyVector(self._impl.ge(other_impl))

    # --------------------------------------------------------
    # API Methods
    # --------------------------------------------------------
    def unique(self):
        return set(self._impl._data)
        
    def fillna(self, value):
        return PyVector(None, _impl=self._impl.fillna(value))

    def size(self):
        """Returns shape tuple (length,) for consistency with PyTable."""
        return (len(self),)

    def isna(self):
        """Returns boolean vector indicating None values."""
        return PyVector(self._impl.isna())

    def dropna(self):
        """Returns a new vector with None values removed."""
        return PyVector(None, _impl=self._impl.dropna())

    def _promote(self, dtype):
        """Internal promotion logic required by type promotion tests."""
        self._impl._promote(dtype)
        
    def cast(self, dtype):
        return PyVector(None, _impl=self._impl.cast(dtype))

    def rename(self, name):
        """Rename the vector in-place."""
        self._impl._name = name  # Uses the setter we added earlier
        return self              # Return self to allow chaining: v.rename("x").copy()

    @property
    def T(self):
        """Transpose. For a 1D vector, this is an identity operation."""
        return self


    # Aggregates
    def sum(self): return self._impl.sum()
    def min(self): return self._impl.min()
    def max(self): return self._impl.max()
    def mean(self): return self._impl.mean()
    def stdev(self): return self._impl.stdev()
    def any(self): return self._impl.any()
    def all(self): return self._impl.all()

# ============================================================
# Typed Subclasses (The "Rich" Layer)
# ============================================================

class _PyInt(PyVector):
    def __getattr__(self, attr):
        # Proxy int attributes (e.g. .real, .imag, .bit_length())
        if not hasattr(int, attr):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

        # Distinguish between property (immediate) and method (callable)
        sample = getattr(int, attr)
        if callable(sample):
            def wrapper(*args, **kwargs):
                # Apply method to every element
                return PyVector([getattr(x, attr)(*args, **kwargs) if x is not None else None for x in self._impl.data])
            return wrapper
        else:
            # Apply property to every element
            return PyVector([getattr(x, attr) if x is not None else None for x in self._impl.data])

class _PyFloat(PyVector):
    def round(self, ndigits=0):
        # Explicit override for standard rounding
        data = self._impl.data
        return PyVector([round(x, ndigits) if x is not None else None for x in data])

    def __getattr__(self, attr):
        # Proxy float attributes (e.g. .is_integer(), .hex())
        if not hasattr(float, attr):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

        sample = getattr(float, attr)
        if callable(sample):
            def wrapper(*args, **kwargs):
                return PyVector([getattr(x, attr)(*args, **kwargs) if x is not None else None for x in self._impl.data])
            return wrapper
        else:
            return PyVector([getattr(x, attr) if x is not None else None for x in self._impl.data])

class _PyBool(PyVector):
    def __invert__(self):
        data = self._impl.data
        return PyVector([not x if x is not None else None for x in data])
        
    def __and__(self, other):
        if isinstance(other, PyVector):
            return PyVector([(a and b) for a, b in zip(self, other)])
        return super().__and__(other)
    
    def __or__(self, other):
        if isinstance(other, PyVector):
            return PyVector([(a or b) for a, b in zip(self, other)])
        return super().__or__(other)

class _PyString(PyVector):
    # Keep explicit methods if you want specific docstrings or optimizations
    # But __getattr__ handles the rest (replace, strip, split, etc.)
    
    def __getattr__(self, attr):
        if not hasattr(str, attr):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

        sample = getattr(str, attr)
        if callable(sample):
            def wrapper(*args, **kwargs):
                return PyVector([getattr(x, attr)(*args, **kwargs) if x is not None else None for x in self._impl.data])
            return wrapper
        else:
            return PyVector([getattr(x, attr) if x is not None else None for x in self._impl.data])

class _PyDate(PyVector):
    # Explicit properties for convenience (optional, but faster than getattr)
    @property
    def year(self):
        return PyVector([x.year if x else None for x in self._impl.data])
    
    @property
    def month(self):
        return PyVector([x.month if x else None for x in self._impl.data])
        
    @property
    def day(self):
        return PyVector([x.day if x else None for x in self._impl.data])

    def __getattr__(self, attr):
        # Handles .replace(), .isoformat(), etc.
        if not hasattr(date, attr):
             # Fallback: check if it might be a datetime method if we actually hold datetimes?
             # For strictness, raise error based on the declared type class.
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")
            
        sample = getattr(date, attr)
        if callable(sample):
            def wrapper(*args, **kwargs):
                return PyVector([getattr(x, attr)(*args, **kwargs) if x is not None else None for x in self._impl.data])
            return wrapper
        else:
            return PyVector([getattr(x, attr) if x is not None else None for x in self._impl.data])

class _PyDateTime(_PyDate):
    @property
    def hour(self):
        return PyVector([x.hour if x else None for x in self._impl.data])
        
    def __getattr__(self, attr):
        # Handles .time(), .dst(), etc.
        if not hasattr(datetime, attr):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")
            
        sample = getattr(datetime, attr)
        if callable(sample):
            def wrapper(*args, **kwargs):
                return PyVector([getattr(x, attr)(*args, **kwargs) if x is not None else None for x in self._impl.data])
            return wrapper
        else:
            return PyVector([getattr(x, attr) if x is not None else None for x in self._impl.data])