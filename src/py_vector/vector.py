# ======================================================================
# PyVector - User-facing vector class delegating to TupleVector backend
# ======================================================================

import warnings
from datetime import date, datetime

from .tuplevector import TupleVector, MethodProxy
from .typing import DataType, infer_dtype
from .errors import PyVectorTypeError, PyVectorValueError, PyVectorIndexError
from .display import _printr
from .alias_tracker import _ALIAS_TRACKER


class PyVector:
    """
    User-facing vector class with optional type safety.
    Delegates storage and operations to TupleVector backend.
    """

    def __new__(cls, initial=(), dtype=None, name=None, as_row=False, **kwargs):
        """
        Decide what type of PyVector to create based on contents.

        - If `initial` is a sequence of PyVector with equal length,
          we construct a PyTable instead.
        - Otherwise we dispatch to typed subclasses based on dtype.kind.
        """
        # Check if we're creating a PyTable (all elements are vectors of same length)
        if initial and all(isinstance(x, PyVector) for x in initial):
            lengths = {len(x) for x in initial}
            if len(lengths) == 1:
                from .table import PyTable
                return PyTable(initial=initial, dtype=dtype, name=name, as_row=as_row)
            warnings.warn(
                "Passing vectors of different length will not produce a PyTable.",
                UserWarning,
            )

        # Convert Python types to DataType if needed
        if dtype is not None and not isinstance(dtype, DataType):
            dtype = DataType(dtype)

        # Infer dtype if not provided
        if dtype is None and initial:
            dtype = infer_dtype(initial)

        # Dispatch to typed subclasses
        target_class = cls
        if dtype is not None:
            kind = dtype.kind
            if kind is str:
                target_class = _PyString
            elif kind is int:
                target_class = _PyInt
            elif kind is float:
                target_class = _PyFloat
            elif kind is date:
                target_class = _PyDate

        instance = super(PyVector, target_class).__new__(target_class)
        instance._dtype = dtype
        return instance

    def __init__(self, initial=(), dtype=None, name=None, as_row=False, **kwargs):
        """Initialize PyVector with TupleVector backend."""
        # Use dtype from __new__ if passed there
        if dtype is not None and not isinstance(dtype, DataType):
            dtype = DataType(dtype)
        if getattr(self, "_dtype", None) is None:
            self._dtype = dtype or (infer_dtype(initial) if initial else None)

        self._backend = TupleVector(initial=initial, dtype=self._dtype, name=name)
        self._name = name
        self._display_as_row = as_row

        # Register with alias tracker
        _ALIAS_TRACKER.register(self, id(self._backend._storage))

    # ======================================================
    # Compatibility properties
    # ======================================================

    @property
    def _underlying(self):
        """Compatibility property - returns backend storage."""
        return self._backend._storage

    # ======================================================
    # Schema and metadata
    # ======================================================

    def schema(self):
        return self._backend.schema()

    def fingerprint(self):
        return self._backend.fingerprint()

    def size(self):
        if len(self) == 0:
            return tuple()
        return (len(self),)

    # ======================================================
    # Basic protocol - delegate to backend
    # ======================================================

    def __len__(self):
        return len(self._backend)

    def __iter__(self):
        return iter(self._backend._storage)

    def __bool__(self):
        # For now, keep simple "non-empty => True" at wrapper level
        return len(self._backend._storage) > 0

    def __repr__(self):
        return _printr(self)

    # ======================================================
    # Indexing - read operations
    # ======================================================

    def __getitem__(self, key):
        # Single integer index
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if not (0 <= key < len(self)):
                raise PyVectorIndexError(f"Index {key} out of range for length {len(self)}")
            return self._backend[key]

        # Slice
        if isinstance(key, slice):
            backend_slice = self._backend[key]
            return self.copy(backend_slice._storage, name=self._name)

        # Boolean mask: PyVector[bool]
        if isinstance(key, PyVector) and key.schema() is not None and key.schema().kind is bool:
            if len(key) != len(self):
                raise PyVectorValueError("Boolean mask length must match vector length.")
            filtered = [val for val, keep in zip(self._backend._storage, key) if keep]
            return self.copy(filtered)

        # Boolean mask: list[bool]
        if isinstance(key, list) and all(isinstance(x, bool) for x in key):
            if len(key) != len(self):
                raise PyVectorValueError("Boolean mask length must match vector length.")
            filtered = [val for val, keep in zip(self._backend._storage, key) if keep]
            return self.copy(filtered)

        # Integer vector indexing: PyVector[int]
        if isinstance(key, PyVector) and key.schema() is not None and key.schema().kind is int:
            selected = [self._backend[i] for i in key]
            return self.copy(selected)

        # Integer vector indexing: list[int]
        if isinstance(key, list) and all(isinstance(x, int) for x in key):
            selected = [self._backend[i] for i in key]
            return self.copy(selected)

        raise PyVectorTypeError(f"Invalid index type: {type(key)}")

    # ======================================================
    # Indexing - write operations (copy-on-write)
    # ======================================================

    def __setitem__(self, key, value):
        """Copy-on-write mutation via backend._set_values."""
        _ALIAS_TRACKER.check_writable(self, id(self._backend._storage))

        updates = []

        # Single index
        if isinstance(key, int):
            idx = key
            if idx < 0:
                idx = len(self) + idx
            if not (0 <= idx < len(self)):
                raise PyVectorIndexError(f"Index {key} out of range for length {len(self)}")
            updates.append((idx, value))

        # Slice
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                values = list(value)
                if len(values) != len(indices):
                    raise PyVectorValueError(
                        "Slice assignment value length must match slice length."
                    )
                for idx, val in zip(indices, values):
                    updates.append((idx, val))
            else:
                for idx in indices:
                    updates.append((idx, value))

        # Boolean mask: PyVector[bool]
        elif isinstance(key, PyVector) and key.schema() is not None and key.schema().kind is bool:
            if len(key) != len(self):
                raise PyVectorValueError("Boolean mask length must match vector length.")
            indices = [i for i, keep in enumerate(key) if keep]
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                values = list(value)
                if len(values) != len(indices):
                    raise PyVectorValueError(
                        "Assigned values length must match number of True entries in mask."
                    )
                for idx, val in zip(indices, values):
                    updates.append((idx, val))
            else:
                for idx in indices:
                    updates.append((idx, value))

        # Boolean mask: list[bool]
        elif isinstance(key, list) and all(isinstance(x, bool) for x in key):
            if len(key) != len(self):
                raise PyVectorValueError("Boolean mask length must match vector length.")
            indices = [i for i, keep in enumerate(key) if keep]
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                values = list(value)
                if len(values) != len(indices):
                    raise PyVectorValueError(
                        "Assigned values length must match number of True entries in mask."
                    )
                for idx, val in zip(indices, values):
                    updates.append((idx, val))
            else:
                for idx in indices:
                    updates.append((idx, value))

        # Integer vector: PyVector[int]
        elif isinstance(key, PyVector) and key.schema() is not None and key.schema().kind is int:
            idx_list = list(key)
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                values = list(value)
                if len(values) != len(idx_list):
                    raise PyVectorValueError(
                        "Assigned values length must match index vector length."
                    )
                for idx, val in zip(idx_list, values):
                    updates.append((idx, val))
            else:
                for idx in idx_list:
                    updates.append((idx, value))

        # Integer vector: list[int]
        elif isinstance(key, list) and all(isinstance(x, int) for x in key):
            idx_list = key
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                values = list(value)
                if len(values) != len(idx_list):
                    raise PyVectorValueError(
                        "Assigned values length must match index list length."
                    )
                for idx, val in zip(idx_list, values):
                    updates.append((idx, val))
            else:
                for idx in idx_list:
                    updates.append((idx, value))

        else:
            raise PyVectorTypeError(f"Invalid index type for assignment: {type(key)}")

        # Apply updates via backend
        old_id = id(self._backend._storage)
        self._backend._set_values(updates)

        # Update alias tracker
        _ALIAS_TRACKER.unregister(self, old_id)
        _ALIAS_TRACKER.register(self, id(self._backend._storage))

    # ======================================================
    # Copy and utility methods
    # ======================================================

    def copy(self, new_values=None, name=...):
        """Create a copy with optional new values and name."""
        use_name = self._name if name is ... else name
        values = list(new_values) if new_values is not None else list(self._backend._storage)
        return PyVector(
            values,
            dtype=self._backend._dtype,
            name=use_name,
            as_row=self._display_as_row,
        )

    def rename(self, new_name):
        """Rename this vector (returns self for chaining)."""
        self._name = new_name
        self._backend._name = new_name
        return self

    # ======================================================
    # Arithmetic operations - delegate to backend
    # ======================================================

    def __add__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend + backend_other
        return self._wrap_result(result)

    def __radd__(self, other):
        result = self._backend.__radd__(other)
        return self._wrap_result(result)

    def __sub__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend - backend_other
        return self._wrap_result(result)

    def __rsub__(self, other):
        result = self._backend.__rsub__(other)
        return self._wrap_result(result)

    def __mul__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend * backend_other
        return self._wrap_result(result)

    def __rmul__(self, other):
        result = self._backend.__rmul__(other)
        return self._wrap_result(result)

    def __truediv__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend / backend_other
        return self._wrap_result(result)

    def __rtruediv__(self, other):
        result = self._backend.__rtruediv__(other)
        return self._wrap_result(result)

    def __floordiv__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend // backend_other
        return self._wrap_result(result)

    def __rfloordiv__(self, other):
        result = self._backend.__rfloordiv__(other)
        return self._wrap_result(result)

    def __mod__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend % backend_other
        return self._wrap_result(result)

    def __rmod__(self, other):
        result = self._backend.__rmod__(other)
        return self._wrap_result(result)

    def __pow__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend ** backend_other
        return self._wrap_result(result)

    def __rpow__(self, other):
        result = self._backend.__rpow__(other)
        return self._wrap_result(result)

    # ======================================================
    # Comparison operations
    # ======================================================

    def __eq__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend == backend_other
        return self._wrap_result(result)

    def __ne__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend != backend_other
        return self._wrap_result(result)

    def __lt__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend < backend_other
        return self._wrap_result(result)

    def __le__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend <= backend_other
        return self._wrap_result(result)

    def __gt__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend > backend_other
        return self._wrap_result(result)

    def __ge__(self, other):
        backend_other = other._backend if isinstance(other, PyVector) else other
        result = self._backend >= backend_other
        return self._wrap_result(result)

    # ======================================================
    # Aggregations
    # ======================================================

    def sum(self):
        return self._backend.sum()

    def min(self):
        return self._backend.min()

    def max(self):
        return self._backend.max()

    def mean(self):
        return self._backend.mean()

    def all(self):
        return self._backend.all()

    def any(self):
        return self._backend.any()

    def stdev(self, population: bool = False):
        """Calculate standard deviation (population=False → sample stdev)."""
        non_none = [v for v in self._backend._storage if v is not None]
        if len(non_none) < 2:
            return None
        m = sum(non_none) / len(non_none)
        num = sum((x - m) * (x - m) for x in non_none)
        denom = len(non_none) if population else (len(non_none) - 1)
        if denom == 0:
            return None
        return (num / denom) ** 0.5

    # ======================================================
    # Utility methods
    # ======================================================

    def cast(self, target_type):
        result = self._backend.cast(target_type)
        return self._wrap_result(result)

    def fillna(self, value):
        result = self._backend.fillna(value)
        return self._wrap_result(result)

    def dropna(self):
        filtered = [x for x in self._backend._storage if x is not None]
        return self.copy(filtered)

    def isna(self):
        result = [x is None for x in self._backend._storage]
        return PyVector(result, dtype=DataType(bool))

    def unique(self):
        result = self._backend.unique()
        return self._wrap_result(result)

    # ======================================================
    # Helper to wrap TupleVector results back to PyVector
    # ======================================================

    def _wrap_result(self, backend_result):
        """Wrap a TupleVector result back into PyVector."""
        if isinstance(backend_result, TupleVector):
            return PyVector(
                backend_result._storage,
                dtype=backend_result._dtype,
                name=backend_result._name,
            )
        return backend_result

    # ======================================================
    # Method proxy for element-wise methods (e.g. .upper(), .strip())
    # ======================================================

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        proxy = MethodProxy(self._backend, name)

        def wrapped_call(*args, **kwargs):
            result = proxy(*args, **kwargs)
            return self._wrap_result(result)

        return wrapped_call


# ======================================================
# Typed subclasses (placeholders for specialization)
# ======================================================

class _PyString(PyVector):
    """Reserved for string-specific behaviors."""
    pass


class _PyInt(PyVector):
    """Reserved for int-specific behaviors."""
    pass


class _PyFloat(PyVector):
    """Reserved for float-specific behaviors."""
    pass


class _PyDate(PyVector):
    """Reserved for date-specific behaviors."""
    pass
